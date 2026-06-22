"""D.2.2 tests — synthesizer merge primitive."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from meta_model.config import MoaProfile, UpstreamConfig
from meta_model.moa.fanout import GeneratorFailure, GeneratorSuccess
from meta_model.moa.synthesizer import (
    SynthesisFailure,
    SynthesizedResponse,
    synthesize,
    _candidate_signature,
    _clone_msg_with_finish,
    _coerce_reasoning_into_content,
    _draft_signature,
    _reasoning_fallback_text,
    _render_candidate,
)

SYNTH_UP = UpstreamConfig(
    model_id="synth-model",
    base_url="http://synthesizer:9999/v1",
    context=8192,
    max_output=2048,
)


def _profile(
    *,
    mode: str = "merge",
    fastpath: bool = True,
    generators: list[str] | None = None,
) -> MoaProfile:
    return MoaProfile(
        type="moa",
        generators=generators or ["a", "b", "c"],
        synthesizer="a",
        synthesis_mode=mode,
        fastpath_on_agreement=fastpath,
    )


def _success(
    name: str, content: str, tool_calls: list[dict[str, Any]] | None = None
) -> GeneratorSuccess:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return GeneratorSuccess(
        upstream_name=name,
        response={
            "id": f"chatcmpl-{name}",
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        },
        elapsed_ms=42,
    )


def _failure(name: str) -> GeneratorFailure:
    return GeneratorFailure(
        upstream_name=name,
        reason="non_2xx",
        detail="upstream returned 503",
        status=503,
        elapsed_ms=12,
    )


# ── Failure / no-quorum ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_synthesize_no_successes_returns_failure() -> None:
    out = await synthesize(
        _profile(),
        [_failure("a"), _failure("b")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
    )
    assert isinstance(out, SynthesisFailure)
    assert out.reason == "no_quorum"


# ── Single-success fast path ──────────────────────────────────────


@pytest.mark.asyncio
async def test_synthesize_single_success_returns_directly() -> None:
    """1 success → return verbatim, no synth call."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("synthesizer should NOT be called on single-success")

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "hello"), _failure("b"), _failure("c")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.fastpath is True
    assert out.fallback_reason == "single_success"
    assert out.response["choices"][0]["message"]["content"] == "hello"


# ── Agreement fast path ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_synthesize_identical_content_skips_synth() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("synthesizer should NOT be called on agreement")

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=True),
        [_success("a", "answer"), _success("b", "answer"), _success("c", "answer")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.fastpath is True
    assert out.fallback_reason == "none"
    assert out.quorum == 3


@pytest.mark.asyncio
async def test_synthesize_normalizes_whitespace_for_agreement() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("synthesizer should NOT be called")

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=True),
        [
            _success("a", "hello world"),
            _success("b", "hello   world  "),
            _success("c", "hello\nworld"),
        ],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.fastpath is True


@pytest.mark.asyncio
async def test_synthesize_fastpath_disabled_runs_synth_even_on_agreement() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["called"] = True
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "synthed"}}],
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=False),
        [_success("a", "agreed"), _success("b", "agreed")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert captured.get("called") is True
    assert isinstance(out, SynthesizedResponse)
    assert out.fastpath is False
    assert out.response["choices"][0]["message"]["content"] == "synthed"


# ── Real synth (merge) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_synthesize_merge_mode_calls_synthesizer() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "synth-1",
                "choices": [{"message": {"role": "assistant", "content": "merged"}}],
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(mode="merge"),
        [_success("a", "answer A"), _success("b", "answer B"), _success("c", "answer C")],
        SYNTH_UP,
        {"messages": [{"role": "user", "content": "what is X?"}]},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.fastpath is False
    assert out.fallback_reason == "none"
    assert out.quorum == 3
    # Synthesizer received system+user, with all 3 candidates listed
    sys_msg = captured["body"]["messages"][0]
    user_msg = captured["body"]["messages"][1]
    assert sys_msg["role"] == "system"
    assert "synthesizing" in sys_msg["content"].lower()
    assert "what is X?" in user_msg["content"]
    assert "candidate 1" in user_msg["content"]
    assert "candidate 3" in user_msg["content"]
    assert "answer A" in user_msg["content"]
    assert "answer C" in user_msg["content"]


@pytest.mark.asyncio
async def test_synthesize_best_of_returns_chosen_candidate_verbatim() -> None:
    """best-of: synth picks an INDEX, server returns that candidate
    verbatim (review r12 — synth is judge, not author)."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        # Synth picks candidate #2
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "2"}}]},
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(mode="best-of"),
        [_success("a", "answer-A"), _success("b", "answer-B"), _success("c", "answer-C")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.response["choices"][0]["message"]["content"] == "answer-B"
    assert out.fastpath is False
    sys_msg = captured["body"]["messages"][0]
    # Best-of prompt asks for an index, not for the answer
    assert "index" in sys_msg["content"].lower()


@pytest.mark.asyncio
async def test_synthesize_best_of_unparseable_index_falls_back() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "I prefer the second"}}]
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(mode="best-of"),
        [_success("a", "first"), _success("b", "second")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    # "second" contains no leading integer; fallback to first candidate
    # (the prose "second" doesn't parse as a 1-based index — the
    # synth is supposed to emit an integer, not English).
    # Actually wait, "I prefer the second" contains no integer chars
    # at all on the first line — index parser returns None → fallback.
    assert isinstance(out, SynthesizedResponse)
    assert out.fallback_reason == "synth_failed_picked_primary"
    assert out.response["choices"][0]["message"]["content"] == "first"


@pytest.mark.asyncio
async def test_synthesize_best_of_out_of_range_index_falls_back() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "9"}}]},
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(mode="best-of"),
        [_success("a", "first"), _success("b", "second")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.fallback_reason == "synth_failed_picked_primary"


# ── Runaway-output guard (incident 2026-05-02) ─────────────────────


@pytest.mark.asyncio
async def test_synthesize_defaults_max_tokens_to_upstream_max_output() -> None:
    """When caller omits max_tokens, synth body must include the upstream's
    max_output as a defensive cap. Without this an unbounded synth call
    can run for the full request_timeout_secs (incident 2026-05-02)."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(mode="merge"),
        [_success("a", "first"), _success("b", "second")],
        SYNTH_UP,
        {"messages": []},  # no max_tokens, no max_completion_tokens
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert captured["body"].get("max_tokens") == SYNTH_UP.max_output
    assert "max_completion_tokens" not in captured["body"]


@pytest.mark.asyncio
async def test_synthesize_caller_max_tokens_wins_over_default() -> None:
    """Inherited max_tokens must NOT be replaced by the upstream default."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(mode="merge"),
        [_success("a", "first"), _success("b", "second")],
        SYNTH_UP,
        {"messages": [], "max_tokens": 64},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert captured["body"].get("max_tokens") == 64


@pytest.mark.asyncio
async def test_synthesize_caller_max_completion_tokens_suppresses_default() -> None:
    """If caller bounded the call via max_completion_tokens (the modern
    OpenAI field), the default max_tokens=max_output must NOT be added."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(mode="merge"),
        [_success("a", "first"), _success("b", "second")],
        SYNTH_UP,
        {"messages": [], "max_completion_tokens": 64},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert captured["body"].get("max_completion_tokens") == 64
    assert "max_tokens" not in captured["body"]


@pytest.mark.asyncio
async def test_synthesize_synth_call_uses_timeout_minus_margin() -> None:
    """Synth call timeout must be < inbound timeout so the fallback path
    has time to return BEFORE httpx ReadTimeout fires from the caller."""
    captured = {}

    class _RecordingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            captured["timeout"] = request.extensions.get("timeout")
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                },
            )

    out = await synthesize(
        _profile(mode="merge"),
        [_success("a", "first"), _success("b", "second")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=600.0,
        transport=_RecordingTransport(),
    )
    assert isinstance(out, SynthesizedResponse)
    timeout = captured["timeout"]
    # httpx records {connect, read, write, pool} timeouts. The READ leg
    # is the per-request deadline forwarded by forward_chat_completion.
    read_timeout = timeout["read"] if isinstance(timeout, dict) else timeout
    assert read_timeout is not None
    assert read_timeout <= 540.0  # 600 - 60 margin
    assert read_timeout >= 300.0  # not floored below half


@pytest.mark.asyncio
async def test_synthesize_synth_call_floors_timeout_at_half() -> None:
    """For tiny inbound timeouts, synth_timeout floors at timeout/2 so we
    don't drive it negative or to zero."""
    captured = {}

    class _RecordingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(
            self, request: httpx.Request
        ) -> httpx.Response:
            captured["timeout"] = request.extensions.get("timeout")
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                },
            )

    out = await synthesize(
        _profile(mode="merge"),
        [_success("a", "first"), _success("b", "second")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=10.0,
        synth_min_viable_secs=0.0,
        transport=_RecordingTransport(),
    )
    assert isinstance(out, SynthesizedResponse)
    timeout = captured["timeout"]
    read_timeout = timeout["read"] if isinstance(timeout, dict) else timeout
    assert read_timeout == 5.0  # floored at timeout/2 since 10-60 < 5


# ── Synth failures fall back to first candidate ──────────────────


@pytest.mark.asyncio
async def test_synthesize_synth_5xx_falls_back_to_first_candidate() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "first"), _success("b", "second")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.fastpath is False
    assert out.fallback_reason == "synth_failed_picked_primary"
    assert out.response["choices"][0]["message"]["content"] == "first"


@pytest.mark.asyncio
async def test_synthesize_synth_transport_error_falls_back() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dial failed", request=request)

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "first"), _success("b", "second")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.fallback_reason == "synth_failed_picked_primary"


@pytest.mark.asyncio
async def test_synthesize_synth_failure_log_includes_elapsed_ms(caplog) -> None:
    """The synth-failed WARN must report elapsed_ms so operators can
    distinguish 'timed out at the deadline' from 'errored fast'.
    Before this landed, the WARN line was timing-blind, which made it
    hard to reconcile the metrics-side `elapsed_ms_avg` field with
    individual fallback events in the log."""
    import logging as _logging

    caplog.set_level(_logging.WARNING, logger="meta_model.moa.synthesizer")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "first"), _success("b", "second")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.fallback_reason == "synth_failed_picked_primary"

    matching = [
        rec for rec in caplog.records
        if "synth failed" in rec.getMessage()
        and "elapsed_ms=" in rec.getMessage()
    ]
    assert matching, (
        f"expected a synth-failed WARN with elapsed_ms=, got messages: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_synthesize_synth_non_json_falls_back() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "first"), _success("b", "second")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.fallback_reason == "synth_failed_picked_primary"


@pytest.mark.asyncio
async def test_synthesize_synth_returns_malformed_shape_falls_back() -> None:
    """Synth 200 with no choices[0].message → fall back to first candidate."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "x", "choices": []})

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "first"), _success("b", "second")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.fallback_reason == "synth_failed_picked_primary"


# ── Tool-call agreement fast path ────────────────────────────────


@pytest.mark.asyncio
async def test_synthesize_identical_tool_calls_fastpaths() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("synthesizer should NOT be called")

    tc = [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [
            _success("a", "", tool_calls=tc),
            _success("b", "", tool_calls=tc),
            _success("c", "", tool_calls=tc),
        ],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.fastpath is True


@pytest.mark.asyncio
async def test_synthesize_propagates_contract_fields() -> None:
    """response_format / stop / max_tokens propagate to the synth call;
    sampling-policy fields (n, stream, temperature override) do not
    (review r12)."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    transport = httpx.MockTransport(handler)
    await synthesize(
        _profile(mode="merge"),
        [_success("a", "x"), _success("b", "y")],
        SYNTH_UP,
        {
            "messages": [{"role": "user", "content": "q"}],
            "response_format": {"type": "json_object"},
            "stop": ["END"],
            "max_tokens": 200,
            "n": 5,  # must NOT propagate
            "stream": True,  # must NOT propagate as-is
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    body = captured["body"]
    assert body["response_format"] == {"type": "json_object"}
    assert body["stop"] == ["END"]
    assert body["max_tokens"] == 200
    assert "n" not in body
    assert body["stream"] is False  # forced false on synth call


@pytest.mark.asyncio
async def test_synthesize_tool_calls_appear_in_synth_prompt_with_empty_content() -> None:
    """OpenAI tool-call responses commonly emit content="" alongside
    tool_calls. Synth prompt must show those tool_calls (review r12)."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "merged"}}]},
        )

    transport = httpx.MockTransport(handler)
    tc_a = [{"id": "1", "type": "function", "function": {"name": "fa", "arguments": "{}"}}]
    tc_b = [{"id": "2", "type": "function", "function": {"name": "fb", "arguments": "{}"}}]
    await synthesize(
        _profile(mode="merge"),
        [_success("a", "", tool_calls=tc_a), _success("b", "", tool_calls=tc_b)],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    user_msg = captured["body"]["messages"][1]["content"]
    assert "fa" in user_msg
    assert "fb" in user_msg


@pytest.mark.asyncio
async def test_synthesize_different_tool_calls_runs_synth() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["called"] = True
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "decided"}}]},
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [
            _success(
                "a",
                "",
                tool_calls=[
                    {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            ),
            _success(
                "b",
                "",
                tool_calls=[
                    {"id": "2", "type": "function", "function": {"name": "g", "arguments": "{}"}}
                ],
            ),
        ],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert captured.get("called") is True
    assert isinstance(out, SynthesizedResponse)


# ── D.3.1 — tool-aware synth, finalize integration ─────────────────


def _tool_def(name: str = "f") -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "parameters": {}}}


def _tc(name: str, args: str = "{}", call_id: str = "1") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


@pytest.mark.asyncio
async def test_d31_tool_aware_merge_keeps_tools_in_synth_body() -> None:
    """When candidates emit tool_calls, the synth body must keep
    tools/tool_choice/parallel_tool_calls so the synth can re-emit
    a constraint-satisfying call."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "", "tool_calls": [_tc("f")]},
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [
            _success("a", "", tool_calls=[_tc("f", '{"x": 1}', "id-a")]),
            _success("b", "", tool_calls=[_tc("g", "{}", "id-b")]),
        ],
        SYNTH_UP,
        {
            "messages": [{"role": "user", "content": "do it"}],
            "tools": [_tool_def("f"), _tool_def("g")],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    body = captured["body"]
    assert body.get("tools") == [_tool_def("f"), _tool_def("g")]
    assert body.get("tool_choice") == "auto"
    assert body.get("parallel_tool_calls") is True


@pytest.mark.asyncio
async def test_d31_text_only_merge_strips_tools() -> None:
    """No tool arbitration → existing text-only merge prompt + body
    (no tools forwarded)."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "merged"}}]}
        )

    transport = httpx.MockTransport(handler)
    await synthesize(
        _profile(),
        [_success("a", "x"), _success("b", "y")],
        SYNTH_UP,
        {"messages": [{"role": "user", "content": "q"}]},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert "tools" not in captured["body"]
    assert "tool_choice" not in captured["body"]


@pytest.mark.asyncio
async def test_d31_required_pre_synth_violation_repairs_via_synth() -> None:
    """Single-success fast-path with text-only candidate under
    tool_choice='required' must NOT 502 immediately. Synth gets a
    repair chance and emits a satisfying tool_call."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["called"] = True
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "", "tool_calls": [_tc("f")]},
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "no tools here")],
        SYNTH_UP,
        {
            "messages": [],
            "tools": [_tool_def("f")],
            "tool_choice": "required",
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert captured.get("called") is True
    assert isinstance(out, SynthesizedResponse)
    msg = out.response["choices"][0]["message"]
    assert msg.get("tool_calls")


@pytest.mark.asyncio
async def test_d31_required_synth_also_fails_returns_502_failure() -> None:
    """If synth ALSO returns no tool_calls and no candidate has any,
    surface SynthesisFailure with tool_required_unmet."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "still text"}}]}
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "text only"), _success("b", "more text")],
        SYNTH_UP,
        {"messages": [], "tools": [_tool_def("f")], "tool_choice": "required"},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesisFailure)
    assert out.reason == "tool_required_unmet"


@pytest.mark.asyncio
async def test_d31_tool_choice_none_strips_synth_calls() -> None:
    """tool_choice='none' → if synth output emits tool_calls, finalize
    strips them and sets finish_reason='stop'."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "", "tool_calls": [_tc("f")]},
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "ans1"), _success("b", "ans2")],
        SYNTH_UP,
        {"messages": [], "tool_choice": "none"},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    msg = out.response["choices"][0]["message"]
    assert "tool_calls" not in msg
    assert out.response["choices"][0]["finish_reason"] == "stop"
    assert out.fallback_reason == "tool_choice_none_stripped"


@pytest.mark.asyncio
async def test_d31_specific_synth_picks_wrong_falls_back_to_right_candidate() -> None:
    """Synth emitted wrong function under tool_choice={name:right} →
    finalize picks the candidate that did call right, deterministically."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [_tc("wrong")],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [
            _success("a", "", tool_calls=[_tc("right", "{}", "ra")]),
            _success("b", "", tool_calls=[_tc("wrong", "{}", "wb")]),
        ],
        SYNTH_UP,
        {
            "messages": [],
            "tools": [_tool_def("right"), _tool_def("wrong")],
            "tool_choice": {"type": "function", "function": {"name": "right"}},
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    chosen = out.response["choices"][0]["message"]
    assert chosen["tool_calls"][0]["function"]["name"] == "right"
    assert out.fallback_reason == "tool_choice_repaired"


@pytest.mark.asyncio
async def test_d31_parallel_false_fastpath_trims() -> None:
    """Single-success candidate with multiple tool_calls under
    parallel_tool_calls=false → trimmed to one in fast-path."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("synthesizer should NOT be called (single-success fast-path)")

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "", tool_calls=[_tc("f", "{}", "1"), _tc("g", "{}", "2")])],
        SYNTH_UP,
        {
            "messages": [],
            "tools": [_tool_def("f"), _tool_def("g")],
            "parallel_tool_calls": False,
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    msg = out.response["choices"][0]["message"]
    assert len(msg["tool_calls"]) == 1
    assert out.fallback_reason == "parallel_violation_trimmed"


@pytest.mark.asyncio
async def test_d31_best_of_with_tool_arbitration_skips_llm() -> None:
    """best-of + tool arbitration → deterministic candidate pick, no
    LLM call. Per review r24 F5: tool schema and integer-index judge
    prompts don't compose."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("synth should NOT be called for tool-aware best-of")

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(mode="best-of", fastpath=False),
        [
            _success("a", "", tool_calls=[_tc("f", '{"x": 1}', "1")]),
            _success("b", "", tool_calls=[_tc("f", '{"x": 1}', "2")]),
            _success("c", "", tool_calls=[_tc("g", "{}", "3")]),
        ],
        SYNTH_UP,
        {"messages": [], "tools": [_tool_def("f"), _tool_def("g")], "tool_choice": "required"},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    msg = out.response["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"]["name"] == "f"


@pytest.mark.asyncio
async def test_d31_best_of_text_path_unchanged_when_no_tool_arbitration() -> None:
    """best-of with tools+auto+all-text-only candidates keeps existing
    integer-judge path. No tool arbitration → no skip."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["called"] = True
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "1"}}]}
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(mode="best-of", fastpath=False),
        [_success("a", "first"), _success("b", "second")],
        SYNTH_UP,
        {"messages": [], "tools": [_tool_def("f")], "tool_choice": "auto"},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert captured.get("called") is True
    assert isinstance(out, SynthesizedResponse)
    assert out.response["choices"][0]["message"]["content"] == "first"


@pytest.mark.asyncio
async def test_d31_finalize_does_not_mutate_source_response() -> None:
    """The original GeneratorSuccess.response carries through to other
    consumers. Finalize must not mutate it via shared dict refs."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("synth not called on parallel-trim fast-path")

    transport = httpx.MockTransport(handler)
    src_calls = [_tc("f", "{}", "1"), _tc("g", "{}", "2")]
    src = _success("a", "", tool_calls=src_calls)
    snapshot = list(src.response["choices"][0]["message"]["tool_calls"])
    await synthesize(
        _profile(),
        [src],
        SYNTH_UP,
        {"messages": [], "parallel_tool_calls": False},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert src.response["choices"][0]["message"]["tool_calls"] == snapshot
    assert len(src.response["choices"][0]["message"]["tool_calls"]) == 2


@pytest.mark.asyncio
async def test_d31_required_with_one_passing_candidate_fastpath_satisfies() -> None:
    """Single success that DID emit a tool_call under tool_choice='required'
    fast-paths cleanly (no synth call needed)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("synth should NOT be called when fast-path satisfies")

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "", tool_calls=[_tc("f")])],
        SYNTH_UP,
        {"messages": [], "tools": [_tool_def("f")], "tool_choice": "required"},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.fastpath is True


@pytest.mark.asyncio
async def test_d31_finish_reason_coerced_after_strip() -> None:
    """When tool_calls are stripped, finish_reason must change from
    tool_calls → stop."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "", "tool_calls": [_tc("f")]},
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "x"), _success("b", "y")],
        SYNTH_UP,
        {"messages": [], "tool_choice": "none"},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    choice = out.response["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]


# ── review r26 follow-ups (P1 wrapper id, P2 best-of route) ─────────


@pytest.mark.asyncio
async def test_d31_fallback_uses_chosen_candidate_response_id() -> None:
    """When finalize falls back to a candidate, the returned response
    must wrap with the CHOSEN candidate's id/model/usage — not the
    synth's wrapper. Review r26 P1: id-mixing is a contract bug."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "synth-wrapper-id",
                "model": "synth-model",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [_tc("wrong")],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    cand_a_resp = {
        "id": "chatcmpl-a",
        "model": "gen-a-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tc("right", "{}", "ra")],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    cand_a = GeneratorSuccess(upstream_name="a", response=cand_a_resp, elapsed_ms=10)
    cand_b = _success("b", "", tool_calls=[_tc("wrong", "{}", "wb")])
    out = await synthesize(
        _profile(),
        [cand_a, cand_b],
        SYNTH_UP,
        {
            "messages": [],
            "tools": [_tool_def("right"), _tool_def("wrong")],
            "tool_choice": {"type": "function", "function": {"name": "right"}},
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    # Wrapper id/model must come from the candidate that satisfied the
    # constraint, NOT the synth wrapper that emitted the wrong call.
    assert out.response["id"] == "chatcmpl-a"
    assert out.response["model"] == "gen-a-model"
    assert out.response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "right"
    assert out.response["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_d31_best_of_pre_synth_violation_skips_llm() -> None:
    """Review r26 P2: in best-of mode, a single-success candidate that
    violates tool_choice="required" must NOT call the integer-judge
    synth (it has no tools). Route to deterministic pick instead."""
    captured = {}

    def handler(_request: httpx.Request) -> httpx.Response:
        captured["called"] = True
        # If reached, this would be the integer-judge prompt — wrong path.
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "1"}}]}
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(mode="best-of", fastpath=False),
        [_success("a", "no tool_calls here")],  # single, text-only
        SYNTH_UP,
        {"messages": [], "tools": [_tool_def("f")], "tool_choice": "required"},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    # No candidate satisfies "required" → 502 SynthesisFailure (no synth-repair in best-of).
    assert isinstance(out, SynthesisFailure)
    assert out.reason == "tool_required_unmet"
    # Crucially, the integer-judge synth was NOT called.
    assert "called" not in captured


@pytest.mark.asyncio
async def test_d31_fallback_finish_reason_synced_when_wrapper_stale() -> None:
    """Review r27: chosen-candidate wrapper may carry stale `stop`
    finish_reason while its message has tool_calls. Final response
    must show `tool_calls` (or no calls + stop, symmetrically)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "synth",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [_tc("wrong")],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    # Candidate A satisfies "right" — but its wrapper has finish_reason=stop
    # (a generator quirk; OpenAI spec says this is invalid but real-world
    # generators do emit it).
    cand_a_resp = {
        "id": "chatcmpl-a",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tc("right", "{}", "ra")],
                },
                "finish_reason": "stop",  # STALE — should coerce to tool_calls
            }
        ],
    }
    cand_a = GeneratorSuccess(upstream_name="a", response=cand_a_resp, elapsed_ms=10)
    cand_b = _success("b", "", tool_calls=[_tc("wrong", "{}", "wb")])
    out = await synthesize(
        _profile(),
        [cand_a, cand_b],
        SYNTH_UP,
        {
            "messages": [],
            "tools": [_tool_def("right"), _tool_def("wrong")],
            "tool_choice": {"type": "function", "function": {"name": "right"}},
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    # Wrapper id from candidate A. finish_reason coerced to tool_calls.
    assert out.response["id"] == "chatcmpl-a"
    assert out.response["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_d31_best_of_with_none_strips_calls_no_502() -> None:
    """Review r28: best-of + tool_choice='none' with candidates that
    have tool_calls must NOT 502 — finalize strips the calls and
    returns. Prior bug: arbitration trigger sent this through
    deterministic pick, where 'none' requires empty-signature so all
    candidates failed."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["called"] = True
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "1"}}]}
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(mode="best-of", fastpath=False),
        [
            _success("a", "", tool_calls=[_tc("f", "{}", "1")]),
            _success("b", "", tool_calls=[_tc("g", "{}", "2")]),
        ],
        SYNTH_UP,
        {"messages": [], "tools": [_tool_def("f"), _tool_def("g")], "tool_choice": "none"},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    msg = out.response["choices"][0]["message"]
    assert "tool_calls" not in msg
    assert out.response["choices"][0]["finish_reason"] == "stop"
    assert out.fallback_reason == "tool_choice_none_stripped"
    assert captured.get("called") is True  # text judge ran (existing behavior)


# ── Per-draft observability (review r103) ──────────────────────────


@pytest.mark.asyncio
async def test_draft_stats_populated_on_single_success() -> None:
    """1 candidate path: hashes/lengths set, decision=single_success."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("synth must not be called on single-success")

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "hello world"), _failure("b"), _failure("c")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.synth_decision == "single_success"
    assert len(out.draft_stats.lengths) == 1
    assert len(out.draft_stats.hashes) == 1
    assert all(len(h) == 16 for h in out.draft_stats.hashes)


@pytest.mark.asyncio
async def test_draft_stats_consensus_label_distinct_from_single_success() -> None:
    """Multiple identical candidates → fastpath_consensus, NOT single_success."""

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("synth must not run on agreement")

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(),
        [_success("a", "same"), _success("b", "same"), _success("c", "same")],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.synth_decision == "fastpath_consensus"
    # Identical drafts → identical hashes (the within-call signal we
    # ship to detect degenerate MoA).
    assert len(set(out.draft_stats.hashes)) == 1
    assert len(out.draft_stats.lengths) == 3


@pytest.mark.asyncio
async def test_draft_stats_diverse_candidates_have_distinct_hashes() -> None:
    """Different drafts → different hashes (the divergence signal).

    Uses a synth handler that returns a merge so the path actually runs
    through _run_synth_and_finalize and exercises the merge label.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-synth",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "merged answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=False),
        [
            _success("a", "draft one is short"),
            _success("b", "draft two is also short but different"),
            _success("c", "third draft, distinct content"),
        ],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    assert out.synth_decision == "merged"
    assert len(out.draft_stats.hashes) == 3
    # Three different drafts must produce three different hashes.
    assert len(set(out.draft_stats.hashes)) == 3
    # Lengths reflect the input drafts, not the synth output.
    assert out.draft_stats.lengths[0] < out.draft_stats.lengths[1]


@pytest.mark.asyncio
async def test_authority_context_passed_to_synth_system(monkeypatch) -> None:
    """Review r106: the request's leading system/developer messages must
    appear in the synth's system prompt. Without this the synthesizer
    can't see the date / persona / mode constraints and merges blind.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["synth_messages"] = body["messages"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-synth",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "merged"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=False),
        [_success("a", "draft alpha"), _success("b", "draft beta"), _success("c", "draft gamma")],
        SYNTH_UP,
        {
            "messages": [
                {"role": "system", "content": "Today: 2026-05-01\nPlatform: linux\n"},
                {"role": "developer", "content": "Plan-mode rules: stage 1 first."},
                {"role": "user", "content": "what's the weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "web_scrape",
                                "arguments": '{"url":"https://example.com"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": "Forecast: 18°C, partly cloudy",
                },
            ]
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    synth_system = captured["synth_messages"][0]["content"]
    assert "Today: 2026-05-01" in synth_system
    assert "Plan-mode rules: stage 1 first." in synth_system
    assert "SYNTHESIS TASK" in synth_system
    # The synth instructions must remain (operational guidance follows
    # the authoritative context).
    assert "candidate" in synth_system.lower()
    # Cutoff-driven refusal regression (2026-05-02): the synth must be
    # told NOT to discard tool-grounded content just because dates
    # post-date its training. Without this clarification, a synth at
    # primary (older training cutoff) refused 3 convergent news drafts
    # from real BBC/AP scrapes. Two structural assertions: the
    # post-cutoff carve-out exists, and the convergence-as-evidence
    # framing exists. r-fab-1 H1: these only fire when the request
    # actually carries a recent tool chain (this turn's
    # assistant→tool messages); the request above includes one.
    assert "post-dates" in synth_system.lower()
    assert "tool results" in synth_system.lower()
    assert (
        "convergent" in synth_system.lower()
        or "convergence" in synth_system.lower()
    )


@pytest.mark.asyncio
async def test_authority_context_no_tool_chain_blocks_fabrication(monkeypatch) -> None:
    """r-fab-1 H1: when the request carries authority context but NO
    recent tool chain (no post-user assistant→tool messages), the synth
    must NOT be told to treat candidate convergence as tool-grounded
    evidence. Three generators agreeing on "tests pass" without any
    test having actually run is fabrication, not evidence — the synth
    must be told to exclude such outcome claims.

    Live failure pattern: auto-mode planning MoA at write-synth time
    converges on "✓ 1 passed (3 s)" / "all Playwright tests passing"
    when generators are asked to draft a forward-looking PLAN.md.
    With no tool chain to anchor on, the old "convergence as evidence"
    rule pulled the synth toward honoring the fabricated consensus.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["synth_messages"] = body["messages"]
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "merged"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=False),
        [
            _success("a", "All tests pass"),
            _success("b", "Tests pass consistently"),
            _success("c", "✓ 1 passed (3 s)"),
        ],
        SYNTH_UP,
        {
            "messages": [
                {"role": "system", "content": "The requested document describes future intent."},
                {"role": "user", "content": "draft the plan"},
            ]
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    synth_system = captured["synth_messages"][0]["content"]
    assert "SYNTHESIS TASK" in synth_system
    # The cutoff carve-out + convergence-as-evidence framing MUST be
    # absent when no tool evidence exists — those rules only make
    # sense when there are tool results to anchor on.
    assert "post-dates" not in synth_system.lower()
    assert "evidence the tools surfaced real data" not in synth_system.lower()
    # The no-evidence branch must structurally tell the synth that
    # convergence is draft agreement only and outcome claims without
    # supporting evidence must be excluded. Review r-fab-1-H1 MED:
    # "supporting evidence" includes user-supplied source/log text in
    # the prompt, not only tool-role messages.
    assert "draft agreement" in synth_system.lower()
    assert "fabricated" in synth_system.lower()
    assert "evidence visible in this prompt" in synth_system.lower()
    assert "user-supplied source/log text" in synth_system.lower()


@pytest.mark.asyncio
async def test_authority_context_prior_tool_result_uses_grounded_branch(monkeypatch) -> None:
    """r-fab-1-H1 HIGH: when a tool-role message exists ANYWHERE in the
    conversation (not just post-latest-user), the synth must see the
    tool-grounded branch (cutoff carve-out + convergence-as-evidence).
    Otherwise a follow-up like "summarize what you scraped" — which has
    earlier tool results but an empty post-user chain — would be
    misclassified as no-evidence and outcome claims falsely flagged
    as fabricated.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["synth_messages"] = body["messages"]
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "merged"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=False),
        [
            _success("a", "BBC reports US-Iran ceasefire announced today"),
            _success("b", "Top story: ceasefire announcement"),
            _success("c", "Today: ceasefire announced per BBC scrape"),
        ],
        SYNTH_UP,
        {
            "messages": [
                {"role": "system", "content": "Today: 2026-05-03"},
                {"role": "user", "content": "what's on BBC?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "web_scrape",
                                "arguments": '{"url":"https://bbc.com"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": "BBC headline: US-Iran ceasefire announced today",
                },
                {"role": "assistant", "content": "(earlier summary)"},
                # Newer user follow-up — `_extract_recent_tool_chain`
                # returns "" because there's no post-this-user tool tail.
                # Without H1's evidence-presence gate, this would land
                # in the no-evidence branch and tell the synth to flag
                # "ceasefire announced today" as fabricated.
                {"role": "user", "content": "summarize what you scraped"},
            ]
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    synth_system = captured["synth_messages"][0]["content"]
    # Tool-grounded branch active — earlier tool message is visible
    # evidence even though post-user chain is empty.
    assert "post-dates" in synth_system.lower()
    assert "tool results" in synth_system.lower()
    assert (
        "convergent" in synth_system.lower()
        or "convergence" in synth_system.lower()
    )
    assert "evidence the tools surfaced real data" in synth_system.lower()
    # No-evidence-specific framing must NOT fire here. "draft agreement"
    # is the discriminator — that phrasing exists only in the
    # no-evidence branch. ("fabricated" now appears in both branches
    # because the r2 MED fix added an unsupported-outcome guard to the
    # grounded branch too — it's no longer a discriminator.)
    assert "draft agreement" not in synth_system.lower()


@pytest.mark.asyncio
async def test_authority_context_grounded_branch_still_excludes_unsupported_outcomes(
    monkeypatch,
) -> None:
    """r-fab-1-H1 r2 MED: in tool-heavy histories (e.g. auto-mode after
    a web_search ran several phases ago), `has_tool_evidence` is True,
    so the grounded branch fires. But the candidates' "tests pass"
    convergence is NOT supported by any test-execution evidence in the
    prompt — only an unrelated earlier scrape exists.

    The grounded branch must keep the cutoff carve-out (legitimate
    tool-grounded content) AND explicitly exclude unsupported outcome
    claims (regardless of candidate agreement). Otherwise three
    generators converging on "✓ 1 passed" inherit the convergence-as-
    evidence framing and the auto-mode planning fabrication regression
    re-opens.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["synth_messages"] = body["messages"]
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "merged"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=False),
        [
            _success("a", "All tests pass"),
            _success("b", "Tests pass consistently"),
            _success("c", "✓ 1 passed (3 s)"),
        ],
        SYNTH_UP,
        {
            "messages": [
                {"role": "system", "content": "The requested document describes future intent."},
                {"role": "user", "content": "draft phase 1 plan"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"q":"example query"}',
                            },
                        }
                    ],
                },
                # Earlier tool result — unrelated to "tests pass" claim.
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": "Wikipedia: chess is played on an 8x8 board.",
                },
                {"role": "user", "content": "now draft the plan"},
            ]
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    synth_system = captured["synth_messages"][0]["content"]
    # Grounded branch active (tool message exists).
    assert "post-dates" in synth_system.lower()
    # AND the unsupported-outcome guard must be present in the grounded
    # branch — not just in no-evidence. Three structural assertions
    # mirror the no-evidence wording for evidence sources.
    assert "evidence visible in this prompt" in synth_system.lower()
    assert "user-supplied source/log text" in synth_system.lower()
    assert "fabricated" in synth_system.lower()
    assert "regardless of how many candidates" in synth_system.lower()


@pytest.mark.asyncio
async def test_authority_context_assistant_only_tail_uses_no_evidence_branch(
    monkeypatch,
) -> None:
    """r-fab-1-H1 LOW: a turn where the post-user tail contains plain
    assistant content (no tool_calls, no tool messages) and there are
    no tool-role messages anywhere in the conversation must use the
    no-evidence branch. Earlier `recent_tool_chain != ""` truthiness
    gate would have falsely classified this as tool-grounded because
    the assistant text would populate the chain string.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["synth_messages"] = body["messages"]
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "merged"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=False),
        [
            _success("a", "draft alpha"),
            _success("b", "draft beta"),
            _success("c", "draft gamma"),
        ],
        SYNTH_UP,
        {
            "messages": [
                {"role": "system", "content": "The requested document describes future intent."},
                {"role": "user", "content": "draft a plan"},
                # Assistant content WITHOUT tool_calls or tool messages.
                # The post-user chain string would be non-empty (contains
                # the assistant text), but `_has_visible_tool_evidence`
                # correctly returns False because no tool role exists.
                {"role": "assistant", "content": "thinking about the plan..."},
            ]
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    synth_system = captured["synth_messages"][0]["content"]
    # No-evidence branch active.
    assert "draft agreement" in synth_system.lower()
    assert "fabricated" in synth_system.lower()
    assert "post-dates" not in synth_system.lower()


@pytest.mark.asyncio
async def test_authority_context_user_pasted_log_no_false_fabrication(
    monkeypatch,
) -> None:
    """r-fab-1-H1 MED: when the user pastes their own logs/source as
    ground truth (no tool calls, no tool messages), the synth's
    no-evidence branch must allow that user-supplied evidence to
    support outcome claims — not flag everything as fabrication.
    Structural assertion: the no-evidence rule mentions user-supplied
    text as a valid evidence source.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["synth_messages"] = body["messages"]
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "merged"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=False),
        [
            _success("a", "Looks like 1 test passes"),
            _success("b", "Test passed: 1"),
            _success("c", "✓ 1 passed"),
        ],
        SYNTH_UP,
        {
            "messages": [
                {"role": "system", "content": "Reading user-supplied test log."},
                {
                    "role": "user",
                    "content": "Here is my test runner output:\n\n========= 1 passed in 0.42s =========",
                },
            ]
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    synth_system = captured["synth_messages"][0]["content"]
    # No-evidence branch is correctly chosen (no tool messages), but
    # the wording must explicitly allow user-supplied source/log text
    # so the synth doesn't flag "1 passed" as fabrication when the
    # user themselves supplied it.
    assert "user-supplied source/log text" in synth_system.lower()
    assert "evidence visible in this prompt" in synth_system.lower()


@pytest.mark.asyncio
async def test_authority_context_absent_when_no_system(monkeypatch) -> None:
    """No authority block in the request → synth system stays generic.
    Backward compat: requests without leading system/developer messages
    must still synthesize cleanly.
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["synth_messages"] = body["messages"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-synth",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "merged"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=False),
        [_success("a", "draft alpha"), _success("b", "draft beta"), _success("c", "draft gamma")],
        SYNTH_UP,
        {"messages": [{"role": "user", "content": "what's the weather?"}]},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    synth_system = captured["synth_messages"][0]["content"]
    assert "SYNTHESIS TASK" not in synth_system  # no authority → no separator
    assert "candidate" in synth_system.lower()


@pytest.mark.asyncio
async def test_recent_tool_chain_passed_to_synth_user() -> None:
    """2026-05-02 #1.b: when the request carries assistant tool_calls
    and tool messages after the latest user message, the synth must
    see the chain so it can verify candidates' fact claims against the
    actual tool output. Without this, a synth produces refusals from
    convergent news drafts (smoke test of "what's the world news?").
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["synth_messages"] = body["messages"]
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "merged"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=False),
        [
            _success("a", "Today's headlines: US-Iran ceasefire"),
            _success("b", "Top story: US-Iran ceasefire"),
            _success("c", "BBC reports US-Iran ceasefire"),
        ],
        SYNTH_UP,
        {
            "messages": [
                {"role": "user", "content": "what's the world news?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "web_scrape",
                                "arguments": '{"url":"https://bbc.com/news/world"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": "BBC headline: US-Iran ceasefire announced today",
                },
            ]
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    user_content = captured["synth_messages"][1]["content"]
    assert "RECENT TOOL CHAIN" in user_content
    assert "web_scrape" in user_content
    assert "BBC headline: US-Iran ceasefire" in user_content
    # Candidate section still present alongside the new tool-chain section.
    assert "CANDIDATE RESPONSES" in user_content


@pytest.mark.asyncio
async def test_recent_tool_chain_truncates_huge_tool_results() -> None:
    """A scraped page can be 100KB+. Per-message cap at 3000 chars
    plus a `…(truncated)` marker keeps the synth body bounded."""
    captured: dict[str, Any] = {}
    big_payload = "X" * 8000

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["synth_messages"] = body["messages"]
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "merged"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    await synthesize(
        _profile(fastpath=False),
        [_success("a", "draft a"), _success("b", "draft b")],
        SYNTH_UP,
        {
            "messages": [
                {"role": "user", "content": "scrape it"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "web_scrape", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": big_payload},
            ]
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    user_content = captured["synth_messages"][1]["content"]
    assert "(truncated)" in user_content
    # 3000 X's max in the truncated section.
    assert user_content.count("X") == 3000


@pytest.mark.asyncio
async def test_recent_tool_chain_absent_when_no_post_user_messages() -> None:
    """Trivial case: only system + user, no tools called yet. Synth
    must not get an empty 'RECENT TOOL CHAIN' header."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["synth_messages"] = body["messages"]
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "merged"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    await synthesize(
        _profile(fastpath=False),
        [_success("a", "draft a"), _success("b", "draft b")],
        SYNTH_UP,
        {"messages": [{"role": "user", "content": "hi"}]},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    user_content = captured["synth_messages"][1]["content"]
    assert "RECENT TOOL CHAIN" not in user_content


@pytest.mark.asyncio
async def test_authority_context_soft_caps_huge_block() -> None:
    """Auto-mode prompts can be 600+ lines / 30K+ chars. The synth
    authority context soft-caps to keep budgets sane while preserving
    head + tail (date is in head, mode-specific tail-rules in tail).
    """
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured["synth_messages"] = body["messages"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-synth",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "merged"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    transport = httpx.MockTransport(handler)
    head = "Today: 2026-05-01\n"
    middle = "X" * 30000
    tail = "End of authority block: emit JSON only."
    huge_system = head + middle + tail
    out = await synthesize(
        _profile(fastpath=False),
        [_success("a", "draft alpha"), _success("b", "draft beta"), _success("c", "draft gamma")],
        SYNTH_UP,
        {
            "messages": [
                {"role": "system", "content": huge_system},
                {"role": "user", "content": "do the thing"},
            ]
        },
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    synth_system = captured["synth_messages"][0]["content"]
    # Head and tail preserved across the elision.
    assert "Today: 2026-05-01" in synth_system
    assert "End of authority block" in synth_system
    assert "elided" in synth_system  # elision marker present
    # The composite total stays well below the original.
    assert len(synth_system) < 25000


# ── r115/r116: thinking-model reasoning fallback ───────────────────


def test_reasoning_fallback_text_extracts_when_content_null() -> None:
    """Thinking models (reasoning-model-20b) put output in `reasoning_content`
    when content is null. Fallback extracts it."""
    msg = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Stepping through the problem: tea balance is key.",
    }
    assert (
        _reasoning_fallback_text(msg)
        == "Stepping through the problem: tea balance is key."
    )


def test_reasoning_fallback_text_legacy_reasoning_field() -> None:
    """Some upstreams (older reasoning-model) emit `reasoning` rather than
    `reasoning_content`. Fallback covers both."""
    msg = {"role": "assistant", "content": None, "reasoning": "legacy field"}
    assert _reasoning_fallback_text(msg) == "legacy field"


def test_reasoning_fallback_skipped_when_tool_calls_present() -> None:
    """Review r115 finding #2: tool_calls are the substantive output;
    reasoning text is private CoT we shouldn't surface as content."""
    msg = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "should be ignored",
        "tool_calls": [{"id": "x", "function": {"name": "f", "arguments": "{}"}}],
    }
    assert _reasoning_fallback_text(msg) == ""


def test_reasoning_fallback_skipped_when_function_call_present() -> None:
    """Same gate for legacy `function_call`."""
    msg = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "should be ignored",
        "function_call": {"name": "f", "arguments": "{}"},
    }
    assert _reasoning_fallback_text(msg) == ""


def test_reasoning_fallback_skipped_when_content_is_string() -> None:
    """Visible content always wins. Reasoning is fallback only."""
    msg = {
        "role": "assistant",
        "content": "the visible answer",
        "reasoning_content": "internal thought",
    }
    assert _reasoning_fallback_text(msg) == ""


# ── empty-string sibling of the r115 fix ───────────────────────────


def test_reasoning_fallback_extracts_when_content_empty_string() -> None:
    """The 9b12fbb fix only handled `content: null`. Live runs hit a
    sibling shape under different vLLM build paths: `content: ""` (or
    whitespace-only) plus full reasoning_content. The original gate
    let `""` slip through `isinstance(str)`, hashing to an 8-byte
    framing-only signature → ZERO contribution from the reasoning
    generator. Widened gate now treats empty/whitespace content as
    structurally missing."""
    msg = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "the actual answer lives here",
    }
    assert _reasoning_fallback_text(msg) == "the actual answer lives here"


def test_reasoning_fallback_extracts_when_content_whitespace_only() -> None:
    msg = {
        "role": "assistant",
        "content": "  \n  \t  ",
        "reasoning": "actual reply",
    }
    assert _reasoning_fallback_text(msg) == "actual reply"


def test_reasoning_fallback_skipped_when_empty_content_but_tool_calls() -> None:
    """Same gate as the None case: tool_calls win even when content
    is the empty-string flavor of missing."""
    msg = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "should be ignored",
        "tool_calls": [{"id": "x", "function": {"name": "f", "arguments": "{}"}}],
    }
    assert _reasoning_fallback_text(msg) == ""


def test_draft_signature_rescues_empty_string_content() -> None:
    """Regression: pre-fix, an empty-string content + reasoning_content
    produced an 8-byte signature (NUL framing only). Post-fix, the
    reasoning text contributes to the body so signatures diverge across
    generators that returned different reasoning."""
    from meta_model.moa.synthesizer import _draft_signature

    empty_msg = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "long chain-of-thought arguing for X",
    }
    sig = _draft_signature(empty_msg)
    # body should contain the reasoning text, not be empty
    assert "long chain-of-thought" in sig
    # Signature length must be much more than 8 bytes
    assert len(sig.encode("utf-8")) > 30


def test_visible_content_missing_empty_list() -> None:
    """Review r1 follow-up: multimodal/list shape. Empty list → missing."""
    from meta_model.moa.synthesizer import _is_visible_content_missing

    assert _is_visible_content_missing({"role": "assistant", "content": []})


def test_visible_content_missing_list_with_empty_text_parts() -> None:
    """List of text parts with all empty/whitespace text → missing."""
    from meta_model.moa.synthesizer import _is_visible_content_missing

    msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": ""},
            {"type": "text", "text": "  \n  "},
        ],
    }
    assert _is_visible_content_missing(msg)


def test_visible_content_present_with_real_text_part() -> None:
    """List with at least one substantive text part → NOT missing."""
    from meta_model.moa.synthesizer import _is_visible_content_missing

    msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": ""},
            {"type": "text", "text": "the actual answer"},
        ],
    }
    assert not _is_visible_content_missing(msg)


def test_visible_content_present_with_image_part() -> None:
    """Non-text parts (image, audio, file) count as substance — even
    when the only text part is empty. Image-only assistant output is
    rare but legal under the multimodal contract; treating it as
    missing would silently swap it for reasoning text, which is the
    wrong behavior for a model that DID emit a real visual answer."""
    from meta_model.moa.synthesizer import _is_visible_content_missing

    msg = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": ""},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
        ],
    }
    assert not _is_visible_content_missing(msg)


def test_visible_content_malformed_list_element_treated_as_substance() -> None:
    """A non-dict element in the list is malformed. Conservatively
    treat as substance — bypassing the rescue here is preferable to
    silently coercing reasoning-content into a malformed message."""
    from meta_model.moa.synthesizer import _is_visible_content_missing

    msg = {"role": "assistant", "content": ["raw string in list", {"type": "text", "text": ""}]}
    assert not _is_visible_content_missing(msg)


def test_reasoning_fallback_extracts_for_empty_list_content() -> None:
    """End-to-end: empty-list content + reasoning_content → rescue fires."""
    msg = {
        "role": "assistant",
        "content": [],
        "reasoning_content": "the actual answer lives here",
    }
    assert _reasoning_fallback_text(msg) == "the actual answer lives here"


def test_reasoning_fallback_extracts_for_all_empty_text_parts() -> None:
    """End-to-end: list-of-empty-text-parts + reasoning_content → rescue."""
    msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}, {"type": "text", "text": "  "}],
        "reasoning_content": "rescued reasoning",
    }
    assert _reasoning_fallback_text(msg) == "rescued reasoning"


def test_candidate_signature_rescues_empty_string_content() -> None:
    """Review r1 follow-up: `_candidate_signature` is the fast-path
    agreement check counterpart. Same widening applies — empty-string
    content with reasoning_content must hash the reasoning text, not
    the framing-only "" body. Without this, two generators returning
    different reasoning chains would falsely agree on the empty-content
    signature and trip degenerate fastpath consensus."""
    from meta_model.moa.synthesizer import _candidate_signature

    empty_a = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "reasoning chain A",
    }
    empty_b = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "reasoning chain B (different)",
    }
    sig_a = _candidate_signature(empty_a)
    sig_b = _candidate_signature(empty_b)
    # Signatures must reflect the reasoning divergence, not collapse
    # to a shared empty-content signature.
    assert sig_a != sig_b
    # The content_sig component (first tuple element) carries reasoning text
    assert "reasoning chain A" in sig_a[0]
    assert "reasoning chain B" in sig_b[0]


def test_render_candidate_uses_reasoning_when_content_null() -> None:
    """The synth prompt should see the reasoning text — not '(empty)' —
    when a thinking model returns null content with no tool calls."""
    msg = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "A good cup of tea balances temperature and steeping time.",
    }
    rendered = _render_candidate(msg)
    assert "A good cup of tea balances" in rendered
    assert "(empty)" not in rendered


def test_render_candidate_uses_reasoning_when_content_empty_string() -> None:
    """F12 sibling: ``_render_candidate`` previously gated rescue on
    ``content is None`` only. With ``content == ""`` (vLLM/template
    paths after include_reasoning was lifted) and a populated
    ``reasoning_content``, the synth merge prompt rendered ``(empty)``
    while ``_draft_signature`` / ``_candidate_signature`` saw the
    rescued reasoning text — inconsistent view across the synth
    pipeline. Gate is now ``_is_visible_content_missing`` so empty
    string and whitespace-only take the same fallback path."""
    msg = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "the actual answer lives here",
    }
    rendered = _render_candidate(msg)
    assert "the actual answer lives here" in rendered
    assert "(empty)" not in rendered


def test_render_candidate_uses_reasoning_when_content_whitespace_only() -> None:
    """Sibling of the empty-string case: whitespace-only content also
    routes to reasoning rescue via ``_is_visible_content_missing``."""
    msg = {
        "role": "assistant",
        "content": "   \n  ",
        "reasoning": "rescued from legacy reasoning field",
    }
    rendered = _render_candidate(msg)
    assert "rescued from legacy reasoning field" in rendered
    assert "(empty)" not in rendered


def test_render_candidate_skips_reasoning_when_tool_calls_present() -> None:
    """When tool_calls are present, only tool_calls render — reasoning
    is private CoT, not part of the user-visible response (review r115)."""
    msg = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "should not appear",
        "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ],
    }
    rendered = _render_candidate(msg)
    assert "should not appear" not in rendered
    assert "tool_calls" in rendered


def test_draft_signature_uses_reasoning_when_content_null() -> None:
    """Telemetry hashing must reflect actual draft content. Pre-fix,
    every reasoning-model-20b draft hashed the same ~8-byte signature because
    body=='' for null content. Post-fix, reasoning text contributes."""
    msg = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "real draft text here",
    }
    sig = _draft_signature(msg)
    assert "real draft text here" in sig
    # Old behavior was sig length ~8 (just NUL-separated nulls). Confirm
    # we're substantially above that with real content.
    assert len(sig.encode("utf-8")) > 20


def test_candidate_signature_uses_reasoning_when_content_null() -> None:
    """Fast-path agreement check should compare reasoning text when
    that's the actual draft (review r115)."""
    msg_a = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Same reasoning",
    }
    msg_b = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Same reasoning",
    }
    msg_c = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Different reasoning",
    }
    assert _candidate_signature(msg_a) == _candidate_signature(msg_b)
    assert _candidate_signature(msg_a) != _candidate_signature(msg_c)


def test_clone_msg_with_finish_lifts_reasoning_into_content() -> None:
    """Direct-return paths (single_success / fastpath / best-of /
    merged synth output) all flow through `_clone_msg_with_finish`.
    A thinking-model candidate with content=null + reasoning text
    must come out with content populated so the wrapped response
    delivers the answer to the caller (review r115 finding #3)."""
    response = {
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "finish_reason": "length"}],
    }
    msg = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "the actual answer",
    }
    cloned = _clone_msg_with_finish(response, msg)
    assert cloned["content"] == "the actual answer"
    assert cloned["finish_reason"] == "length"


def test_coerce_reasoning_preserves_tool_calls_and_does_not_mutate_content() -> None:
    """When tool_calls are present, content stays null even if
    reasoning has text (gate per review r115)."""
    msg = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "should not lift",
        "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ],
    }
    _coerce_reasoning_into_content(msg)
    assert msg["content"] is None  # untouched
    assert msg["tool_calls"] == [
        {"id": "x", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    ]


def test_multimodal_list_signature_unchanged_no_regression() -> None:
    """Review r115 finding #1: signatures must keep JSON-serializing
    list content so different image_urls don't collapse together.
    The reasoning fallback only applies to content=None, not list."""
    msg_a = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "https://a/img.png"}},
        ],
    }
    msg_b = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "https://b/img.png"}},
        ],
    }
    # Different image URLs must produce different signatures.
    assert _candidate_signature(msg_a) != _candidate_signature(msg_b)
    assert _draft_signature(msg_a) != _draft_signature(msg_b)


@pytest.mark.asyncio
async def test_synth_failure_label_survives_finalize_relabel() -> None:
    """Review r103 finding: synth_failed_picked_primary must NOT be
    relabeled by finalize tool-repair. The decision field is set
    authoritatively by the call site, not derived from fallback_reason.
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)  # synth fails → fallback_primary path

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=False),
        [
            _success("a", "draft one"),
            _success("b", "draft two"),
        ],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=2.0, synth_min_viable_secs=0.0,
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    # Even if fallback_reason gets relabeled to a tool-repair value,
    # the authoritative decision label remains fallback_primary.
    assert out.synth_decision == "fallback_primary"


@pytest.mark.asyncio
async def test_synthesize_short_circuits_below_min_viable_budget() -> None:
    """If `timeout_secs < synth_min_viable_secs`, synth call is skipped
    entirely and we fall back to first-candidate. Diagnoses the
    'fanout drained the budget' case explicitly instead of waiting for
    a sub-second httpx ReadTimeout (observed in prod 2026-05-04 as
    `synth failed: ReadTimeout elapsed_ms=566`).
    """

    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("synth should NOT be called below viable budget")

    transport = httpx.MockTransport(handler)
    out = await synthesize(
        _profile(fastpath=False),
        [
            _success("a", "draft one"),
            _success("b", "draft two"),
        ],
        SYNTH_UP,
        {"messages": []},
        timeout_secs=1.13,                # mirrors the 599s-fanout pathology
        synth_min_viable_secs=30.0,       # production default
        transport=transport,
    )
    assert isinstance(out, SynthesizedResponse)
    # Should be the first-candidate fallback path. Synth was never
    # called (handler raises AssertionError if invoked); the
    # `insufficient_budget` reason is logged at WARNING level by
    # `_synth_failed_fallback` (visible in test logs).
    assert out.synth_decision == "fallback_primary"
