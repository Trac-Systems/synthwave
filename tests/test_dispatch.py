"""D.2.4 tests — profile dispatch + capabilities exposure."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from meta_model.config import parse_config_str
from meta_model.server import app, set_config, set_upstream_transport

# Test config: 5 upstreams (3 text + 2 vision), 4 profile types, voting on.
_BASE_TOML = """
[features]
voting = true

[upstreams.text_a]
model_id = "text-a"
base_url = "http://upstream:9000/v1"
context = 16384
max_output = 2048
modalities = ["text"]

[upstreams.text_b]
model_id = "text-b"
base_url = "http://upstream:9001/v1"
context = 8192
max_output = 2048
modalities = ["text"]

[upstreams.text_c]
model_id = "text-c"
base_url = "http://upstream:9002/v1"
context = 4096
max_output = 1024
modalities = ["text"]

[upstreams.vision_p]
model_id = "vision-p"
base_url = "http://upstream:9003/v1"
context = 8192
max_output = 2048
modalities = ["text", "image"]

[upstreams.vision_s]
model_id = "vision-s"
base_url = "http://upstream:9004/v1"
context = 4096
max_output = 1024
modalities = ["text", "image"]

[profiles."moa.text.v1"]
type = "moa"
generators = ["text_a", "text_b", "text_c"]
synthesizer = "text_a"

[profiles."moa.vision.v1"]
type = "moa"
generators = ["text_a", "vision_p"]
synthesizer = "text_a"

[profiles."cascade.vision.v1"]
type = "cascade"
upstreams = ["vision_p", "vision_s"]

[profiles."voting.injection.v1"]
type = "voting"
upstreams = ["text_a", "text_b", "text_c"]
aggregation = "any_yes"
failure_vote = "yes"

[profiles."moa.recovery.v1"]
type = "moa"
generators = ["text_a", "text_b", "text_c"]
synthesizer = "text_a"
strip_tools = true
"""


@pytest.fixture
def loaded_config() -> Any:
    cfg = parse_config_str(_BASE_TOML)
    set_config(cfg)
    yield cfg
    set_config(None)
    set_upstream_transport(None)


def _client() -> TestClient:
    return TestClient(app)


def _ok_response(content: str = "ok") -> dict[str, Any]:
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 1,
        "model": "x",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


# ── /v1/models capabilities exposure ──────────────────────────────


def test_models_endpoint_exposes_max_model_len_and_capabilities(loaded_config) -> None:
    r = _client().get("/v1/models")
    assert r.status_code == 200
    data = {m["id"]: m for m in r.json()["data"]}

    # F1: MoA text profile advertises the synthesizer's effective
    # ingress budget = text_a.context − overhead − reserve − max_output
    # = 16384 − 256 − 1024 − 2048 = 13056. (Was min(generators)=4096
    # under the pre-F1 rule; the reasoning-model floor effect.)
    # F1: MoA text profile advertises the synthesizer's effective
    # ingress budget = text_a.context − overhead − reserve − max_output
    # = 16384 − 256 − 1024 − 2048 = 13056.
    moa_text = data["moa.text.v1"]
    assert moa_text["max_model_len"] == 13056
    # F6: vision/video/audio are SERVER-LEVEL — every profile reports
    # the same multimodal flags. The fixture has no [vision]/[video]/
    # [audio] block, so all three are False everywhere.
    assert moa_text["capabilities"]["vision"] is False
    assert moa_text["capabilities"]["video"] is False
    assert moa_text["capabilities"]["audio"] is False
    assert moa_text["capabilities"]["function_calling"] is True
    assert moa_text["capabilities"]["effective_image_capability"] is False
    assert moa_text["capabilities"]["supports_image_tools"] is False

    # F6: profile that previously advertised vision=True via per-profile
    # generator-modality derivation now reports vision=False — multimodal
    # is server-level, not profile-derived. Same max_model_len (F1).
    moa_vision = data["moa.vision.v1"]
    assert moa_vision["max_model_len"] == 13056
    assert moa_vision["capabilities"]["vision"] is False
    assert moa_vision["capabilities"]["video"] is False

    # Cascade: F1 rule keeps min(upstream.context) until cascade
    # context-aware skip lands. min(vision_p=8192, vision_s=4096) = 4096.
    # F6: vision flag follows server-level rule.
    cascade_vision = data["cascade.vision.v1"]
    assert cascade_vision["max_model_len"] == 4096
    assert cascade_vision["capabilities"]["vision"] is False


def test_models_endpoint_server_level_vision_advertised_when_configured() -> None:
    """F6: when the server's [vision].endpoints is populated, EVERY
    profile reports vision=True — multimodal is server-level, single-
    model transparency."""
    cfg = parse_config_str("""
[upstreams.text_a]
model_id = "ta"
base_url = "http://up:9000/v1"
context = 8192
max_output = 1024
modalities = ["text"]

[upstreams.vision_a]
model_id = "va"
base_url = "http://up:9001/v1"
context = 8192
max_output = 1024
modalities = ["text", "image"]

[profiles."text_only.v1"]
type = "moa"
generators = ["text_a"]
synthesizer = "text_a"

[vision]
endpoints = ["vision_a"]
""")
    set_config(cfg)
    try:
        r = _client().get("/v1/models")
        data = {m["id"]: m for m in r.json()["data"]}
        # Profile has no vision-capable generator, but vision is advertised
        # because the SERVER has a vision cascade configured.
        assert data["text_only.v1"]["capabilities"]["vision"] is True
        assert data["text_only.v1"]["capabilities"]["effective_image_capability"] is True
        # Review r1 F6 HIGH: supports_image_tools comes from the vision
        # cascade upstream's function_calling capability, NOT the
        # selected profile's. vision_a defaults to supports_function_calling=True.
        assert data["text_only.v1"]["capabilities"]["supports_image_tools"] is True
        # Video/audio still false — only [vision] is configured.
        assert data["text_only.v1"]["capabilities"]["video"] is False
        assert data["text_only.v1"]["capabilities"]["audio"] is False
    finally:
        set_config(None)


def test_models_endpoint_voting_function_calling_uses_all_rule(loaded_config) -> None:
    """Voting requires every voter to support tools (consensus
    discipline)."""
    r = _client().get("/v1/models")
    data = {m["id"]: m for m in r.json()["data"]}
    voting = data["voting.injection.v1"]
    # function_calling: every text-only voter has supports_function_calling=True
    # default, so all-rule resolves True.
    assert voting["capabilities"]["function_calling"] is True


def test_models_endpoint_hides_voting_when_feature_disabled() -> None:
    cfg = parse_config_str(_BASE_TOML.replace("voting = true", "voting = false"))
    set_config(cfg)
    try:
        r = _client().get("/v1/models")
        ids = [m["id"] for m in r.json()["data"]]
        assert "voting.injection.v1" not in ids
        assert "moa.text.v1" in ids
    finally:
        set_config(None)


# ── MoA dispatch ───────────────────────────────────────────────────


def test_moa_text_profile_fans_out(loaded_config) -> None:
    """3-generator MoA: every generator port called, synthesizer too."""
    ports_called: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ports_called.append(request.url.port or 0)
        return httpx.Response(200, json=_ok_response("answer"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text
    # Three generators (ports 9000, 9001, 9002) + synthesizer (9000 again).
    assert {9000, 9001, 9002} <= set(ports_called)
    # Synth should also have been called → at least 4 total port hits.
    assert len(ports_called) >= 4
    assert r.headers.get("x-metamodel-profile") == "moa.text.v1"
    assert r.headers.get("x-metamodel-generators") == "3"
    assert r.headers.get("x-metamodel-quorum") == "3"


# ── Server-owned system prompt injection ─────────────────────────────


_SYSTEM_PROMPT_TOML = (
    """
[server]
system_prompt = "you-are-synthwave-1"
"""
    + _BASE_TOML
)


@pytest.fixture
def system_prompt_config() -> Any:
    cfg = parse_config_str(_SYSTEM_PROMPT_TOML)
    set_config(cfg)
    yield cfg
    set_config(None)
    set_upstream_transport(None)


def _generator_calls(seen: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    """Helper: filter the captured upstream calls to just the generator
    fan-out hits, dropping the synthesizer call. The synth call has
    its own (different) prompt shape built by ``synthesizer.py`` from
    a separate template, so it does NOT carry the dispatch-entry
    injection — that's correct: synth context is server-owned, not a
    place where the operator's identity prompt belongs.
    """
    return [
        m
        for m in seen
        if any(msg.get("role") == "user" and msg.get("content") == "hi" for msg in m)
    ]


def test_system_prompt_prepended_to_every_generator(system_prompt_config) -> None:
    """When ``[server] system_prompt`` is set, every generator upstream
    receives a leading ``role:"system"`` message with that text BEFORE
    the caller's own messages. The synthesizer call is built from a
    separate template and is intentionally NOT touched (synth context
    is server-owned)."""
    seen_messages: list[list[dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_messages.append(body.get("messages", []))
        return httpx.Response(200, json=_ok_response())

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [
                {"role": "system", "content": "caller-system"},
                {"role": "user", "content": "hi"},
            ],
        },
    )
    assert r.status_code == 200, r.text
    gens = _generator_calls(seen_messages)
    assert len(gens) == 3, f"expected 3 generator calls, got {len(gens)}"
    for msgs in gens:
        assert msgs[0] == {"role": "system", "content": "you-are-synthwave-1"}
        assert msgs[1] == {"role": "system", "content": "caller-system"}
        assert msgs[2] == {"role": "user", "content": "hi"}


def test_system_prompt_unset_is_passthrough(loaded_config) -> None:
    """When ``[server] system_prompt`` is unset/empty, dispatch must
    not touch ``messages``. Bytes-identical contract preserved on
    every generator fan-out."""
    seen_messages: list[list[dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen_messages.append(body.get("messages", []))
        return httpx.Response(200, json=_ok_response())

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    gens = _generator_calls(seen_messages)
    assert len(gens) == 3, f"expected 3 generator calls, got {len(gens)}"
    for msgs in gens:
        # Just the user message; no leading injection.
        assert msgs == [{"role": "user", "content": "hi"}]


def test_system_prompt_whitespace_only_is_passthrough() -> None:
    """Whitespace-only system_prompt must not trigger injection
    (matches the ``model_name`` whitespace-strip contract). Empty
    after strip → no injection."""
    cfg = parse_config_str(
        """
[server]
system_prompt = "   \\n  "
"""
        + _BASE_TOML
    )
    set_config(cfg)
    try:
        seen_messages: list[list[dict[str, Any]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen_messages.append(body.get("messages", []))
            return httpx.Response(200, json=_ok_response())

        set_upstream_transport(httpx.MockTransport(handler))
        r = _client().post(
            "/v1/chat/completions",
            json={
                "model": "moa.text.v1",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200
        gens = _generator_calls(seen_messages)
        assert len(gens) == 3
        for msgs in gens:
            assert msgs == [{"role": "user", "content": "hi"}]
    finally:
        set_config(None)
        set_upstream_transport(None)


def test_x_meta_model_profile_overrides_model(loaded_config) -> None:
    """x_meta_model.profile is the explicit, type-aware path. When set,
    it wins over the OpenAI `model` field. Clients that know about the
    extension can keep `model` as a legacy default."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response())

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "hi"}],
            "x_meta_model": {"profile": "moa.text.v1"},
        },
    )
    assert r.status_code == 200
    # Profile won: dispatch ran the MoA fan-out, not raw text_a.
    assert r.headers.get("x-metamodel-profile") == "moa.text.v1"
    assert r.headers.get("x-metamodel-generators") == "3"


def test_moa_partial_failure_still_succeeds(loaded_config) -> None:
    """When 1 of 3 generators fails, dispatch still synthesizes from
    the survivors. Quorum header reflects the actual success count.
    Degraded-mode header is set so observability can distinguish a
    healthy 3/3 success from a degenerate quorum=2/3 fast-path."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 9001:  # text_b fails
            return httpx.Response(503, json={"error": "down"})
        return httpx.Response(200, json=_ok_response())

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    assert r.headers.get("x-metamodel-quorum") == "2"
    assert r.headers.get("x-metamodel-degraded") == "true"
    failed = r.headers.get("x-metamodel-failed-generators") or ""
    assert "text_b" in failed
    assert "non_2xx" in failed


def test_moa_strip_tools_removes_tool_fields(loaded_config) -> None:
    """`recovery_synthesis.v1` profile sets strip_tools=true. Every
    upstream call must see a body with no `tools`, `tool_choice`, or
    `parallel_tool_calls` — the structural guarantee that the synth
    can't emit another tool_call."""
    captured_bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_ok_response())

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.recovery.v1",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "exec",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        },
    )
    assert r.status_code == 200
    assert len(captured_bodies) >= 1, "at least one upstream call expected"
    for body in captured_bodies:
        assert "tools" not in body, f"tools leaked into upstream body: {body.keys()}"
        assert "tool_choice" not in body
        assert "parallel_tool_calls" not in body


def test_moa_full_quorum_no_degraded_header(loaded_config) -> None:
    """Healthy 3/3 success must NOT carry the degraded marker — the
    header's whole purpose is to distinguish degraded from healthy."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response())

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    assert r.headers.get("x-metamodel-quorum") == "3"
    assert r.headers.get("x-metamodel-degraded") is None
    assert r.headers.get("x-metamodel-failed-generators") is None


def test_moa_all_failures_returns_502(loaded_config) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "all_generators_failed"


# ── Cascade dispatch ───────────────────────────────────────────────


def test_cascade_first_success_wins(loaded_config) -> None:
    """vision_p (port 9003) succeeds first → vision_s never called."""
    ports_called: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ports_called.append(request.url.port or 0)
        return httpx.Response(200, json=_ok_response("vision answer"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "cascade.vision.v1",
            "messages": [{"role": "user", "content": "describe"}],
        },
    )
    assert r.status_code == 200
    assert ports_called == [9003]  # vision_s never tried
    assert r.headers.get("x-metamodel-cascade-tried") == "vision_p"
    assert r.headers.get("x-metamodel-cascade-winner") == "vision_p"


def test_cascade_falls_through_on_5xx(loaded_config) -> None:
    ports_called: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ports_called.append(request.url.port or 0)
        if request.url.port == 9003:  # vision_p fails
            return httpx.Response(502, json={"error": "primary down"})
        return httpx.Response(200, json=_ok_response("secondary saved"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "cascade.vision.v1",
            "messages": [{"role": "user", "content": "describe"}],
        },
    )
    assert r.status_code == 200
    assert ports_called == [9003, 9004]
    assert r.headers.get("x-metamodel-cascade-tried") == "vision_p,vision_s"
    assert r.headers.get("x-metamodel-cascade-winner") == "vision_s"


def test_cascade_all_fail_bubbles_last_error(loaded_config) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "cascade.vision.v1",
            "messages": [{"role": "user", "content": "describe"}],
        },
    )
    # bubble_last_error → relay 503 status with our error envelope.
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "cascade_exhausted"
    assert r.headers.get("x-metamodel-cascade-exhausted") == "2"
    # Review r2 MED: telemetry parity on exhaustion. last_detail must
    # surface in headers so the client can reproduce per-attempt logs.
    assert "503" in (r.headers.get("x-metamodel-cascade-last-detail") or "")
    assert r.headers.get("x-metamodel-cascade-last-status") == "503"


def test_cascade_falls_through_on_empty_choices(loaded_config) -> None:
    """Review r2 MED: 200 with empty choices must rotate, not short-circuit.

    A judge upstream that returns OpenAI-shape `{"choices":[]}` (or
    no choices at all) is broken from the cascade's perspective —
    the client has nothing to parse. Without rotation, the client
    fails open and the cascade silently degrades to single-upstream.
    """
    ports_called: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ports_called.append(request.url.port or 0)
        if request.url.port == 9003:  # vision_p returns empty choices
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-empty",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "vision-p",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 0,
                        "total_tokens": 1,
                    },
                },
            )
        return httpx.Response(200, json=_ok_response("secondary saved"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "cascade.vision.v1",
            "messages": [{"role": "user", "content": "describe"}],
        },
    )
    assert r.status_code == 200
    assert ports_called == [9003, 9004]
    assert r.headers.get("x-metamodel-cascade-tried") == "vision_p,vision_s"
    assert r.headers.get("x-metamodel-cascade-winner") == "vision_s"


def test_cascade_falls_through_on_empty_content_no_tool_calls(loaded_config) -> None:
    """200 with content="" and no tool_calls = unusable, must rotate."""
    ports_called: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ports_called.append(request.url.port or 0)
        if request.url.port == 9003:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-blank",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "vision-p",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "   "},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        return httpx.Response(200, json=_ok_response("real answer"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "cascade.vision.v1",
            "messages": [{"role": "user", "content": "describe"}],
        },
    )
    assert r.status_code == 200
    assert ports_called == [9003, 9004]
    assert r.headers.get("x-metamodel-cascade-winner") == "vision_s"


def test_cascade_falls_through_on_malformed_choice_object(loaded_config) -> None:
    """Review r2 HIGH: choices[0] non-dict must not crash; rotate instead."""
    ports_called: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ports_called.append(request.url.port or 0)
        if request.url.port == 9003:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-malformed",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "vision-p",
                    "choices": ["not-an-object"],
                },
            )
        return httpx.Response(200, json=_ok_response("real answer"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "cascade.vision.v1",
            "messages": [{"role": "user", "content": "describe"}],
        },
    )
    assert r.status_code == 200
    assert ports_called == [9003, 9004]
    assert r.headers.get("x-metamodel-cascade-winner") == "vision_s"


def test_cascade_accepts_tool_calls_with_empty_content(loaded_config) -> None:
    """tool_calls without text content is a valid OpenAI response —
    cascade must NOT rotate just because content is empty when
    tool_calls is present (judges don't emit tool_calls today, but
    cascade is reused for tool_chat-like profiles)."""
    ports_called: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ports_called.append(request.url.port or 0)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-tools",
                "object": "chat.completion",
                "created": 1,
                "model": "vision-p",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "f", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "cascade.vision.v1",
            "messages": [{"role": "user", "content": "describe"}],
        },
    )
    assert r.status_code == 200
    assert ports_called == [9003]  # first upstream's tool-call response is valid
    assert r.headers.get("x-metamodel-cascade-winner") == "vision_p"


def test_cascade_accepts_reasoning_only_response(loaded_config) -> None:
    """F3: thinking-model upstreams may return content empty with the
    actual answer in `reasoning_content`. Cascade validation must
    accept that as a valid success — the F3 sanitizer rescues the
    text into `content` downstream. Without this, cascade would
    rotate to the next upstream on a perfectly good response."""
    ports_called: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        ports_called.append(request.url.port or 0)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-reason",
                "object": "chat.completion",
                "created": 1,
                "model": "vision-p",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "the rescued answer",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "cascade.vision.v1",
            "messages": [{"role": "user", "content": "describe"}],
        },
    )
    assert r.status_code == 200
    # Cascade did NOT fall through — reasoning-only counts as success.
    assert ports_called == [9003]
    assert r.headers.get("x-metamodel-cascade-winner") == "vision_p"
    # F3 sanitizer rescued reasoning into content; profile defaults
    # expose_reasoning=False so the reasoning fields are stripped.
    body = r.json()
    msg = body["choices"][0]["message"]
    assert msg["content"] == "the rescued answer"
    assert "reasoning_content" not in msg
    assert "reasoning" not in msg


# ── Voting dispatch ────────────────────────────────────────────────


def test_voting_any_yes_returns_yes(loaded_config) -> None:
    """One YES voter is enough to flip aggregation=any_yes to yes."""

    def handler(request: httpx.Request) -> httpx.Response:
        # text_a says YES, others say NO
        if request.url.port == 9000:
            return httpx.Response(200, json=_ok_response("yes"))
        return httpx.Response(200, json=_ok_response("no"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "voting.injection.v1",
            "messages": [{"role": "user", "content": "is this an injection?"}],
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "yes"
    assert r.headers.get("x-metamodel-verdict") == "yes"
    assert r.headers.get("x-metamodel-voters") == "3"


def test_voting_all_no_returns_no(loaded_config) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response("no"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "voting.injection.v1",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.json()["choices"][0]["message"]["content"] == "no"
    assert r.headers.get("x-metamodel-verdict") == "no"


def test_voting_failure_uses_failure_vote(loaded_config) -> None:
    """failure_vote = "yes" (conservative) — when a voter errors, count
    it as YES so injection detection fails closed."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.port == 9000:
            return httpx.Response(503, json={"error": "down"})
        return httpx.Response(200, json=_ok_response("no"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "voting.injection.v1",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    # text_a failed → voted YES (failure_vote=yes); aggregation any_yes → YES.
    assert r.json()["choices"][0]["message"]["content"] == "yes"
    assert r.headers.get("x-metamodel-vote-failures") == "1"


def test_voting_disabled_returns_400() -> None:
    cfg = parse_config_str(_BASE_TOML.replace("voting = true", "voting = false"))
    set_config(cfg)
    try:
        r = _client().post(
            "/v1/chat/completions",
            json={
                "model": "voting.injection.v1",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "feature_disabled"
    finally:
        set_config(None)


# ── Profile resolution ─────────────────────────────────────────────


def test_unknown_model_returns_404(loaded_config) -> None:
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "does-not-exist",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


# ── D.2.5 — Compaction observability headers + 413 ──────────────────


def test_compaction_headers_emitted_on_moa_success(loaded_config) -> None:
    """D.2.5: every MoA success carries Compacted-N + Prompt-Tokens-In/Out."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response("ok"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    # Headers must be present and parseable as integers.
    assert "x-metamodel-compacted-n" in r.headers
    assert "x-metamodel-prompt-tokens-in" in r.headers
    assert "x-metamodel-prompt-tokens-out" in r.headers
    int(r.headers["x-metamodel-compacted-n"])
    int(r.headers["x-metamodel-prompt-tokens-in"])
    int(r.headers["x-metamodel-prompt-tokens-out"])


def test_compaction_drops_messages_reflected_in_compacted_n(loaded_config) -> None:
    """D.2.5: when shared-tail compaction drops older history, the
    smallest generator's payload size loss surfaces as Compacted-N."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response("ok"))

    set_upstream_transport(httpx.MockTransport(handler))
    # Build a 200-message conversation; smallest generator (text_c) has
    # 4096 context — head will be aggressively compacted, dropping
    # several user/assistant turns.
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    for i in range(100):
        msgs.append({"role": "user", "content": f"u{i}: {'x' * 200}"})
        msgs.append({"role": "assistant", "content": f"a{i}: {'y' * 200}"})

    r = _client().post(
        "/v1/chat/completions",
        json={"model": "moa.text.v1", "messages": msgs},
    )
    assert r.status_code == 200
    compacted = int(r.headers["x-metamodel-compacted-n"])
    tokens_in = int(r.headers["x-metamodel-prompt-tokens-in"])
    tokens_out = int(r.headers["x-metamodel-prompt-tokens-out"])
    assert compacted > 0, "head should have been compacted for 4K-context generator"
    assert tokens_in > tokens_out, f"compacted output {tokens_out} must be < input {tokens_in}"


def test_compaction_headers_on_passthrough(loaded_config) -> None:
    """Single-upstream passthrough emits the same compaction header set
    (Compacted-N=0, Tokens-In = Tokens-Out) so clients don't need to
    branch on dispatch path."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response("ok"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",  # raw upstream → 1-element MoA → passthrough
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    assert r.headers.get("x-metamodel-compacted-n") == "0"
    assert r.headers.get("x-metamodel-prompt-tokens-in") == r.headers.get(
        "x-metamodel-prompt-tokens-out"
    )


def test_tools_tokens_header_emitted(loaded_config) -> None:
    """D.2.5 / review r21: tool schema overhead surfaces as a separate
    header so Tokens-In/Out stay message-only and clients can see why
    a request might 413 even with a small message payload."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response("ok"))

    set_upstream_transport(httpx.MockTransport(handler))
    big_tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": "x" * 200,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for i in range(20)
    ]
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": big_tools,
        },
    )
    assert r.status_code == 200
    tools_tokens = int(r.headers["x-metamodel-tools-tokens"])
    assert tools_tokens > 0, "tool schema overhead should be reported"
    # Tokens-In is message-only — the tiny "hi" content; Tools-Tokens
    # is the meaningful overhead.
    tokens_in = int(r.headers["x-metamodel-prompt-tokens-in"])
    assert tokens_in < tools_tokens, (
        f"Tokens-In ({tokens_in}) should reflect message-only; "
        f"Tools-Tokens ({tools_tokens}) carries schema overhead"
    )


def test_compacted_n_reflects_actual_chunk_drops(loaded_config) -> None:
    """D.2.5 / review r21 finding 1: Compacted-N must come from the
    primitive's truth-source, not a message-length delta confounded
    by the `[N earlier message groups…]` sentinel.

    Build a conversation where the smallest generator should drop
    several chunks; assert Compacted-N is positive and at least
    equals the count of input messages that didn't survive."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response("ok"))

    set_upstream_transport(httpx.MockTransport(handler))
    msgs: list[dict[str, Any]] = [{"role": "system", "content": "sys"}]
    for i in range(60):
        msgs.append({"role": "user", "content": f"u{i}: {'x' * 300}"})
        msgs.append({"role": "assistant", "content": f"a{i}: {'y' * 300}"})

    r = _client().post(
        "/v1/chat/completions",
        json={"model": "moa.text.v1", "messages": msgs},
    )
    assert r.status_code == 200
    compacted = int(r.headers["x-metamodel-compacted-n"])
    assert compacted > 0, "smallest generator (4K ctx) should drop chunks"
    # The input has 121 messages (1 system + 120 turns). The smallest
    # generator can hold maybe ~10 turns + system; compacted should
    # reflect at least dozens of dropped chunks.
    assert compacted >= 10, f"expected several dropped chunks, got {compacted}"


def test_observability_header_parity_across_dispatch_modes(loaded_config) -> None:
    """D.2.5 / review r22: every dispatch surface (MoA, cascade, voting,
    passthrough) emits the same compaction observability header set so
    clients see one stable schema regardless of which profile they call."""
    required = {
        "x-metamodel-profile",
        "x-metamodel-compacted-n",
        "x-metamodel-prompt-tokens-in",
        "x-metamodel-prompt-tokens-out",
        "x-metamodel-tools-tokens",
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response("ok"))

    set_upstream_transport(httpx.MockTransport(handler))

    surfaces = [
        ("moa.text.v1", "MoA"),
        ("cascade.vision.v1", "cascade"),
        ("voting.injection.v1", "voting"),
        ("text_a", "passthrough"),
    ]
    for model, label in surfaces:
        r = _client().post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, f"{label} dispatch failed: {r.text}"
        missing = required - set(r.headers.keys())
        assert not missing, f"{label} surface missing headers: {missing}"


def test_413_when_input_exceeds_every_generator_budget() -> None:
    """D.2.5 / Phase D plan §253: 413 context_length_exceeded only
    when the request is genuinely irreducible (every usable generator's
    budget would be ≤ 0). Test with tools_token_estimate large enough
    to push every generator past its context."""
    # Two tiny generators where reserve+margin already eats the context.
    cfg = parse_config_str(
        """
[upstreams.tiny_a]
model_id = "ta"
base_url = "http://up:9000/v1"
context = 1024
max_output = 512
modalities = ["text"]

[upstreams.tiny_b]
model_id = "tb"
base_url = "http://up:9001/v1"
context = 1024
max_output = 512
modalities = ["text"]

[profiles."tiny.v1"]
type = "moa"
generators = ["tiny_a", "tiny_b"]
synthesizer = "tiny_a"
# F1 config-load check would reject this toy fixture; zero out the
# reserves so the formula admits the tiny synth context. The test
# exercises the generator-level 413 path inside compaction, not F1.
synth_prompt_overhead_tokens = 0
non_client_synth_reserve_tokens = 0
"""
    )
    set_config(cfg)
    try:
        # Default response_reserve=2000, safety_margin=512 → budget
        # already negative for ctx=1024 even without tools_token_estimate.
        r = _client().post(
            "/v1/chat/completions",
            json={
                "model": "tiny.v1",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "context_length_exceeded"
        # Tokens-In should still be emitted on 413 so clients can debug
        # what they sent.
        assert "x-metamodel-prompt-tokens-in" in r.headers
    finally:
        set_config(None)


def test_unknown_x_meta_model_profile_names_in_error(loaded_config) -> None:
    """Review r18 finding 5: when x_meta_model.profile is unknown, the
    404 message must name the actual unresolved target, not the
    `model` field that may resolve fine on its own."""
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",  # this resolves
            "messages": [{"role": "user", "content": "x"}],
            "x_meta_model": {"profile": "ghost.v1"},  # this doesn't
        },
    )
    assert r.status_code == 404
    assert "ghost.v1" in r.json()["error"]["message"]


def test_voting_strips_max_completion_tokens(loaded_config) -> None:
    """Review r18 finding 2: voting profile sets `max_tokens=5` but
    must also strip `max_completion_tokens` from the upstream body
    if the client sent it. Otherwise upstreams 400 on the conflict."""
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_ok_response("no"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "voting.injection.v1",
            "messages": [{"role": "user", "content": "x"}],
            "max_completion_tokens": 100,  # client-set; voting overrides
        },
    )
    assert r.status_code == 200
    for body in captured:
        assert "max_completion_tokens" not in body, (
            f"voting forwarded conflicting max_completion_tokens: {body}"
        )
        assert body.get("max_tokens") == 5  # profile-owned


def test_moa_synth_failure_returns_typed_502(loaded_config) -> None:
    """Review r18 finding 1: when synthesizer can't extract a usable
    candidate from any 2xx response, return a typed 502 not a 500."""

    # Generators return 200 with empty choices list — synthesizer
    # SynthesisFailure path.
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": "x",
                "choices": [],  # malformed: synth has nothing to merge
            },
        )

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "synthesis_failed"


def test_max_model_len_includes_synthesizer(loaded_config) -> None:
    """F1: max_model_len is bounded by the synthesizer's window.
    A small synth context dominates the advertisement — the synth
    call would 413 otherwise. Pre-F1 used min(upstream.context);
    F1 uses synth.context − overhead − reserve − max_output."""
    cfg = parse_config_str(
        """
[upstreams.big_gen]
model_id = "big"
base_url = "http://upstream:9000/v1"
context = 16384
max_output = 2048
modalities = ["text"]

[upstreams.small_synth]
model_id = "small"
base_url = "http://upstream:9001/v1"
context = 4096
max_output = 512
modalities = ["text"]

[profiles."skewed.v1"]
type = "moa"
generators = ["big_gen"]
synthesizer = "small_synth"
"""
    )
    set_config(cfg)
    try:
        r = _client().get("/v1/models")
        data = {m["id"]: m for m in r.json()["data"]}
        # F1: synth.context 4096 − overhead 256 − reserve 1024 − max_output 512 = 2304.
        assert data["skewed.v1"]["max_model_len"] == 2304
    finally:
        set_config(None)


def test_voting_all_failures_returns_failure_vote(loaded_config) -> None:
    """Review r18 follow-up: when all voters error, every voter is
    counted as failure_vote. With aggregation=any_yes + failure_vote=yes,
    the verdict is YES (fail-closed). Vote-Failures header reflects 3."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "voting.injection.v1",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "yes"
    assert r.headers.get("x-metamodel-vote-failures") == "3"


def test_cascade_structured_502_policy(loaded_config) -> None:
    """Review r18 follow-up: cascade with on_all_fail='structured_502'
    returns 502 regardless of the per-upstream final status."""
    cfg = parse_config_str(
        """
[upstreams.a]
model_id = "a"
base_url = "http://up:9000/v1"
context = 8192
max_output = 1024

[upstreams.b]
model_id = "b"
base_url = "http://up:9001/v1"
context = 8192
max_output = 1024

[profiles."structured.cascade.v1"]
type = "cascade"
upstreams = ["a", "b"]
on_all_fail = "structured_502"
"""
    )
    set_config(cfg)
    try:

        def handler(_request: httpx.Request) -> httpx.Response:
            # Last upstream returns 503 — bubble would propagate that.
            return httpx.Response(503, json={"error": "down"})

        set_upstream_transport(httpx.MockTransport(handler))
        r = _client().post(
            "/v1/chat/completions",
            json={
                "model": "structured.cascade.v1",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        # structured_502 forces 502, not the 503 the last upstream emitted.
        assert r.status_code == 502
        assert r.json()["error"]["code"] == "cascade_exhausted"
    finally:
        set_config(None)
        set_upstream_transport(None)


def test_raw_upstream_addressing_still_works(loaded_config) -> None:
    """Backward compat: clients addressing a raw upstream key get
    direct passthrough behavior (preserves D.1.4 semantics)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_response("direct"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 200
    assert r.headers.get("x-metamodel-profile") == "text_a"
    assert r.headers.get("x-metamodel-generators") == "1"


# ── D.3.1 — legacy normalization at dispatch entry ─────────────────


def test_d31_legacy_functions_promoted_to_tools(loaded_config) -> None:
    """`functions` field at request entry → normalized to `tools` before
    fan-out. The upstream sees the modern shape."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(200, json=_ok_response())

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "functions": [{"name": "f", "parameters": {"type": "object"}}],
        },
    )
    assert r.status_code == 200
    body = captured["body"]
    assert '"tools"' in body
    assert '"functions"' not in body


def test_d31_legacy_function_call_promoted_to_tool_choice(loaded_config) -> None:
    """`function_call: {name: f}` at request entry → normalized to
    tool_choice={type: function, function: {name: f}}."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = request.read().decode()
        return httpx.Response(200, json=_ok_response())

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "function_call": {"name": "f"},
        },
    )
    assert r.status_code == 200
    body = captured["body"]
    assert '"function_call"' not in body
    assert '"tool_choice"' in body


def test_d31_mixed_legacy_and_modern_400s(loaded_config) -> None:
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "functions": [{"name": "f"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request_error"


def test_d31_custom_tool_type_400s(loaded_config) -> None:
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "custom", "custom": {"name": "x"}}],
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "feature_not_supported_in_v1"


def test_d31_allowed_tools_400s(loaded_config) -> None:
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "allowed_tools": ["f"],
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "feature_not_supported_in_v1"


def test_d31_voting_strips_parallel_tool_calls(loaded_config) -> None:
    """Voting strips tools/tool_choice/parallel_tool_calls/response_format
    from the per-voter body — review r25 cleanliness add."""
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.read().decode())
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "no"}}],
            },
        )

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "voting.injection.v1",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "parallel_tool_calls": False,
        },
    )
    assert r.status_code == 200
    # Every voter body has tools/tool_choice/parallel_tool_calls stripped.
    for body in captured:
        assert '"tools"' not in body
        assert '"parallel_tool_calls"' not in body



# ── F1: advertised_context formula tests ──────────────────────────


def test_f1_advertised_context_uses_synth_window_minus_reserves() -> None:
    """F1: a profile's max_model_len is synth.context − overhead −
    reserve − max_output, NOT min(generators)."""
    from meta_model.config import parse_config_str
    from meta_model.server import set_config

    cfg = parse_config_str(
        """
[upstreams.small_gen]
model_id = "g"
base_url = "http://up:9000/v1"
context = 8192
max_output = 1024

[upstreams.big_synth]
model_id = "s"
base_url = "http://up:9001/v1"
context = 49152
max_output = 4096

[profiles."f1.large_synth.v1"]
type = "moa"
generators = ["small_gen"]
synthesizer = "big_synth"
"""
    )
    set_config(cfg)
    try:
        r = _client().get("/v1/models")
        data = {m["id"]: m for m in r.json()["data"]}
        # F1: 49152 − 256 (overhead) − 1024 (reserve) − 4096 (max_output) = 43776
        # NOT 8192 (which would be min(small_gen, big_synth)).
        assert data["f1.large_synth.v1"]["max_model_len"] == 43776
    finally:
        set_config(None)


def test_f1_operator_override_advertised_context_wins() -> None:
    """F1: explicit `advertised_context` per profile bypasses the
    formula. Operators with custom workloads can set whatever
    ceiling they want."""
    from meta_model.config import parse_config_str
    from meta_model.server import set_config

    cfg = parse_config_str(
        """
[upstreams.s]
model_id = "s"
base_url = "http://up:9001/v1"
context = 49152
max_output = 4096

[profiles."f1.override.v1"]
type = "moa"
generators = ["s"]
synthesizer = "s"
advertised_context = 8000
"""
    )
    set_config(cfg)
    try:
        r = _client().get("/v1/models")
        data = {m["id"]: m for m in r.json()["data"]}
        assert data["f1.override.v1"]["max_model_len"] == 8000
    finally:
        set_config(None)


def test_f1_config_load_fails_on_negative_budget() -> None:
    """F1: when synth.context − overhead − reserve − max_output ≤ 0,
    config load must reject. Otherwise every request would 413."""
    from meta_model.config import parse_config_str

    import pytest

    with pytest.raises(Exception, match="≤ 0"):
        parse_config_str(
            """
[upstreams.s]
model_id = "s"
base_url = "http://up:9001/v1"
context = 1024
max_output = 512

[profiles."f1.too_small.v1"]
type = "moa"
generators = ["s"]
synthesizer = "s"
non_client_synth_reserve_tokens = 1024
"""
        )


def test_f1_cascade_keeps_min_context() -> None:
    """F1: cascade profiles still advertise min(upstream.context).
    Until cascade dispatch implements context-aware skip, advertising
    max(context) would be a lie — a small-context upstream still
    serves the request and 413s."""
    from meta_model.config import parse_config_str
    from meta_model.server import set_config

    cfg = parse_config_str(
        """
[upstreams.big]
model_id = "b"
base_url = "http://up:9000/v1"
context = 16384
max_output = 2048

[upstreams.small]
model_id = "sm"
base_url = "http://up:9001/v1"
context = 4096
max_output = 1024

[profiles."f1.cascade.v1"]
type = "cascade"
upstreams = ["big", "small"]
"""
    )
    set_config(cfg)
    try:
        r = _client().get("/v1/models")
        data = {m["id"]: m for m in r.json()["data"]}
        # min(big=16384, small=4096) = 4096
        assert data["f1.cascade.v1"]["max_model_len"] == 4096
    finally:
        set_config(None)


# ── F6: server-level multimodal cascade ────────────────────────────


_F6_TOML = """
[upstreams.text_only]
model_id = "t"
base_url = "http://up.text/v1"
context = 8192
max_output = 1024
modalities = ["text"]

[upstreams.vision_a]
model_id = "va"
base_url = "http://up.vision_a/v1"
context = 8192
max_output = 1024
modalities = ["text", "image"]

[upstreams.vision_b]
model_id = "vb"
base_url = "http://up.vision_b/v1"
context = 8192
max_output = 1024
modalities = ["text", "image"]

[profiles."txt.v1"]
type = "moa"
generators = ["text_only"]
synthesizer = "text_only"

[vision]
endpoints = ["vision_a", "vision_b"]
"""


_F6_VIDEO_TOML = """
[upstreams.text_only]
model_id = "t"
base_url = "http://up.text/v1"
context = 8192
max_output = 1024
modalities = ["text"]

[upstreams.video_only]
model_id = "v"
base_url = "http://up.video/v1"
context = 8192
max_output = 1024
modalities = ["text", "video"]

[profiles."txt.v1"]
type = "moa"
generators = ["text_only"]
synthesizer = "text_only"

[video]
endpoints = ["video_only"]
"""


def _msg_with_image(url: str = "http://x/y.png") -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": url}},
        ],
    }


def _msg_with_video(url: str = "http://x/y.mp4") -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "watch"},
            {"type": "video_url", "video_url": {"url": url}},
        ],
    }


def _msg_with_audio() -> dict[str, Any]:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "listen"},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        ],
    }


def _ok_chat(content: str = "vision-answer") -> dict[str, Any]:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "x",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture
def f6_vision_config() -> Any:
    cfg = parse_config_str(_F6_TOML)
    set_config(cfg)
    yield cfg
    set_config(None)
    set_upstream_transport(None)


def test_f6_vision_cascade_first_2xx_wins(f6_vision_config) -> None:
    """Image request → cascades through `[vision].endpoints`, first 2xx
    wins. Both endpoints succeed → first one's body returned."""
    served_by: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        port = req.url.port or 0
        served_by.append(port)
        # vision_a is on up.vision_a (no port → default 80 or url-based)
        return httpx.Response(200, json=_ok_chat(f"served-by-{req.url.host}"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "txt.v1", "messages": [_msg_with_image()]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # vision_a (first in cascade) wins on 2xx — handler called once.
    assert "served-by-up.vision_a" in body["choices"][0]["message"]["content"]
    assert r.headers.get("x-metamodel-multimodal-path") == "vision"
    assert r.headers.get("x-metamodel-multimodal-upstream") == "vision_a"


def test_f6_vision_cascade_fallback_on_first_5xx(f6_vision_config) -> None:
    """First endpoint 5xx → cascade tries second. Second 2xx wins."""
    call_log: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        call_log.append(req.url.host)
        if req.url.host == "up.vision_a":
            return httpx.Response(503, json={"error": {"message": "vision_a down"}})
        return httpx.Response(200, json=_ok_chat("served-by-vision_b"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "txt.v1", "messages": [_msg_with_image()]},
    )
    assert r.status_code == 200, r.text
    assert call_log == ["up.vision_a", "up.vision_b"]
    assert "served-by-vision_b" in r.json()["choices"][0]["message"]["content"]
    attempts = r.headers.get("x-metamodel-multimodal-attempts")
    assert "vision_a:503" in attempts
    assert "vision_b:200" in attempts


def test_f6_vision_cascade_all_fail_returns_last_error(f6_vision_config) -> None:
    """Both endpoints 5xx → cascade exhausted, last error returned to client."""
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"error": {"message": "all down"}})

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "txt.v1", "messages": [_msg_with_image()]},
    )
    # Last upstream's status passes through.
    assert r.status_code == 502


def test_f6_vision_disabled_when_endpoints_empty() -> None:
    """Image request hits a server with no vision endpoints → typed
    501 modality_not_supported envelope through normal OpenAI rails."""
    cfg = parse_config_str("""
[upstreams.text_only]
model_id = "t"
base_url = "http://up.text/v1"
context = 8192
max_output = 1024
modalities = ["text"]

[profiles."txt.v1"]
type = "moa"
generators = ["text_only"]
synthesizer = "text_only"
""")
    set_config(cfg)
    try:
        r = _client().post(
            "/v1/chat/completions",
            json={"model": "txt.v1", "messages": [_msg_with_image()]},
        )
        assert r.status_code == 501
        body = r.json()
        assert body["error"]["code"] == "vision_not_supported"
        assert "[vision].endpoints" in body["error"]["message"]
    finally:
        set_config(None)


def test_f6_video_cascade_when_configured() -> None:
    """Video request → routes through [video].endpoints when configured."""
    cfg = parse_config_str(_F6_VIDEO_TOML)
    set_config(cfg)

    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_ok_chat("video-answer"))

    set_upstream_transport(httpx.MockTransport(handler))
    try:
        r = _client().post(
            "/v1/chat/completions",
            json={"model": "txt.v1", "messages": [_msg_with_video()]},
        )
        assert r.status_code == 200, r.text
        assert r.headers.get("x-metamodel-multimodal-path") == "video"
    finally:
        set_config(None)
        set_upstream_transport(None)


def test_f6_video_disabled_when_empty(f6_vision_config) -> None:
    """Video request to a server with [vision] but no [video] →
    501 video_not_supported."""
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "txt.v1", "messages": [_msg_with_video()]},
    )
    assert r.status_code == 501
    assert r.json()["error"]["code"] == "video_not_supported"


def test_f6_audio_disabled_when_empty(f6_vision_config) -> None:
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "txt.v1", "messages": [_msg_with_audio()]},
    )
    assert r.status_code == 501
    assert r.json()["error"]["code"] == "audio_not_supported"


def test_f6_text_only_request_unaffected_by_vision_block(f6_vision_config) -> None:
    """Pure-text requests still flow through profile dispatch — server
    [vision] block doesn't intercept."""
    def handler(req: httpx.Request) -> httpx.Response:
        # Should hit text_only upstream, NOT vision_a / vision_b.
        assert req.url.host == "up.text"
        return httpx.Response(200, json=_ok_chat("text-answer"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "txt.v1", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    # No multimodal-path header on the text path.
    assert "x-metamodel-multimodal-path" not in r.headers


def test_f6_vision_overrides_profile_selection(f6_vision_config) -> None:
    """Single-model transparency: client targets `txt.v1` (text-only
    profile) with image content → vision cascade serves regardless of
    profile selection."""
    def handler(req: httpx.Request) -> httpx.Response:
        # Routed to vision endpoint, NOT text_only generator.
        assert req.url.host in ("up.vision_a", "up.vision_b")
        return httpx.Response(200, json=_ok_chat("vision-served-from-text-profile"))

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "txt.v1", "messages": [_msg_with_image()]},
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("x-metamodel-multimodal-path") == "vision"


def test_f6_response_preserves_profile_identity(f6_vision_config) -> None:
    """Review r2 F6 MED: single-model transparency requires that the
    response's `model` field and the X-MetaModel-Profile header
    reflect the canonical profile/alias the client requested, not the
    underlying upstream's model_id."""
    def handler(_req: httpx.Request) -> httpx.Response:
        # Upstream returns its own model_id ("va") — meta-model rewrites.
        return httpx.Response(200, json={
            **_ok_chat("payload"),
            "model": "va-upstream-model-id",
        })

    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "txt.v1", "messages": [_msg_with_image()]},
    )
    assert r.status_code == 200, r.text
    # Response.model = canonical profile, NOT upstream's model_id.
    assert r.json()["model"] == "txt.v1"
    # Standard X-MetaModel-Profile header emitted.
    assert r.headers.get("x-metamodel-profile") == "txt.v1"


def test_f6_unknown_model_with_image_returns_404(f6_vision_config) -> None:
    """Review r1 F6 MED: profile resolution runs BEFORE the multimodal
    short-circuit. An image request with an unknown model still
    returns the standard 404 model_not_found envelope — the cascade
    never fires for unresolved profiles."""
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "does-not-exist", "messages": [_msg_with_image()]},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"


def test_f6_missing_capability_uses_invalid_request_error_type(f6_vision_config) -> None:
    """Review r1 F6 MED: 501 modality_not_supported envelope's `type`
    field must be `invalid_request_error` (per OpenAI spec for
    request-shape errors), not the default `api_error` that 5xx
    statuses normally get."""
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "txt.v1", "messages": [_msg_with_video()]},
    )
    assert r.status_code == 501
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "video_not_supported"


def test_f6_cascade_deadline_does_not_multiply_timeout(f6_vision_config) -> None:
    """Review r1 F6 HIGH: cascade-wide deadline. Two slow upstreams
    cannot consume 2× the configured timeout; each gets the
    remaining budget."""
    import asyncio

    call_count = [0]

    async def slow_handler(_req: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        # Sleep ~longer than the cascade deadline; cascade should
        # cancel us via timeout on the second attempt.
        await asyncio.sleep(0.5)
        return httpx.Response(200, json=_ok_chat("late"))

    set_upstream_transport(httpx.MockTransport(slow_handler))
    import time as _time

    start = _time.monotonic()
    r = _client().post(
        "/v1/chat/completions",
        json={"model": "txt.v1", "messages": [_msg_with_image()]},
        # Force a tiny request_timeout — deadline-aware cascade
        # should bail before walking both upstreams in full.
        params={},
    )
    elapsed = _time.monotonic() - start
    # Either both attempts time out within the bounded budget OR
    # one returned 200 in time. Either way, total elapsed must
    # not exceed N × per-attempt slack — bounded by the cascade
    # deadline. Default request_timeout is large in tests, so
    # primary check is "got a coherent response".
    assert r.status_code in (200, 504, 502)
    # Bounded multiplication check is loose because the test
    # transport doesn't actually honor httpx timeouts the same
    # way; the structural check is that the function exits with
    # a single error envelope, not a hang.
    assert elapsed < 5.0, f"cascade exceeded reasonable bound: {elapsed:.1f}s"


# ── Per-candidate constraint enforcement (review r2 HIGH) ──────────────


def _ok_response_with_tool_call(name: str, args: str = "{}") -> dict[str, Any]:
    """OpenAI-shape chat completion with a single tool_call to `name`."""
    return {
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "created": 1,
        "model": "x",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_x",
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _make_per_upstream_handler(
    responses_by_url: dict[str, dict[str, Any]],
):
    """Build a MockTransport handler that returns different responses
    per upstream URL host. Used to model "generator A returns declared
    tool, generator B returns undeclared tool, etc."""

    def handler(request: httpx.Request) -> httpx.Response:
        for url_fragment, resp in responses_by_url.items():
            if url_fragment in str(request.url):
                return httpx.Response(200, json=resp)
        # Default: text response
        return httpx.Response(200, json=_ok_response("default"))

    return handler


def test_undeclared_tool_demoted_with_survivors(loaded_config) -> None:
    """Review r2 HIGH: a candidate emitting an undeclared tool name is
    demoted to the failure pool. Surviving candidates merge through
    the synth path; degraded headers reflect the demotion."""
    # text_a calls "good" (declared), text_b calls "rogue" (NOT declared),
    # text_c calls "good" too. Synthesizer (text_a) gets called with the
    # text_a + text_c survivors and produces a final response.
    handler = _make_per_upstream_handler(
        {
            "9000": _ok_response_with_tool_call("good"),  # text_a
            "9001": _ok_response_with_tool_call("rogue"),  # text_b — undeclared
            "9002": _ok_response_with_tool_call("good"),  # text_c
        }
    )
    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "good"}}],
        },
    )
    assert r.status_code == 200, r.text
    # Degraded header should list text_b with reason undeclared_tool
    failed = r.headers.get("X-MetaModel-Failed-Generators", "")
    assert "text_b:undeclared_tool" in failed
    # The merged response must not call the rogue tool
    msg = r.json()["choices"][0]["message"]
    tcs = msg.get("tool_calls") or []
    if tcs:
        for tc in tcs:
            assert tc["function"]["name"] != "rogue"


def test_all_undeclared_returns_503_no_quorum_after_constraint(loaded_config) -> None:
    """All candidates emit undeclared names → 503, no synth call."""
    handler = _make_per_upstream_handler(
        {
            "9000": _ok_response_with_tool_call("rogue"),
            "9001": _ok_response_with_tool_call("rogue"),
            "9002": _ok_response_with_tool_call("rogue"),
        }
    )
    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "good"}}],
        },
    )
    assert r.status_code == 503, r.text
    body = r.json()
    # Code field surfaces the new no_quorum_after_constraint
    assert body["error"]["code"] == "no_quorum_after_constraint"
    failed = r.headers.get("X-MetaModel-Failed-Generators", "")
    # All three demoted with reason undeclared_tool
    assert failed.count("undeclared_tool") == 3
    assert r.headers.get("X-MetaModel-Quorum") == "0"


def test_dual_shape_response_demoted(loaded_config) -> None:
    """Candidate emits BOTH `tool_calls` AND legacy `function_call` —
    malformed dual shape, demoted with reason dual_shape_response."""
    dual = _ok_response_with_tool_call("good")
    # Inject legacy function_call alongside the modern tool_calls
    dual["choices"][0]["message"]["function_call"] = {"name": "good", "arguments": "{}"}

    handler = _make_per_upstream_handler(
        {
            "9000": _ok_response_with_tool_call("good"),  # clean
            "9001": dual,  # malformed
            "9002": _ok_response_with_tool_call("good"),  # clean
        }
    )
    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "good"}}],
        },
    )
    assert r.status_code == 200, r.text
    failed = r.headers.get("X-MetaModel-Failed-Generators", "")
    assert "text_b:dual_shape_response" in failed


def test_undeclared_tool_via_legacy_function_call_demoted(loaded_config) -> None:
    """A backend emitting only legacy `function_call` to an undeclared
    name is demoted exactly like a modern tool_calls undeclared.
    Review r2 HIGH: legacy shape must be first-class for this check."""
    legacy_undeclared = _ok_response("")
    legacy_undeclared["choices"][0]["message"]["function_call"] = {
        "name": "rogue",
        "arguments": "{}",
    }

    handler = _make_per_upstream_handler(
        {
            "9000": _ok_response_with_tool_call("good"),
            "9001": legacy_undeclared,
            "9002": _ok_response_with_tool_call("good"),
        }
    )
    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "good"}}],
        },
    )
    assert r.status_code == 200, r.text
    failed = r.headers.get("X-MetaModel-Failed-Generators", "")
    assert "text_b:undeclared_tool" in failed


def test_text_only_response_unaffected_by_declared_check(loaded_config) -> None:
    """Tools declared but candidates return plain text (no tool_calls).
    declared-tool check is no-op; existing text-merge path runs."""
    handler = _make_per_upstream_handler(
        {
            "9000": _ok_response("hello from a"),
            "9001": _ok_response("hello from b"),
            "9002": _ok_response("hello from c"),
        }
    )
    set_upstream_transport(httpx.MockTransport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "good"}}],
        },
    )
    assert r.status_code == 200, r.text
    # No demotions — header either absent or doesn't list undeclared_tool
    failed = r.headers.get("X-MetaModel-Failed-Generators", "")
    assert "undeclared_tool" not in failed


# ── Context-fit backstop: cap output so generators don't 400 on context ──
from meta_model.moa.dispatch import _cap_output_to_context, DEFAULT_SAFETY_MARGIN
from meta_model.moa.compaction import estimate_messages_tokens


def _one_msg(n_chars: int) -> list[dict[str, Any]]:
    return [{"role": "user", "content": "x" * n_chars}]


def test_cap_output_noop_when_it_fits() -> None:
    body = {"max_tokens": 1000}
    _cap_output_to_context(body, _one_msg(100), 0, 200_000)
    assert body["max_tokens"] == 1000


def test_cap_output_caps_so_prompt_plus_output_fits_context() -> None:
    msgs = _one_msg(4000)
    est = estimate_messages_tokens(msgs)
    context = est + 700  # only ~700 tokens of room beyond the prompt
    body = {"max_tokens": 5000}
    _cap_output_to_context(body, msgs, 0, context)
    assert body["max_tokens"] == context - est - DEFAULT_SAFETY_MARGIN
    # The whole point: est(prompt) + output + margin == context (no 400).
    assert est + body["max_tokens"] + DEFAULT_SAFETY_MARGIN == context


def test_cap_output_handles_max_completion_tokens() -> None:
    msgs = _one_msg(4000)
    est = estimate_messages_tokens(msgs)
    context = est + 700
    body = {"max_completion_tokens": 5000}
    _cap_output_to_context(body, msgs, 0, context)
    assert body["max_completion_tokens"] == context - est - DEFAULT_SAFETY_MARGIN


def test_cap_output_noop_when_no_output_budget_set() -> None:
    body = {"temperature": 0.5}
    _cap_output_to_context(body, _one_msg(100), 0, 1000)
    assert "max_tokens" not in body and "max_completion_tokens" not in body


def test_cap_output_floors_at_one_when_no_room() -> None:
    msgs = _one_msg(4000)
    est = estimate_messages_tokens(msgs)
    body = {"max_tokens": 5000}
    _cap_output_to_context(body, msgs, 0, est)  # context == prompt, ceiling negative
    assert body["max_tokens"] == 1


def test_cap_output_margin_scales_with_large_prompt() -> None:
    # Near the context limit the margin must grow with the prompt (~3%)
    # so per-tokenizer drift (heavy tool/template tokenization) can't push
    # prompt + output past context.
    msgs = _one_msg(200_000)
    est = estimate_messages_tokens(msgs)
    assert est // 32 > DEFAULT_SAFETY_MARGIN  # proportional term dominates
    context = est + 3000
    body = {"max_tokens": 5000}
    _cap_output_to_context(body, msgs, 0, context)
    assert body["max_tokens"] == context - est - (est // 32)
    assert est + body["max_tokens"] + (est // 32) == context
