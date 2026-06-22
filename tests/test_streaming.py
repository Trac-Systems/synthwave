"""D.3.3 — simulated SSE streaming tests.

Three categories of test:

1. **ASGI-buffered tests** drive the full stack via FastAPI's
   ``TestClient`` + a ``MockTransport`` for upstream calls. ``httpx``'s
   ASGI transport buffers the entire body before yielding bytes, so
   these tests assert *final-state SSE ordering*: the right events,
   in the right order, with the right shape. They CANNOT prove
   live heartbeat delivery — see (3) for that.
2. **Chunking-helper unit tests** poke ``_chunk_content`` directly to
   verify the round-trip invariant ``"".join(chunks) == text``.
3. **Direct-generator-drive tests** instantiate ``stream_chat_completion``
   directly with a stubbed dispatch callable. These reveal heartbeat
   timing (interval injectable) and client-disconnect cancellation
   semantics that the buffered ASGI path obscures.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from meta_model.config import parse_config_str
from meta_model.moa.dispatch import DispatchResult
from meta_model.server import app, set_config, set_upstream_transport
from meta_model.streaming import (
    MalformedSynthResponse,
    _chunk_content,
    _dispatch_result_to_error_envelope,
    _format_sse_comment,
    _format_sse_data,
    _format_sse_done,
    _response_to_chunks,
    new_chunk_id,
    stream_chat_completion,
)

# ── TOML fixture (mirrors test_chat_completions but adds a
# multi-generator MoA profile for richer coverage) ──────────────────

_BASE_TOML = """
[upstreams.text_a]
model_id = "text-a-actual"
base_url = "http://upstream:9000/v1"
context = 8192
max_output = 2048

[upstreams.text_b]
model_id = "text-b-actual"
base_url = "http://upstream:9001/v1"
context = 8192
max_output = 2048

[features]
voting = true

[profiles."single.v1"]
type = "moa"
generators = ["text_a"]
synthesizer = "text_a"

[profiles."multi.v1"]
type = "moa"
generators = ["text_a", "text_b"]
synthesizer = "text_a"

[profiles."best.v1"]
type = "moa"
generators = ["text_a", "text_b"]
synthesizer = "text_a"
synthesis_mode = "best-of"

[profiles."single_cascade.v1"]
type = "cascade"
upstreams = ["text_a"]

[profiles."voting.v1"]
type = "voting"
upstreams = ["text_a", "text_b"]
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


def _ok_text(content: str = "Hello world") -> dict[str, Any]:
    """Synthetic upstream response — text content."""
    return {
        "id": "upstream-1",
        "object": "chat.completion",
        "created": 1,
        "model": "text-a-actual",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def _ok_tool_call(*, calls: int = 1) -> dict[str, Any]:
    """Synthetic upstream response — tool_calls."""
    tool_calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": json.dumps({"q": f"q{i}"}),
            },
        }
        for i in range(calls)
    ]
    return {
        "id": "upstream-1",
        "object": "chat.completion",
        "created": 1,
        "model": "text-a-actual",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def _parse_sse(raw: bytes) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Parse an SSE byte stream into ``(events, comments, saw_done)``.

    ``events`` is the parsed JSON of every ``data:`` event other than
    ``[DONE]``. ``comments`` is the text of every ``:`` comment line.
    """
    events: list[dict[str, Any]] = []
    comments: list[str] = []
    saw_done = False
    text = raw.decode("utf-8", errors="replace")
    for chunk in text.split("\n\n"):
        if not chunk.strip():
            continue
        # Each chunk is one or more lines.
        for line in chunk.split("\n"):
            if line.startswith(":"):
                comments.append(line[1:].strip())
            elif line.startswith("data: "):
                payload = line[len("data: ") :]
                if payload == "[DONE]":
                    saw_done = True
                else:
                    events.append(json.loads(payload))
    return events, comments, saw_done


# ── _chunk_content helper tests ─────────────────────────────────────


def test_chunk_content_empty_yields_nothing() -> None:
    assert list(_chunk_content("")) == []


def test_chunk_content_short_yields_one() -> None:
    chunks = list(_chunk_content("hi"))
    assert chunks == ["hi"]


def test_chunk_content_round_trip_long_text() -> None:
    text = "The quick brown fox jumps over the lazy dog. " * 5
    assert "".join(_chunk_content(text)) == text


def test_chunk_content_preserves_leading_whitespace() -> None:
    text = "    leading"
    assert "".join(_chunk_content(text)) == text


def test_chunk_content_preserves_trailing_whitespace() -> None:
    text = "trailing    "
    assert "".join(_chunk_content(text)) == text


def test_chunk_content_preserves_double_spaces() -> None:
    text = "word1  word2  word3   word4"
    assert "".join(_chunk_content(text)) == text


def test_chunk_content_preserves_newlines() -> None:
    text = "line one\nline two\nline three\n"
    assert "".join(_chunk_content(text)) == text


def test_chunk_content_round_trip_unicode() -> None:
    text = "héllo wörld — über kürzlich naïve résumé café"
    assert "".join(_chunk_content(text)) == text


def test_chunk_content_long_token_emits_oversized() -> None:
    # No whitespace at all; chunker must emit at least one chunk per
    # target_size window rather than infinite-loop.
    text = "A" * 250
    chunks = list(_chunk_content(text, target_size=40))
    assert "".join(chunks) == text
    assert len(chunks) >= 5


# ── _format_* helpers ───────────────────────────────────────────────


def test_format_sse_data_shape() -> None:
    out = _format_sse_data({"a": 1})
    assert out == b'data: {"a":1}\n\n'


def test_format_sse_comment_shape() -> None:
    assert _format_sse_comment("heartbeat") == b": heartbeat\n\n"


def test_format_sse_done_shape() -> None:
    assert _format_sse_done() == b"data: [DONE]\n\n"


# ── _response_to_chunks (direct unit tests) ──────────────────────────


def test_response_to_chunks_text_emits_role_content_final() -> None:
    chunks = list(
        _response_to_chunks(
            _ok_text("Hello"),
            chunk_id="chatcmpl-x",
            model_label="single.v1",
            created_ts=42,
            include_usage=False,
        )
    )
    # role + content* + final
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert chunks[0]["choices"][0]["finish_reason"] is None
    middle = [c for c in chunks[1:-1] if "content" in c["choices"][0]["delta"]]
    assert "".join(c["choices"][0]["delta"]["content"] for c in middle) == "Hello"
    final = chunks[-1]
    assert final["choices"][0]["delta"] == {}
    assert final["choices"][0]["finish_reason"] == "stop"


def test_response_to_chunks_tool_calls_role_has_null_content() -> None:
    chunks = list(
        _response_to_chunks(
            _ok_tool_call(calls=2),
            chunk_id="chatcmpl-x",
            model_label="single.v1",
            created_ts=42,
            include_usage=False,
        )
    )
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": None}
    tool_chunks = [c for c in chunks if "tool_calls" in c["choices"][0]["delta"]]
    assert len(tool_chunks) == 2
    assert tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert tool_chunks[1]["choices"][0]["delta"]["tool_calls"][0]["index"] == 1
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_response_to_chunks_include_usage_emits_usage_chunk() -> None:
    chunks = list(
        _response_to_chunks(
            _ok_text("Hello"),
            chunk_id="chatcmpl-x",
            model_label="single.v1",
            created_ts=42,
            include_usage=True,
        )
    )
    # Every non-usage chunk has usage: null.
    for c in chunks[:-1]:
        assert c["usage"] is None
    last = chunks[-1]
    assert last["choices"] == []
    assert last["usage"] == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}


def test_response_to_chunks_no_include_usage_no_usage_field() -> None:
    chunks = list(
        _response_to_chunks(
            _ok_text("Hi"),
            chunk_id="chatcmpl-x",
            model_label="single.v1",
            created_ts=42,
            include_usage=False,
        )
    )
    for c in chunks:
        assert "usage" not in c


def test_response_to_chunks_share_id_and_created() -> None:
    chunks = list(
        _response_to_chunks(
            _ok_text("Hello world this is a longer response"),
            chunk_id="chatcmpl-shared",
            model_label="single.v1",
            created_ts=100,
            include_usage=True,
        )
    )
    # All chunks (incl. final usage chunk) share id + created.
    for c in chunks:
        assert c["id"] == "chatcmpl-shared"
        assert c["created"] == 100


def test_response_to_chunks_rejects_payload_without_choices() -> None:
    """Review r36 #1: 2xx error-shaped payload must not become a fake stream."""
    with pytest.raises(MalformedSynthResponse):
        list(
            _response_to_chunks(
                {"error": {"message": "bad"}},
                chunk_id="chatcmpl-x",
                model_label="x",
                created_ts=1,
                include_usage=False,
            )
        )


def test_response_to_chunks_rejects_empty_choices() -> None:
    with pytest.raises(MalformedSynthResponse):
        list(
            _response_to_chunks(
                {"choices": []},
                chunk_id="chatcmpl-x",
                model_label="x",
                created_ts=1,
                include_usage=False,
            )
        )


def test_response_to_chunks_rejects_missing_message() -> None:
    with pytest.raises(MalformedSynthResponse):
        list(
            _response_to_chunks(
                {"choices": [{"finish_reason": "stop"}]},
                chunk_id="chatcmpl-x",
                model_label="x",
                created_ts=1,
                include_usage=False,
            )
        )


def test_response_to_chunks_rejects_non_list_tool_calls() -> None:
    """Review r37: tool_calls must be a list — non-list = protocol error."""
    with pytest.raises(MalformedSynthResponse):
        list(
            _response_to_chunks(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "tool_calls": "oops"},
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                chunk_id="chatcmpl-x",
                model_label="m",
                created_ts=1,
                include_usage=False,
            )
        )


def test_response_to_chunks_rejects_non_dict_tool_call_item() -> None:
    with pytest.raises(MalformedSynthResponse):
        list(
            _response_to_chunks(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "tool_calls": ["nope"]},
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                chunk_id="chatcmpl-x",
                model_label="m",
                created_ts=1,
                include_usage=False,
            )
        )


def test_response_to_chunks_rejects_non_dict_function() -> None:
    with pytest.raises(MalformedSynthResponse):
        list(
            _response_to_chunks(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "tool_calls": [{"id": "c", "type": "function", "function": "bad"}],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                chunk_id="chatcmpl-x",
                model_label="m",
                created_ts=1,
                include_usage=False,
            )
        )


def test_response_to_chunks_rejects_non_dict_function_call() -> None:
    with pytest.raises(MalformedSynthResponse):
        list(
            _response_to_chunks(
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "function_call": "bad"},
                            "finish_reason": "function_call",
                        }
                    ]
                },
                chunk_id="chatcmpl-x",
                model_label="m",
                created_ts=1,
                include_usage=False,
            )
        )


def test_malformed_tool_calls_2xx_emits_protocol_error(loaded_config) -> None:
    """Review r37 end-to-end: 2xx body with malformed tool_calls must
    NOT leak as internal_error — must be upstream_protocol_error."""
    bad_body = {
        "id": "u",
        "object": "chat.completion",
        "created": 1,
        "model": "text-a-actual",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "tool_calls": "not-a-list"},
                "finish_reason": "tool_calls",
            }
        ],
    }
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=bad_body)))
    r = _stream_post({"model": "text_a", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    events, _c, done = _parse_sse(r.content)
    assert done is False
    assert events[0]["error"]["code"] == "upstream_protocol_error"


def test_response_to_chunks_legacy_function_call_emits_delta() -> None:
    """Review r36 #2: legacy ``message.function_call`` must surface as a
    ``delta.function_call`` chunk (legacy OpenAI streaming shape)."""
    resp = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {"name": "lookup", "arguments": '{"q":"x"}'},
                },
                "finish_reason": "function_call",
            }
        ]
    }
    chunks = list(
        _response_to_chunks(
            resp,
            chunk_id="chatcmpl-x",
            model_label="m",
            created_ts=1,
            include_usage=False,
        )
    )
    fc_chunks = [c for c in chunks if "function_call" in c["choices"][0]["delta"]]
    assert len(fc_chunks) == 1
    fc = fc_chunks[0]["choices"][0]["delta"]["function_call"]
    assert fc["name"] == "lookup"
    assert fc["arguments"] == '{"q":"x"}'
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": None}
    assert chunks[-1]["choices"][0]["finish_reason"] == "function_call"


def test_response_to_chunks_empty_content_no_content_deltas() -> None:
    chunks = list(
        _response_to_chunks(
            _ok_text(""),
            chunk_id="chatcmpl-x",
            model_label="single.v1",
            created_ts=42,
            include_usage=False,
        )
    )
    # role + final only.
    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[1]["choices"][0]["delta"] == {}


# ── _dispatch_result_to_error_envelope ──────────────────────────────


def test_error_envelope_from_explicit_error_tuple() -> None:
    res = DispatchResult(payload=None, status_code=429, error=("rate limited", "rate_limited"))
    env = _dispatch_result_to_error_envelope(res)
    assert env == {
        "error": {
            "message": "rate limited",
            "type": "rate_limit_error",
            "param": None,
            "code": "rate_limited",
        }
    }


def test_error_envelope_from_status_only_passthrough() -> None:
    """Review r34 F1: raw passthrough may relay 429 with payload set
    and error=None. Streaming must synthesize a meta-model envelope."""
    res = DispatchResult(
        payload={"some": "upstream body"},  # discarded
        status_code=502,
        error=None,
    )
    env = _dispatch_result_to_error_envelope(res)
    assert env["error"]["message"] == "upstream returned HTTP 502"
    assert env["error"]["type"] == "api_error"
    assert env["error"]["code"] == "upstream_error"


# ── ASGI-buffered tests ─────────────────────────────────────────────


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def _stream_post(body: dict[str, Any]) -> httpx.Response:
    return _client().post("/v1/chat/completions", json={**body, "stream": True})


def test_basic_content_stream_role_content_final_done(loaded_config) -> None:
    set_upstream_transport(
        _mock_transport(lambda req: httpx.Response(200, json=_ok_text("Hello world")))
    )
    r = _stream_post({"model": "text_a", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    events, _comments, done = _parse_sse(r.content)
    assert done is True
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    middle = [e for e in events[1:-1] if "content" in e["choices"][0]["delta"]]
    assert "".join(e["choices"][0]["delta"]["content"] for e in middle) == "Hello world"


def test_chunk_envelope_shape_each_chunk(loaded_config) -> None:
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=_ok_text("Hi"))))
    r = _stream_post({"model": "text_a", "messages": [{"role": "user", "content": "x"}]})
    events, _c, _d = _parse_sse(r.content)
    for e in events:
        assert e["object"] == "chat.completion.chunk"
        assert isinstance(e["created"], int)
        assert e["model"]
        assert re.match(r"^chatcmpl-[0-9a-f]{24}$", e["id"]), e["id"]
        assert e["choices"][0]["index"] == 0


def test_chunks_share_created_timestamp(loaded_config) -> None:
    """Review r34 minor add: same created across non-usage chunks."""
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=_ok_text("Hello"))))
    r = _stream_post(
        {
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "stream_options": {"include_usage": True},
        }
    )
    events, _c, _d = _parse_sse(r.content)
    timestamps = {e["created"] for e in events}
    assert len(timestamps) == 1


def test_initial_delta_role_finish_reason_null(loaded_config) -> None:
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=_ok_text("Hi"))))
    r = _stream_post({"model": "text_a", "messages": [{"role": "user", "content": "x"}]})
    events, _c, _d = _parse_sse(r.content)
    first = events[0]
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert first["choices"][0]["finish_reason"] is None


def test_final_delta_empty_with_finish_reason(loaded_config) -> None:
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=_ok_text("Hi"))))
    r = _stream_post({"model": "text_a", "messages": [{"role": "user", "content": "x"}]})
    events, _c, _d = _parse_sse(r.content)
    last = events[-1]
    assert last["choices"][0]["delta"] == {}
    assert last["choices"][0]["finish_reason"] == "stop"


def test_tool_call_stream_indexed_deltas(loaded_config) -> None:
    set_upstream_transport(
        _mock_transport(lambda req: httpx.Response(200, json=_ok_tool_call(calls=1)))
    )
    r = _stream_post(
        {
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
        }
    )
    events, _c, done = _parse_sse(r.content)
    assert done is True
    tool_events = [e for e in events if "tool_calls" in e["choices"][0]["delta"]]
    assert len(tool_events) == 1
    tc = tool_events[0]["choices"][0]["delta"]["tool_calls"][0]
    assert tc["index"] == 0
    assert tc["id"] == "call_0"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "lookup"
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_multi_tool_call_stream_two_indexed_deltas(loaded_config) -> None:
    set_upstream_transport(
        _mock_transport(lambda req: httpx.Response(200, json=_ok_tool_call(calls=2)))
    )
    r = _stream_post(
        {
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
        }
    )
    events, _c, done = _parse_sse(r.content)
    assert done is True
    tool_events = [e for e in events if "tool_calls" in e["choices"][0]["delta"]]
    assert len(tool_events) == 2
    assert [t["choices"][0]["delta"]["tool_calls"][0]["index"] for t in tool_events] == [0, 1]


def test_error_mid_stream_emits_error_no_done(loaded_config) -> None:
    """Dispatch returns DispatchResult.error tuple → SSE error event,
    NO [DONE]. Review r34/r35 confirmed."""

    # Cause dispatch to error: text_a transport raises.
    def handler(req):
        raise httpx.ConnectError("boom")

    set_upstream_transport(_mock_transport(handler))
    r = _stream_post({"model": "text_a", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    events, _c, done = _parse_sse(r.content)
    assert done is False  # NO [DONE] after error
    assert any("error" in e for e in events)


def test_raw_upstream_429_emits_error_no_done(loaded_config) -> None:
    """Review r34 F1: passthrough relays HTTP 429 with payload set and
    error=None. Streaming must NOT chunk that as a ChatCompletion."""
    set_upstream_transport(
        _mock_transport(
            lambda req: httpx.Response(429, json={"error": {"message": "rate limited"}})
        )
    )
    r = _stream_post({"model": "text_a", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    events, _c, done = _parse_sse(r.content)
    assert done is False
    assert events[0].get("error", {}).get("type") == "rate_limit_error"


def test_pre_dispatch_error_stays_4xx(loaded_config) -> None:
    """stream:true + n=2 still 400 (JSON envelope, not SSE)."""
    r = _stream_post({"model": "text_a", "messages": [{"role": "user", "content": "x"}], "n": 2})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_v1"


def test_unsupported_content_part_preflight_4xx(loaded_config) -> None:
    """Review r35 F3: preflight unsupported parts BEFORE opening SSE."""
    r = _stream_post(
        {
            "model": "text_a",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "audio_url", "audio_url": {"url": "u"}}],
                }
            ],
        }
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_content_part"


def test_include_usage_emits_usage_chunk(loaded_config) -> None:
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=_ok_text("Hi"))))
    r = _stream_post(
        {
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "stream_options": {"include_usage": True},
        }
    )
    events, _c, done = _parse_sse(r.content)
    assert done is True
    # Final event should be the usage chunk: empty choices + usage object.
    usage_events = [e for e in events if e.get("choices") == [] and e.get("usage")]
    assert len(usage_events) == 1
    assert usage_events[0]["usage"]["total_tokens"] == 12


def test_no_include_usage_no_usage_chunk(loaded_config) -> None:
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=_ok_text("Hi"))))
    r = _stream_post(
        {
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
        }
    )
    events, _c, _d = _parse_sse(r.content)
    for e in events:
        assert "usage" not in e


def test_stream_and_stream_options_stripped_from_upstream_body(loaded_config) -> None:
    """Review r34 F2: neither stream nor stream_options leaks to upstream."""
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=_ok_text("Hi"))

    set_upstream_transport(_mock_transport(handler))
    _stream_post(
        {
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "stream_options": {"include_usage": True},
        }
    )
    assert seen, "upstream was not called"
    body = seen[0]
    assert "stream" not in body
    assert "stream_options" not in body


def test_done_sentinel_format(loaded_config) -> None:
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=_ok_text("Hi"))))
    r = _stream_post({"model": "text_a", "messages": [{"role": "user", "content": "x"}]})
    assert b"data: [DONE]\n\n" in r.content


def test_empty_content_role_then_final_no_content_deltas(loaded_config) -> None:
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=_ok_text(""))))
    r = _stream_post({"model": "text_a", "messages": [{"role": "user", "content": "x"}]})
    events, _c, done = _parse_sse(r.content)
    assert done is True
    assert len(events) == 2
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    assert events[1]["choices"][0]["delta"] == {}


def test_voting_profile_streams_uniformly(loaded_config) -> None:
    """Review r35: voting profile + stream:true streams the synthetic
    yes/no answer uniformly."""
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=_ok_text("yes"))))
    r = _stream_post({"model": "voting.v1", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    events, _c, done = _parse_sse(r.content)
    assert done is True
    assert events[-1]["choices"][0]["finish_reason"]


def test_cascade_profile_streams_uniformly(loaded_config) -> None:
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=_ok_text("Hello"))))
    r = _stream_post(
        {
            "model": "single_cascade.v1",
            "messages": [{"role": "user", "content": "x"}],
        }
    )
    assert r.status_code == 200
    _events, _c, done = _parse_sse(r.content)
    assert done is True


def test_multi_generator_moa_streams_synth_response(loaded_config) -> None:
    """Multi-generator profile runs through synthesizer; stream the
    synthesized response uniformly."""
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=_ok_text("Synth"))))
    r = _stream_post({"model": "multi.v1", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    events, _c, done = _parse_sse(r.content)
    assert done is True
    contents = [e["choices"][0]["delta"].get("content") for e in events]
    assert "".join(c for c in contents if c) == "Synth"


def test_best_of_profile_streams_chosen_candidate(loaded_config) -> None:
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=_ok_text("Best"))))
    r = _stream_post({"model": "best.v1", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    _events, _c, done = _parse_sse(r.content)
    assert done is True


def test_malformed_2xx_passthrough_emits_protocol_error(loaded_config) -> None:
    """Review r36 #1 end-to-end: passthrough relays 200 with an error
    body. Streaming must NOT treat that as a ChatCompletion."""
    set_upstream_transport(
        _mock_transport(lambda req: httpx.Response(200, json={"error": {"message": "x"}}))
    )
    r = _stream_post({"model": "text_a", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 200
    events, _c, done = _parse_sse(r.content)
    assert done is False
    assert events[0]["error"]["code"] == "upstream_protocol_error"


def test_streaming_response_headers_minimal(loaded_config) -> None:
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=_ok_text("Hi"))))
    r = _stream_post({"model": "text_a", "messages": [{"role": "user", "content": "x"}]})
    assert r.headers.get("content-type", "").startswith("text/event-stream")
    assert r.headers.get("cache-control") == "no-cache"
    # Review r35 F5: dispatch-specific observability headers (Compacted-N
    # / Quorum / Fastpath) are intentionally NOT present in streaming
    # mode — Starlette sends response.start before dispatch runs.
    assert "x-metamodel-quorum" not in {k.lower() for k in r.headers.keys()}


# ── Direct-generator-drive tests (review r34 F4) ─────────────────────


@pytest.mark.asyncio
async def test_heartbeats_during_slow_dispatch(loaded_config) -> None:
    """Review r34 F4: ASGI buffer obscures live heartbeats. Drive the
    generator directly and verify ≥3 heartbeat comments arrive before
    any data event when dispatch is slow."""

    async def slow_dispatch(*_a, **_kw):
        await asyncio.sleep(0.3)
        return DispatchResult(payload=_ok_text("late"), status_code=200, headers={})

    cfg = loaded_config
    bytes_seen: list[bytes] = []
    with patch("meta_model.streaming.dispatch", side_effect=slow_dispatch):
        gen = stream_chat_completion(
            cfg,
            {"model": "text_a", "messages": [{"role": "user", "content": "x"}]},
            model="text_a",
            ext_profile=None,
            timeout_secs=10.0,
            transport=None,
            chunk_id=new_chunk_id(),
            model_label="text_a",
            include_usage=False,
            heartbeat_interval=0.05,
        )
        async for piece in gen:
            bytes_seen.append(piece)

    # First several events should be heartbeats; final events are
    # the SSE data chunks.
    text = b"".join(bytes_seen).decode("utf-8")
    heartbeats_before_data = 0
    for line in text.split("\n\n"):
        if not line.strip():
            continue
        if line.startswith(":"):
            heartbeats_before_data += 1
        elif line.startswith("data:"):
            break
    assert heartbeats_before_data >= 3, f"only saw {heartbeats_before_data} heartbeats"
    assert text.endswith("data: [DONE]\n\n")


@pytest.mark.asyncio
async def test_client_disconnect_cancels_dispatch(loaded_config) -> None:
    """Review r34 F4: ASGI buffer hides cancellation. Drive the
    generator directly, abort mid-iteration, verify dispatch task
    received cancel."""

    state = {"cancelled": False, "started": False}

    async def long_dispatch(*_a, **_kw):
        state["started"] = True
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise
        return DispatchResult(payload=_ok_text("never"), status_code=200, headers={})

    cfg = loaded_config
    with patch("meta_model.streaming.dispatch", side_effect=long_dispatch):
        gen = stream_chat_completion(
            cfg,
            {"model": "text_a", "messages": [{"role": "user", "content": "x"}]},
            model="text_a",
            ext_profile=None,
            timeout_secs=20.0,
            transport=None,
            chunk_id=new_chunk_id(),
            model_label="text_a",
            include_usage=False,
            heartbeat_interval=0.05,
        )
        # Pull at least one heartbeat.
        first = await gen.__anext__()
        assert first.startswith(b":")
        # Now abort.
        await gen.aclose()

    assert state["started"]
    assert state["cancelled"]


@pytest.mark.asyncio
async def test_dispatch_exception_emits_500_error_envelope(loaded_config) -> None:
    """Review r35 F3: unhandled dispatch exception → SSE 500 envelope, no [DONE]."""

    async def broken_dispatch(*_a, **_kw):
        raise RuntimeError("boom")

    cfg = loaded_config
    bytes_seen: list[bytes] = []
    with patch("meta_model.streaming.dispatch", side_effect=broken_dispatch):
        gen = stream_chat_completion(
            cfg,
            {"model": "text_a", "messages": [{"role": "user", "content": "x"}]},
            model="text_a",
            ext_profile=None,
            timeout_secs=10.0,
            transport=None,
            chunk_id=new_chunk_id(),
            model_label="text_a",
            include_usage=False,
            heartbeat_interval=0.05,
        )
        async for piece in gen:
            bytes_seen.append(piece)

    blob = b"".join(bytes_seen).decode("utf-8")
    assert "[DONE]" not in blob
    assert '"type":"api_error"' in blob or '"type": "api_error"' in blob


# ── F3: streaming reasoning deltas (expose_reasoning=True path) ────


def _ok_reasoning_only(reasoning_content: str = "the answer") -> dict[str, Any]:
    """Synthetic upstream response — content empty, answer in
    reasoning_content (thinking-model shape)."""
    return {
        "id": "upstream-r",
        "object": "chat.completion",
        "created": 1,
        "model": "thinker",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": reasoning_content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def _ok_text_with_reasoning(
    content: str = "the answer",
    reasoning_content: str = "thinking out loud",
) -> dict[str, Any]:
    """Synthetic upstream response — both content and
    reasoning_content present (typical thinking-model shape)."""
    return {
        "id": "upstream-tr",
        "object": "chat.completion",
        "created": 1,
        "model": "thinker",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning_content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }


def test_response_to_chunks_emits_reasoning_delta_when_present() -> None:
    """When reasoning fields survive sanitize (expose=True path),
    the chunker emits a single delta carrying reasoning_content."""
    chunks = list(
        _response_to_chunks(
            _ok_text_with_reasoning("answer text", "thinking text"),
            chunk_id="chatcmpl-x",
            model_label="p",
            created_ts=42,
            include_usage=False,
        )
    )
    # Role chunk first; reasoning delta before content delta(s).
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    reasoning_chunks = [
        c for c in chunks if "reasoning_content" in c["choices"][0]["delta"]
    ]
    assert len(reasoning_chunks) == 1
    assert (
        reasoning_chunks[0]["choices"][0]["delta"]["reasoning_content"]
        == "thinking text"
    )
    # Content delta still emitted.
    content_chunks = [
        c
        for c in chunks
        if "content" in c["choices"][0]["delta"]
        and c["choices"][0]["delta"].get("content")
    ]
    assert content_chunks, "content delta should still be emitted"
    # Reasoning chunk arrives before content chunks.
    reasoning_idx = chunks.index(reasoning_chunks[0])
    first_content_idx = chunks.index(content_chunks[0])
    assert reasoning_idx < first_content_idx


def test_response_to_chunks_no_reasoning_delta_when_field_absent() -> None:
    """Default path (no reasoning fields on the message): no
    reasoning delta emitted. Existing tests already pass; this
    pins the behavior so adding reasoning emission did NOT
    regress the no-reasoning case."""
    chunks = list(
        _response_to_chunks(
            _ok_text("just text"),
            chunk_id="chatcmpl-x",
            model_label="p",
            created_ts=42,
            include_usage=False,
        )
    )
    for c in chunks:
        delta = c["choices"][0]["delta"]
        assert "reasoning" not in delta
        assert "reasoning_content" not in delta


def test_response_to_chunks_reasoning_only_emits_reasoning_then_no_content() -> None:
    """Response with content empty + reasoning present (expose=True
    path where sanitize did NOT rescue): reasoning delta emits,
    content deltas don't (no content to chunk)."""
    chunks = list(
        _response_to_chunks(
            _ok_reasoning_only("the rescued answer"),
            chunk_id="chatcmpl-x",
            model_label="p",
            created_ts=42,
            include_usage=False,
        )
    )
    reasoning_chunks = [
        c for c in chunks if "reasoning_content" in c["choices"][0]["delta"]
    ]
    assert len(reasoning_chunks) == 1
    content_chunks = [
        c
        for c in chunks
        if "content" in c["choices"][0]["delta"]
        and c["choices"][0]["delta"].get("content")
    ]
    assert not content_chunks
