"""F8 — legacy /v1/completions response-shape adapter.

Coverage:
- Non-streaming: chat-shape body → legacy text_completion shape.
  - object="text_completion"
  - id rewritten "chatcmpl-..." → "cmpl-..."
  - choices[i].text (extracted from message.content)
  - choices[i].logprobs present (null when not requested)
  - usage preserved
  - multimodal content list stitched into text via {type:"text"} parts
  - tool_calls / function_call → empty text (legacy has no tool surface)
  - error envelopes pass through untouched

- Streaming: SSE chunks rewritten line-by-line.
  - role chunk dropped
  - content delta → text choice
  - final chunk preserved with finish_reason
  - usage chunk preserved
  - [DONE] sentinel passes through
  - heartbeat comments pass through
  - malformed JSON passes through verbatim
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from meta_model.legacy_completion import (
    _reshape_stream_chunk,
    chat_to_legacy_completion,
    transform_chat_sse_to_legacy,
)


# ── Non-streaming ───────────────────────────────────────────────────


def test_chat_to_legacy_basic() -> None:
    chat = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "text_synth.v1",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello world"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    legacy = chat_to_legacy_completion(chat)
    assert legacy["object"] == "text_completion"
    assert legacy["id"] == "cmpl-abc123"
    assert legacy["created"] == 1234567890
    assert legacy["model"] == "text_synth.v1"
    assert legacy["choices"][0]["text"] == "Hello world"
    assert legacy["choices"][0]["index"] == 0
    assert legacy["choices"][0]["logprobs"] is None
    assert legacy["choices"][0]["finish_reason"] == "stop"
    assert legacy["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
    }


def test_chat_to_legacy_id_rewrite_variations() -> None:
    """id rewriting handles chatcmpl- prefix, cmpl- already, raw, missing."""
    assert chat_to_legacy_completion({"id": "chatcmpl-x"})["id"] == "cmpl-x"
    assert chat_to_legacy_completion({"id": "cmpl-y"})["id"] == "cmpl-y"
    assert chat_to_legacy_completion({"id": "raw"})["id"] == "cmpl-raw"
    assert chat_to_legacy_completion({})["id"] == "cmpl-?"
    assert chat_to_legacy_completion({"id": ""})["id"] == "cmpl-?"


def test_chat_to_legacy_multimodal_content_list() -> None:
    """Multimodal content list (text + image_url parts) stitches text only."""
    chat = {
        "id": "chatcmpl-z",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "look: "},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                        {"type": "text", "text": "a tree."},
                    ],
                },
                "finish_reason": "stop",
            }
        ],
    }
    legacy = chat_to_legacy_completion(chat)
    assert legacy["choices"][0]["text"] == "look: a tree."


def test_chat_to_legacy_tool_calls_empty_text() -> None:
    """tool_calls in chat → empty text in legacy (no tool surface).
    finish_reason ``"tool_calls"`` is preserved so clients can detect."""
    chat = {
        "id": "chatcmpl-tc",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_1", "function": {"name": "f"}}],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    legacy = chat_to_legacy_completion(chat)
    assert legacy["choices"][0]["text"] == ""
    assert legacy["choices"][0]["finish_reason"] == "tool_calls"
    # tool_calls/message NOT present at top of choice
    assert "tool_calls" not in legacy["choices"][0]
    assert "message" not in legacy["choices"][0]


def test_chat_to_legacy_no_usage() -> None:
    """usage absent in chat → omitted from legacy (matches OpenAI shape)."""
    legacy = chat_to_legacy_completion({"id": "chatcmpl-x", "choices": []})
    assert "usage" not in legacy


def test_chat_to_legacy_preserves_logprobs_when_present() -> None:
    """If the chat choice carries logprobs (some OpenAI calls do),
    legacy should preserve it. Default null otherwise."""
    chat = {
        "id": "chatcmpl-lp",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "x"},
                "logprobs": {"content": [{"token": "x"}]},
                "finish_reason": "stop",
            }
        ],
    }
    legacy = chat_to_legacy_completion(chat)
    assert legacy["choices"][0]["logprobs"] == {"content": [{"token": "x"}]}


# ── Streaming chunk rewriter ────────────────────────────────────────


def test_stream_chunk_role_chunk_dropped() -> None:
    """Role chunk has empty content — drops entirely."""
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
    }
    assert _reshape_stream_chunk(chunk) is None


def test_stream_chunk_content_delta_to_text() -> None:
    chunk = {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
    }
    legacy = _reshape_stream_chunk(chunk)
    assert legacy["object"] == "text_completion"
    assert legacy["id"] == "cmpl-1"
    assert legacy["choices"][0]["text"] == "Hi"
    assert legacy["choices"][0]["logprobs"] is None
    assert legacy["choices"][0]["finish_reason"] is None


def test_stream_chunk_final_chunk_preserves_finish_reason() -> None:
    chunk = {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    legacy = _reshape_stream_chunk(chunk)
    assert legacy["choices"][0]["text"] == ""
    assert legacy["choices"][0]["finish_reason"] == "stop"


def test_stream_chunk_usage_chunk_preserved() -> None:
    """Usage-only chunk has empty choices and a top-level usage."""
    chunk = {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    legacy = _reshape_stream_chunk(chunk)
    assert legacy["choices"] == []
    assert legacy["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 1,
        "total_tokens": 6,
    }


def test_stream_chunk_error_envelope_passthrough() -> None:
    """``{"error": {...}}`` is the same shape on both endpoints — passthrough."""
    err = {"error": {"message": "x", "type": "api_error", "code": None, "param": None}}
    assert _reshape_stream_chunk(err) == err


def test_stream_chunk_tool_calls_emit_empty_text() -> None:
    """tool_calls delta → empty-text choice (legacy has no tool surface)."""
    chunk = {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": "c1", "function": {"name": "f"}}
                    ]
                },
                "finish_reason": None,
            }
        ],
    }
    legacy = _reshape_stream_chunk(chunk)
    assert legacy["choices"][0]["text"] == ""


# ── Streaming SSE byte-stream wrapper ───────────────────────────────


async def _collect(it: AsyncIterator[bytes]) -> bytes:
    out = b""
    async for chunk in it:
        out += chunk
    return out


def _async_iter(parts: list[bytes]) -> AsyncIterator[bytes]:
    async def gen():
        for p in parts:
            yield p

    return gen()


def test_sse_stream_basic_roundtrip() -> None:
    """One role chunk + two content chunks + final + [DONE]. Role drops."""
    chunks = [
        '{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}',
        '{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
        '{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
        '{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    ]
    payload = b""
    for c in chunks:
        payload += b"data: " + c.encode() + b"\n\n"
    payload += b"data: [DONE]\n\n"

    out = asyncio.run(_collect(transform_chat_sse_to_legacy(_async_iter([payload]))))
    out_text = out.decode()
    # Role chunk dropped
    assert '"role":"assistant"' not in out_text
    # Two content chunks preserved with text
    assert '"text": "Hello"' in out_text or '"text":"Hello"' in out_text.replace(" ", "")
    assert '"text": " world"' in out_text or '"text":" world"' in out_text
    # Object renamed
    assert '"object": "text_completion"' in out_text or '"object":"text_completion"' in out_text
    # No "chat.completion.chunk" anywhere
    assert "chat.completion.chunk" not in out_text
    # id rewritten
    assert '"id": "cmpl-1"' in out_text or '"id":"cmpl-1"' in out_text
    # [DONE] preserved
    assert "[DONE]" in out_text


def test_sse_stream_heartbeat_passthrough() -> None:
    """Comment lines (": heartbeat") pass through unchanged."""
    payload = b": heartbeat\n\ndata: [DONE]\n\n"
    out = asyncio.run(_collect(transform_chat_sse_to_legacy(_async_iter([payload]))))
    assert b": heartbeat" in out
    assert b"[DONE]" in out


def test_sse_stream_malformed_json_passthrough() -> None:
    """Unparseable data: lines pass through verbatim — better than dropping."""
    payload = b"data: {garbage\n\ndata: [DONE]\n\n"
    out = asyncio.run(_collect(transform_chat_sse_to_legacy(_async_iter([payload]))))
    assert b"{garbage" in out
    assert b"[DONE]" in out


def test_sse_stream_split_across_buffer_boundaries() -> None:
    """SSE producers may chunk bytes mid-line. Wrapper buffers correctly."""
    full = (
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk",'
        b'"created":1,"model":"m","choices":[{"index":0,"delta":'
        b'{"content":"Hi"},"finish_reason":null}]}\n\n'
        b"data: [DONE]\n\n"
    )
    # Split into many small fragments
    parts = [full[i : i + 17] for i in range(0, len(full), 17)]
    out = asyncio.run(_collect(transform_chat_sse_to_legacy(_async_iter(parts))))
    out_text = out.decode()
    assert '"text": "Hi"' in out_text or '"text":"Hi"' in out_text
    assert "[DONE]" in out_text
    assert "chat.completion.chunk" not in out_text
