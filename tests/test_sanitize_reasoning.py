"""Tests for the F3 centralized reasoning sanitizer.

Verifies the policy:
  - When ``expose_reasoning=False`` (default):
    1. Rescue: empty content + reasoning text + no tool_calls →
       lift reasoning into content.
    2. Strip: drop ``reasoning`` and ``reasoning_content`` from
       ``message``. Same for streaming ``delta``. Tool calls
       untouched.
  - When ``expose_reasoning=True``: pass-through; nothing
    stripped.
"""

from __future__ import annotations

from meta_model.sanitize import sanitize_reasoning


def _wrap(message: dict) -> dict:
    """Build a minimal ChatCompletion-shaped dict around a message."""
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 0,
        "model": "p",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


# ── Strip-only path (content present) ──────────────────────────────


def test_strip_with_content_present_drops_reasoning_keeps_content():
    resp = _wrap(
        {
            "role": "assistant",
            "content": "the actual answer",
            "reasoning": "internal CoT",
            "reasoning_content": "more CoT",
        }
    )
    out = sanitize_reasoning(resp, expose_reasoning=False)
    msg = out["choices"][0]["message"]
    assert msg["content"] == "the actual answer"
    assert "reasoning" not in msg
    assert "reasoning_content" not in msg


# ── Rescue path (content empty, reasoning has the answer) ─────────


def test_rescue_lifts_reasoning_content_into_content():
    resp = _wrap(
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "the rescued answer",
        }
    )
    out = sanitize_reasoning(resp, expose_reasoning=False)
    msg = out["choices"][0]["message"]
    assert msg["content"] == "the rescued answer"
    assert "reasoning" not in msg
    assert "reasoning_content" not in msg


def test_rescue_lifts_legacy_reasoning_into_content():
    """Legacy `reasoning` key (sibling of reasoning_content)."""
    resp = _wrap(
        {
            "role": "assistant",
            "content": "",
            "reasoning": "legacy CoT answer",
        }
    )
    out = sanitize_reasoning(resp, expose_reasoning=False)
    msg = out["choices"][0]["message"]
    assert msg["content"] == "legacy CoT answer"
    assert "reasoning" not in msg


# ── Tool calls preserved + reasoning still stripped ────────────────


def test_tool_calls_preserved_no_rescue():
    resp = _wrap(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
            "reasoning_content": "should not be lifted",
        }
    )
    out = sanitize_reasoning(resp, expose_reasoning=False)
    msg = out["choices"][0]["message"]
    # Tool calls intact.
    assert msg["tool_calls"][0]["id"] == "call_1"
    # Rescue did NOT fire (tool_calls gate).
    assert msg["content"] is None
    # Strip still ran.
    assert "reasoning_content" not in msg
    assert "reasoning" not in msg


# ── No content-quality heuristic ───────────────────────────────────


def test_content_present_means_content_wins_no_rescue_attempt():
    """If content is non-empty, rescue does NOT run even if
    reasoning has plausibly substantive text."""
    resp = _wrap(
        {
            "role": "assistant",
            "content": "short",
            "reasoning_content": "a much longer and arguably more substantive reasoning text",
        }
    )
    out = sanitize_reasoning(resp, expose_reasoning=False)
    assert out["choices"][0]["message"]["content"] == "short"


# ── expose_reasoning=True: passthrough ─────────────────────────────


def test_expose_true_passes_reasoning_fields_through():
    resp = _wrap(
        {
            "role": "assistant",
            "content": "answer",
            "reasoning": "visible CoT",
            "reasoning_content": "visible CoT v2",
        }
    )
    out = sanitize_reasoning(resp, expose_reasoning=True)
    msg = out["choices"][0]["message"]
    assert msg["content"] == "answer"
    assert msg["reasoning"] == "visible CoT"
    assert msg["reasoning_content"] == "visible CoT v2"


# ── Streaming-delta sibling ────────────────────────────────────────


def test_streaming_delta_strips_reasoning_keys():
    """A response carrying `delta` (streaming chunk shape) gets
    reasoning fields stripped from the delta too."""
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "p",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "hi",
                    "reasoning": "hidden",
                    "reasoning_content": "hidden v2",
                },
                "finish_reason": None,
            }
        ],
    }
    out = sanitize_reasoning(chunk, expose_reasoning=False)
    delta = out["choices"][0]["delta"]
    assert delta["content"] == "hi"
    assert "reasoning" not in delta
    assert "reasoning_content" not in delta


def test_streaming_delta_passes_through_when_exposed():
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "p",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "hi",
                    "reasoning": "visible",
                },
                "finish_reason": None,
            }
        ],
    }
    out = sanitize_reasoning(chunk, expose_reasoning=True)
    assert out["choices"][0]["delta"]["reasoning"] == "visible"


# ── Robustness against malformed shapes ───────────────────────────


def test_no_choices_returns_unchanged():
    resp = {"id": "x", "object": "chat.completion"}
    out = sanitize_reasoning(resp, expose_reasoning=False)
    assert out == resp


def test_choices_not_list_returns_unchanged():
    resp = {"choices": "oops"}
    out = sanitize_reasoning(resp, expose_reasoning=False)
    assert out == resp


def test_choice_without_message_or_delta_skipped_safely():
    resp = {"choices": [{"index": 0, "finish_reason": "stop"}]}
    # Should not raise; nothing to sanitize.
    out = sanitize_reasoning(resp, expose_reasoning=False)
    assert out["choices"][0]["finish_reason"] == "stop"


def test_does_not_mutate_input_when_stripping():
    """Caller's dict must remain intact after sanitization."""
    resp = _wrap(
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "CoT",
        }
    )
    sanitize_reasoning(resp, expose_reasoning=False)
    # Original dict unchanged.
    assert "reasoning_content" in resp["choices"][0]["message"]
