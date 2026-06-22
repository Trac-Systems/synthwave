"""Centralized reasoning sanitizer (F3).

Profile policy ``expose_reasoning: bool`` (default false) controls
whether ``reasoning`` and ``reasoning_content`` fields appear in
response bodies and streaming deltas.

Used at every emission point that hands a chat-completion-shaped
payload back to a client:
  - ``server.py`` JSON chat-completion response
  - ``streaming.py._response_to_chunks`` (per-message rechunking)
  - any future ``/v1/responses`` adapter output (F4-Responses)

Rules (only when ``expose_reasoning=False``):
  1. **Rescue** — if ``message.content`` is null/empty AND
     ``message.reasoning`` OR ``message.reasoning_content`` is
     non-empty AND ``message.tool_calls`` is empty/null →
     lift the reasoning text into ``message.content``.
  2. **Strip** — drop both ``reasoning`` AND
     ``reasoning_content`` from ``message`` and from any
     streaming ``delta`` sibling. **Do NOT touch
     ``tool_calls``** — strip reasoning while leaving tool
     calls intact.

No content-quality heuristic: if ``content`` is non-empty, content
wins; rescue does NOT run; reasoning is just stripped.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .reasoning import coerce_reasoning_into_content


def sanitize_reasoning(
    response: dict[str, Any],
    *,
    expose_reasoning: bool,
) -> dict[str, Any]:
    """Apply ``expose_reasoning`` policy to a chat-completion-shaped
    payload.

    Returns a deep-copied response with reasoning fields handled.
    The input is not mutated. When ``expose_reasoning`` is True
    the original (un-copied) response is returned for efficiency —
    callers MUST NOT rely on the returned object being a copy in
    that case.

    Handles both finalized message shapes
    (``choices[*].message.{content, reasoning, reasoning_content,
    tool_calls}``) and streaming chunk shapes
    (``choices[*].delta.{content, reasoning, reasoning_content,
    tool_calls}``).
    """
    if expose_reasoning:
        return response

    out = deepcopy(response)
    choices = out.get("choices")
    if not isinstance(choices, list):
        return out
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        # Finalized message path. Rescue first, then strip.
        msg = choice.get("message")
        if isinstance(msg, dict):
            coerce_reasoning_into_content(msg)
            msg.pop("reasoning", None)
            msg.pop("reasoning_content", None)
        # Streaming-delta path. No rescue here — the stream is
        # already mid-flight; the rescue path is the responsibility
        # of the pre-stream sanitize call. Just strip the keys so
        # they never appear on the wire.
        delta = choice.get("delta")
        if isinstance(delta, dict):
            delta.pop("reasoning", None)
            delta.pop("reasoning_content", None)
    return out


__all__ = ["sanitize_reasoning"]
