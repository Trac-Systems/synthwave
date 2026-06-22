"""Shared reasoning-rescue primitives.

Used by:
  - the MoA synthesizer to widen candidate-draft signatures over
    thinking models that put the answer in ``reasoning_content``;
  - the F3 centralized sanitizer to lift reasoning into ``content``
    BEFORE stripping when ``expose_reasoning=False``;
  - cascade dispatch validation, which must accept a reasoning-only
    upstream response as valid output (sanitize rescues it
    downstream).

Boundary: this module owns "what counts as visible content" and
"how to rescue when content is missing". Other modules import; do
NOT duplicate the gating logic.
"""

from __future__ import annotations

from typing import Any


def is_visible_content_missing(msg: dict[str, Any]) -> bool:
    """True when ``content`` carries no user-visible substance.

    Missing when ANY of:
      - ``content is None``
      - ``content`` is a string and empty/whitespace-only
      - ``content`` is a list AND has no element carrying substance,
        where "substance" means: a ``text``-type part with
        non-empty/non-whitespace ``text``, OR any non-text part
        (image_url, input_audio, file, etc.). A malformed (non-dict)
        list element is conservatively treated as substance to
        avoid silent bypass.
    """
    content = msg.get("content")
    if content is None:
        return True
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        if not content:
            return True
        for part in content:
            if not isinstance(part, dict):
                # Malformed element — conservatively treat as substance.
                return False
            ptype = part.get("type")
            if ptype == "text":
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return False
            else:
                # Non-text parts carry substantive output.
                return False
        return True
    # Unknown content type — conservative default.
    return False


def reasoning_fallback_text(msg: dict[str, Any]) -> str:
    """Return reasoning_content / reasoning text when visible content
    is missing and there are no tool/function calls.

    Thinking models (reasoning-model-20b, thinking-enabled chat templates,
    some hosted reasoning models) put chain-of-thought in ``reasoning_content`` (or
    legacy ``reasoning``) and the user-visible answer in
    ``content``. With ``finish_reason=length``, the CoT can eat the
    full token budget before content gets populated — leaving
    ``content`` empty plus reasoning text the answer effectively
    lives in.

    Gated:
      - content is missing per :func:`is_visible_content_missing`.
      - no ``tool_calls`` (substantive structured output; do not
        surface CoT as a user-visible body).
      - no legacy ``function_call`` (same gate).

    Returns ``""`` when fallback is not applicable.
    """
    if not is_visible_content_missing(msg):
        return ""
    if msg.get("tool_calls"):
        return ""
    if msg.get("function_call"):
        return ""
    rc = msg.get("reasoning_content")
    if not isinstance(rc, str) or not rc.strip():
        rc = msg.get("reasoning")
    if isinstance(rc, str) and rc.strip():
        return rc.strip()
    return ""


def has_reasoning_rescue_text(msg: dict[str, Any]) -> bool:
    """True when :func:`reasoning_fallback_text` would return a
    non-empty rescue. Used by cascade validation to treat a
    reasoning-only successful response as valid before
    :func:`coerce_reasoning_into_content` runs downstream in the
    sanitizer.
    """
    return bool(reasoning_fallback_text(msg))


def coerce_reasoning_into_content(msg: dict[str, Any]) -> None:
    """In-place sibling of :func:`reasoning_fallback_text`: lifts
    reasoning text into ``content`` so downstream consumers deliver
    populated content to the caller.

    Mutates ``msg``. Gated identically (only fires when content is
    missing and no tool/function calls). No-op when nothing to
    coerce.
    """
    fallback = reasoning_fallback_text(msg)
    if fallback:
        msg["content"] = fallback


__all__ = [
    "is_visible_content_missing",
    "reasoning_fallback_text",
    "has_reasoning_rescue_text",
    "coerce_reasoning_into_content",
]
