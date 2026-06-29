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

import re
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


# ── In-band chain-of-thought stripping (F-thinking) ─────────────────
# reasoning.py above owns the `reasoning_content` *field*. The helpers
# below strip thinking that leaks INSIDE the `content` string itself —
# the shape `<think>..</think>answer`, prompt-prefilled opens that
# leave a trailing `</think>`, cut-off opens with no close, DeepSeek
# `◁think▷` unicode tags, harmony `<|channel|>` blocks, and a
# conservative natural-language "thinking process:" preamble.

_REASON_TAGS = "think|thinking|thought|thoughts|reasoning|scratchpad|reflection|analysis|inner_monologue|rationale"
_PAIRED_TAG_RE = re.compile(r"<(" + _REASON_TAGS + r")\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(r"</(?:" + _REASON_TAGS + r")\s*>", re.IGNORECASE)
_OPEN_TAG_RE = re.compile(r"<(?:" + _REASON_TAGS + r")\b[^>]*>", re.IGNORECASE)
_UNICODE_THINK_RE = re.compile(r"◁\s*think\s*▷.*?◁\s*/\s*think\s*▷", re.DOTALL)
_SPECIAL_THOUGHT_RE = re.compile(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", re.DOTALL | re.IGNORECASE)
_SPECIAL_SOLUTION_RE = re.compile(r"<\|begin_of_solution\|>(.*?)(?:<\|end_of_solution\|>|\Z)", re.DOTALL | re.IGNORECASE)
_HARMONY_FINAL_RE = re.compile(r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|start\|>|\Z)", re.DOTALL | re.IGNORECASE)
_HARMONY_ANY_CHANNEL_RE = re.compile(r"<\|channel\|>\s*\w+\s*<\|message\|>.*?(?:<\|end\|>|<\|return\|>|\Z)", re.DOTALL | re.IGNORECASE)
_LEADING_PREAMBLE_RE = re.compile(
    r"^\s*(?:here(?:'s| is)\s+(?:a|my|the)?\s*(?:thinking|thought|reasoning|chain[- ]of[- ]thought)[^\n:]*:"
    r"|let(?:'s| us| me)\s+think[^\n]*"
    r"|thinking:|reasoning:|thought process:|chain of thought:|step[- ]by[- ]step reasoning:)",
    re.IGNORECASE,
)
_ANSWER_MARKER_RE = re.compile(
    r"\n\s*(?:#{1,6}\s*)?(?:\*\*\s*)?(?:final answer|final response|answer|response|solution|conclusion|here(?:'s| is) the (?:final )?answer)\s*(?:\*\*)?\s*[:\-]?\s*\n",
    re.IGNORECASE,
)


def strip_inband_reasoning(text: Any) -> Any:
    """Remove in-band chain-of-thought from an assistant content string.

    Never raises; returns the input unchanged when nothing matches or
    the input is not a non-empty string.
    """
    if not isinstance(text, str) or not text:
        return text
    if "<" not in text and "◁" not in text and not _LEADING_PREAMBLE_RE.match(text):
        return text
    s = text
    # harmony channels — keep only the final channel's message
    if "<|channel|>" in s:
        finals = _HARMONY_FINAL_RE.findall(s)
        if finals:
            s = finals[-1]
        else:
            s = _HARMONY_ANY_CHANNEL_RE.sub("", s)
    # special begin/end solution/thought blocks
    if "<|begin_of_solution|>" in s:
        sol = _SPECIAL_SOLUTION_RE.findall(s)
        if sol:
            s = sol[-1]
    s = _SPECIAL_THOUGHT_RE.sub("", s)
    # paired reasoning tags (closed)
    s = _PAIRED_TAG_RE.sub("", s)
    s = _UNICODE_THINK_RE.sub("", s)
    # prompt-prefilled open: content begins mid-think, ends ...</think>answer
    closes = list(_CLOSE_TAG_RE.finditer(s))
    if closes:
        s = s[closes[-1].end():]
    # cut-off open with no matching close: drop from the orphan open onward
    opn = _OPEN_TAG_RE.search(s)
    if opn:
        s = s[: opn.start()]
    # conservative natural-language preamble + explicit answer marker
    if _LEADING_PREAMBLE_RE.match(s):
        markers = list(_ANSWER_MARKER_RE.finditer(s))
        if markers:
            s = s[markers[-1].end():]
    return s.strip()


def strip_message_inband_reasoning(msg: dict[str, Any]) -> None:
    """Apply :func:`strip_inband_reasoning` to a chat message's content
    in place. Touches only string content and the text parts of list
    content; never touches ``tool_calls`` or non-text parts. No-op on a
    non-dict ``msg``."""
    if not isinstance(msg, dict):
        return
    content = msg.get("content")
    if isinstance(content, str):
        cleaned = strip_inband_reasoning(content)
        if cleaned != content:
            msg["content"] = cleaned
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                part["text"] = strip_inband_reasoning(part["text"])


__all__ += ["strip_inband_reasoning", "strip_message_inband_reasoning"]
