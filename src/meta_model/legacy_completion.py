"""Legacy `/v1/completions` response-shape adapter. F8.

OpenAI's classic text-completion endpoint returns a different envelope
than `/v1/chat/completions`:

    chat completion           legacy text completion
    ─────────────────────     ─────────────────────────
    object: "chat.completion" object: "text_completion"
    id:     "chatcmpl-..."    id:     "cmpl-..."
    choices[i].message.content  choices[i].text
                                choices[i].logprobs (null when not requested)

Drop-in clients written against the legacy SDK do
``response.choices[0].text``; on a chat-shaped body they crash with
``AttributeError``. F8 reshapes the chat body before
``/v1/completions`` returns, both non-streaming (a single JSON body)
and streaming (per-chunk SSE rewrites).

Streaming chunk taxonomy on input is OpenAI's chat stream:
- role chunk: ``{delta:{role:"assistant",content:""}, finish_reason:null}``
- content chunks: ``{delta:{content:"…"}, finish_reason:null}``
- final chunk: ``{delta:{}, finish_reason:"stop"|"length"|"tool_calls"|…}``
- optional usage chunk (when ``stream_options.include_usage``):
  ``{choices:[], usage:{…}}``

On output we drop the role chunk (legacy text completion has no role
concept), rewrite content chunks to ``choices[i].text``, and keep
finish/usage chunks under the ``text_completion`` object name.

Tool-call chunks (``delta.tool_calls`` or ``delta.function_call``) are
rewritten as empty-text chunks. The legacy completions API has no
tool surface, so any tool_calls would be unparseable by legacy
clients. Forwarding empty text and a finish_reason of ``"tool_calls"``
keeps the wire shape valid; clients that expect tools should use
``/v1/chat/completions``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any


def _legacy_id(chat_id: Any) -> str:
    """Rewrite ``chatcmpl-…`` to ``cmpl-…`` (legacy convention)."""
    if not isinstance(chat_id, str) or not chat_id:
        return "cmpl-?"
    if chat_id.startswith("chatcmpl-"):
        return "cmpl-" + chat_id[len("chatcmpl-"):]
    if chat_id.startswith("cmpl-"):
        return chat_id
    return "cmpl-" + chat_id


def _content_text(content: Any) -> str:
    """Extract a plain-text view of a chat message's ``content``.

    Strings are returned as-is. Multimodal content lists are stitched
    by concatenating ``{type:"text", text:…}`` parts. Non-text parts
    (image_url etc.) are dropped — legacy text completion has no part
    concept and the synthesizer never returns those anyway.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _reshape_choice(chat_choice: dict[str, Any]) -> dict[str, Any]:
    """Rewrite one chat completion choice → legacy text completion choice."""
    msg = chat_choice.get("message") or {}
    return {
        "text": _content_text(msg.get("content")),
        "index": chat_choice.get("index", 0),
        "logprobs": chat_choice.get("logprobs"),  # null unless requested
        "finish_reason": chat_choice.get("finish_reason"),
    }


def chat_to_legacy_completion(chat_body: dict[str, Any]) -> dict[str, Any]:
    """Non-streaming: rewrite a chat body to legacy text-completion shape.

    Drops chat-only fields (``message.role``, ``message.tool_calls``,
    ``message.function_call``, reasoning extras) — the legacy
    completions wire has no place for them. Keeps ``usage`` verbatim.
    """
    legacy: dict[str, Any] = {
        "id": _legacy_id(chat_body.get("id")),
        "object": "text_completion",
        "created": chat_body.get("created", 0),
        "model": chat_body.get("model", ""),
        "choices": [_reshape_choice(c) for c in chat_body.get("choices") or []],
    }
    usage = chat_body.get("usage")
    if usage is not None:
        legacy["usage"] = usage
    return legacy


# ── SSE chunk rewriter ──────────────────────────────────────────────


def _reshape_stream_chunk(chunk: dict[str, Any]) -> dict[str, Any] | None:
    """Rewrite one chat stream chunk into legacy completion shape.

    Returns ``None`` when the chunk should be dropped (role chunks).
    Tool-call / function-call chunks degrade to empty-text chunks.
    Error envelopes (``{"error": {...}}``) pass through untouched —
    they're already the same shape on both endpoints.
    """
    if "error" in chunk:
        return chunk
    new: dict[str, Any] = {
        "id": _legacy_id(chunk.get("id")),
        "object": "text_completion",
        "created": chunk.get("created", 0),
        "model": chunk.get("model", ""),
        "choices": [],
    }
    chat_choices = chunk.get("choices") or []

    # Usage-only chunk has empty choices and a top-level `usage` field.
    if not chat_choices and "usage" in chunk:
        new["usage"] = chunk["usage"]
        return new

    drop_chunk = True
    for c in chat_choices:
        delta = c.get("delta") or {}
        finish = c.get("finish_reason")
        # Role chunk: ``{role: "assistant", content: ""}`` with no
        # finish reason. Drop entirely — legacy clients don't expect
        # an opening "role" event.
        if (
            "role" in delta
            and not delta.get("content")
            and finish is None
        ):
            continue
        text = ""
        content = delta.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = _content_text(content)
        # tool_calls / function_call deltas: emit empty-text choice so
        # the chunk count stays sane and a final `finish_reason:
        # "tool_calls"` chunk still terminates the stream cleanly.
        new["choices"].append(
            {
                "text": text,
                "index": c.get("index", 0),
                "logprobs": c.get("logprobs"),
                "finish_reason": finish,
            }
        )
        drop_chunk = False
    if drop_chunk:
        return None
    if "usage" in chunk and "usage" not in new:
        new["usage"] = chunk["usage"]
    return new


async def transform_chat_sse_to_legacy(
    stream: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Wrap a chat-shaped SSE byte stream, emitting legacy-shaped chunks.

    Parses incoming SSE line-by-line. ``data: {…}`` lines are JSON-decoded
    and reshaped via ``_reshape_stream_chunk``. ``data: [DONE]`` and
    comment lines (``: heartbeat``) pass through unchanged. Malformed
    lines pass through too — better to forward something than to drop
    a frame and confuse the client.
    """
    buf = b""
    async for raw in stream:
        if not isinstance(raw, (bytes, bytearray)):
            raw = str(raw).encode("utf-8")
        buf += raw
        # SSE events are line-delimited with \n; empty line terminates
        # an event. Process complete lines only; keep trailing partial.
        while b"\n" in buf:
            line, _sep, rest = buf.partition(b"\n")
            buf = rest
            yielded = _process_sse_line(line)
            if yielded is not None:
                yield yielded
    # Flush any trailing partial line (rare — SSE producers usually
    # terminate frames with \n\n).
    if buf:
        yielded = _process_sse_line(buf)
        if yielded is not None:
            yield yielded


def _process_sse_line(line: bytes) -> bytes | None:
    """Process one SSE line. Returns the bytes to emit, or None to drop.

    The wire format expects events terminated by `\\n`. We always
    re-emit `<line>\\n` even for lines that pass through unchanged.
    """
    text = line.rstrip(b"\r").decode("utf-8", errors="replace")
    if text == "":
        return b"\n"  # event terminator — preserve framing
    if not text.startswith("data: "):
        # comment (": heartbeat"), event, retry, id field — pass through
        return text.encode("utf-8") + b"\n"
    payload = text[len("data: "):]
    if payload.strip() == "[DONE]":
        return b"data: [DONE]\n"
    try:
        chunk = json.loads(payload)
    except (ValueError, TypeError):
        # Unparseable — pass through verbatim. Better than dropping.
        return text.encode("utf-8") + b"\n"
    if not isinstance(chunk, dict):
        return text.encode("utf-8") + b"\n"
    new = _reshape_stream_chunk(chunk)
    if new is None:
        return None
    return ("data: " + json.dumps(new)).encode("utf-8") + b"\n"


__all__ = [
    "chat_to_legacy_completion",
    "transform_chat_sse_to_legacy",
]
