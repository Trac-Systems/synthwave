"""D.3.3 — wire-compatible simulated SSE streaming.

The server runs MoA dispatch (or any other ensemble) to completion,
then emits the synthesized response as OpenAI-format SSE chunks. The
chunking is artificial — by the time the first delta event arrives,
the full response is already sitting in server memory. Plan §209-236
locked this as the v1 contract: clients see standard OpenAI streaming
wire shape, latency stays honest (no fake first-token), and MoA's
architectural serialization (synthesizer waits for quorum) is
preserved. v2 will add real intrinsic streaming via a separate plan.

**Heartbeats during the wait.** Long MoA paths can blow past default
30s client read timeouts. The generator emits SSE comments
(`: heartbeat\\n\\n` per spec — clients silently ignore lines starting
with ``:``) every ``HEARTBEAT_INTERVAL_SECS`` while dispatch is
pending, so the connection stays alive even if dispatch takes ~60s.

**Wire shape per chunk:**

    data: {
        "id": "chatcmpl-<24-hex>",
        "object": "chat.completion.chunk",
        "created": <unix-ts>,
        "model": "<profile_name or model>",
        "choices": [{"index": 0, "delta": {...}, "finish_reason": null|...}]
    }\\n\\n

Sequence (text response):
1. role chunk: ``delta = {"role": "assistant", "content": ""}``.
2. N content chunks: ``delta = {"content": "<chunk>"}``.
3. final chunk: ``delta = {}``, ``finish_reason = "stop"|...``.
4. (optional) usage chunk if ``stream_options.include_usage`` true.
5. ``data: [DONE]\\n\\n``.

Sequence (tool_call response):
1. role chunk with ``content: null``.
2. one delta per tool_call (full id + type + name + arguments — the
   minimum-valid OpenAI chunk shape per the schema. Plan §232 explicitly
   permits "minimum number of chunks needed for valid tool_calls
   indexed deltas").
3. final chunk with ``finish_reason = "tool_calls"``.
4. (optional) usage chunk.
5. ``[DONE]``.

**Errors after SSE opens.** When ``StreamingResponse`` returns from
the route handler, Starlette commits ``http.response.start`` with
status 200 *before* iterating the generator (see
``starlette.responses.StreamingResponse``). Anything that goes wrong
after that becomes an SSE error event:

    data: {"error": {...OpenAI envelope...}}\\n\\n

and the stream ends — **no ``[DONE]``** after an error. This mirrors
OpenAI's actual streaming-error behavior (the SDK throws on the error
chunk and treats it as the terminator).

Pre-dispatch validation (max_tokens conflict, n!=1, no_upstream,
unsupported_content_part) stays synchronous in ``server.py`` and
returns plain JSON 4xx — those errors never open SSE in the first place.

**Result-specific dispatch headers (Compacted-N, Quorum, Fastpath,
Modality, etc.) are LOST in streaming mode** — initial response opens
with status + minimal headers BEFORE dispatch runs. Document tradeoff:
clients that need observability stay on ``stream:false``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from .config import MetaModelConfig
from .errors import error_envelope, error_type_for_status
from .moa.dispatch import DispatchResult, dispatch
from .sanitize import sanitize_reasoning

log = logging.getLogger(__name__)


HEARTBEAT_INTERVAL_SECS = 10.0
"""Default seconds between SSE heartbeats while dispatch is pending.

Default-30s client read timeouts won't fire because at least one byte
arrives within ``HEARTBEAT_INTERVAL_SECS``. Tests inject smaller values
via the ``heartbeat_interval`` keyword to ``stream_chat_completion``.
"""

CONTENT_CHUNK_TARGET_CHARS = 40
"""Target size for content chunks in characters.

Small enough to feel "streaming-ish" for a typical response (10-50
chunks per ~500-char reply). Word boundaries respected when possible.
The exact size is approximate — chunks vary as ``_chunk_content`` slices
at whitespace.
"""


# ── Wire-format helpers ─────────────────────────────────────────────


def _format_sse_data(obj: dict[str, Any]) -> bytes:
    """Encode an SSE ``data:`` event from a dict payload.

    Output: ``data: <json>\\n\\n``. JSON has no embedded newlines (the
    default ``json.dumps`` writes a single line), so the SSE parser
    sees one event boundary per blank-line.
    """
    return f"data: {json.dumps(obj, separators=(',', ':'), ensure_ascii=False)}\n\n".encode()


def _format_sse_comment(text: str) -> bytes:
    """Encode an SSE comment line (``: <text>\\n\\n``).

    Per the EventSource spec, lines beginning with ``:`` are comments
    and clients ignore them. Used for heartbeats so the connection
    stays warm during the MoA wait without producing parseable events
    on the client side.
    """
    return f": {text}\n\n".encode()


def _format_sse_done() -> bytes:
    """Encode the OpenAI streaming sentinel: ``data: [DONE]\\n\\n``."""
    return b"data: [DONE]\n\n"


# ── Content chunking ────────────────────────────────────────────────


def _chunk_content(text: str, target_size: int = CONTENT_CHUNK_TARGET_CHARS) -> Iterator[str]:
    """Split ``text`` into approximately ``target_size``-char chunks at
    whitespace boundaries.

    Invariant: ``"".join(_chunk_content(t, n)) == t`` for any ``t`` and
    ``n > 0``. Concatenation is byte-exact — leading/trailing
    whitespace, double spaces, internal newlines all preserved. Runs
    of non-whitespace longer than ``target_size`` emit verbatim as a
    single oversized chunk rather than being split mid-token.
    """
    if not text:
        return
    n = len(text)
    i = 0
    while i < n:
        end = min(i + target_size, n)
        if end >= n:
            yield text[i:n]
            return
        # Try to slide ``end`` back to the last whitespace at or after
        # the midpoint of the current window. If no whitespace exists
        # in that range, emit the full target_size chunk anyway —
        # better to emit slightly oversized than split a token.
        midpoint = i + max(1, target_size // 2)
        split = -1
        for j in range(end - 1, midpoint - 1, -1):
            if text[j].isspace():
                split = j + 1  # include the whitespace in the current chunk
                break
        if split == -1:
            yield text[i:end]
            i = end
        else:
            yield text[i:split]
            i = split


# ── Response → chunks ────────────────────────────────────────────────


class MalformedSynthResponse(Exception):
    """Raised when ``_response_to_chunks`` is handed a 2xx payload that
    isn't a recognizable ChatCompletion (no ``choices[0].message``).

    Review r36 #1: 2xx non-ChatCompletion payloads (e.g.
    ``{"error": {...}}`` or ``{"choices": []}``) reach the streaming
    re-chunker via single-upstream passthrough, which accepts any 2xx
    JSON object (dispatch.py:528). Without this guard the stream
    would emit a fake role+stop+DONE sequence on top of an upstream
    error body. The caller catches this exception and emits an SSE
    error envelope (``upstream_protocol_error``) with no ``[DONE]``.
    """


def _new_chunk_envelope(
    *,
    chunk_id: str,
    model_label: str,
    created_ts: int,
    delta: dict[str, Any],
    finish_reason: str | None,
    include_usage: bool,
) -> dict[str, Any]:
    """Build one SSE chunk dict.

    All non-usage chunks in a stream share the same ``id`` and
    ``created`` (per OpenAI's actual streams). When
    ``include_usage`` is true, every non-usage chunk carries
    ``usage: null`` at top level — the final usage chunk replaces
    that with the real usage object and clears ``choices``.
    """
    chunk: dict[str, Any] = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created_ts,
        "model": model_label,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if include_usage:
        chunk["usage"] = None
    return chunk


def _response_to_chunks(
    response: dict[str, Any],
    *,
    chunk_id: str,
    model_label: str,
    created_ts: int,
    include_usage: bool,
) -> Iterator[dict[str, Any]]:
    """Convert a synthesized ChatCompletion to a sequence of SSE chunks.

    Walks ``response.choices[0].message`` and yields:
    - one role chunk (with ``content=""`` for text, ``content=None``
      for tool-call responses),
    - content deltas (text only) OR tool_calls deltas (one per call,
      indexed),
    - one final chunk with ``finish_reason`` and empty delta,
    - one usage chunk if ``include_usage`` true.

    The synthesizer always emits at least one choice with a message
    object (D.2.2 guarantees this; ``SynthesisFailure`` would have
    been turned into a DispatchResult error before we reach here).
    Single-upstream passthrough is laxer — it accepts any 2xx JSON
    object — so this function validates the ChatCompletion shape and
    raises ``MalformedSynthResponse`` on any payload missing
    ``choices[0].message``. The caller turns that into an SSE
    ``upstream_protocol_error``.
    """
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise MalformedSynthResponse(
            "2xx response missing choices[0]: cannot stream as a ChatCompletion"
        )
    choice = choices[0]
    msg = choice.get("message")
    if not isinstance(msg, dict):
        raise MalformedSynthResponse(
            "2xx response missing choices[0].message: cannot stream as a ChatCompletion"
        )
    finish_reason = choice.get("finish_reason")

    content = msg.get("content")
    # Review r37: validate tool_calls substructure here so an upstream
    # body with non-list tool_calls or non-dict items / function turns
    # into ``upstream_protocol_error`` (caller catches MalformedSynth-
    # Response) instead of leaking as ``internal server error`` from
    # the generic catch-all in ``stream_chat_completion``.
    tool_calls_raw = msg.get("tool_calls")
    if tool_calls_raw is None:
        tool_calls: list[dict[str, Any]] = []
    elif not isinstance(tool_calls_raw, list):
        raise MalformedSynthResponse(
            f"choices[0].message.tool_calls is {type(tool_calls_raw).__name__}, expected list"
        )
    else:
        for i, tc in enumerate(tool_calls_raw):
            if not isinstance(tc, dict):
                raise MalformedSynthResponse(
                    f"choices[0].message.tool_calls[{i}] is {type(tc).__name__}, expected object"
                )
            fn = tc.get("function")
            if fn is not None and not isinstance(fn, dict):
                raise MalformedSynthResponse(
                    f"choices[0].message.tool_calls[{i}].function is "
                    f"{type(fn).__name__}, expected object"
                )
        tool_calls = tool_calls_raw
    # Review r36 #2: legacy `message.function_call` (deprecated but still
    # produced by some upstreams + recognized by the synthesizer at
    # synthesizer.py:172) must surface as `delta.function_call` chunks
    # in OpenAI's legacy streaming format. Otherwise the response
    # collapses to role + finish_reason with the call body silently
    # dropped. We emit it AS-IS rather than promoting to tool_calls so
    # clients that asked for the legacy shape get the legacy shape back.
    function_call = msg.get("function_call")
    if function_call is not None and not isinstance(function_call, dict):
        raise MalformedSynthResponse(
            f"choices[0].message.function_call is {type(function_call).__name__}, expected object"
        )
    has_legacy_function_call = isinstance(function_call, dict) and not tool_calls and not content

    # F3: surfacing reasoning fields when present. The pre-stream
    # sanitizer strips them when ``expose_reasoning=false``; when they
    # survive into chunking, the profile is opted-in and the deltas
    # belong on the wire.
    reasoning_text = msg.get("reasoning")
    reasoning_content = msg.get("reasoning_content")
    has_reasoning = (
        isinstance(reasoning_text, str) and reasoning_text
    ) or (isinstance(reasoning_content, str) and reasoning_content)

    # Role chunk. ``content: ""`` for text responses (matches OpenAI),
    # ``content: null`` for tool-call / function-call responses.
    role_delta: dict[str, Any] = {"role": "assistant"}
    if (tool_calls or has_legacy_function_call) and not content:
        role_delta["content"] = None
    else:
        role_delta["content"] = ""
    yield _new_chunk_envelope(
        chunk_id=chunk_id,
        model_label=model_label,
        created_ts=created_ts,
        delta=role_delta,
        finish_reason=None,
        include_usage=include_usage,
    )

    # F3: reasoning deltas (single-burst). Emit before content so
    # clients see the chain-of-thought stream before the answer,
    # mirroring how thinking models stream natively.
    if has_reasoning:
        reasoning_delta: dict[str, Any] = {}
        if isinstance(reasoning_content, str) and reasoning_content:
            reasoning_delta["reasoning_content"] = reasoning_content
        if isinstance(reasoning_text, str) and reasoning_text:
            reasoning_delta["reasoning"] = reasoning_text
        yield _new_chunk_envelope(
            chunk_id=chunk_id,
            model_label=model_label,
            created_ts=created_ts,
            delta=reasoning_delta,
            finish_reason=None,
            include_usage=include_usage,
        )

    # Content deltas (only for text responses with non-empty content).
    if isinstance(content, str) and content:
        for piece in _chunk_content(content):
            yield _new_chunk_envelope(
                chunk_id=chunk_id,
                model_label=model_label,
                created_ts=created_ts,
                delta={"content": piece},
                finish_reason=None,
                include_usage=include_usage,
            )

    # Tool-call deltas — one per call, full id+type+name+arguments. The
    # plan permits "minimum number of chunks needed for valid tool_calls
    # indexed deltas" (§232). Single-burst per call avoids JSON-fragmen-
    # tation complexity and is schema-compliant.
    for idx, tc in enumerate(tool_calls):
        tc_delta = {
            "tool_calls": [
                {
                    "index": idx,
                    "id": tc.get("id"),
                    "type": tc.get("type", "function"),
                    "function": {
                        "name": (tc.get("function") or {}).get("name"),
                        "arguments": (tc.get("function") or {}).get("arguments", ""),
                    },
                }
            ]
        }
        yield _new_chunk_envelope(
            chunk_id=chunk_id,
            model_label=model_label,
            created_ts=created_ts,
            delta=tc_delta,
            finish_reason=None,
            include_usage=include_usage,
        )

    # Legacy function_call delta (single-burst, name + arguments).
    if has_legacy_function_call:
        fc_delta = {
            "function_call": {
                "name": function_call.get("name"),
                "arguments": function_call.get("arguments", ""),
            }
        }
        yield _new_chunk_envelope(
            chunk_id=chunk_id,
            model_label=model_label,
            created_ts=created_ts,
            delta=fc_delta,
            finish_reason=None,
            include_usage=include_usage,
        )

    # Final chunk. Empty delta, finish_reason set.
    if finish_reason is None:
        if tool_calls:
            finish_reason = "tool_calls"
        elif has_legacy_function_call:
            finish_reason = "function_call"
        else:
            finish_reason = "stop"
    yield _new_chunk_envelope(
        chunk_id=chunk_id,
        model_label=model_label,
        created_ts=created_ts,
        delta={},
        finish_reason=finish_reason,
        include_usage=include_usage,
    )

    # Usage chunk — emitted ONLY when stream_options.include_usage true.
    # OpenAI's wire shape: empty choices array, usage object populated.
    if include_usage:
        usage = response.get("usage") or {}
        yield {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": model_label,
            "choices": [],
            "usage": usage,
        }


# ── Error envelope from DispatchResult ──────────────────────────────


def _dispatch_result_to_error_envelope(result: DispatchResult) -> dict[str, Any]:
    """Translate a non-success DispatchResult into an OpenAI envelope.

    Two cases:
    - ``result.error is not None``: explicit ``(message, code)`` from
      the dispatcher. Use it directly.
    - ``result.error is None`` but status is non-2xx (raw upstream
      passthrough returning 429/500 etc.): synthesize a meta-model
      envelope. The actual upstream body is discarded — clients
      addressing via streaming get our error shape; if they need the
      verbatim upstream body they should use ``stream:false`` (review
      r34 F1: streaming is meta-model's contract).
    """
    if result.error is not None:
        message, code = result.error
        # Review r2 F6 MED: forward DispatchResult.error_type override
        # to streaming envelopes too — keep error envelope `type` field
        # consistent across stream/non-stream paths.
        return error_envelope(
            message,
            status=result.status_code,
            code=code,
            type_=result.error_type,
        )
    return error_envelope(
        f"upstream returned HTTP {result.status_code}",
        status=result.status_code,
        code="upstream_error",
        type_=error_type_for_status(result.status_code),
    )


# ── Main streaming generator ─────────────────────────────────────────


async def stream_chat_completion(
    cfg: MetaModelConfig,
    request_body: dict[str, Any],
    *,
    model: str,
    ext_profile: str | None,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None,
    chunk_id: str,
    model_label: str,
    include_usage: bool,
    heartbeat_interval: float = HEARTBEAT_INTERVAL_SECS,
) -> AsyncIterator[bytes]:
    """Run dispatch and stream the synthesized response as SSE.

    Yields raw bytes ready for SSE wire transmission. The caller wraps
    this in a ``StreamingResponse`` with media type
    ``text/event-stream``.

    Behavior:
    1. Spawn dispatch as a background task.
    2. While the task is pending, emit ``: heartbeat\\n\\n`` every
       ``heartbeat_interval`` seconds.
    3. When dispatch completes successfully (DispatchResult.error is
       None AND status 2xx): convert the response payload to SSE
       chunks and emit them, ending with ``[DONE]``.
    4. On any failure (DispatchResult.error set OR status not 2xx OR
       unhandled exception): emit one ``data: {"error": ...}\\n\\n``
       and close. **No ``[DONE]`` after error** — this mirrors OpenAI
       SDK behavior.
    5. On generator cancellation (client disconnect via Starlette):
       cancel the dispatch task in finally so we don't leak HTTP calls
       to upstreams.

    `chunk_id` is the OpenAI-shaped identifier embedded in every chunk
    (``chatcmpl-<hex>``), distinct from the meta-model request id used
    for the ``X-MetaModel-Request-Id`` header. Review r35 F4: keep them
    separate so clients see standard OpenAI ``id`` shape.
    """
    created_ts = int(time.time())

    dispatch_task = asyncio.create_task(
        dispatch(
            cfg,
            request_body,
            model=model,
            ext_profile=ext_profile,
            timeout_secs=timeout_secs,
            transport=transport,
        )
    )
    try:
        # Heartbeat while waiting. Reusing the same Task across
        # ``asyncio.wait`` calls is safe — it does not cancel on
        # timeout (unlike ``wait_for``). Review r34 confirmed.
        while not dispatch_task.done():
            done, _pending = await asyncio.wait({dispatch_task}, timeout=heartbeat_interval)
            if not done:
                yield _format_sse_comment("heartbeat")

        # Dispatch returned (or raised). Handle both.
        try:
            result = dispatch_task.result()
        except Exception as e:  # top-of-stack catch by design
            # Review r35 F3: any unhandled dispatch exception must NOT
            # leak a stack trace. Emit a generic 500 SSE error event.
            log.exception("dispatch raised in streaming path: %s", e)
            yield _format_sse_data(error_envelope("internal server error", status=500))
            return

        # Review r35 F2: check status NOT 2xx (covers 3xx/4xx/5xx —
        # raw upstream passthrough may relay 429/500 with payload set
        # and error=None; never treat that as a ChatCompletion).
        if result.error is not None or not (200 <= result.status_code < 300):
            yield _format_sse_data(_dispatch_result_to_error_envelope(result))
            return

        # F3: apply expose_reasoning policy BEFORE chunking. Resolved
        # profile name lives in result.headers; default false when no
        # profile is in play (e.g. raw-upstream passthrough). The
        # rescue branch (lift reasoning_content into content when
        # content is empty) needs the finalized message shape to fire,
        # so it MUST run before the chunker walks the message.
        sanitized_payload = result.payload or {}
        profile_name = result.headers.get("X-MetaModel-Profile")
        expose_reasoning = False
        if profile_name and profile_name in cfg.profiles:
            expose_reasoning = getattr(
                cfg.profiles[profile_name], "expose_reasoning", False
            )
        if isinstance(sanitized_payload, dict):
            sanitized_payload = sanitize_reasoning(
                sanitized_payload, expose_reasoning=expose_reasoning
            )

        # Success — re-chunk the synthesized payload.
        # Review r36 #1: success-path payloads from passthrough may not
        # actually be ChatCompletions (any 2xx JSON is accepted by
        # ``_passthrough_single``). ``MalformedSynthResponse`` is the
        # validator's signal that we must NOT proceed to a fake stop+
        # DONE — emit ``upstream_protocol_error`` (mirrors dispatch.py's
        # non-JSON / non-object code) and close without ``[DONE]``.
        try:
            chunks = list(
                _response_to_chunks(
                    sanitized_payload,
                    chunk_id=chunk_id,
                    model_label=model_label,
                    created_ts=created_ts,
                    include_usage=include_usage,
                )
            )
        except MalformedSynthResponse as e:
            yield _format_sse_data(
                error_envelope(
                    f"upstream returned a malformed response body: {e}",
                    status=502,
                    code="upstream_protocol_error",
                )
            )
            return
        except Exception as e:
            # Defensive: if the success payload format is otherwise
            # broken, fail gracefully rather than half-streaming.
            log.exception("failed to format synthesized response as SSE: %s", e)
            yield _format_sse_data(error_envelope("internal server error", status=500))
            return

        for chunk in chunks:
            yield _format_sse_data(chunk)

        yield _format_sse_done()

    finally:
        # Review r34 F4: client disconnect raises CancelledError in this
        # generator. Cancel dispatch + drain so we don't leak upstream
        # HTTP calls. ``asyncio.CancelledError`` re-raises through the
        # ``finally`` so Starlette sees the proper cancellation signal.
        if not dispatch_task.done():
            dispatch_task.cancel()
            try:
                await dispatch_task
            except (asyncio.CancelledError, Exception):
                pass


def new_chunk_id() -> str:
    """OpenAI-shaped chat-completion chunk id (``chatcmpl-<24 hex>``)."""
    return "chatcmpl-" + uuid.uuid4().hex[:24]


__all__ = [
    "CONTENT_CHUNK_TARGET_CHARS",
    "HEARTBEAT_INTERVAL_SECS",
    "_chunk_content",  # exported for tests
    "_dispatch_result_to_error_envelope",  # exported for tests
    "_format_sse_comment",
    "_format_sse_data",
    "_format_sse_done",
    "_response_to_chunks",  # exported for tests
    "new_chunk_id",
    "stream_chat_completion",
]
