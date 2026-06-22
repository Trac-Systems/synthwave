"""F4-Responses — `/v1/responses` minimal adapter.

Converts OpenAI Responses API requests to / from the meta-model's
Chat Completions backbone. Stateless: `previous_response_id` is
typed-501 because we don't persist conversation history. The
adapter is "drop-in" only in the sense that vanilla OpenAI clients
(SDK ≥ 1.30 defaults to Responses) get a working response — not a
full Responses implementation.

Two halves:

1. **Request**: `responses_to_chat(body)` flattens the Responses
   `input` (string or list of input items) + `instructions` +
   `tools` + `tool_choice` into a Chat Completions body. Typed-501
   surfaces unsupported part types / tool selectors with explicit
   advisories so the client knows what to remove.

2. **Response**: `chat_to_responses(chat_response, request_id, ...)`
   reshapes the synthesized chat body into a Responses-shaped
   envelope: per-message items, per-function-call items, flattened
   `output_text` convenience field, `usage`, `status`, `id`.

Streaming uses the Responses event taxonomy:
`response.created` → `response.in_progress` →
`response.output_item.added` → (text / function_call deltas) →
`response.output_item.done` → `response.completed`. The simulated
SSE emitter runs AFTER dispatch finishes (mirrors the chat-side
post-dispatch chunking), so clients receive a coherent event
sequence even though the upstream synthesis itself isn't streamed.

Honest scope (TYPED_501 below):
- `previous_response_id` — stateful chaining not implemented.
- Built-in tools (web_search / file_search / computer_use).
- `background: true`, `conversations` resource.
- Multimodal parts beyond `input_text` / `input_image` /
  `output_text` echo.

The adapter routes through the SHARED `routing.resolve_profile()`
so unknown models surface the same typed 404 envelope as
`/v1/chat/completions` and `/v1/completions`.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from .config import MetaModelConfig


class ResponsesAdapterError(Exception):
    """Typed adapter error. Carries a structured envelope so the
    endpoint converts it into the OpenAI error body without
    re-stringifying the message."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        param: str | None = None,
        type_: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.param = param
        self.type_ = type_


# ── Top-level unsupported features ──────────────────────────────────


# Review r1 F4-Responses MED: only reject these when truthy. `null` /
# `false` / missing-default should pass through silently — clients
# often send the field with a null default. The actual API field is
# `conversation` (singular), not `conversations` — fixed.
_UNSUPPORTED_TOP_LEVEL = {
    "previous_response_id": (
        "stateful Responses chaining is not implemented; send the full "
        "conversation in `input`"
    ),
    "background": "background: true is not implemented",
    "conversation": "the conversation resource is not implemented",
}


_UNSUPPORTED_PART_TYPES = {
    "input_file",
    "input_audio",
    "input_video",
    "computer_call",
    "computer_call_output",
    "web_search_call",
    "file_search_call",
    "code_interpreter_call",
}


_UNSUPPORTED_OUTPUT_PART_TYPES = {
    "input_image",
    "input_file",
    "input_audio",
    "input_video",
}


_UNSUPPORTED_BUILTIN_TOOL_TYPES = {
    "web_search",
    "file_search",
    "computer_use",
    "code_interpreter",
}


# ── Helpers ─────────────────────────────────────────────────────────


def _typed_501(code: str, message: str, *, param: str | None = None) -> ResponsesAdapterError:
    return ResponsesAdapterError(501, code, message, param=param)


def _typed_400(code: str, message: str, *, param: str | None = None) -> ResponsesAdapterError:
    return ResponsesAdapterError(400, code, message, param=param)


def new_response_id() -> str:
    """OpenAI Responses-shaped id (`resp_<24 hex>`)."""
    return "resp_" + uuid.uuid4().hex[:24]


# ── Request: input items → chat messages ────────────────────────────


def _map_input_text_part(part: dict) -> dict:
    return {"type": "text", "text": part.get("text", "")}


def _map_input_image_part(part: dict) -> dict:
    """Responses `input_image` → Chat Completions `image_url`.

    OpenAI Responses `input_image` carries `image_url` as either a
    string or `{url, detail}`. Chat Completions wants
    `{type:"image_url", "image_url":{url, detail?}}`.
    """
    image_url = part.get("image_url")
    if isinstance(image_url, str):
        chat_image_url: dict[str, Any] = {"url": image_url}
    elif isinstance(image_url, dict):
        chat_image_url = {"url": image_url.get("url", "")}
        if "detail" in image_url:
            chat_image_url["detail"] = image_url["detail"]
    else:
        # Some SDKs send `file_id` or base64 separately. Out of scope
        # for the minimal adapter — typed 501 with advisory.
        raise _typed_501(
            "unsupported_image_shape",
            "input_image without `image_url` (e.g. file_id only) is not "
            "implemented; pass `image_url` as URL or data URI",
            param="input",
        )
    detail = part.get("detail")
    if detail and "detail" not in chat_image_url:
        chat_image_url["detail"] = detail
    return {"type": "image_url", "image_url": chat_image_url}


def _map_message_content_parts(parts: list, *, role: str) -> str | list[dict]:
    """Map a Responses message item's content list to Chat Completions
    content (string or list of typed parts)."""
    chat_parts: list[dict] = []
    text_only = True
    for part in parts:
        if not isinstance(part, dict):
            raise _typed_400(
                "invalid_input_part",
                f"input message content part must be an object, got {type(part).__name__}",
                param="input",
            )
        ptype = part.get("type")
        if ptype == "input_text":
            chat_parts.append(_map_input_text_part(part))
        elif ptype == "output_text":
            # Conversation echo of an assistant message — flatten as text.
            chat_parts.append({"type": "text", "text": part.get("text", "")})
        elif ptype == "input_image":
            chat_parts.append(_map_input_image_part(part))
            text_only = False
        elif ptype in _UNSUPPORTED_PART_TYPES:
            raise _typed_501(
                "unsupported_input_part_type",
                f"input part type {ptype!r} is not implemented",
                param="input",
            )
        else:
            raise _typed_501(
                "unsupported_input_part_type",
                f"input part type {ptype!r} is not recognized",
                param="input",
            )

    if text_only and len(chat_parts) == 1 and chat_parts[0]["type"] == "text":
        # Collapse single-text-part to plain string for cleaner upstream.
        return chat_parts[0]["text"]
    if text_only and not chat_parts:
        return ""
    return chat_parts


def _map_function_call_output(item: dict) -> dict:
    """Responses `function_call_output` → Chat `tool` message.

    `output` may be:
      - str → tool message content directly.
      - list of `{type: "input_text", text: ...}` parts → flatten by
        concatenation. (OpenAI Agents SDK convention.)
      - list of `{type: "output_text", text: ...}` parts → also
        flatten (compat kindness).
      - any list containing `input_image`/`input_file`/audio →
        typed 501 advisory.
    """
    output = item.get("output")
    call_id = item.get("call_id") or item.get("id")
    if call_id is None:
        raise _typed_400(
            "missing_call_id",
            "function_call_output is missing `call_id`",
            param="input",
        )
    if isinstance(output, str):
        content = output
    elif isinstance(output, list):
        chunks: list[str] = []
        for part in output:
            if not isinstance(part, dict):
                raise _typed_400(
                    "invalid_function_call_output",
                    "function_call_output.output list parts must be objects",
                    param="input",
                )
            ptype = part.get("type")
            if ptype in ("input_text", "output_text"):
                chunks.append(part.get("text", ""))
            elif ptype in _UNSUPPORTED_OUTPUT_PART_TYPES:
                raise _typed_501(
                    "unsupported_function_call_output_part",
                    f"function_call_output part type {ptype!r} is not "
                    f"implemented; tool outputs must be text",
                    param="input",
                )
            else:
                raise _typed_501(
                    "unsupported_function_call_output_part",
                    f"function_call_output part type {ptype!r} is not recognized",
                    param="input",
                )
        content = "".join(chunks)
    else:
        raise _typed_400(
            "invalid_function_call_output",
            "function_call_output.output must be string or list",
            param="input",
        )
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _map_function_call_item(item: dict) -> dict:
    """Responses `function_call` input → Chat assistant message with
    `tool_calls`. Each input function_call item is its own assistant
    message in Chat Completions semantics."""
    call_id = item.get("call_id") or item.get("id")
    name = item.get("name")
    arguments = item.get("arguments")
    if call_id is None or not isinstance(name, str):
        raise _typed_400(
            "invalid_function_call",
            "function_call input item must have `call_id` and `name`",
            param="input",
        )
    if arguments is None:
        arguments = ""
    if not isinstance(arguments, str):
        # Spec says JSON-string; some clients send dict — re-encode.
        import json

        arguments = json.dumps(arguments)
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _input_to_messages(req_input: Any) -> list[dict]:
    """Flatten the Responses `input` field to Chat messages.

    String input → single user message. List of items → walk per
    documented item types; reject unknown shapes with typed-501.
    """
    if req_input is None:
        return []
    if isinstance(req_input, str):
        return [{"role": "user", "content": req_input}]
    if not isinstance(req_input, list):
        raise _typed_400(
            "invalid_input",
            "`input` must be a string or list of input items",
            param="input",
        )
    messages: list[dict] = []
    for item in req_input:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise _typed_400(
                "invalid_input_item",
                "input list items must be strings or objects",
                param="input",
            )
        itype = item.get("type", "message")
        if itype == "message":
            role = item.get("role", "user")
            if role not in ("system", "user", "assistant", "developer"):
                raise _typed_400(
                    "invalid_role",
                    f"input message role {role!r} is not valid",
                    param="input",
                )
            # OpenAI's "developer" role maps to system in Chat semantics.
            chat_role = "system" if role == "developer" else role
            content = item.get("content")
            if isinstance(content, str):
                messages.append({"role": chat_role, "content": content})
            elif isinstance(content, list):
                messages.append(
                    {
                        "role": chat_role,
                        "content": _map_message_content_parts(content, role=chat_role),
                    }
                )
            elif content is None:
                messages.append({"role": chat_role, "content": ""})
            else:
                raise _typed_400(
                    "invalid_input_content",
                    "message content must be string or list",
                    param="input",
                )
        elif itype == "function_call":
            messages.append(_map_function_call_item(item))
        elif itype == "function_call_output":
            messages.append(_map_function_call_output(item))
        elif itype in _UNSUPPORTED_PART_TYPES:
            raise _typed_501(
                "unsupported_input_item",
                f"input item type {itype!r} is not implemented",
                param="input",
            )
        else:
            raise _typed_501(
                "unsupported_input_item",
                f"input item type {itype!r} is not recognized",
                param="input",
            )
    return messages


# ── tool_choice mapping ─────────────────────────────────────────────


def _map_tool_choice(tc: Any) -> Any:
    """Responses `tool_choice` → Chat `tool_choice`.

    - `"auto"` / `"none"` / `"required"` → identical (Chat already
      accepts these strings).
    - `{"type":"function","name":"<fn>"}` →
      `{"type":"function","function":{"name":"<fn>"}}`.
    - typed 501 for `allowed_tools`, built-in selectors, novel types.
    """
    if tc is None:
        return None
    if isinstance(tc, str):
        if tc in ("auto", "none", "required"):
            return tc
        raise _typed_501(
            "unsupported_tool_choice",
            f"tool_choice {tc!r} is not implemented",
            param="tool_choice",
        )
    if not isinstance(tc, dict):
        raise _typed_400(
            "invalid_tool_choice",
            "tool_choice must be a string or object",
            param="tool_choice",
        )
    ttype = tc.get("type")
    if ttype == "function":
        name = tc.get("name")
        if not isinstance(name, str):
            raise _typed_400(
                "invalid_tool_choice",
                "tool_choice {type:function} requires `name`",
                param="tool_choice",
            )
        return {"type": "function", "function": {"name": name}}
    if ttype == "allowed_tools":
        raise _typed_501(
            "unsupported_tool_choice",
            "tool_choice.allowed_tools is not implemented; pass `tool_choice` as "
            '"auto" / "required" / {type:"function", name:...}',
            param="tool_choice",
        )
    if ttype in _UNSUPPORTED_BUILTIN_TOOL_TYPES:
        raise _typed_501(
            "unsupported_tool_choice",
            f"tool_choice.{ttype!r} (built-in tool selector) is not implemented",
            param="tool_choice",
        )
    raise _typed_501(
        "unsupported_tool_choice",
        f"tool_choice.type {ttype!r} is not recognized",
        param="tool_choice",
    )


# ── tools mapping ───────────────────────────────────────────────────


def _map_tools(tools: Any) -> list[dict] | None:
    """Responses `tools` → Chat Completions `tools`.

    Review r1 F4-Responses HIGH: Responses uses a FLAT function tool
    shape (`{type:"function", name, description?, parameters}`). Chat
    Completions expects the NESTED shape
    (`{type:"function", function:{name, ...}}`) and the meta-model's
    own dispatch tools layer at moa/tools.py rejects flat shapes as
    malformed. Rewrite flat → nested here. Tolerate already-nested
    tools (some clients send chat-shaped tools for compat).

    Built-in tools (web_search etc.) are typed-501.
    """
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise _typed_400(
            "invalid_tools",
            "`tools` must be a list",
            param="tools",
        )
    out: list[dict] = []
    for t in tools:
        if not isinstance(t, dict):
            raise _typed_400(
                "invalid_tools",
                "each tool must be an object",
                param="tools",
            )
        ttype = t.get("type")
        if ttype == "function":
            inner = t.get("function")
            if isinstance(inner, dict):
                # Already chat-shaped — passthrough.
                out.append(t)
            else:
                # Flat Responses shape: lift name/description/parameters
                # under `function`, keep `type` at top level.
                fn: dict[str, Any] = {}
                if "name" in t:
                    fn["name"] = t["name"]
                if "description" in t:
                    fn["description"] = t["description"]
                if "parameters" in t:
                    fn["parameters"] = t["parameters"]
                if "strict" in t:
                    fn["strict"] = t["strict"]
                if not fn.get("name"):
                    raise _typed_400(
                        "invalid_tools",
                        "function tool must have `name`",
                        param="tools",
                    )
                out.append({"type": "function", "function": fn})
        elif ttype in _UNSUPPORTED_BUILTIN_TOOL_TYPES:
            raise _typed_501(
                "unsupported_tool",
                f"built-in tool {ttype!r} is not implemented",
                param="tools",
            )
        else:
            raise _typed_501(
                "unsupported_tool",
                f"tool type {ttype!r} is not recognized",
                param="tools",
            )
    return out


# ── Public: responses_to_chat ───────────────────────────────────────


def responses_to_chat(body: dict) -> dict:
    """Convert a Responses request body → Chat Completions body.

    Caller is responsible for resolving `model` separately (so the
    typed 404 envelope from `routing.resolve_profile` happens before
    this function runs). This function only does the body shape
    conversion + raises typed 501/400 for unsupported features.
    """
    if not isinstance(body, dict):
        raise _typed_400(
            "invalid_request",
            "request body must be a JSON object",
        )

    # Reject unsupported top-level features up-front so partial
    # mappings don't run and produce confusing errors. Review r1
    # F4-Responses MED: only reject TRUTHY values — clients commonly
    # send `previous_response_id: null` / `background: false` as
    # part of a default request shape. The semantic payload is
    # absent in those cases; passing through is harmless.
    for field, message in _UNSUPPORTED_TOP_LEVEL.items():
        if field in body and body[field]:
            raise _typed_501(
                "unsupported_responses_param",
                message,
                param=field,
            )

    # text.format → response_format mapping. Plain text format passes
    # through silently; json_schema / json_object are typed-501 since
    # we don't translate to chat's response_format yet (the synthesizer
    # has no JSON-schema grounding step). Review r1 F4-Responses MED.
    text = body.get("text")
    if isinstance(text, dict):
        fmt = text.get("format")
        if isinstance(fmt, dict):
            ftype = fmt.get("type", "text")
            if ftype != "text":
                raise _typed_501(
                    "unsupported_text_format",
                    f"text.format.type={ftype!r} (structured outputs) is not "
                    f"implemented; pass plain text or omit `text.format`",
                    param="text.format",
                )

    chat: dict[str, Any] = {}

    # `instructions` → leading system message, prepended.
    instructions = body.get("instructions")
    messages: list[dict] = []
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})

    messages.extend(_input_to_messages(body.get("input")))
    chat["messages"] = messages

    # Top-level scalars that map 1:1.
    for src, dst in (
        ("model", "model"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_output_tokens", "max_tokens"),
        ("seed", "seed"),
        ("user", "user"),
    ):
        if src in body:
            chat[dst] = body[src]

    if "stop" in body:
        chat["stop"] = body["stop"]

    tools = _map_tools(body.get("tools"))
    if tools is not None:
        chat["tools"] = tools

    tc = _map_tool_choice(body.get("tool_choice"))
    if tc is not None:
        chat["tool_choice"] = tc

    if body.get("stream"):
        # Adapter consumer (server.py /v1/responses) handles streaming
        # AFTER dispatch — chat dispatch itself runs non-streaming so
        # we can rebuild the Responses event sequence around the
        # synthesized result. Strip stream so chat dispatch doesn't
        # wrap into SSE.
        chat["stream"] = False

    return chat


# ── Response: chat → responses shape ────────────────────────────────


def _build_message_item(content: str, *, item_id: str) -> dict:
    """Responses message output item with one text content part."""
    return {
        "type": "message",
        "id": item_id,
        "status": "completed",
        "role": "assistant",
        "content": [
            {"type": "output_text", "text": content, "annotations": []}
        ],
    }


def _build_function_call_item(tc: dict, *, item_id: str) -> dict:
    """Responses function_call output item from a Chat tool_call."""
    fn = tc.get("function") or {}
    return {
        "type": "function_call",
        "id": item_id,
        "status": "completed",
        "call_id": tc.get("id", item_id),
        "name": fn.get("name", ""),
        "arguments": fn.get("arguments", ""),
    }


def chat_to_responses(
    chat_response: dict,
    *,
    response_id: str,
    model_name: str,
    created_at: int | None = None,
) -> dict:
    """Reshape a synthesized Chat Completions response → Responses
    envelope.

    `chat_response` is the dict produced by `dispatch.dispatch()`
    and (when applicable) sanitized by `sanitize_reasoning`. We
    walk `choices[0].message` and emit one or more output items:

    - Text content → one `message` item.
    - Each `tool_calls[i]` → one `function_call` item.

    `output_text` is the flattened concatenation of all message
    items' text — the convenience field SDKs read.
    """
    choices = chat_response.get("choices") or []
    msg = choices[0].get("message") if choices and isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        msg = {}

    output_items: list[dict] = []
    output_text_parts: list[str] = []

    content = msg.get("content")
    if isinstance(content, str) and content:
        item_id = "msg_" + uuid.uuid4().hex[:24]
        output_items.append(_build_message_item(content, item_id=item_id))
        output_text_parts.append(content)
    elif isinstance(content, list):
        # Multimodal-shaped assistant message — flatten text parts.
        text = "".join(
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") in ("text", "output_text")
        )
        if text:
            item_id = "msg_" + uuid.uuid4().hex[:24]
            output_items.append(_build_message_item(text, item_id=item_id))
            output_text_parts.append(text)

    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        item_id = "fc_" + uuid.uuid4().hex[:24]
        output_items.append(_build_function_call_item(tc, item_id=item_id))

    finish_reason = (choices[0].get("finish_reason") if choices else None) or "stop"
    status = "completed"
    # Review r1 F4-Responses MED: Responses spec uses its own enum for
    # incomplete_details.reason (`max_output_tokens`, `content_filter`,
    # ...). Chat uses (`length`, `content_filter`, `tool_calls`, ...).
    # Map the chat finish_reason taxonomy to the Responses one so SDK
    # branching on `incomplete_details.reason` matches the official
    # values.
    incomplete_reason: str | None = None
    if finish_reason == "length":
        status = "incomplete"
        incomplete_reason = "max_output_tokens"
    elif finish_reason == "content_filter":
        status = "incomplete"
        incomplete_reason = "content_filter"

    usage = chat_response.get("usage") or {}
    responses_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at if created_at is not None else int(time.time()),
        "status": status,
        "model": model_name,
        "output": output_items,
        "output_text": "".join(output_text_parts),
        "usage": responses_usage,
        "incomplete_details": (
            None if status == "completed" else {"reason": incomplete_reason}
        ),
    }


# ── Streaming SSE event taxonomy ────────────────────────────────────


def stream_responses_events(
    response_body: dict,
    *,
    chunk_size: int = 32,
):
    """Yield Responses-API SSE event dicts in the documented order.

    Run AFTER `chat_to_responses(...)` so the response body is fully
    materialized; we only chunk the text/arguments to look like a
    streamed answer to drop-in clients. Each yielded dict carries
    `event` (the SSE event type) and `data` (the JSON payload).

    Sequence (per OpenAI Responses streaming spec):
      response.created
      response.in_progress
      for each output_item:
        response.output_item.added
        for message text items:
          response.content_part.added
          response.output_text.delta (repeated)
          response.output_text.done
          response.content_part.done
        for function_call items:
          response.function_call_arguments.delta (repeated)
          response.function_call_arguments.done
        response.output_item.done
      response.completed
    """
    response_id = response_body["id"]
    sequence = 0

    def event(name: str, payload: dict):
        nonlocal sequence
        sequence += 1
        return {
            "event": name,
            "data": {
                "type": name,
                "sequence_number": sequence,
                **payload,
            },
        }

    # Review r1 F4-Responses HIGH: created / in_progress snapshots
    # carry the IN-PROGRESS shape — empty output array, no usage —
    # because the deltas haven't been emitted yet. The final shape
    # ships only in `.done` / `response.completed` / `.incomplete`.
    in_progress_envelope = _envelope_for_stream(
        response_body,
        status="in_progress",
        clear_output=True,
        clear_usage=True,
    )
    yield event("response.created", {"response": in_progress_envelope})
    yield event("response.in_progress", {"response": in_progress_envelope})

    for out_index, item in enumerate(response_body.get("output", [])):
        if item["type"] == "message":
            # Added with empty content; populated via deltas; done event
            # carries the final item.
            in_progress_item = {
                **item,
                "status": "in_progress",
                "content": [],
            }
            yield event(
                "response.output_item.added",
                {"output_index": out_index, "item": in_progress_item},
            )
            content_index = 0
            final_part = item["content"][0] if item.get("content") else {
                "type": "output_text",
                "text": "",
                "annotations": [],
            }
            empty_part = {**final_part, "text": ""}
            yield event(
                "response.content_part.added",
                {
                    "item_id": item["id"],
                    "output_index": out_index,
                    "content_index": content_index,
                    "part": empty_part,
                },
            )
            text = final_part.get("text", "")
            for chunk in _chunk_string(text, chunk_size):
                yield event(
                    "response.output_text.delta",
                    {
                        "item_id": item["id"],
                        "output_index": out_index,
                        "content_index": content_index,
                        "delta": chunk,
                    },
                )
            yield event(
                "response.output_text.done",
                {
                    "item_id": item["id"],
                    "output_index": out_index,
                    "content_index": content_index,
                    "text": text,
                },
            )
            yield event(
                "response.content_part.done",
                {
                    "item_id": item["id"],
                    "output_index": out_index,
                    "content_index": content_index,
                    "part": final_part,
                },
            )

        elif item["type"] == "function_call":
            in_progress_item = {
                **item,
                "status": "in_progress",
                "arguments": "",
            }
            yield event(
                "response.output_item.added",
                {"output_index": out_index, "item": in_progress_item},
            )
            arguments = item.get("arguments", "")
            for chunk in _chunk_string(arguments, chunk_size):
                yield event(
                    "response.function_call_arguments.delta",
                    {
                        "item_id": item["id"],
                        "output_index": out_index,
                        "delta": chunk,
                    },
                )
            yield event(
                "response.function_call_arguments.done",
                {
                    "item_id": item["id"],
                    "output_index": out_index,
                    "arguments": arguments,
                },
            )

        yield event(
            "response.output_item.done",
            {"output_index": out_index, "item": item},
        )

    # Review r1 F4-Responses MED: terminal event mirrors the response
    # status — `response.completed` for "completed", `response.incomplete`
    # for "incomplete" (max_output_tokens, content_filter).
    final_status = response_body.get("status", "completed")
    final_envelope = _envelope_for_stream(response_body, status=final_status)
    if final_status == "incomplete":
        yield event("response.incomplete", {"response": final_envelope})
    else:
        yield event("response.completed", {"response": final_envelope})


def _chunk_string(s: str, size: int):
    """Yield consecutive `size`-char chunks. Empty string → no yields."""
    if not s:
        return
    for i in range(0, len(s), size):
        yield s[i : i + size]


def _envelope_for_stream(
    response_body: dict,
    *,
    status: str,
    clear_output: bool = False,
    clear_usage: bool = False,
) -> dict:
    """Stream-side response envelope — same shape as the final body
    but with status / progress fields overridden.

    Review r1 F4-Responses HIGH: created/in_progress envelopes drop
    `output` (deltas haven't shipped) and `usage` (tokens not
    counted yet); final completed/incomplete envelope keeps both.
    """
    out = dict(response_body)
    out["status"] = status
    if clear_output:
        out["output"] = []
        out["output_text"] = ""
    if clear_usage:
        out["usage"] = None
    # Review r2 F4-Responses MED: when overriding to in_progress,
    # `incomplete_details` MUST be cleared even if the final response
    # is incomplete. Otherwise created/in_progress events leak the
    # terminal reason before `response.incomplete` lands. The reason
    # belongs on the final envelope only.
    if status == "in_progress":
        out["incomplete_details"] = None
    return out


# ── Convenience: empty-config sentinel ──────────────────────────────


def empty_config_envelope(cfg: MetaModelConfig | None) -> ResponsesAdapterError | None:
    """Return a typed adapter error if no upstreams are configured.
    Used by the `/v1/responses` endpoint to surface the same 503
    contract `/v1/chat/completions` uses."""
    if cfg is None or not cfg.upstreams:
        return ResponsesAdapterError(
            503,
            "no_upstream",
            "no upstreams configured",
            type_="service_unavailable_error",
        )
    return None


__all__ = [
    "ResponsesAdapterError",
    "chat_to_responses",
    "empty_config_envelope",
    "new_response_id",
    "responses_to_chat",
    "stream_responses_events",
]
