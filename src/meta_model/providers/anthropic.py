"""Anthropic Messages API provider adapter.

Translates synthwave's internal OpenAI Chat-Completions request/response
shape to and from the Anthropic `/v1/messages` wire protocol, so an
Anthropic model (e.g. claude-opus-4-8) can serve as an MoA generator or
the synthesizer transparently — including tool calling, image input, and
extended-thinking / effort control, none of which survive Anthropic's
OpenAI-compatibility shim.

`forward_anthropic_messages` is the public entry point and mirrors
`upstream.forward_chat_completion`'s signature. It returns a synthetic
`httpx.Response` whose JSON body is OpenAI-shaped, so callers stay
protocol-agnostic. Non-2xx and non-JSON upstream responses are relayed
verbatim, letting the existing fanout error classifier (non_2xx /
non_json) work unchanged.

This adapter is generic to any Anthropic Messages model — no per-model
assumptions live here; model id, version, thinking mode and effort all
come from the upstream config.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from ..config import AnthropicOptions, UpstreamConfig
from ..upstream import prepare_upstream_body

MESSAGES_PATH = "/messages"

# OpenAI `finish_reason` ← Anthropic `stop_reason`.
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "stop",
}


def build_auth_headers(upstream: UpstreamConfig) -> dict[str, str]:
    """Anthropic auth + version headers (`x-api-key` + `anthropic-version`).

    Shared by the request adapter and the health probe so both speak the
    same auth scheme — Anthropic does not accept the OpenAI-style
    `Authorization: Bearer` header, so a generic probe would false-fail
    with 401."""
    opts = upstream.anthropic or AnthropicOptions()
    headers = {"anthropic-version": opts.version}
    api_key = upstream.resolved_api_key()
    if api_key:
        headers["x-api-key"] = api_key
    return headers


# ── Request translation (OpenAI → Anthropic) ────────────────────────


def _text_of(content: Any) -> str:
    """Flatten a message's content to plain text (used for system
    messages, which Anthropic carries as a single top-level string)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return "\n".join(parts)
    return ""


def _translate_image_url(url: str) -> dict[str, Any]:
    """OpenAI `image_url` → Anthropic image block. Handles both
    `data:` base64 URIs and remote http(s) URLs."""
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        meta = header[len("data:") :]  # e.g. "image/png;base64"
        media_type = meta.split(";", 1)[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _translate_content_blocks(content: Any) -> list[dict[str, Any]]:
    """OpenAI message content → Anthropic content blocks. A bare string
    becomes a single text block; a multimodal list maps text and image
    parts (audio/other parts are dropped — Anthropic has no equivalent).
    Empty text is skipped (Anthropic rejects empty text blocks)."""
    blocks: list[dict[str, Any]] = []
    if isinstance(content, str):
        if content:
            blocks.append({"type": "text", "text": content})
        return blocks
    if isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text = str(part.get("text", ""))
                if text:
                    blocks.append({"type": "text", "text": text})
            elif ptype == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url:
                    blocks.append(_translate_image_url(url))
            # input_audio / other modalities: no Anthropic mapping → drop.
    return blocks


def _assistant_blocks(msg: dict[str, Any]) -> list[dict[str, Any]]:
    """Assistant turn → Anthropic content blocks: text first, then a
    `tool_use` block per OpenAI tool_call."""
    blocks = _translate_content_blocks(msg.get("content"))
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args else {}
        except ValueError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "input": args,
            }
        )
    return blocks


def _tool_result_block(msg: dict[str, Any]) -> dict[str, Any]:
    """OpenAI tool-result message → Anthropic `tool_result` block."""
    return {
        "type": "tool_result",
        "tool_use_id": msg.get("tool_call_id", ""),
        "content": _text_of(msg.get("content")),
    }


def _split_system_and_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Partition OpenAI messages into Anthropic's top-level `system`
    string plus an alternating user/assistant message list.

    System messages (at any position) are concatenated into `system`.
    Tool-result messages are gathered into a following user turn.
    Consecutive same-role turns are merged, since Anthropic expects a
    single content list per turn.
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    def _push(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            text = _text_of(msg.get("content"))
            if text:
                system_parts.append(text)
        elif role == "assistant":
            _push("assistant", _assistant_blocks(msg))
        elif role == "tool":
            _push("user", [_tool_result_block(msg)])
        else:  # "user" (and any unknown role treated as user input)
            _push("user", _translate_content_blocks(msg.get("content")))

    return "\n\n".join(system_parts), out


def _translate_tools(tools: Any) -> list[dict[str, Any]]:
    """OpenAI function tools → Anthropic tools."""
    out: list[dict[str, Any]] = []
    if not isinstance(tools, list):
        return out
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        out.append(
            {
                "name": name,
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return out


def _translate_tool_choice(tool_choice: Any) -> dict[str, Any] | None:
    """OpenAI tool_choice → Anthropic tool_choice. ("none" is handled by
    the caller dropping tools entirely, so it never reaches here.)"""
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        if tool_choice == "required":
            return {"type": "any"}
        return None
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        name = (tool_choice.get("function") or {}).get("name")
        if name:
            return {"type": "tool", "name": name}
    return None


def build_messages_request(
    prepared_body: dict[str, Any], upstream: UpstreamConfig
) -> dict[str, Any]:
    """Build the Anthropic `/v1/messages` request body from an
    already-`prepare_upstream_body`'d (OpenAI-shaped) body."""
    opts = upstream.anthropic or AnthropicOptions()
    messages_in = prepared_body.get("messages")
    if not isinstance(messages_in, list):
        messages_in = []
    system_text, a_messages = _split_system_and_messages(messages_in)

    # Anthropic requires max_tokens. Honor the caller's output budget,
    # clamped to the upstream's configured ceiling.
    requested = prepared_body.get("max_tokens") or prepared_body.get("max_completion_tokens")
    max_tokens = int(requested) if requested else upstream.max_output
    if upstream.max_output:
        max_tokens = min(max_tokens, upstream.max_output)
    max_tokens = max(max_tokens, 1)

    out: dict[str, Any] = {
        "model": upstream.model_id,
        "messages": a_messages,
        "max_tokens": max_tokens,
    }
    if system_text:
        out["system"] = system_text

    thinking_on = False
    if opts.thinking == "adaptive":
        out["thinking"] = {"type": "adaptive"}
        thinking_on = True
        if opts.effort:
            out["output_config"] = {"effort": opts.effort}
    elif opts.thinking == "enabled":
        budget = opts.budget_tokens or max(1024, max_tokens // 2)
        if out["max_tokens"] <= budget:
            out["max_tokens"] = budget + 1024
        out["thinking"] = {"type": "enabled", "budget_tokens": budget}
        thinking_on = True

    # Sampling params: Anthropic rejects a custom temperature/top_p while
    # thinking is engaged, so only forward them when thinking is off.
    if not thinking_on:
        temperature = prepared_body.get("temperature")
        if temperature is not None:
            out["temperature"] = temperature
        top_p = prepared_body.get("top_p")
        if top_p is not None:
            out["top_p"] = top_p

    stop = prepared_body.get("stop")
    if stop:
        out["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)

    tools = prepared_body.get("tools")
    tool_choice = prepared_body.get("tool_choice")
    if tools and tool_choice != "none":
        a_tools = _translate_tools(tools)
        if a_tools:
            out["tools"] = a_tools
            tc = _translate_tool_choice(tool_choice)
            if tc is not None:
                out["tool_choice"] = tc
    return out


# ── Response translation (Anthropic → OpenAI) ───────────────────────


def translate_response(anthropic_json: dict[str, Any], model_label: str) -> dict[str, Any]:
    """Anthropic message response → OpenAI chat.completion dict."""
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in anthropic_json.get("content") or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(str(block.get("text", "")))
        elif btype == "thinking":
            reasoning_parts.append(str(block.get("thinking", "")))
        elif btype == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                }
            )
        # redacted_thinking and any unknown block types are dropped.

    text = "".join(text_parts)
    message: dict[str, Any] = {"role": "assistant", "content": text if text else None}
    reasoning_text = "".join(reasoning_parts)
    if reasoning_text:  # omit when thinking is redacted/empty (opus-4.x)
        message["reasoning_content"] = reasoning_text
    if tool_calls:
        message["tool_calls"] = tool_calls

    finish = _STOP_REASON_MAP.get(anthropic_json.get("stop_reason"), "stop")
    usage = anthropic_json.get("usage") or {}
    prompt_tokens = int(usage.get("input_tokens", 0) or 0)
    completion_tokens = int(usage.get("output_tokens", 0) or 0)
    usage_out: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    # Extended thinking: models like claude-opus-4.x bill the reasoning as
    # output tokens and report the count under output_tokens_details rather
    # than returning the (redacted) thinking text. Surface it under the
    # OpenAI reasoning-usage field so it mirrors OpenAI reasoning models.
    details = usage.get("output_tokens_details") or {}
    thinking_tokens = details.get("thinking_tokens")
    if thinking_tokens is not None:
        usage_out["completion_tokens_details"] = {"reasoning_tokens": int(thinking_tokens)}

    return {
        "id": anthropic_json.get("id", ""),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_label,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage_out,
    }


# ── Transport ───────────────────────────────────────────────────────


async def forward_anthropic_messages(
    upstream: UpstreamConfig,
    body: dict[str, Any],
    *,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.Response:
    """POST an OpenAI-shaped body to the Anthropic Messages API and
    return an OpenAI-shaped `httpx.Response`.

    Mirrors `upstream.forward_chat_completion`. Non-2xx / non-JSON
    upstream responses are returned verbatim so the fanout error
    classifier handles them exactly as it does for OpenAI upstreams.
    """
    prepared = prepare_upstream_body(body, upstream)
    request_body = build_messages_request(prepared, upstream)
    headers = {"Content-Type": "application/json", **build_auth_headers(upstream)}
    url = upstream.base_url.rstrip("/") + MESSAGES_PATH
    async with httpx.AsyncClient(transport=transport, timeout=timeout_secs) as client:
        raw = await client.post(url, json=request_body, headers=headers)

    if raw.status_code < 200 or raw.status_code >= 300:
        return raw  # relay verbatim → fanout classifies as non_2xx
    try:
        anthropic_json = raw.json()
    except ValueError:
        return raw  # non-JSON → fanout classifies as non_json
    if not isinstance(anthropic_json, dict):
        return raw

    openai_json = translate_response(
        anthropic_json, anthropic_json.get("model") or upstream.model_id
    )
    return httpx.Response(
        status_code=raw.status_code, json=openai_json, request=raw.request
    )
