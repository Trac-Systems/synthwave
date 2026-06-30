"""Upstream HTTP client.

Forwards a Chat Completions request to a single upstream's
OpenAI-compatible /chat/completions endpoint. Replaces the request's
`model` field with the upstream's `model_id` (clients address upstreams
by their config-key name; the upstream itself expects its own model_id).

Auth headers come from UpstreamConfig.resolved_api_key() and
resolved_basic_auth(). Bearer wins over basic if somehow both
configured (but the validator forbids that combination).

Streaming is not handled here — the endpoint rejects `stream: true`
in D.1.4. D.3.3 will add SSE support.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from .config import UpstreamConfig

CHAT_COMPLETIONS_PATH = "/chat/completions"


def _build_auth_headers(upstream: UpstreamConfig) -> dict[str, str]:
    """Resolve auth headers from the upstream config."""
    api_key = upstream.resolved_api_key()
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    basic = upstream.resolved_basic_auth()
    if basic:
        user, password = basic
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        return {"Authorization": f"Basic {token}"}
    return {}


def _demote_non_leading_system_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Demote every system message after index 0 to role="user" with a
    "[SYSTEM]: " content prefix.

    Some chat templates return 400
    with "System message must be at the beginning" when a system message
    appears mid-conversation. Mid-stream system messages are valid in
    the OpenAI Chat Completions contract and are widely used by clients
    (mid-conversation system messages: nudges, audit feedback, stage
    gates, in-flight directives), so the fix can't be "stop sending
    them" — it has to be a per-upstream shim.

    Position is preserved (the demoted message lands at the same index).
    Non-string content (multimodal parts) gets the prefix prepended to
    the first text part if present; otherwise a synthetic text part is
    inserted at the front.
    """
    if not messages:
        return messages
    out: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role != "system" or idx == 0:
            out.append(msg)
            continue
        demoted = dict(msg)
        demoted["role"] = "user"
        content = demoted.get("content")
        if isinstance(content, str):
            demoted["content"] = f"[SYSTEM]: {content}"
        elif isinstance(content, list):
            new_content: list[Any] = []
            prefixed = False
            for part in content:
                if not prefixed and isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    new_part = dict(part)
                    new_part["text"] = f"[SYSTEM]: {text}"
                    new_content.append(new_part)
                    prefixed = True
                else:
                    new_content.append(part)
            if not prefixed:
                new_content.insert(0, {"type": "text", "text": "[SYSTEM]:"})
            demoted["content"] = new_content
        else:
            demoted["content"] = "[SYSTEM]:"
        out.append(demoted)
    return out


def prepare_upstream_body(body: dict[str, Any], upstream: UpstreamConfig) -> dict[str, Any]:
    """Mutate-by-copy: swap `model` to upstream.model_id, stamp the
    server-owned chat_template_kwargs, and apply per-upstream
    request_overrides (e.g., `reasoning_effort`, `include_reasoning`).

    Server config wins over caller-provided chat_template_kwargs:
    clients should not be able to flip thinking on/off against
    configured upstream policy. Unknown caller kwargs are dropped.
    Strip the server-owned `x_meta_model` extension before forwarding.

    `request_overrides` is applied BEFORE `model` and chat_template_kwargs
    handling so those dedicated fields cannot be silently clobbered by
    a misconfigured override block.

    When `requires_leading_system_only` is set on the upstream, demote
    every non-leading system message to role="user" with a "[SYSTEM]: "
    prefix BEFORE returning. Done last so the demotion sees the body
    after request_overrides have been applied (in case an override ever
    rewrites messages — currently it cannot).
    """
    out = dict(body)
    out.pop("x_meta_model", None)
    if upstream.request_overrides:
        out.update(upstream.request_overrides)
    out["model"] = upstream.model_id

    if upstream.protocol != "openai":
        # Non-OpenAI protocols own their own wire translation (system
        # extraction, thinking, image blocks) in their provider adapter.
        # The vLLM-specific shims below (chat_template_kwargs, leading-
        # system demotion, reasoning-model param renames) don't apply.
        out.pop("chat_template_kwargs", None)
        return out

    # OpenAI reasoning-model normalization (opt-in via the [openai] block).
    oai = upstream.openai
    if oai is not None:
        if oai.reasoning_effort is not None:
            out["reasoning_effort"] = oai.reasoning_effort
        if oai.max_tokens_param == "max_completion_tokens" and "max_tokens" in out:
            # Reasoning models reject `max_tokens`; carry the caller's
            # output budget under the field the model accepts.
            out["max_completion_tokens"] = out.pop("max_tokens")
        for param in oai.drop_params:
            out.pop(param, None)

    if upstream.supports_thinking and upstream.chat_template_kwargs:
        out["chat_template_kwargs"] = dict(upstream.chat_template_kwargs)
    else:
        out.pop("chat_template_kwargs", None)
    if upstream.requires_leading_system_only:
        msgs = out.get("messages")
        if isinstance(msgs, list):
            out["messages"] = _demote_non_leading_system_messages(msgs)
    return out


async def forward_chat_completion(
    upstream: UpstreamConfig,
    body: dict[str, Any],
    *,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.Response:
    """POST the Chat Completions body to the upstream and return the
    raw response. Caller owns translating the response back to the
    client (status + body relayed verbatim in D.1.4).

    `transport` is an injection point for tests using
    `httpx.MockTransport`.

    Non-"openai" upstreams are delegated to their provider adapter,
    which speaks the backend's native wire protocol and returns a
    synthetic OpenAI-shaped `httpx.Response` so every caller here is
    protocol-agnostic.
    """
    if upstream.protocol == "anthropic":
        # Imported lazily to keep the default (OpenAI) path free of the
        # provider package and avoid any import cycle.
        from .providers.anthropic import forward_anthropic_messages

        return await forward_anthropic_messages(
            upstream, body, timeout_secs=timeout_secs, transport=transport
        )

    upstream_body = prepare_upstream_body(body, upstream)
    headers = {"Content-Type": "application/json", **_build_auth_headers(upstream)}
    url = upstream.base_url.rstrip("/") + CHAT_COMPLETIONS_PATH
    async with httpx.AsyncClient(transport=transport, timeout=timeout_secs) as client:
        return await client.post(url, json=upstream_body, headers=headers)
