"""Unit tests for the Anthropic Messages provider adapter.

Covers the request translation (OpenAI Chat-Completions → Anthropic
`/v1/messages`), the response translation back to OpenAI shape, and the
transport (headers, URL, success / error / non-JSON relay) via
`httpx.MockTransport`. The adapter is generic to any Anthropic model;
these tests pin the wire contract, not a specific model.
"""

from __future__ import annotations

import json

import httpx
import pytest

from meta_model.config import AnthropicOptions, UpstreamConfig
from meta_model.providers.anthropic import (
    build_messages_request,
    forward_anthropic_messages,
    translate_response,
)


def _up(**kw) -> UpstreamConfig:
    base = dict(
        model_id="claude-opus-4-8",
        base_url="https://api.anthropic.com/v1",
        context=200000,
        max_output=8192,
        modalities=["text", "image"],
        protocol="anthropic",
        api_key="sk-ant-test",
    )
    base.update(kw)
    return UpstreamConfig(**base)


# ── Request translation ─────────────────────────────────────────────


def test_system_messages_lift_to_top_level() -> None:
    body = {
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "be terse"},
        ],
        "max_tokens": 100,
    }
    out = build_messages_request(body, _up())
    assert out["system"] == "you are helpful\n\nbe terse"
    # System messages are gone from the message list; only the user turn.
    assert [m["role"] for m in out["messages"]] == ["user"]
    assert out["messages"][0]["content"] == [{"type": "text", "text": "hi"}]


def test_consecutive_same_role_turns_merge() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ],
        "max_tokens": 10,
    }
    out = build_messages_request(body, _up())
    assert len(out["messages"]) == 1
    assert out["messages"][0]["content"] == [
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
    ]


def test_assistant_tool_calls_become_tool_use_blocks() -> None:
    body = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "checking",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "72F"},
        ],
        "max_tokens": 50,
    }
    out = build_messages_request(body, _up())
    assistant = out["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "text", "text": "checking"}
    tool_use = assistant["content"][1]
    assert tool_use == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {"city": "NYC"},
    }
    # Tool result lands in a following user turn as a tool_result block.
    tool_turn = out["messages"][2]
    assert tool_turn["role"] == "user"
    assert tool_turn["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": "72F",
    }


def test_image_data_uri_and_remote_url() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD"},
                    },
                    {"type": "image_url", "image_url": {"url": "https://x/y.jpg"}},
                ],
            }
        ],
        "max_tokens": 50,
    }
    out = build_messages_request(body, _up())
    blocks = out["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "what is this"}
    assert blocks[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
    }
    assert blocks[2] == {
        "type": "image",
        "source": {"type": "url", "url": "https://x/y.jpg"},
    }


def test_tools_and_tool_choice_translate() -> None:
    body = {
        "messages": [{"role": "user", "content": "go"}],
        "max_tokens": 50,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "search the web",
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                },
            }
        ],
        "tool_choice": "required",
    }
    out = build_messages_request(body, _up())
    assert out["tools"][0] == {
        "name": "search",
        "description": "search the web",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
    assert out["tool_choice"] == {"type": "any"}


def test_tool_choice_none_drops_tools() -> None:
    body = {
        "messages": [{"role": "user", "content": "go"}],
        "max_tokens": 50,
        "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
        "tool_choice": "none",
    }
    out = build_messages_request(body, _up())
    assert "tools" not in out
    assert "tool_choice" not in out


def test_specific_tool_choice() -> None:
    body = {
        "messages": [{"role": "user", "content": "go"}],
        "max_tokens": 50,
        "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
        "tool_choice": {"type": "function", "function": {"name": "x"}},
    }
    out = build_messages_request(body, _up())
    assert out["tool_choice"] == {"type": "tool", "name": "x"}


def test_adaptive_thinking_with_effort_and_drops_temperature() -> None:
    body = {
        "messages": [{"role": "user", "content": "hard"}],
        "max_tokens": 4000,
        "temperature": 0.6,
        "top_p": 0.9,
    }
    up = _up(anthropic=AnthropicOptions(thinking="adaptive", effort="xhigh"))
    out = build_messages_request(body, up)
    assert out["thinking"] == {"type": "adaptive"}
    assert out["output_config"] == {"effort": "xhigh"}
    # Custom sampling params are dropped while thinking is engaged.
    assert "temperature" not in out
    assert "top_p" not in out


def test_enabled_thinking_bumps_max_tokens_above_budget() -> None:
    body = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 500}
    up = _up(
        max_output=20000,
        anthropic=AnthropicOptions(thinking="enabled", budget_tokens=2000),
    )
    out = build_messages_request(body, up)
    assert out["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert out["max_tokens"] > 2000


def test_temperature_forwarded_when_thinking_off() -> None:
    body = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 100, "temperature": 0.3}
    out = build_messages_request(body, _up())  # no thinking configured
    assert out["temperature"] == 0.3


def test_max_tokens_required_and_clamped_to_ceiling() -> None:
    # No caller max_tokens → fall back to upstream.max_output.
    out = build_messages_request({"messages": [{"role": "user", "content": "x"}]}, _up())
    assert out["max_tokens"] == 8192
    # Caller asks for more than the ceiling → clamped.
    out2 = build_messages_request(
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 999999}, _up()
    )
    assert out2["max_tokens"] == 8192


def test_stop_sequences_translation() -> None:
    out = build_messages_request(
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 10, "stop": "STOP"},
        _up(),
    )
    assert out["stop_sequences"] == ["STOP"]
    out2 = build_messages_request(
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 10, "stop": ["a", "b"]},
        _up(),
    )
    assert out2["stop_sequences"] == ["a", "b"]


# ── Response translation ────────────────────────────────────────────


def test_response_text_and_usage() -> None:
    a = {
        "id": "msg_1",
        "model": "claude-opus-4-8",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }
    out = translate_response(a, "claude-opus-4-8")
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}


def test_response_thinking_to_reasoning_content() -> None:
    a = {
        "id": "m",
        "content": [
            {"type": "thinking", "thinking": "let me think"},
            {"type": "text", "text": "answer"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    out = translate_response(a, "m")
    msg = out["choices"][0]["message"]
    assert msg["content"] == "answer"
    assert msg["reasoning_content"] == "let me think"


def test_response_tool_use_to_tool_calls() -> None:
    a = {
        "id": "m",
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "x"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }
    out = translate_response(a, "m")
    msg = out["choices"][0]["message"]
    assert msg["content"] is None
    assert msg["tool_calls"][0]["id"] == "toolu_1"
    assert msg["tool_calls"][0]["function"]["name"] == "search"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"q": "x"}
    assert out["choices"][0]["finish_reason"] == "tool_calls"


def test_response_max_tokens_maps_to_length() -> None:
    a = {"id": "m", "content": [{"type": "text", "text": "x"}], "stop_reason": "max_tokens"}
    out = translate_response(a, "m")
    assert out["choices"][0]["finish_reason"] == "length"


def test_response_thinking_tokens_map_to_reasoning_tokens() -> None:
    """Opus-4.x adaptive thinking returns an empty thinking block + a
    token count; surface the count as OpenAI reasoning-usage."""
    a = {
        "id": "m",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "abc"},
            {"type": "text", "text": "answer"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 62, "output_tokens": 1268,
                  "output_tokens_details": {"thinking_tokens": 740}},
    }
    out = translate_response(a, "m")
    assert out["usage"]["completion_tokens"] == 1268
    assert out["usage"]["completion_tokens_details"] == {"reasoning_tokens": 740}
    # Empty thinking text → no reasoning_content key (nothing to surface).
    assert "reasoning_content" not in out["choices"][0]["message"]


def test_response_no_reasoning_usage_when_absent() -> None:
    a = {"id": "m", "content": [{"type": "text", "text": "x"}], "stop_reason": "end_turn",
         "usage": {"input_tokens": 5, "output_tokens": 2}}
    out = translate_response(a, "m")
    assert "completion_tokens_details" not in out["usage"]


# ── Transport ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forward_success_path() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_x",
                "model": "claude-opus-4-8",
                "content": [{"type": "text", "text": "hi there"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 4, "output_tokens": 2},
            },
        )

    up = _up(anthropic=AnthropicOptions(thinking="adaptive", effort="high", version="2099-01-01"))
    resp = await forward_anthropic_messages(
        up,
        {"model": "client_name", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 50},
        timeout_secs=5,
        transport=httpx.MockTransport(handler),
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["choices"][0]["message"]["content"] == "hi there"

    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    headers = captured["headers"]
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2099-01-01"
    sent = captured["body"]
    assert sent["model"] == "claude-opus-4-8"  # rewritten from client_name
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_forward_relays_non_2xx_verbatim() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"type": "error", "error": {"message": "bad"}})

    resp = await forward_anthropic_messages(
        _up(),
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 10},
        timeout_secs=5,
        transport=httpx.MockTransport(handler),
    )
    assert resp.status_code == 400
    assert "bad" in resp.text


@pytest.mark.asyncio
async def test_health_probe_uses_x_api_key_for_anthropic() -> None:
    """The readiness probe must speak Anthropic's auth scheme (x-api-key +
    anthropic-version), not OpenAI Bearer — otherwise every Anthropic
    upstream false-fails the probe with 401."""
    from meta_model.health import probe_upstream

    captured: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured["url"] = str(req.url)
        captured["headers"] = dict(req.headers)
        return httpx.Response(200, json={"data": [{"id": "claude-opus-4-8"}]})

    h = await probe_upstream(_up(), timeout_secs=2.0, transport=httpx.MockTransport(handler))
    assert h.state == "up"
    assert str(captured["url"]).endswith("/v1/models")
    headers = captured["headers"]
    assert headers["x-api-key"] == "sk-ant-test"
    assert "anthropic-version" in headers
    assert "authorization" not in headers  # must NOT send Bearer


@pytest.mark.asyncio
async def test_forward_relays_non_json_verbatim() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    resp = await forward_anthropic_messages(
        _up(),
        {"messages": [{"role": "user", "content": "x"}], "max_tokens": 10},
        timeout_secs=5,
        transport=httpx.MockTransport(handler),
    )
    assert resp.status_code == 200
    assert resp.text == "not json"
