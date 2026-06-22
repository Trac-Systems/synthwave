"""D.1.4 tests — /v1/chat/completions single-upstream passthrough."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from meta_model.config import parse_config_str
from meta_model.server import app, set_config, set_upstream_transport

_BASE_TOML = """
[upstreams.text_a]
model_id = "text-a-actual"
base_url = "http://upstream:9000/v1"
context = 8192
max_output = 2048

[upstreams.with_key]
model_id = "secret-model"
base_url = "http://upstream:9001/v1"
context = 8192
max_output = 2048
api_key = "sk-test-1234"

[upstreams.with_basic]
model_id = "basic-model"
base_url = "http://upstream:9002/v1"
context = 8192
max_output = 2048
basic_auth_user = "alice"
basic_auth_pass = "wonderland"

[upstreams.with_thinking]
model_id = "thinker"
base_url = "http://upstream:9003/v1"
context = 8192
max_output = 2048
supports_thinking = true
chat_template_kwargs = { enable_thinking = false }

[upstreams.text_b]
model_id = "text-b-actual"
base_url = "http://upstream:9004/v1"
context = 8192
max_output = 2048

[upstreams.with_overrides]
model_id = "override-model"
base_url = "http://upstream:9005/v1"
context = 8192
max_output = 2048
request_overrides = { reasoning_effort = "low", include_reasoning = false }

# Defensive case: an override block that tries to clobber model and
# chat_template_kwargs MUST NOT win over their dedicated handling.
[upstreams.with_dangerous_overrides]
model_id = "real-model-id"
base_url = "http://upstream:9006/v1"
context = 8192
max_output = 2048
supports_thinking = true
chat_template_kwargs = { enable_thinking = false }
request_overrides = { model = "spoofed", chat_template_kwargs = { enable_thinking = true }, reasoning_effort = "high" }

# Single-upstream profiles for the request_overrides tests.
[profiles."with_overrides.v1"]
type = "moa"
generators = ["with_overrides"]
synthesizer = "with_overrides"

[profiles."with_dangerous_overrides.v1"]
type = "moa"
generators = ["with_dangerous_overrides"]
synthesizer = "with_dangerous_overrides"

# Degenerate single-upstream MoA profile — D.1.4 should accept this.
[profiles."single.v1"]
type = "moa"
generators = ["text_a"]
synthesizer = "text_a"

# Real multi-generator MoA — D.1.4 should reject (D.2.4 will dispatch).
[profiles."multi.v1"]
type = "moa"
generators = ["text_a", "text_b"]
synthesizer = "text_a"

# Single-cascade — degenerate, accept.
[profiles."single_cascade.v1"]
type = "cascade"
upstreams = ["text_a"]
"""


@pytest.fixture
def loaded_config() -> Any:
    cfg = parse_config_str(_BASE_TOML)
    set_config(cfg)
    yield cfg
    set_config(None)
    set_upstream_transport(None)


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def _client() -> TestClient:
    return TestClient(app)


# ── Happy path ─────────────────────────────────────────────────────


def test_chat_completion_passthrough_success(loaded_config) -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-xyz",
                "object": "chat.completion",
                "created": 1730000000,
                "model": "text-a-actual",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

    set_upstream_transport(_mock_transport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hi"
    # The upstream saw model_id, not the upstream-key name
    assert captured["body"]["model"] == "text-a-actual"
    assert captured["url"] == "http://upstream:9000/v1/chat/completions"
    # Meta-model headers present
    assert r.headers["x-metamodel-request-id"].startswith("metamodel-")
    assert r.headers["x-metamodel-profile"] == "text_a"


def test_chat_completion_forwards_authorization_bearer(loaded_config) -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"id": "x", "choices": []})

    set_upstream_transport(_mock_transport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "with_key",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    assert captured["auth"] == "Bearer sk-test-1234"


# ── Validation errors ─────────────────────────────────────────────


def test_chat_completion_missing_messages_returns_400(loaded_config) -> None:
    r = _client().post("/v1/chat/completions", json={"model": "text_a"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_chat_completion_unknown_model_returns_404(loaded_config) -> None:
    """OpenAI convention: missing model is 404, not 400 (review r10)."""
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "ghost",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"
    assert body["error"]["type"] == "not_found_error"


def test_chat_completion_degenerate_moa_profile_passthrough(loaded_config) -> None:
    """Profile with single generator+synthesizer routes to that one upstream."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"id": "ok", "choices": []})

    set_upstream_transport(_mock_transport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "single.v1",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 200
    assert captured["url"] == "http://upstream:9000/v1/chat/completions"


def test_chat_completion_degenerate_cascade_profile_passthrough(loaded_config) -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "id": "ok",
                "object": "chat.completion",
                "created": 1,
                "model": "x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    set_upstream_transport(_mock_transport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "single_cascade.v1",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 200
    assert captured["url"] == "http://upstream:9000/v1/chat/completions"


def test_chat_completion_multi_voice_profile_dispatches(loaded_config) -> None:
    """D.2.4: real multi-generator MoA fans out + synthesizes.

    Two generators return distinct content; the synthesizer is asked
    to merge. With both generator and synthesizer endpoints reachable,
    the response carries the merged content + per-profile X-MetaModel-*
    headers reflecting the actual fan-out (Generators=2, Quorum=2)."""

    call_log: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        call_log.append(request.url.host)
        # All upstreams in this test config share the same host
        # `upstream`; differentiate by port if needed. For dispatch
        # smoke testing, returning a parseable ChatCompletion is enough.
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-x",
                "object": "chat.completion",
                "created": 1,
                "model": "x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "synthesized answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    set_upstream_transport(_mock_transport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "multi.v1",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "synthesized answer"
    # Headers reflect actual MoA dispatch (2 generators, 2 succeeded).
    assert r.headers.get("x-metamodel-profile") == "multi.v1"
    assert r.headers.get("x-metamodel-generators") == "2"
    assert r.headers.get("x-metamodel-quorum") == "2"


def test_chat_completion_streaming_returns_sse(loaded_config) -> None:
    """D.3.3: ``stream:true`` opens an SSE response.

    Review r36 #3: install a mock upstream so the test exercises the
    full streaming path (not just headers — assert real SSE content
    + ``[DONE]`` so a regression that returns an empty error event
    doesn't pass silently).
    """
    upstream_body = {
        "id": "upstream-1",
        "object": "chat.completion",
        "created": 1,
        "model": "text-a-actual",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    set_upstream_transport(_mock_transport(lambda req: httpx.Response(200, json=upstream_body)))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        },
    )
    assert r.status_code == 200
    assert r.headers.get("content-type", "").startswith("text/event-stream")
    assert b"data: [DONE]\n\n" in r.content
    assert b'"role":"assistant"' in r.content


def test_chat_completion_n_greater_than_one_rejected(loaded_config) -> None:
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "n": 2,
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_v1"


def test_chat_completion_conflicting_token_params_rejected(loaded_config) -> None:
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 100,
            "max_completion_tokens": 100,
        },
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "conflicting_params"


# ── Upstream failure modes ────────────────────────────────────────


def test_chat_completion_upstream_5xx_relayed_with_metamodel_headers(
    loaded_config,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"message": "model loading"}})

    set_upstream_transport(_mock_transport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    # Body relayed verbatim — server is a passthrough at this layer
    assert r.status_code == 503
    assert r.json() == {"error": {"message": "model loading"}}
    assert "x-metamodel-request-id" in r.headers


def test_chat_completion_upstream_timeout_returns_504(loaded_config) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("upstream timed out", request=request)

    set_upstream_transport(_mock_transport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 504
    assert r.json()["error"]["code"] == "upstream_timeout"


def test_chat_completion_upstream_connect_error_returns_502(loaded_config) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("cannot connect", request=request)

    set_upstream_transport(_mock_transport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "upstream_error"


def test_chat_completion_upstream_non_json_returns_502(loaded_config) -> None:
    """Review r10: a 200-with-error-body misleads clients. Non-JSON
    upstream response is a protocol failure regardless of upstream
    status — return 502."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json at all")

    set_upstream_transport(_mock_transport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 502
    assert r.json()["error"]["code"] == "upstream_protocol_error"
    assert r.json()["error"]["type"] == "api_error"


# ── Auth + body-mutation tests ─────────────────────────────────────


def test_chat_completion_forwards_basic_auth(loaded_config) -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"id": "x", "choices": []})

    set_upstream_transport(_mock_transport(handler))
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "with_basic",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200
    # Basic YWxpY2U6d29uZGVybGFuZA== = base64("alice:wonderland")
    assert captured["auth"] == "Basic YWxpY2U6d29uZGVybGFuZA=="


def test_chat_completion_does_not_leak_pydantic_defaults(loaded_config) -> None:
    """Review r10: the forwarded body must not contain fields the
    client never set (e.g., parallel_tool_calls=True default)."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "x", "choices": []})

    set_upstream_transport(_mock_transport(handler))
    _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    # Caller did not send parallel_tool_calls; it must NOT appear.
    assert "parallel_tool_calls" not in captured["body"]
    assert "stream" not in captured["body"]
    # Caller did set messages and model — those must be present.
    assert captured["body"]["messages"][0]["role"] == "user"


def test_chat_completion_strips_x_meta_model(loaded_config) -> None:
    """x_meta_model is server-owned; never forward to upstream
    (review r10). The trace flag passes through dispatch but doesn't
    leak to the upstream body."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "x", "choices": []})

    set_upstream_transport(_mock_transport(handler))
    _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
            "x_meta_model": {"trace": True},
        },
    )
    assert "x_meta_model" not in captured["body"]


def test_chat_completion_server_thinking_policy_overrides_caller(loaded_config) -> None:
    """Caller's chat_template_kwargs must not override server config
    (review r10). For the with_thinking upstream, server forces
    enable_thinking=false even if the client requests true."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "x", "choices": []})

    set_upstream_transport(_mock_transport(handler))
    _client().post(
        "/v1/chat/completions",
        json={
            "model": "with_thinking",
            "messages": [{"role": "user", "content": "x"}],
            "chat_template_kwargs": {"enable_thinking": True, "rogue": "value"},
        },
    )
    # Server config wins — enable_thinking false; rogue caller key dropped.
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_completion_request_overrides_applied(loaded_config) -> None:
    """`request_overrides` flows into every forwarded request body."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "x", "choices": []})

    set_upstream_transport(_mock_transport(handler))
    _client().post(
        "/v1/chat/completions",
        json={
            "model": "with_overrides.v1",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert captured["body"]["reasoning_effort"] == "low"
    assert captured["body"]["include_reasoning"] is False


def test_chat_completion_request_overrides_cannot_clobber_model_or_thinking(
    loaded_config,
) -> None:
    """Even a misconfigured override block cannot bypass the dedicated
    `model` swap or the server-owned `chat_template_kwargs` policy."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "x", "choices": []})

    set_upstream_transport(_mock_transport(handler))
    _client().post(
        "/v1/chat/completions",
        json={
            "model": "with_dangerous_overrides.v1",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    # Dedicated handling wins.
    assert captured["body"]["model"] == "real-model-id"
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    # Other override keys still flow through.
    assert captured["body"]["reasoning_effort"] == "high"


# ── No config / no upstreams ──────────────────────────────────────


def test_chat_completion_no_config_returns_503() -> None:
    set_config(None)
    r = _client().post(
        "/v1/chat/completions",
        json={
            "model": "text_a",
            "messages": [{"role": "user", "content": "x"}],
        },
    )
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "no_upstream"
