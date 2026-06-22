"""D.1.2 tests — FastAPI app shell."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from meta_model import __version__
from meta_model.server import app


# Module-level Pydantic models for the validation-envelope test.
# Defined at module scope so FastAPI's introspection sees stable
# class objects (function-local Pydantic classes can confuse the
# body/query inference path in some FastAPI versions).
class _Item(BaseModel):
    role: str


class _ValidationBody(BaseModel):
    messages: list[_Item]


def _client() -> TestClient:
    return TestClient(app)


# ── /v1/health ──────────────────────────────────────────────────────


def test_health_returns_503_when_no_monitor_configured() -> None:
    """F5: with no HealthMonitor on app.state (TestClient without
    `with` skips lifespan, no monitor installed), readiness must
    report unhealthy. Reporting "ok" would lie — the server can't
    actually serve chat completions in this state."""
    from meta_model.server import app

    if hasattr(app.state, "upstream_health"):
        app.state.upstream_health = None
    r = _client().get("/v1/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unhealthy"
    assert body["reason"] == "no_monitor_configured"
    assert body["version"] == __version__


# ── /v1/models ──────────────────────────────────────────────────────


def test_models_empty_when_no_config_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure no config bleed from previous tests (config is cached on app.state).
    from meta_model.server import set_config

    set_config(None)
    monkeypatch.setenv("META_MODEL_CONFIG", "/no/such/path.toml")
    r = _client().get("/v1/models")
    assert r.status_code == 200
    assert r.json() == {"object": "list", "data": []}
    set_config(None)


def test_models_lists_loaded_profiles() -> None:
    from meta_model.config import parse_config_str
    from meta_model.server import set_config

    cfg = parse_config_str(
        """
[upstreams.a]
model_id = "a"
base_url = "http://a"
context = 8192
max_output = 512

[upstreams.b]
model_id = "b"
base_url = "http://b"
context = 8192
max_output = 512

[profiles."some.profile.v1"]
type = "moa"
generators = ["a", "b"]
synthesizer = "a"

[profiles."another.cascade.v1"]
type = "cascade"
upstreams = ["a", "b"]
"""
    )
    set_config(cfg)
    try:
        r = _client().get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        ids = [m["id"] for m in body["data"]]
        # Sorted alphabetically
        assert ids == ["another.cascade.v1", "some.profile.v1"]
        for entry in body["data"]:
            assert entry["object"] == "model"
            assert entry["owned_by"] == "meta-model"
            assert isinstance(entry["created"], int)
    finally:
        set_config(None)


def test_models_hides_voting_profiles_when_feature_disabled() -> None:
    from meta_model.config import parse_config_str
    from meta_model.server import set_config

    cfg = parse_config_str(
        """
[features]
voting = false

[upstreams.a]
model_id = "a"
base_url = "http://a"
context = 8192
max_output = 512

[upstreams.b]
model_id = "b"
base_url = "http://b"
context = 8192
max_output = 512

[profiles."text.v1"]
type = "moa"
generators = ["a", "b"]
synthesizer = "a"

[profiles."vote.v1"]
type = "voting"
upstreams = ["a", "b"]
"""
    )
    set_config(cfg)
    try:
        ids = [m["id"] for m in _client().get("/v1/models").json()["data"]]
        assert ids == ["text.v1"]  # vote.v1 hidden
    finally:
        set_config(None)


# ── Error envelope ──────────────────────────────────────────────────


def test_404_returns_openai_error_envelope() -> None:
    r = _client().get("/v1/does-not-exist")
    assert r.status_code == 404
    body = r.json()
    assert "error" in body
    err = body["error"]
    assert err["type"] == "not_found_error"
    assert isinstance(err["message"], str) and err["message"]
    # Stable shape — both fields always present, possibly null.
    assert "param" in err
    assert "code" in err


def test_405_returns_openai_error_envelope() -> None:
    # POST to a GET-only endpoint → 405 method not allowed
    r = _client().post("/v1/health")
    assert r.status_code == 405
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"


def test_error_envelope_shape_for_500_path() -> None:
    """Sanity-check the type mapping for a hypothetical 500.

    No live 500 path exists yet, so this exercises the helper directly
    rather than going through FastAPI. Keeps the type-mapping
    contract pinned even before upstream calls land.
    """
    from meta_model.server import error_envelope

    body = error_envelope("oops", status=500)
    assert body["error"]["type"] == "api_error"
    assert body["error"]["message"] == "oops"
    assert body["error"]["param"] is None
    assert body["error"]["code"] is None


def test_error_envelope_type_overrides_status_default() -> None:
    from meta_model.server import error_envelope

    body = error_envelope("nope", status=400, type_="authentication_error")
    assert body["error"]["type"] == "authentication_error"


def test_503_maps_to_service_unavailable_error() -> None:
    from meta_model.server import error_envelope

    body = error_envelope("upstream down", status=503)
    assert body["error"]["type"] == "service_unavailable_error"


def test_500_maps_to_api_error() -> None:
    from meta_model.server import error_envelope

    body = error_envelope("kaboom", status=500)
    assert body["error"]["type"] == "api_error"
    body502 = error_envelope("kaboom", status=502)
    assert body502["error"]["type"] == "api_error"


def test_loc_flattening_uses_bracketed_indices() -> None:
    from meta_model.server import _format_loc_path

    # ("body", "messages", 0, "role") → "messages[0].role"
    assert _format_loc_path(("body", "messages", 0, "role")) == "messages[0].role"
    assert _format_loc_path(("body", "tools", 1, "function", "name")) == "tools[1].function.name"
    assert _format_loc_path(("body", "model")) == "model"
    assert _format_loc_path(("body",)) is None
    assert _format_loc_path(()) is None


# ── Live unhandled-exception envelope ───────────────────────────────


def test_unhandled_exception_returns_500_envelope() -> None:
    """A real unhandled exception in a route flows through the
    catch-all handler, not the bare ASGI 500. We mount a separate
    test app with the same handlers and a route that raises.
    """
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from meta_model.server import (
        http_exception_handler,
        unhandled_exception_handler,
        validation_exception_handler,
    )

    test_app = FastAPI()
    test_app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    test_app.add_exception_handler(RequestValidationError, validation_exception_handler)
    test_app.add_exception_handler(Exception, unhandled_exception_handler)

    @test_app.get("/_boom")
    async def _boom() -> None:
        raise RuntimeError("test boom — should never reach the client")

    client = TestClient(test_app, raise_server_exceptions=False)
    r = client.get("/_boom")
    assert r.status_code == 500
    body = r.json()
    assert body["error"]["type"] == "api_error"
    assert body["error"]["message"] == "internal server error"
    assert "test boom" not in body["error"]["message"]  # stack detail stays out


# ── Live validation envelope ────────────────────────────────────────


def test_validation_error_returns_400_with_bracketed_param() -> None:
    """Mount a body-validating route and trigger a Pydantic error.
    Confirms validation handler returns 400 (not 422) with bracketed
    index in `param`.
    """
    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from meta_model.server import http_exception_handler, validation_exception_handler

    test_app = FastAPI()
    test_app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    test_app.add_exception_handler(RequestValidationError, validation_exception_handler)

    @test_app.post("/_v")
    async def _v(payload: _ValidationBody) -> dict[str, int]:
        return {"ok": 1}

    client = TestClient(test_app)
    # Missing required field "role" on messages[0]
    r = client.post("/_v", json={"messages": [{}]})
    assert r.status_code == 400
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "messages[0].role"


# ── F3: thinking + reasoning_visible advertising ───────────────────


def test_models_advertises_thinking_false_by_default() -> None:
    """A profile with a thinking-capable upstream but
    expose_reasoning=false (the default) advertises thinking=false.
    The thinking signal is gated on BOTH upstream support AND
    the profile opting to surface reasoning."""
    from meta_model.config import parse_config_str
    from meta_model.server import set_config

    cfg = parse_config_str(
        """
[upstreams.thinker]
model_id = "thinker"
base_url = "http://t"
context = 8192
max_output = 512
supports_thinking = true

[profiles."plain.v1"]
type = "cascade"
upstreams = ["thinker"]
"""
    )
    set_config(cfg)
    try:
        body = _client().get("/v1/models").json()
        entry = next(m for m in body["data"] if m["id"] == "plain.v1")
        caps = entry["capabilities"]
        assert caps["thinking"] is False
        assert caps["reasoning_visible"] is False
    finally:
        set_config(None)


def test_models_advertises_thinking_true_when_exposed_and_supported() -> None:
    """expose_reasoning=true + a thinking-capable upstream → thinking=true."""
    from meta_model.config import parse_config_str
    from meta_model.server import set_config

    cfg = parse_config_str(
        """
[upstreams.thinker]
model_id = "thinker"
base_url = "http://t"
context = 8192
max_output = 512
supports_thinking = true

[profiles."exposed.v1"]
type = "cascade"
upstreams = ["thinker"]
expose_reasoning = true
"""
    )
    set_config(cfg)
    try:
        body = _client().get("/v1/models").json()
        entry = next(m for m in body["data"] if m["id"] == "exposed.v1")
        caps = entry["capabilities"]
        assert caps["thinking"] is True
        assert caps["reasoning_visible"] is True
    finally:
        set_config(None)


def test_models_thinking_false_when_exposed_but_no_thinking_upstream() -> None:
    """expose_reasoning=true but no thinking-capable upstream →
    thinking=false (still honest); reasoning_visible mirrors
    the policy regardless."""
    from meta_model.config import parse_config_str
    from meta_model.server import set_config

    cfg = parse_config_str(
        """
[upstreams.plain]
model_id = "plain"
base_url = "http://p"
context = 8192
max_output = 512

[profiles."exposed_no_thinker.v1"]
type = "cascade"
upstreams = ["plain"]
expose_reasoning = true
"""
    )
    set_config(cfg)
    try:
        body = _client().get("/v1/models").json()
        entry = next(
            m for m in body["data"] if m["id"] == "exposed_no_thinker.v1"
        )
        caps = entry["capabilities"]
        assert caps["thinking"] is False
        assert caps["reasoning_visible"] is True
    finally:
        set_config(None)


def test_models_caps_block_includes_all_eight_keys() -> None:
    """Regression: clients depend on the capabilities shape; make
    sure F2 (effective_image_capability + supports_image_tools)
    and F3 (thinking + reasoning_visible) are always present."""
    from meta_model.config import parse_config_str
    from meta_model.server import set_config

    cfg = parse_config_str(
        """
[upstreams.a]
model_id = "a"
base_url = "http://a"
context = 8192
max_output = 512

[profiles."p.v1"]
type = "cascade"
upstreams = ["a"]
"""
    )
    set_config(cfg)
    try:
        body = _client().get("/v1/models").json()
        caps = body["data"][0]["capabilities"]
        assert set(caps.keys()) == {
            "vision",
            "video",
            "audio",
            "function_calling",
            "effective_image_capability",
            "supports_image_tools",
            "thinking",
            "reasoning_visible",
        }
    finally:
        set_config(None)
