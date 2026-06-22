"""F7 — bearer auth enforcement.

Coverage:
- bearer_token unset → all routes accessible without Authorization
  header (current dev contract preserved).
- bearer_token set → protected routes 401 without Authorization.
- bearer_token set → protected routes 401 with wrong token.
- bearer_token set → protected routes pass with right token.
- bearer_token set → exempt routes (health/openapi/metrics root)
  stay public.
- 401 envelope is OpenAI-typed: ``invalid_request_error`` /
  ``missing_api_key`` or ``invalid_api_key``.
- Constant-time compare path covered (parametrize wrong tokens of
  varying length so a length-leak bug would skew the path).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from meta_model.config import parse_config_str
from meta_model.server import app, set_config


_FIXTURE_NO_AUTH = """
[upstreams.a]
model_id = "ma"
base_url = "http://a.local/v1"
context = 8192
max_output = 512

[profiles."plain.v1"]
type = "moa"
generators = ["a"]
synthesizer = "a"
"""

_FIXTURE_WITH_AUTH = """
[server]
bearer_token = "sk-test-secret-token"

[upstreams.a]
model_id = "ma"
base_url = "http://a.local/v1"
context = 8192
max_output = 512

[profiles."plain.v1"]
type = "moa"
generators = ["a"]
synthesizer = "a"
"""

_FIXTURE_WHITESPACE_TOKEN = """
[server]
bearer_token = "   "

[upstreams.a]
model_id = "ma"
base_url = "http://a.local/v1"
context = 8192
max_output = 512

[profiles."plain.v1"]
type = "moa"
generators = ["a"]
synthesizer = "a"
"""


@pytest.fixture
def no_auth_config():
    cfg = parse_config_str(_FIXTURE_NO_AUTH)
    set_config(cfg)
    yield cfg
    set_config(None)


@pytest.fixture
def with_auth_config():
    cfg = parse_config_str(_FIXTURE_WITH_AUTH)
    set_config(cfg)
    yield cfg
    set_config(None)


@pytest.fixture
def whitespace_token_config():
    cfg = parse_config_str(_FIXTURE_WHITESPACE_TOKEN)
    set_config(cfg)
    yield cfg
    set_config(None)


def _client() -> TestClient:
    return TestClient(app)


# ── bearer unset → drop-in dev contract ──────────────────────────────


def test_bearer_unset_allows_protected_routes_without_auth(no_auth_config) -> None:
    """Default dev contract: no token configured, no auth required.
    /v1/models works with no Authorization header."""
    r = _client().get("/v1/models")
    assert r.status_code == 200


def test_bearer_unset_allows_models_with_garbage_auth_header(no_auth_config) -> None:
    """When auth is disabled, even malformed Authorization headers
    are ignored (they just pass through to the route)."""
    r = _client().get("/v1/models", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 200


def test_whitespace_only_token_treated_as_unset(whitespace_token_config) -> None:
    """``bearer_token = "   "`` strips to empty → auth disabled.
    Defends against operator-typo configs that leave whitespace."""
    r = _client().get("/v1/models")
    assert r.status_code == 200


# ── bearer set → 401 paths ───────────────────────────────────────────


def test_protected_route_missing_authorization_returns_401(with_auth_config) -> None:
    r = _client().get("/v1/models")
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "missing_api_key"
    assert "Authorization" in body["error"]["message"]


def test_protected_route_malformed_authorization_returns_401(with_auth_config) -> None:
    """``Authorization: Basic ...`` (not Bearer) is missing_api_key."""
    r = _client().get("/v1/models", headers={"Authorization": "Basic xyz"})
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["code"] == "missing_api_key"


@pytest.mark.parametrize(
    "wrong_token",
    [
        "sk-different-token",  # same length-ish, different content
        "x",  # very short
        "sk-test-secret-token-but-longer",  # longer than expected
        "",  # empty after Bearer
    ],
)
def test_protected_route_wrong_token_returns_401(with_auth_config, wrong_token) -> None:
    """Wrong token of varying length all return ``invalid_api_key``.
    Constant-time compare via ``hmac.compare_digest`` means the
    branch is taken regardless of length — this just exercises the
    common reject path."""
    r = _client().get(
        "/v1/models", headers={"Authorization": f"Bearer {wrong_token}"}
    )
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "invalid_api_key"


def test_protected_route_right_token_returns_200(with_auth_config) -> None:
    r = _client().get(
        "/v1/models",
        headers={"Authorization": "Bearer sk-test-secret-token"},
    )
    assert r.status_code == 200


# ── per-route coverage (every protected route enforces) ──────────────


@pytest.mark.parametrize(
    "method,path,extra",
    [
        ("get", "/v1/models", {}),
        ("get", "/v1/metrics/moa", {}),
        ("post", "/v1/chat/completions", {"json": {"model": "plain.v1", "messages": []}}),
        ("post", "/v1/completions", {"json": {"model": "plain.v1", "prompt": "x"}}),
        ("post", "/v1/responses", {"json": {"model": "plain.v1", "input": "x"}}),
        ("post", "/tokenize", {"json": {"model": "plain.v1", "prompt": "x"}}),
    ],
)
def test_every_protected_route_requires_auth(with_auth_config, method, path, extra) -> None:
    """Per-route enforcement: each protected route returns 401
    without Authorization, regardless of body validity. Body content
    doesn't matter — auth runs before body validation."""
    client = _client()
    fn = getattr(client, method)
    r = fn(path, **extra)
    assert r.status_code == 401, f"{method.upper()} {path} should require auth"
    body = r.json()
    assert body["error"]["code"] == "missing_api_key"


# ── exempt routes stay public ────────────────────────────────────────


def test_health_v1_is_exempt(with_auth_config) -> None:
    r = _client().get("/v1/health")
    # Health may 503 (no monitor) or 200 — both signal the route ran.
    # The exempt contract is simply: NOT 401.
    assert r.status_code != 401


def test_health_root_is_exempt(with_auth_config) -> None:
    r = _client().get("/health")
    assert r.status_code != 401


def test_metrics_root_alias_is_exempt(with_auth_config) -> None:
    """Root-level ``/metrics`` is the prometheus-style scrape path.
    Stays public so scrapers don't need to bear the API token.
    Versioned ``/v1/metrics/moa`` is protected (covered above)."""
    r = _client().get("/metrics")
    assert r.status_code == 200


def test_openapi_json_is_exempt(with_auth_config) -> None:
    """``/openapi.json`` is for tooling introspection — stays public
    even when bearer is configured. Matches OpenAI which exposes
    its own OpenAPI spec without auth."""
    r = _client().get("/openapi.json")
    assert r.status_code == 200
    body = r.json()
    assert body["openapi"].startswith("3.")


# ── envelope shape ───────────────────────────────────────────────────


def test_invalid_body_with_wrong_token_returns_401_not_400(with_auth_config) -> None:
    """Review r1 F7 HIGH regression. Auth must run BEFORE FastAPI body
    parsing — otherwise a request with a malformed JSON body and a
    wrong/missing token would 400 (invalid_json) instead of 401, letting
    unauthenticated callers probe routes by inspecting which envelope
    they get back. Middleware closes that timing window.

    Sends a malformed JSON body to ``/v1/chat/completions`` with a
    wrong bearer; expects 401 invalid_api_key, not 400.
    """
    r = _client().post(
        "/v1/chat/completions",
        content=b"{not-json",
        headers={
            "Authorization": "Bearer sk-wrong-token",
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 401, (
        f"expected 401 (auth before body parse); got {r.status_code} "
        f"body={r.text[:200]!r}"
    )
    body = r.json()
    assert body["error"]["code"] == "invalid_api_key"


def test_non_ascii_bearer_token_returns_401_not_500(with_auth_config) -> None:
    """Review r2 F7 HIGH regression. ``hmac.compare_digest(str, str)``
    raises TypeError on non-ASCII codepoints, and Starlette decodes
    Authorization header bytes as latin-1 — so a raw byte > 0x7f in
    the header lands as a non-ASCII str. Without bytes-side compare,
    such a request would crash the middleware to 500 instead of
    returning 401 invalid_api_key. The middleware encodes both sides
    to UTF-8 with ``errors="replace"`` so the compare always runs on
    bytes.

    httpx TestClient refuses to send non-ASCII header values (it
    pre-encodes as ASCII), so we exercise the middleware directly
    with a stub Request — the realistic path on the wire would be
    Starlette decoding raw header bytes as latin-1, mirrored by
    ``"Bearer ÿ"`` here.
    """
    import asyncio
    from types import SimpleNamespace

    from meta_model.auth import bearer_auth_middleware

    request = SimpleNamespace(
        headers={"authorization": "Bearer ÿ"},  # latin-1 ÿ → 0xff
        url=SimpleNamespace(path="/v1/models"),
    )

    async def _should_not_be_called(_req):  # pragma: no cover
        raise AssertionError("call_next must not run on bad auth")

    response = asyncio.run(
        bearer_auth_middleware(request, _should_not_be_called)
    )
    assert response.status_code == 401
    # JSONResponse stores the rendered bytes; decode and parse.
    import json as _json
    body = _json.loads(response.body)
    assert body["error"]["code"] == "invalid_api_key"


def test_invalid_body_no_token_returns_401_not_400(with_auth_config) -> None:
    """Same regression but for the no-token case: middleware must short
    -circuit on missing Authorization header even when the body is
    malformed."""
    r = _client().post(
        "/v1/chat/completions",
        content=b"{not-json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 401
    body = r.json()
    assert body["error"]["code"] == "missing_api_key"


def test_401_envelope_shape_full(with_auth_config) -> None:
    """OpenAI-shaped error body: ``{"error": {message, type, param, code}}``.
    All four keys present; ``param`` is null for auth errors."""
    r = _client().get("/v1/models")
    assert r.status_code == 401
    body = r.json()
    err = body["error"]
    assert set(err.keys()) >= {"message", "type", "param", "code"}
    assert err["param"] is None
    assert err["type"] == "invalid_request_error"
    assert isinstance(err["message"], str) and err["message"]
