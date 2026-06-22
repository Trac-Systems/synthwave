"""F11 — CORS middleware tests.

Coverage:
- Default config: cors_allow_origins = ["*"]; preflight 204 + Allow-Origin: *.
- Locked-down origin list: matched origin echoes back; unmatched origin
  → no CORS headers (browser blocks).
- Preflight (OPTIONS) bypasses bearer auth — must return 204 + headers
  even when bearer is configured + Authorization header is absent.
- Real (non-OPTIONS) requests get CORS headers attached on the way out.
- Bearer auth is NOT skipped on real requests; it just runs after CORS
  pre-attaches origin info on the response (well — bearer still raises
  401, then CORS attaches headers on the 401 response too so the
  browser shows the real error instead of "Failed to fetch").
- Wildcard config + non-OPTIONS: Allow-Origin: *.
- Vary: Origin attached on real responses to defeat shared caches.
- The base allowlist of headers (authorization, content-type) is echoed
  back on a wildcard preflight when the client doesn't send
  Access-Control-Request-Headers.
- When the client DOES send Access-Control-Request-Headers, those
  values are echoed verbatim (so SDK extras like X-Stainless-* pass).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from meta_model.config import parse_config_str
from meta_model.server import app, set_config


_BASE = """
[upstreams.a]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."alpha.v1"]
type = "moa"
generators = ["a"]
synthesizer = "a"
"""

_DEFAULT_FIXTURE = _BASE  # cors_allow_origins defaults to ["*"]

_LOCKED_FIXTURE = f"""
[server]
cors_allow_origins = ["https://app.example.com", "http://localhost:3000"]

{_BASE}
"""

_BRAND_FIXTURE = f"""
[server]
bearer_token = "sk-test-secret"
cors_allow_origins = ["https://app.example.com"]

{_BASE}
"""


@pytest.fixture
def default_config():
    cfg = parse_config_str(_DEFAULT_FIXTURE)
    set_config(cfg)
    yield cfg
    set_config(None)


@pytest.fixture
def locked_config():
    cfg = parse_config_str(_LOCKED_FIXTURE)
    set_config(cfg)
    yield cfg
    set_config(None)


@pytest.fixture
def auth_locked_config():
    cfg = parse_config_str(_BRAND_FIXTURE)
    set_config(cfg)
    yield cfg
    set_config(None)


def _client() -> TestClient:
    return TestClient(app)


# ── Default wildcard origins ────────────────────────────────────────


def test_default_cors_allow_origins_is_wildcard() -> None:
    cfg = parse_config_str(_BASE)
    assert cfg.server.cors_allow_origins == ["*"]


def test_preflight_wildcard_returns_204_with_cors_headers(default_config) -> None:
    """OPTIONS preflight from any origin → 204 + Access-Control-* headers."""
    r = _client().options(
        "/v1/models",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code == 204, r.text
    assert r.headers.get("access-control-allow-origin") == "*"
    assert "GET" in r.headers.get("access-control-allow-methods", "")
    assert "POST" in r.headers.get("access-control-allow-methods", "")
    assert "OPTIONS" in r.headers.get("access-control-allow-methods", "")
    # Default header allowlist when client didn't send Access-Control-
    # Request-Headers.
    allow_h = r.headers.get("access-control-allow-headers", "").lower()
    assert "authorization" in allow_h
    assert "content-type" in allow_h
    assert r.headers.get("access-control-max-age") == "86400"
    assert r.headers.get("vary") == "Origin"


def test_preflight_echoes_request_headers_back(default_config) -> None:
    """When client sends Access-Control-Request-Headers, middleware
    echoes them back so SDK-specific headers (X-Stainless-*, OpenAI-Beta)
    pass preflight."""
    r = _client().options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization, x-stainless-os, openai-beta",
        },
    )
    assert r.status_code == 204
    echoed = r.headers.get("access-control-allow-headers", "")
    assert "x-stainless-os" in echoed.lower()
    assert "openai-beta" in echoed.lower()
    assert "authorization" in echoed.lower()


def test_real_request_gets_cors_headers_on_response(default_config) -> None:
    """GET /v1/health (exempt from auth) returns 200 + Allow-Origin
    on the actual response so the browser accepts it."""
    r = _client().get("/v1/health", headers={"Origin": "https://example.com"})
    # health may 503 (no monitor) or 200; either way it's not a CORS
    # issue and the headers must be present.
    assert r.headers.get("access-control-allow-origin") == "*"
    assert "Origin" in r.headers.get("vary", "")


# ── Locked-down origin list ─────────────────────────────────────────


def test_locked_preflight_matched_origin_echoed(locked_config) -> None:
    """Origin is in cors_allow_origins → echo it (NOT '*'). Browser
    accepts because the echo matches what it sent."""
    r = _client().options(
        "/v1/models",
        headers={
            "Origin": "https://app.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code == 204
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"


def test_locked_preflight_unmatched_origin_no_cors(locked_config) -> None:
    """Origin NOT in cors_allow_origins → 204 with no CORS headers.
    Browser blocks the request."""
    r = _client().options(
        "/v1/models",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code == 204
    assert "access-control-allow-origin" not in r.headers


def test_locked_real_request_matched_origin_gets_header(locked_config) -> None:
    r = _client().get(
        "/v1/health", headers={"Origin": "http://localhost:3000"}
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_locked_real_request_unmatched_origin_no_header(locked_config) -> None:
    r = _client().get(
        "/v1/health", headers={"Origin": "https://attacker.example"}
    )
    assert "access-control-allow-origin" not in r.headers


# ── Preflight bypasses bearer auth ──────────────────────────────────


def test_preflight_bypasses_bearer_auth(auth_locked_config) -> None:
    """Critical regression: when bearer is configured AND a preflight
    OPTIONS arrives without Authorization header, CORS middleware
    answers 204 directly. If preflight reached bearer, it'd return 401
    with no CORS headers and the browser would block — surfacing as
    "Failed to fetch" with no diagnostic."""
    r = _client().options(
        "/v1/models",
        headers={
            "Origin": "https://app.example.com",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization, content-type",
        },
    )
    assert r.status_code == 204, (
        f"preflight must succeed without auth; got {r.status_code} {r.text}"
    )
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"
    # Crucially: no 401 envelope.
    assert "missing_api_key" not in r.text
    assert "invalid_api_key" not in r.text


def test_real_request_with_bearer_still_authenticated(auth_locked_config) -> None:
    """CORS doesn't disable bearer auth on real requests. A GET
    without Authorization returns 401 — but with CORS headers attached
    so the browser reports the real error instead of 'Failed to fetch'."""
    r = _client().get(
        "/v1/models", headers={"Origin": "https://app.example.com"}
    )
    assert r.status_code == 401
    # 401 envelope still emitted.
    body = r.json()
    assert body["error"]["code"] == "missing_api_key"
    # But CORS headers ARE present on the 401 — browser sees the real
    # error envelope instead of failing the fetch silently.
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"


def test_real_request_with_correct_bearer_passes_through(auth_locked_config) -> None:
    r = _client().get(
        "/v1/models",
        headers={
            "Origin": "https://app.example.com",
            "Authorization": "Bearer sk-test-secret",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://app.example.com"


# ── Vary: Origin propagation ────────────────────────────────────────


def test_vary_origin_added_to_existing_vary_header(default_config) -> None:
    """If a route handler already sets Vary, CORS appends Origin
    instead of overwriting. Defends against shared caches serving
    wrong-origin responses from cache."""
    r = _client().get("/v1/models", headers={"Origin": "https://example.com"})
    # /v1/models doesn't set Vary today, but CORS must still add it.
    vary = r.headers.get("vary", "")
    assert "Origin" in vary


def test_vary_origin_added_when_origin_rejected(locked_config) -> None:
    """Review r1 F11 MED regression. When the origin is rejected
    (locked-down list, no match), the response carries no CORS
    headers — but a DIFFERENT origin would have produced a CORS
    response. Without `Vary: Origin`, a shared cache (proxy edge,
    CDN) could store the no-CORS variant and later serve it to an
    allowed browser origin, breaking that legitimate client.

    So `Vary: Origin` MUST be present even on the rejected-origin
    response."""
    r = _client().get(
        "/v1/health", headers={"Origin": "https://attacker.example"}
    )
    # No CORS allow-origin (rejected), but Vary still present.
    assert "access-control-allow-origin" not in r.headers
    vary = r.headers.get("vary", "")
    assert "Origin" in vary, f"Vary missing Origin token: {vary!r}"


def test_vary_origin_added_when_request_has_no_origin(locked_config) -> None:
    """Review r2 F11 MED regression. A request without an Origin
    header (server-to-server, curl) gets a no-CORS response. A
    later request from the SAME shared cache with Origin would
    expect a CORS response. Without `Vary: Origin` on the no-Origin
    response, a shared cache could serve the no-CORS variant to a
    browser, breaking the legitimate cross-origin client.

    `Vary: Origin` MUST be present on ALL non-OPTIONS responses,
    regardless of whether the request had an Origin header."""
    r = _client().get("/v1/health")  # no Origin header
    vary = r.headers.get("vary", "")
    assert "Origin" in vary, f"Vary missing Origin (no-Origin request): {vary!r}"
    # No Allow-Origin (no request origin to match), but Vary still set.
    assert "access-control-allow-origin" not in r.headers


def test_vary_origin_token_match_is_exact_not_substring(default_config) -> None:
    """Review r1 F11 LOW regression. The Vary append guard must
    compare comma-separated tokens exactly (lowercase). A header
    like `Vary: User-Origin-Sent` should NOT suppress adding the
    real `Origin` token via substring match.

    Direct unit-level test on `_append_vary_token`."""
    from meta_model.auth import _append_vary_token

    # No-Vary case → set fresh.
    headers = {}
    _append_vary_token(headers, "Origin")
    assert headers["Vary"] == "Origin"

    # Already has Origin → no-op (case-insensitive).
    headers = {"Vary": "origin"}
    _append_vary_token(headers, "Origin")
    assert headers["Vary"] == "origin"  # untouched

    # Already has Origin in a list → no-op.
    headers = {"Vary": "Accept, Origin, User-Agent"}
    _append_vary_token(headers, "Origin")
    assert headers["Vary"] == "Accept, Origin, User-Agent"

    # Substring-match-shaped header that does NOT contain Origin as a
    # comma-separated token → must append.
    headers = {"Vary": "User-Origin-Sent"}
    _append_vary_token(headers, "Origin")
    assert headers["Vary"] == "User-Origin-Sent, Origin"
