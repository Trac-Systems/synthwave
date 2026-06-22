"""Bearer-token auth for the meta-model API. F7.

Drop-in compatible with OpenAI's auth shape: clients send
``Authorization: Bearer <token>``; missing or wrong token returns
HTTP 401 with the OpenAI error envelope (``type: invalid_request_error``,
``code: missing_api_key | invalid_api_key``).

Enforcement is opt-in: when ``[server] bearer_token`` is unset (or
empty after strip), the middleware is a no-op and the API stays
open. This matches the development/local-dev contract — set the
token in production, leave it unset for local probes.

Implemented as **ASGI middleware** (not a FastAPI ``Depends``) so
auth runs BEFORE body parsing. With a route dependency, FastAPI's
body validator can race the dependency and a malformed-JSON request
to a protected route produces a 400 (invalid_json) instead of a 401
— that lets unauthenticated callers probe routes by inspecting
which error envelope they get back. Middleware closes the timing.

Constant-time comparison via ``hmac.compare_digest`` defends against
token-content timing oracles.

Routing model is **fail-closed**: every route is protected by
default, with an explicit exempt set for probes/scrapers/
introspection. New routes added later inherit auth automatically;
adding a new exempt path is a deliberate edit here.
"""

from __future__ import annotations

import hmac
from typing import Awaitable, Callable

from fastapi.responses import JSONResponse
from starlette.requests import Request
from starlette.responses import Response

# Paths that stay public even when ``[server] bearer_token`` is set.
# Health probes (k8s readiness/liveness), the OpenAPI schema (tooling
# introspection), and the root-level metrics alias (Prometheus
# scrapers) all bypass auth. The versioned ``/v1/metrics/moa`` form
# stays protected because it's under the API surface clients
# authenticate against.
EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        "/v1/health",
        "/health",
        "/openapi.json",
        "/metrics",
    }
)


def _envelope_401(message: str, *, code: str) -> JSONResponse:
    """Build the OpenAI-typed 401 envelope."""
    return JSONResponse(
        status_code=401,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": code,
            }
        },
    )


async def bearer_auth_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """ASGI middleware: enforce ``Authorization: Bearer <token>``.

    No-op when ``[server] bearer_token`` is unset or empty (after
    strip). Runs before FastAPI dependency resolution and body
    parsing, so a missing/invalid token always returns 401 — never
    a 400 from the body validator preempting the auth check.
    """
    # Imported lazily so the ``server.py`` ↔ ``auth.py`` cycle stays
    # broken at module-load time (server registers this middleware,
    # then this calls back into server for the loaded config).
    from .server import get_config

    cfg = get_config()
    expected = (cfg.server.bearer_token or "").strip() if cfg is not None else ""
    if not expected:
        return await call_next(request)

    if request.url.path in EXEMPT_PATHS:
        return await call_next(request)

    authorization = request.headers.get("authorization")
    if not authorization or not authorization.startswith("Bearer "):
        return _envelope_401(
            "Authorization header missing or malformed (expected 'Bearer <token>')",
            code="missing_api_key",
        )
    presented = authorization[len("Bearer "):].strip()
    # Compare bytes, not str: ``hmac.compare_digest`` raises TypeError
    # on str inputs containing non-ASCII codepoints, and Starlette
    # decodes incoming header bytes as latin-1 so a raw byte > 0x7f
    # in the Authorization header lands as a non-ASCII str. Without
    # this, a request like ``Authorization: Bearer <0xff>`` would 500
    # instead of returning 401 invalid_api_key (review r2 HIGH).
    if not hmac.compare_digest(
        presented.encode("utf-8", "replace"),
        expected.encode("utf-8", "replace"),
    ):
        return _envelope_401("Invalid Bearer token", code="invalid_api_key")

    return await call_next(request)


__all__ = ["bearer_auth_middleware", "EXEMPT_PATHS", "cors_middleware"]


# ── CORS middleware ─────────────────────────────────────────────────


# Methods clients can call. The OpenAI API surface is GET (for
# /v1/models, /v1/health, /openapi.json) and POST (for everything
# else); OPTIONS is the preflight method browsers use before the
# real request. Other methods aren't part of the contract — listing
# only what's actually served keeps the preflight allowlist honest.
_CORS_ALLOW_METHODS = "GET, POST, OPTIONS"

# Headers clients are permitted to send. Authorization carries the
# bearer token (F7); content-type identifies request bodies. SDK
# additions (X-Stainless-*, OpenAI-Beta, etc.) commonly travel along
# — listing them keeps the preflight allowlist matched against what
# real clients send. The permissive default is ``*`` only when the
# allow-origins list is also wildcard; for locked-down origins we
# echo the request's Access-Control-Request-Headers verbatim so the
# preflight passes whatever the client actually wants to send.
_CORS_BASE_ALLOW_HEADERS = "authorization, content-type"

# Cache preflight responses for 1 day — preflights are pure
# bookkeeping; refreshing them more often just adds latency.
_CORS_MAX_AGE = "86400"


def _resolve_cors_origin(
    cfg_origins: list[str], request_origin: str | None
) -> str | None:
    """Decide the value of ``Access-Control-Allow-Origin`` for this
    request, or ``None`` if CORS shouldn't fire.

    - ``["*"]`` in config → echo ``*`` (any origin allowed).
    - explicit origin list → echo the request's Origin header back if
      it's in the list, else None (browser will block).
    - empty / missing Origin header on a non-wildcard config → None.
    """
    if "*" in cfg_origins:
        return "*"
    if request_origin and request_origin in cfg_origins:
        return request_origin
    return None


async def cors_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """ASGI middleware: CORS handling for browser clients (F11).

    Registered AFTER ``bearer_auth_middleware`` so it sits OUTERMOST in
    the middleware stack — preflight ``OPTIONS`` requests are answered
    here directly with 204 + CORS headers, and never reach bearer auth
    (a 401 on preflight breaks the browser before the real request
    even leaves the page, with no CORS headers attached, surfacing as
    "Failed to fetch" with no diagnostic).

    Origins are read from ``cfg.server.cors_allow_origins`` at request
    time, so config changes take effect on the next request without a
    restart of the middleware. Default ``["*"]`` matches "drop-in
    OpenAI" behavior — any origin allowed.
    """
    from .server import get_config

    cfg = get_config()
    cfg_origins = (
        cfg.server.cors_allow_origins
        if cfg is not None
        else ["*"]
    )
    request_origin = request.headers.get("origin")
    allow_origin = _resolve_cors_origin(cfg_origins, request_origin)

    # Preflight: answer here directly, never forward to bearer / route.
    if request.method == "OPTIONS":
        # If origin doesn't match, return 204 with NO CORS headers —
        # browser will block; we don't pretend to allow it. Don't
        # forward to the route (most routes wouldn't accept OPTIONS
        # anyway and would 405).
        if allow_origin is None:
            return Response(status_code=204)
        # Echo the requested headers back (some clients ask for
        # X-Stainless-*, OpenAI-Beta, etc.). Falls back to the base
        # allowlist when the client doesn't send the request-headers
        # preflight field.
        requested_headers = (
            request.headers.get("access-control-request-headers")
            or _CORS_BASE_ALLOW_HEADERS
        )
        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": allow_origin,
                "Access-Control-Allow-Methods": _CORS_ALLOW_METHODS,
                "Access-Control-Allow-Headers": requested_headers,
                "Access-Control-Max-Age": _CORS_MAX_AGE,
                "Vary": "Origin",
            },
        )

    # Non-OPTIONS: let the request flow down (bearer + route). On the
    # way out, attach CORS headers so the browser accepts the response.
    response = await call_next(request)
    if allow_origin is not None:
        response.headers["Access-Control-Allow-Origin"] = allow_origin
    # Review r1+r2 F11 MED: ``Vary: Origin`` MUST be present on EVERY
    # non-OPTIONS response, regardless of whether the request carried
    # an Origin header. Three variants of the response exist:
    #   - request with matched Origin → has ``Allow-Origin: <origin>``
    #   - request with rejected Origin → no Allow-Origin header
    #   - request with NO Origin (server-to-server, curl) → no Allow-Origin
    # Any of these can be cached. Without Vary, a cache can serve the
    # no-CORS variant (case 2 or 3) to a browser request in case 1 and
    # break the legitimate cross-origin client. Attaching Vary
    # unconditionally is the safe rule.
    _append_vary_token(response.headers, "Origin")
    return response


def _append_vary_token(headers, token: str) -> None:
    """Add ``token`` to the response's Vary header without duplicates.

    Review r1 F11 LOW: parse the existing Vary as comma-separated
    header names and compare lowercase tokens exactly. Substring
    matching would let ``Vary: User-Origin-Sent`` suppress adding
    the real ``Origin`` token (false positive).
    """
    existing = headers.get("Vary")
    if not existing:
        headers["Vary"] = token
        return
    tokens = {t.strip().lower() for t in existing.split(",") if t.strip()}
    if token.lower() in tokens:
        return
    headers["Vary"] = f"{existing}, {token}"
