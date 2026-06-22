"""FastAPI application shell for the meta-model server.

Scope through D.1.4: app object, /v1/health, /v1/models, OpenAI
error envelope, config loading, and a single-upstream passthrough
on /v1/chat/completions. Profile dispatch + MoA arrive in D.2.

Config loading happens at startup via the lifespan handler. Invalid
configs fail the process — uvicorn workers won't start, systemd
sees the failure, no half-up server in production. Missing config
file is allowed (empty profile catalog) for local dev and tests.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .auth import bearer_auth_middleware, cors_middleware
from .chat import ChatRequest
from .legacy_completion import (
    chat_to_legacy_completion,
    transform_chat_sse_to_legacy,
)
from .config import (
    CascadeProfile,
    MetaModelConfig,
    MoaProfile,
    Profile,
    load_config_from_env,
)
from .errors import error_envelope, error_type_for_status
from .health import HealthMonitor
from .moa.dispatch import dispatch
from .moa.multimodal import detect_message_modality
from .responses import (
    ResponsesAdapterError,
    chat_to_responses,
    empty_config_envelope,
    new_response_id,
    responses_to_chat,
    stream_responses_events,
)
from .routing import ProfileResolutionError, error_response, resolve_profile
from .sanitize import sanitize_reasoning
from .streaming import new_chunk_id, stream_chat_completion

# Re-exported for any caller still importing from server (kept private
# alias for backwards compatibility — tests reach in directly).
_error_type_for_status = error_type_for_status


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: load config eagerly so invalid configs fail fast.

    Test injection: callers that have already set `app.state.config`
    via `set_config(...)` are respected — the lifespan only loads
    when no config is present. Missing config file is allowed (empty
    profile catalog) so tests + local dev work without a TOML.

    F5: starts the upstream-health background probe loop after
    config is resolved. Tests that do NOT want the loop running
    (most of them — they stub `app.state.upstream_health` directly)
    can pre-set `app.state.upstream_health` to a custom monitor or a
    sentinel; the lifespan only constructs/starts a default monitor
    when nothing is already present.
    """
    if not getattr(app.state, "config", None):
        try:
            app.state.config = load_config_from_env()
        except FileNotFoundError:
            app.state.config = None
        # ValidationError + other parse failures propagate → startup fails.
    cfg = getattr(app.state, "config", None)
    if cfg is not None and not getattr(app.state, "upstream_health", None):
        monitor = HealthMonitor(
            upstreams=dict(cfg.upstreams),
            interval_sec=cfg.server.health_probe_interval_sec,
            timeout_sec=cfg.server.health_probe_timeout_sec,
            transport_provider=lambda: getattr(app.state, "upstream_transport", None),
        )
        await monitor.start()
        app.state.upstream_health = monitor
    try:
        yield
    finally:
        monitor = getattr(app.state, "upstream_health", None)
        if isinstance(monitor, HealthMonitor):
            await monitor.stop()
            app.state.upstream_health = None


app = FastAPI(
    title="meta-model",
    version=__version__,
    docs_url=None,  # no Swagger UI in production
    redoc_url=None,
    # F4-core: enable /openapi.json (FastAPI default). Operators
    # building tooling that introspects the API surface get a
    # discoverable schema; previously suppressed for "OpenAI-compat
    # surface" reasons but the OpenAI server itself exposes
    # /openapi.json so suppression diverged from the contract.
    lifespan=_lifespan,
)


# ── Error envelope (OpenAI-compatible) ────────────────────────────────
# `error_envelope` + `error_type_for_status` live in `meta_model.errors`
# so non-server modules (streaming.py) can build the same shape without
# importing back into server. The handlers below stay here because they
# are FastAPI-specific (Request/Response types).


def _format_loc_path(loc: tuple) -> str | None:
    """Flatten a Pydantic `loc` tuple into an OpenAI-style dotted path.

    Conventions: drop the leading `body` segment; integer segments
    become bracketed indices on the previous part
    (e.g. `("body", "messages", 0, "role")` → `"messages[0].role"`).
    Returns None when the resulting path is empty.
    """
    parts: list[str] = []
    for p in loc:
        if p == "body":
            continue
        if isinstance(p, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{p}]"
            else:
                parts.append(f"[{p}]")
        else:
            parts.append(str(p))
    return ".".join(parts) if parts else None


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=error_envelope(str(exc.detail), status=exc.status_code),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Render Pydantic validation errors as OpenAI-style 400s.

    OpenAI's `BadRequestError` is HTTP 400; FastAPI's default for
    body validation is 422 (different SDK exception class). For
    OpenAI-client compatibility we collapse to 400 with
    `invalid_request_error`.
    """
    errors = exc.errors()
    if errors:
        first = errors[0]
        param = _format_loc_path(tuple(first.get("loc", ())))
        message = first.get("msg", "Invalid request")
    else:
        param = None
        message = "Invalid request"
    return JSONResponse(
        status_code=400,
        content=error_envelope(message, status=400, param=param),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: any unhandled exception → 500 with OpenAI envelope.

    The exception detail is intentionally generic — internal stack
    traces stay in server logs (uvicorn already logs them). Clients
    see a stable shape, not implementation noise.
    """
    return JSONResponse(
        status_code=500,
        content=error_envelope("internal server error", status=500),
    )


app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# F7: bearer-token enforcement runs as ASGI middleware so auth fires
# BEFORE FastAPI body parsing. With a route-level dependency, malformed
# JSON to a protected route can produce a 400 (invalid_json) instead
# of 401 — that lets unauthenticated callers probe routes by inspecting
# which envelope they get. Middleware closes the timing window. Exempt
# paths (health/openapi/metrics root) are listed in `auth.EXEMPT_PATHS`.
app.middleware("http")(bearer_auth_middleware)
# F11: CORS handling. Registered AFTER bearer so it sits OUTERMOST
# (Starlette wraps middleware in reverse-add order — last-added is
# the first request-side hop). Preflight OPTIONS is answered here
# directly with 204 + CORS headers; bearer auth never sees it. Real
# requests pass through bearer, hit the route, and get CORS headers
# attached on the response on the way out.
app.middleware("http")(cors_middleware)


# ── Endpoints ────────────────────────────────────────────────────────


@app.get("/v1/health")
async def health() -> JSONResponse:
    """Readiness probe — HTTP 503 when ANY configured upstream is
    failing its health check, OR when no monitor is configured.

    F5 contract:
    - All upstreams up → 200, ``{"status":"ok","unhealthy_upstreams":[]}``
    - Any upstream failing → 503, ``{"status":"unhealthy",
      "unhealthy_upstreams":[{"name":..., "reason":...}]}``
    - No monitor / no config / no upstreams → 503,
      ``{"status":"unhealthy","reason":"no_monitor_configured"|
      "no_upstreams_configured", ...}`` — the server can't serve
      `/v1/chat/completions` requests in this state, so reporting
      "ready" would lie. Review r1 F5 finding HIGH.

    Reason values: ``"down"`` (connection refused / 5xx / timeout),
    ``"auth_failed"`` (401/403), ``"misconfigured"`` (DNS not
    resolvable, malformed base_url, 4xx other than auth).

    Wire as **readiness** in your orchestrator, not liveness.
    Repeated 503s on liveness would force a restart even when the
    upstream (not the pod) is the problem.
    """
    monitor = getattr(app.state, "upstream_health", None)
    if not isinstance(monitor, HealthMonitor):
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": "no_monitor_configured",
                "unhealthy_upstreams": [],
                "version": __version__,
            },
        )
    body = await monitor.aggregate()
    status_code = 200 if body.get("status") == "ok" else 503
    return JSONResponse(status_code=status_code, content=body)


# F4-core: root-level `/health` alias. Some load-balancers and
# orchestration platforms expect health at root, not under `/v1`.
# Both routes register the same handler so semantics stay identical.
@app.get("/health")
async def health_root() -> JSONResponse:
    """Alias of `/v1/health` — same semantics, root path."""
    return await health()


def get_config() -> MetaModelConfig | None:
    """Return the loaded config (set by the lifespan handler at startup).

    Returns None when no config was reachable (test/dev with no
    META_MODEL_CONFIG and no ./meta-model.toml). Endpoints handle this
    gracefully — /v1/models lists an empty catalog rather than 500ing.
    """
    return getattr(app.state, "config", None)


def set_config(cfg: MetaModelConfig | None) -> None:
    """Override the loaded config (test injection point).

    Setting before the lifespan runs preempts the startup load;
    setting after replaces it. None clears.
    """
    if cfg is None:
        if hasattr(app.state, "config"):
            del app.state.config
    else:
        app.state.config = cfg


@app.get("/v1/metrics/moa")
async def metrics_moa() -> dict[str, Any]:
    """Per-profile MoA dispatch metrics. #7 carry-over from client application pt. 23.

    Aggregates the in-memory ringbuffer (last N=100 calls per profile)
    into per-profile stats: call count, average quorum, degraded rate,
    synth-decision distribution, tool-call rate, per-position draft
    length stats. Operator-facing — proves each MoA generator is
    earning its compute or surfaces when one is silently degenerate.
    """
    from .metrics import aggregate_metrics

    return aggregate_metrics()


# F4-core: root-level `/metrics` alias. Matches the convention of
# scrape-style metrics endpoints sitting at root. JSON shape (not
# Prometheus text); switching to text exposition would be a separate
# arc. Same handler so semantics stay identical.
@app.get("/metrics")
async def metrics_root() -> dict[str, Any]:
    """Alias of `/v1/metrics/moa` — same JSON body, root path."""
    return await metrics_moa()


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    """OpenAI-compatible model list.

    Each callable profile is exposed as one "model" from the client's
    perspective; raw upstreams are also listed so clients addressing
    a single model directly can introspect its limits. Voting
    profiles are hidden when [features].voting is false (per surface
    inventory: clients shouldn't see profiles they can't call).

    Each entry carries vLLM-compat extension fields:
    - ``max_model_len``: smallest input context across the profile's
      generators/upstreams. Honest input ceiling — shared-tail
      compaction will truncate to the smallest anyway, so reporting
      the largest or primary's context misleads clients.
    - ``capabilities``: ``{vision, video, audio, function_calling,
      effective_image_capability, supports_image_tools, thinking,
      reasoning_visible}`` booleans derived from the profile
      composition. ``vision`` answers "any reachable upstream
      declares image modality"; ``effective_image_capability`` and
      ``supports_image_tools`` (F2) answer the post-policy question
      "would an image-bearing (+tools) request actually succeed",
      which can be False even when ``vision`` and ``function_calling``
      are both True if no single upstream supports both.
      ``thinking`` is True only when an upstream supports it AND the
      profile opts to surface reasoning content via
      ``expose_reasoning=true``. ``reasoning_visible`` mirrors
      ``expose_reasoning`` directly so clients can tell whether
      ``reasoning`` / ``reasoning_content`` keys may appear in
      responses.
    """
    cfg = get_config()
    if cfg is None:
        return {"object": "list", "data": []}
    now = int(time.time())
    callable_ = cfg.callable_profiles()

    # F6: multimodal capabilities are server-level (single-model
    # transparency). Every profile reports the same vision/video/audio
    # flags, derived from the `[vision]/[video]/[audio]` cascade
    # blocks.
    #
    # `supports_image_tools` (review r1 F6 HIGH): cannot be derived
    # from the *profile*'s function_calling because multimodal dispatch
    # bypasses profiles entirely — what matters is whether ANY upstream
    # in the vision cascade supports tools. If any does, an
    # image+tools request can be served by the cascade; if none does,
    # advertising support would mislead.
    server_vision = bool(cfg.vision.endpoints)
    server_video = bool(cfg.video.endpoints)
    server_audio = bool(cfg.audio.endpoints)
    vision_tools_capable = any(
        cfg.upstreams[name].supports_function_calling
        for name in cfg.vision.endpoints
        if name in cfg.upstreams
    )

    def _entry(model_id: str, prof: Profile, *, alias_of: str | None = None) -> dict[str, Any]:
        caps = prof.capabilities(cfg.upstreams)
        body: dict[str, Any] = {
            "id": model_id,
            "object": "model",
            "created": now,
            "owned_by": "meta-model",
            "max_model_len": caps.max_model_len,
            "capabilities": {
                "vision": server_vision,
                "video": server_video,
                "audio": server_audio,
                "function_calling": caps.function_calling,
                "effective_image_capability": server_vision,
                "supports_image_tools": vision_tools_capable,
                "thinking": caps.thinking,
                "reasoning_visible": caps.reasoning_visible,
            },
        }
        if alias_of is not None:
            # F4-A: clients introspecting /v1/models can tell aliases
            # from canonicals at a glance. Resolution is transparent
            # (response.model still reports `alias_of`), but the catalog
            # entry is honest about its shape.
            body["alias_of"] = alias_of
        return body

    data: list[dict[str, Any]] = []
    # F10: when the server brand is set, surface it as the FIRST
    # entry. Capabilities mirror the profile that actually serves the
    # request (the first callable non-voting profile in config order
    # — matches the resolver's brand fallback). Empty fleet edge:
    # bands without any non-voting callable profile still emit the
    # entry (capabilities default to false) so clients can see the
    # name even if it would 404 on call — the dispatch resolver
    # handles the actual error envelope.
    brand = cfg.server.model_name
    brand_target_name: str | None = None
    if brand:
        for pname, prof in cfg.callable_profiles().items():
            from .config import VotingProfile as _VP

            if isinstance(prof, _VP):
                continue
            brand_target_name = pname
            break
        if brand_target_name is not None:
            data.append(_entry(brand, cfg.profiles[brand_target_name]))
    for name in sorted(callable_):
        data.append(_entry(name, cfg.profiles[name]))
    # F4-A: append one entry per alias of a callable profile. Aliases
    # of hidden voting profiles are skipped — same rule as
    # callable_profiles for the canonical entries.
    for alias, canonical in sorted(cfg.alias_entries()):
        if canonical not in callable_:
            continue
        data.append(_entry(alias, cfg.profiles[canonical], alias_of=canonical))
    return {"object": "list", "data": data}


# ── Test injection point for httpx upstream calls ───────────────────


def get_upstream_transport() -> httpx.AsyncBaseTransport | None:
    """Return a custom httpx transport, or None for the real network.

    Tests inject `httpx.MockTransport(...)` here via `set_upstream_
    transport()`. Production keeps it None (real HTTP).
    """
    return getattr(app.state, "upstream_transport", None)


def set_upstream_transport(transport: httpx.AsyncBaseTransport | None) -> None:
    if transport is None:
        if hasattr(app.state, "upstream_transport"):
            del app.state.upstream_transport
    else:
        app.state.upstream_transport = transport


# ── /v1/chat/completions ────────────────────────────────────────────


def _err(status: int, message: str, *, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=error_envelope(message, status=status, code=code),
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, request: Request) -> Any:
    """OpenAI Chat Completions — profile-aware multi-upstream dispatch.

    `model` may be a profile name OR a raw upstream key.
    `x_meta_model.profile` (when set) wins over `model` — it's the
    explicit, type-aware path for clients that know about extension
    fields. ``n != 1`` is unsupported v1.

    Profile dispatch routes:
    - **MoA**: shared-tail compact + parallel fan-out + synthesize.
    - **Cascade**: try upstreams in priority order; first 2xx wins.
    - **Voting**: parallel YES/NO consensus (gated by [features].voting).

    Streaming (D.3.3): ``stream:true`` runs MoA dispatch to completion
    and emits the synthesized response as wire-compatible OpenAI SSE
    chunks with periodic ``: heartbeat\\n\\n`` comments while dispatch
    is pending. Pre-dispatch validation (this function's 4xx checks)
    runs BEFORE constructing ``StreamingResponse`` so malformed
    requests still surface as JSON 4xx, not ``200 + SSE error event``.
    Once Starlette sends ``http.response.start`` with status 200,
    failures from inside dispatch become an SSE error event with NO
    ``[DONE]`` sentinel (mirrors OpenAI's actual streaming-error wire
    behavior).
    """
    cfg = get_config()
    if cfg is None or not cfg.upstreams:
        return _err(503, "no upstreams configured", code="no_upstream")

    if req.max_tokens is not None and req.max_completion_tokens is not None:
        return _err(
            400,
            "set max_tokens or max_completion_tokens, not both",
            code="conflicting_params",
        )

    if req.n is not None and req.n != 1:
        return _err(400, "n != 1 unsupported in v1", code="unsupported_v1")

    # Forward the original request body (only what the client sent)
    # so dispatch can shape per-upstream payloads without pydantic
    # default-value leakage. Review r10 caught this pattern as load-
    # bearing for the passthrough path; it's also right for dispatch.
    try:
        forwarded = await request.json()
    except ValueError:
        forwarded = req.model_dump(exclude_unset=True)

    # Review r35 F3 + F1: preflight unsupported_content_part on the raw
    # dict body (NOT ``req.messages``, which is a list of Pydantic
    # models — ``detect_message_modality`` skips non-dict messages).
    # Symmetrizes streaming + non-streaming: both 4xx synchronously,
    # neither produces a ``200 + SSE error event`` UX.
    preflight_modality = detect_message_modality(forwarded.get("messages") or [])
    if preflight_modality.unsupported_parts:
        return _err(
            400,
            "request contains unsupported content part type(s): "
            + ", ".join(preflight_modality.unsupported_parts),
            code="unsupported_content_part",
        )

    ext_profile = req.x_meta_model.profile if req.x_meta_model else None
    request_id = "metamodel-" + uuid.uuid4().hex[:24]

    if req.stream:
        # Review r34 F2: read include_usage BEFORE stripping
        # stream_options. Strip BOTH stream and stream_options from the
        # body forwarded into dispatch so neither leaks to upstreams
        # (``prepare_upstream_body`` only filters ``x_meta_model``).
        include_usage = bool((req.stream_options or {}).get("include_usage", False))
        forwarded_for_dispatch = {
            k: v for k, v in forwarded.items() if k not in {"stream", "stream_options"}
        }
        # Result-specific dispatch headers (Compacted-N, Quorum, etc.)
        # CANNOT be preserved in streaming mode — Starlette commits the
        # 200 response head BEFORE the generator runs, so headers must
        # be set up-front. Document tradeoff: clients that need
        # observability should use ``stream:false`` (review r35 F5).
        sse_headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-MetaModel-Request-Id": request_id,
            # Generic OpenAI-compatible request-id mirrors so clients
            # that don't know about `X-MetaModel-*` can correlate.
            # Review r-rip-out-1 MED.
            "X-Request-Id": request_id,
            "OpenAI-Request-Id": request_id,
        }
        ext_profile_or_model = ext_profile or req.model
        # F10: server-wide brand override. When `[server] model_name`
        # is set, every SSE chunk's `model` field reports the brand
        # instead of the profile/upstream label.
        brand = cfg.server.model_name
        sse_model_label = brand if brand else ext_profile_or_model
        return StreamingResponse(
            stream_chat_completion(
                cfg,
                forwarded_for_dispatch,
                model=req.model,
                ext_profile=ext_profile,
                timeout_secs=float(cfg.server.request_timeout_secs),
                transport=get_upstream_transport(),
                chunk_id=new_chunk_id(),
                model_label=sse_model_label,
                include_usage=include_usage,
            ),
            status_code=200,
            media_type="text/event-stream",
            headers=sse_headers,
        )

    result = await dispatch(
        cfg,
        forwarded,
        model=req.model,
        ext_profile=ext_profile,
        timeout_secs=float(cfg.server.request_timeout_secs),
        transport=get_upstream_transport(),
    )

    headers = dict(result.headers)
    headers["X-MetaModel-Request-Id"] = request_id
    # Also emit standard OpenAI-compatible request-id headers so
    # generic clients (which don't know about `X-MetaModel-*`) can
    # correlate. Review r-rip-out-1 MED.
    headers["X-Request-Id"] = request_id
    headers["OpenAI-Request-Id"] = request_id

    if result.error is not None:
        message, code = result.error
        # Review r1 F6 MED: dispatcher can override the OpenAI error
        # envelope `type` field via DispatchResult.error_type — used by
        # the F6 modality_not_supported (501) path which wants
        # `invalid_request_error` instead of the status-derived
        # `api_error` default.
        return JSONResponse(
            status_code=result.status_code,
            content=error_envelope(
                message,
                status=result.status_code,
                code=code,
                type_=result.error_type,
            ),
            headers=headers,
        )

    # F3: apply expose_reasoning policy. Resolved profile name lives
    # in the headers `X-MetaModel-Profile`; default false when no
    # profile is in play (e.g. raw-upstream passthrough).
    payload = result.payload
    profile_name = result.headers.get("X-MetaModel-Profile")
    expose_reasoning = False
    if profile_name and profile_name in cfg.profiles:
        expose_reasoning = getattr(
            cfg.profiles[profile_name], "expose_reasoning", False
        )
    if isinstance(payload, dict):
        payload = sanitize_reasoning(payload, expose_reasoning=expose_reasoning)

    # F10: server-wide brand override. Whatever the dispatch path set
    # (profile name for cascade/voting, upstream model_id for synth
    # passthrough) gets rewritten to `[server] model_name` if set.
    # The canonical profile name stays in the `X-MetaModel-Profile`
    # header so introspection-aware clients can still see it.
    if isinstance(payload, dict) and cfg.server.model_name:
        payload["model"] = cfg.server.model_name

    return JSONResponse(
        status_code=result.status_code,
        content=payload,
        headers=headers,
    )


# ── /v1/completions (legacy adapter) ────────────────────────────────


_COMPLETIONS_UNSUPPORTED = {
    "echo": "echo not supported; use chat completions for prompt continuation",
    "suffix": "suffix not supported; legacy `prompt + suffix` insertion is not in v1",
    "best_of": "best_of not supported; n != 1 is rejected",
    "logprobs": "logprobs not supported in v1",
    "response_format": "response_format on legacy /v1/completions not supported; use /v1/chat/completions",
    "max_completion_tokens": (
        "max_completion_tokens is a Chat Completions field; use max_tokens here"
    ),
    "stream_options": "stream_options on legacy /v1/completions not supported",
}


@app.post("/v1/completions")
async def completions(request: Request) -> Any:
    """Legacy OpenAI completions adapter.

    Translates `prompt` (string) → `messages=[{role:"user", content:prompt}]`
    and forwards to `/v1/chat/completions`. Pass-throughs:
    `max_tokens`, `temperature`, `top_p`, `presence_penalty`,
    `frequency_penalty`, `seed`, `logit_bias`, `user`, `stop`,
    `stream`, `model`.

    Typed 501 (`unsupported_legacy_param`) for fields that have no
    chat-completions equivalent or would silently change semantics:
    `prompt: array`, `echo`, `suffix`, `best_of`, `logprobs`,
    `n>1`, `response_format`, `max_completion_tokens`,
    `stream_options`.

    F4-core. Body validation here is intentionally light — the
    payload is shaped into a chat request and re-validated by the
    Chat Completions endpoint, so OpenAI-shaped errors come from
    one place.
    """
    cfg = get_config()
    if cfg is None or not cfg.upstreams:
        return _err(503, "no upstreams configured", code="no_upstream")
    try:
        body = await request.json()
    except ValueError:
        return _err(400, "request body is not valid JSON", code="invalid_json")
    if not isinstance(body, dict):
        return _err(400, "request body must be a JSON object", code="invalid_request")

    for field, message in _COMPLETIONS_UNSUPPORTED.items():
        if field in body:
            return JSONResponse(
                status_code=501,
                content=error_envelope(
                    message,
                    status=501,
                    param=field,
                    code="unsupported_legacy_param",
                    type_="invalid_request_error",
                ),
            )
    n = body.get("n")
    if n is not None and n != 1:
        return JSONResponse(
            status_code=501,
            content=error_envelope(
                "n != 1 unsupported on /v1/completions; pin n=1 or omit",
                status=501,
                param="n",
                code="unsupported_legacy_param",
                type_="invalid_request_error",
            ),
        )
    prompt = body.get("prompt")
    if prompt is None:
        return _err(400, "missing 'prompt' field", code="invalid_request")
    if not isinstance(prompt, str):
        return JSONResponse(
            status_code=501,
            content=error_envelope(
                "prompt: array unsupported; pass a single string",
                status=501,
                param="prompt",
                code="unsupported_legacy_param",
                type_="invalid_request_error",
            ),
        )

    # Resolve profile up-front so /v1/completions emits the same
    # typed 404 model_not_found as /v1/chat/completions instead of
    # leaking the chat path's 503 "no upstreams" surface. Review r1
    # F4-core MED: `x_meta_model.profile` overrides `model` here too,
    # otherwise an invalid extension profile bypasses the typed
    # resolver and returns the chat-path's untyped 404 shape.
    ext_profile = None
    xmm = body.get("x_meta_model")
    if isinstance(xmm, dict):
        ep = xmm.get("profile")
        if isinstance(ep, str):
            ext_profile = ep
    try:
        resolve_profile(cfg, body.get("model"), ext_profile)
    except ProfileResolutionError as exc:
        return error_response(exc)

    chat_body = {k: v for k, v in body.items() if k != "prompt"}
    chat_body["messages"] = [{"role": "user", "content": prompt}]

    # Review r1 F4-core HIGH: validate the rewritten body against
    # `ChatRequest` BEFORE in-process forwarding. FastAPI's body
    # validation only fires on requests entering through the router;
    # an in-process invocation that calls `chat_completions(req,
    # request)` directly with a hand-built `ChatRequest` skips it.
    # Converting ValidationError here keeps clients seeing the same
    # typed 400 envelope they'd get hitting `/v1/chat/completions`.
    try:
        chat_req = ChatRequest.model_validate(chat_body)
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            first = errors[0]
            param = _format_loc_path(tuple(first.get("loc", ())))
            message = first.get("msg", "Invalid request")
        else:
            param = None
            message = "Invalid request"
        return JSONResponse(
            status_code=400,
            content=error_envelope(message, status=400, param=param),
        )

    # Build a fresh ChatRequest + Request-shaped wrapper so we reuse
    # the existing dispatch + streaming codepath verbatim. Constructing
    # a bare Starlette Request from a scope is the cheapest way to
    # forward an in-process call without re-marshalling through HTTP.
    from starlette.requests import Request as StarletteRequest

    new_scope = dict(request.scope)
    new_scope["path"] = "/v1/chat/completions"
    new_request = StarletteRequest(new_scope, request._receive)
    # Cache the parsed JSON on the new request so chat_completions's
    # `await request.json()` returns the rewritten body without
    # re-reading the (already-consumed) byte stream. Verified against
    # Starlette 0.52.x: `Request.json()` calls `body()`, which
    # short-circuits when `_body` is set.
    new_request._body = (
        __import__("json").dumps(chat_body).encode("utf-8")  # noqa: PLE0501
    )
    chat_resp = await chat_completions(chat_req, new_request)

    # F8: reshape chat response into legacy text-completion shape.
    # OpenAI's classic /v1/completions returns ``object: text_completion``
    # with ``choices[i].text`` (not ``choices[i].message.content``).
    # Drop-in clients targeting the legacy SDK do ``resp.choices[0].text``;
    # without this reshape they crash with AttributeError on a chat body.
    #
    # Review r1 F8 HIGH: copying chat_resp.headers verbatim leaks a stale
    # Content-Length onto the new response (Starlette only auto-computes
    # Content-Length when the header is absent). The reshape changes the
    # body length, so the original header would mis-frame the wire and
    # break HTTP/1.1 keep-alive. Strip Content-Length + body-encoding
    # headers and let Starlette/Uvicorn recompute.
    forwarded_headers = {
        k: v
        for k, v in chat_resp.headers.items()
        if k.lower() not in {"content-length", "content-encoding", "transfer-encoding"}
    }
    if isinstance(chat_resp, StreamingResponse):
        return StreamingResponse(
            transform_chat_sse_to_legacy(chat_resp.body_iterator),
            status_code=chat_resp.status_code,
            media_type=chat_resp.media_type,
            headers=forwarded_headers,
        )
    if isinstance(chat_resp, JSONResponse):
        if chat_resp.status_code >= 400:
            return chat_resp  # error envelopes pass through unchanged
        import json as _json

        chat_body_resp = _json.loads(chat_resp.body)
        legacy_body = chat_to_legacy_completion(chat_body_resp)
        return JSONResponse(
            status_code=chat_resp.status_code,
            content=legacy_body,
            headers=forwarded_headers,
        )
    return chat_resp


# ── /tokenize (per-profile tokenizer) ───────────────────────────────


def _resolve_tokenizer_upstream(
    cfg: MetaModelConfig, profile: Profile, profile_name: str
) -> tuple[str, dict[str, Any]] | tuple[None, JSONResponse]:
    """Pick the upstream that answers `/tokenize` for this profile.

    Resolution order:
    1. Explicit `tokenizer_upstream` field on the profile.
    2. Single-upstream profile (only one candidate to pick from).
    3. Homogeneous fleet (all candidates share one ``model_id``).
    4. **F9**: principled per-profile-type default —
       - **MoA** → `synthesizer`. Synthesizer's context governs the
         advertised `max_model_len` (F1's `effective_ingress_budget`)
         and produces the response the client receives, so its
         tokenizer is the most useful default for budget reasoning.
       - **Cascade** → `upstreams[0]`. The first cascade entry is the
         one always tried first; if it serves, that's what the client
         got.
    5. Voting → 409. No obvious "which voter answered" semantic
       (every voter contributes; consensus is the output, not a
       single upstream's tokens).

    Returns `(upstream_name, {})` on success or `(None, JSONResponse)`
    on the 409 path.
    """
    explicit = getattr(profile, "tokenizer_upstream", None)
    if explicit is not None:
        return explicit, {}

    if isinstance(profile, MoaProfile):
        candidates = list(dict.fromkeys([*profile.generators, profile.synthesizer]))
    elif isinstance(profile, CascadeProfile):
        candidates = list(profile.upstreams)
    else:
        # VotingProfile — same shape as cascade.
        candidates = list(profile.upstreams)  # type: ignore[union-attr]

    if len(candidates) == 1:
        return candidates[0], {}

    model_ids = {cfg.upstreams[u].model_id for u in candidates if u in cfg.upstreams}
    if len(model_ids) == 1:
        return candidates[0], {}

    # F9: principled per-profile-type fallback before 409.
    if isinstance(profile, MoaProfile):
        # Synthesizer's tokenizer is the natural default. F1
        # advertised_context is computed against the synth's context,
        # and the synth produces the wire response. Operators can
        # still override via explicit `tokenizer_upstream`.
        if profile.synthesizer in cfg.upstreams:
            return profile.synthesizer, {}
    elif isinstance(profile, CascadeProfile):
        # First-tried upstream wins by convention.
        if candidates and candidates[0] in cfg.upstreams:
            return candidates[0], {}
    # Voting profiles fall through — no principled default.

    return None, JSONResponse(
        status_code=409,
        content=error_envelope(
            (
                f"profile {profile_name!r} routes to {len(model_ids)} distinct "
                f"model_ids; set `tokenizer_upstream` to pick which one answers "
                f"/tokenize"
            ),
            status=409,
            param="model",
            code="heterogeneous_tokenizer",
            type_="invalid_request_error",
        ),
    )


@app.post("/tokenize")
async def tokenize(request: Request) -> Any:
    """Per-profile tokenizer. F4-core.

    Forwards to the resolved upstream's `/tokenize` endpoint
    (vLLM-style — sibling of `/v1/chat/completions` at the
    base-url root). Strips a trailing `/v1` from `base_url` before
    appending `/tokenize`.

    Body: `{"model": "<profile|upstream>", "prompt": "..."}` or
    `{"model": "...", "messages": [...]}`. The `model` field is
    rewritten to the resolved upstream's `model_id` before
    forwarding so the upstream sees the name it expects.

    Errors:
    - 404 `model_not_found` — unknown profile/upstream.
    - 409 `heterogeneous_tokenizer` — MoA/cascade routes to multiple
      `model_id`s and no `tokenizer_upstream` override is configured.
    - 503 `no_upstreams` — empty config.
    - Upstream HTTP status / body relayed verbatim on success or
      upstream-side errors (mirrors the chat-completions passthrough
      behavior at D.1.4).
    """
    cfg = get_config()
    if cfg is None or not cfg.upstreams:
        return _err(503, "no upstreams configured", code="no_upstream")
    try:
        body = await request.json()
    except ValueError:
        return _err(400, "request body is not valid JSON", code="invalid_json")
    if not isinstance(body, dict):
        return _err(400, "request body must be a JSON object", code="invalid_request")

    try:
        profile, profile_name = resolve_profile(cfg, body.get("model"), None)
    except ProfileResolutionError as exc:
        return error_response(exc)

    upstream_name, err_resp = _resolve_tokenizer_upstream(cfg, profile, profile_name)
    if upstream_name is None:
        return err_resp

    upstream = cfg.upstreams[upstream_name]
    forwarded = dict(body)
    forwarded["model"] = upstream.model_id
    forwarded.pop("x_meta_model", None)

    # Strip a trailing /v1 from base_url so /tokenize lands at the
    # vLLM root sibling, not at /v1/tokenize (which doesn't exist).
    base = upstream.base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    target_url = base + "/tokenize"
    headers = {"Content-Type": "application/json"}
    from .upstream import _build_auth_headers as _auth

    headers.update(_auth(upstream))

    transport = get_upstream_transport()
    timeout = float(cfg.server.request_timeout_secs)
    try:
        async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
            resp = await client.post(target_url, json=forwarded, headers=headers)
    except (httpx.UnsupportedProtocol, httpx.InvalidURL) as exc:
        return _err(
            502,
            f"tokenizer upstream URL invalid: {exc!s}",
            code="upstream_misconfigured",
        )
    except (
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.TimeoutException,
    ) as exc:
        return _err(
            502,
            f"tokenizer upstream unreachable: {type(exc).__name__}: {exc!s}",
            code="upstream_unreachable",
        )

    # Relay status + body verbatim (matches chat-completions
    # passthrough D.1.4 contract).
    try:
        upstream_body = resp.json()
    except ValueError:
        upstream_body = {"raw": resp.text}
    # F9: surface which upstream's tokenizer answered. Useful when the
    # default fallback (synthesizer for MoA, first-upstream for cascade)
    # picked one — clients counting tokens across calls can verify the
    # tokenizer is stable and matches their budget assumptions.
    response_headers = {"X-MetaModel-Tokenizer-Upstream": upstream_name}
    return JSONResponse(
        status_code=resp.status_code,
        content=upstream_body,
        headers=response_headers,
    )


# ── /v1/responses (F4-Responses) ────────────────────────────────────


@app.post("/v1/responses")
async def responses_endpoint(request: Request) -> Any:
    """OpenAI Responses API minimal adapter. F4-Responses.

    Translates Responses request shape → Chat Completions, runs
    dispatch, reshapes the synthesized response back into the
    Responses envelope. Routes through the SHARED resolver so
    unknown models surface the same typed 404 envelope as
    `/v1/chat/completions`.

    Streaming uses the documented Responses event taxonomy
    (`response.created` → `response.in_progress` →
    `response.output_item.added` → text/function-call deltas →
    `response.output_item.done` → `response.completed`).

    Honest scope: stateful chaining (`previous_response_id`),
    built-in tools, `background:true`, and most non-text input parts
    are typed-501 with explicit advisories. The minimal implementation
    keeps drop-in clients (OpenAI SDK ≥1.30 defaults to Responses)
    working without overpromising features we don't have.
    """
    cfg = get_config()
    err = empty_config_envelope(cfg)
    if err is not None:
        return JSONResponse(
            status_code=err.status_code,
            content=error_envelope(
                err.message,
                status=err.status_code,
                code=err.code,
                type_=err.type_,
            ),
        )

    try:
        body = await request.json()
    except ValueError:
        return _err(400, "request body is not valid JSON", code="invalid_json")
    if not isinstance(body, dict):
        return _err(400, "request body must be a JSON object", code="invalid_request")

    # Resolve up-front for typed 404 (matches /v1/completions + chat).
    try:
        resolve_profile(cfg, body.get("model"), None)
    except ProfileResolutionError as exc:
        return error_response(exc)

    # Convert Responses → Chat. Adapter raises typed 501/400 on
    # unsupported features.
    try:
        chat_body = responses_to_chat(body)
    except ResponsesAdapterError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_envelope(
                exc.message,
                status=exc.status_code,
                code=exc.code,
                param=exc.param,
                type_=exc.type_,
            ),
        )

    # Validate the rewritten body via ChatRequest (same path as
    # /v1/completions HIGH fix).
    try:
        chat_req = ChatRequest.model_validate(chat_body)
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            first = errors[0]
            param = _format_loc_path(tuple(first.get("loc", ())))
            message = first.get("msg", "Invalid request")
        else:
            param = None
            message = "Invalid request"
        return JSONResponse(
            status_code=400,
            content=error_envelope(message, status=400, param=param),
        )

    # Forward in-process to chat_completions. Build a Starlette
    # request with the rewritten body cached on `_body` so the
    # handler's `await request.json()` short-circuits.
    from starlette.requests import Request as StarletteRequest

    new_scope = dict(request.scope)
    new_scope["path"] = "/v1/chat/completions"
    new_request = StarletteRequest(new_scope, request._receive)
    new_request._body = (
        __import__("json").dumps(chat_body).encode("utf-8")  # noqa: PLE0501
    )
    chat_resp = await chat_completions(chat_req, new_request)

    # JSONResponse.body is the rendered bytes. Inspect status and
    # decode body for reshape. Streaming response shape differs —
    # client side hit `stream:true` so we re-emit the chunks.
    is_streaming = bool(body.get("stream"))

    # Errors and non-JSON pass-through unchanged (chat-side typed
    # errors are already OpenAI-shaped). The Responses spec doesn't
    # diverge here.
    if isinstance(chat_resp, StreamingResponse):
        # Chat dispatched in streaming mode — but responses_to_chat
        # forces stream=False. Reaching this branch means dispatch
        # internally returned StreamingResponse for an unrelated
        # reason. Pass through as-is.
        return chat_resp

    if not isinstance(chat_resp, JSONResponse):
        return chat_resp

    if chat_resp.status_code >= 400:
        return chat_resp

    import json

    chat_body_resp = json.loads(chat_resp.body)
    response_id = new_response_id()
    # F10: when `[server] model_name` is set, the Responses-API
    # response carries the brand instead of whatever model the client
    # asked for. Stays internally consistent with the chat-side
    # rewrite (chat_resp.body already has the brand in `model`).
    response_model = cfg.server.model_name or body.get("model", "")
    responses_body = chat_to_responses(
        chat_body_resp,
        response_id=response_id,
        model_name=response_model,
    )

    if not is_streaming:
        return JSONResponse(
            status_code=200,
            content=responses_body,
            headers={"X-MetaModel-Request-Id": response_id},
        )

    # Streaming: emit Responses-API SSE events.
    async def sse_generator():
        for ev in stream_responses_events(responses_body):
            data_json = json.dumps(ev["data"])
            yield f"event: {ev['event']}\ndata: {data_json}\n\n".encode("utf-8")

    return StreamingResponse(
        sse_generator(),
        status_code=200,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-MetaModel-Request-Id": response_id,
        },
    )


# ── CLI entry ────────────────────────────────────────────────────────


def main() -> None:
    """Run the server with uvicorn (used by `[project.scripts]`).

    Reads `[server] host / port / log_level` from `META_MODEL_CONFIG` so
    operators can flip log levels via the deployed TOML without code
    changes. Without this, the previously-hardcoded `log_level="info"`
    silently masked DEBUG logs (including the `moa.draft` / `moa.synth`
    observability lines added 2026-05-02).
    """
    import logging
    import os

    import uvicorn

    from .config import load_config

    cfg_path = os.environ.get("META_MODEL_CONFIG")
    log_level = "info"
    host = "127.0.0.1"
    port = 8400
    if cfg_path:
        try:
            cfg = load_config(cfg_path)
            log_level = cfg.server.log_level
            host = cfg.server.host
            port = cfg.server.port
        except Exception as exc:  # noqa: BLE001 — startup failure prints + falls through
            print(f"[meta-model] WARNING: failed to read {cfg_path}: {exc}; using defaults")

    # Configure the meta_model namespace logger explicitly. uvicorn
    # owns its own loggers (uvicorn / uvicorn.access) — we only set
    # ours so DEBUG observability lines emit when the operator has
    # set log_level=debug in the config. Without this the root logger
    # default (WARNING) silently swallows everything below INFO.
    py_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=py_level)
    logging.getLogger("meta_model").setLevel(py_level)

    uvicorn.run(
        "meta_model.server:app",
        host=host,
        port=port,
        log_level=log_level,
    )
