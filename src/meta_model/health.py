"""F5 — upstream health probe with background loop + coalesced cache.

Drop-in semantic: `/v1/health` returns HTTP 503 when ANY configured
upstream fails its probe. The body lists which upstreams are down
and why ("down", "auth_failed", "misconfigured"). HTTP 200 only
when every upstream answers the readiness probe AND has the
configured ``model_id`` loaded.

The probe is **`GET /v1/models`**. Reasons:
- It is decoupled from the inference queue — vLLM serves it from
  the API thread without waiting on a free model slot, so a busy
  upstream does not look slow to the prober. The earlier
  `POST /v1/chat/completions max_tokens=1` probe was load-coupled:
  under traffic it queued behind real requests and tripped the
  default 5 s deadline, classifying a healthy-but-busy upstream as
  ``down``.
- Verifying the model is **loaded** (presence in ``data[*].id``)
  catches the failure mode where the upstream is up but serving a
  different model than the one our config addresses — that case
  used to surface as a 4xx on chat with a confusing reason; now it
  is `misconfigured` directly from the readiness probe.
- It matches the readiness convention every other OpenAI-compatible
  client uses (LiteLLM, OpenAI Python SDK warm-up, vLLM router).

Cheap enough to run on a 15 s background tick — NOT one probe per
inbound health request.

Single timing knob — `interval_sec` is BOTH the probe interval and
the cache TTL. The cache is fresh for the entire interval between
ticks. If the background loop is delayed (startup, GIL pressure,
etc.) and a request arrives stale, the request triggers an
out-of-band probe via the same coalescing primitive — multiple
concurrent stale-cache requests share one probe.

Concurrency model:
- One `asyncio.Task` per upstream max (`_inflight: dict`). When a
  probe is in flight, additional callers `await` the same Task
  rather than launching a duplicate.
- The background loop kicks off probes for all upstreams on every
  tick; if a previous tick is still running for some upstream, the
  loop falls through (the in-flight check in `_ensure_fresh` is the
  same coalescing primitive the loop and request paths share).

Test injection: pass an `httpx.AsyncBaseTransport` via the
`transport` argument so tests can use `httpx.MockTransport` without
hitting the real network. Production passes None and httpx uses its
default transport.

Deploy note: configure `/v1/health` as the **readiness** probe, not
liveness. Repeated 503s on liveness would make k8s/systemd restart
the pod when the upstream (not the pod) is the problem.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import httpx

from .config import UpstreamConfig
from .upstream import _build_auth_headers

# OpenAI-spec catalog endpoint. vLLM, proxy-fronted vLLM, hosted APIs
# all expose this and answer in milliseconds without queueing behind
# inference. Probing here decouples readiness latency from load.
_MODELS_PATH = "/models"

logger = logging.getLogger(__name__)


HealthState = Literal["up", "down", "auth_failed", "misconfigured"]
"""Probe outcome.

- ``up`` — upstream answered with HTTP 2xx.
- ``down`` — connection refused / timeout / 5xx response.
- ``auth_failed`` — 401/403 (credentials issue, not config typo).
- ``misconfigured`` — DNS unresolvable, malformed URL, 4xx other
  than auth (e.g. wrong endpoint path, model not found).
"""


@dataclass
class UpstreamHealth:
    """Cached probe result for one upstream."""

    state: HealthState
    reason: str
    """Human-readable reason. Equal to `state` for non-``up`` outcomes
    so `/v1/health` body is self-describing."""
    probed_at_monotonic: float
    """``time.monotonic()`` at the moment the probe completed.
    Compared against the configured interval to detect stale cache."""
    last_error: str | None = None
    """Free-form detail (status code, exception class). Logged but
    NOT exposed in the public body — the body's `reason` is the
    typed enum value. Operators read the daemon log for specifics."""


async def probe_upstream(
    upstream: UpstreamConfig,
    *,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> UpstreamHealth:
    """Issue ``GET {base_url}/models`` against the upstream and
    classify the outcome.

    Never raises — every error is mapped to a typed `HealthState`.
    Callers (the monitor) just record the result.

    Classification:
      - 200 + body lists ``upstream.model_id`` in ``data[*].id`` → ``up``
      - 200 + body parses but model not in catalog        → ``misconfigured``
      - 200 + body unparsable / wrong shape               → ``misconfigured``
      - 401 / 403                                         → ``auth_failed``
      - other 4xx                                         → ``misconfigured``
      - 5xx                                               → ``down``
      - transport / DNS / timeout                         → ``down``
      - malformed ``base_url`` / auth-resolution failure  → ``misconfigured``

    Auth-policy invariant: this probe assumes the upstream applies
    the same auth policy to ``/v1/models`` as it does to
    ``/v1/chat/completions`` (vLLM with ``--api-key`` does; OpenAI
    proper does). A reverse proxy that leaves ``/models`` public
    while protecting chat would let a credential-misconfiguration
    pass readiness silently — the operator is responsible for
    keeping those policies aligned. Review r1 F5 MED.
    """
    # Review r1 F5 HIGH: auth-header resolution can raise (e.g.,
    # ``api_key_env`` / ``basic_auth_pass_env`` is configured but
    # the env var isn't set at runtime). Wrap so the function
    # honors its "Never raises" contract.
    try:
        if upstream.protocol == "anthropic":
            # Anthropic rejects Bearer auth; its catalog probe needs
            # x-api-key + anthropic-version (same scheme the adapter uses).
            from .providers.anthropic import build_auth_headers as _anthropic_auth_headers

            headers = _anthropic_auth_headers(upstream)
        else:
            headers = _build_auth_headers(upstream)
    except Exception as exc:  # noqa: BLE001 — defense in depth
        return UpstreamHealth(
            state="misconfigured",
            reason="misconfigured",
            probed_at_monotonic=time.monotonic(),
            last_error=f"auth header resolution failed: {type(exc).__name__}: {exc!s}",
        )
    # Build URL via httpx so malformed base_url raises at construction
    # time and gets classified as misconfigured rather than 500ing the
    # health endpoint itself. `httpx.URL()` is lenient — it accepts
    # `"localhost:8000/v1"`, `"http://"`, and `"/v1"` without raising.
    # Operator typos like those land at request time as
    # `httpx.UnsupportedProtocol` or `httpx.InvalidURL`. Review r1 F5
    # MED finding: gate on scheme + host explicitly so the typed
    # outcome is `misconfigured`, not `down`.
    try:
        base = upstream.base_url.rstrip("/")
        url_str = base + _MODELS_PATH
        url = httpx.URL(url_str)
    except (httpx.InvalidURL, ValueError) as exc:
        return UpstreamHealth(
            state="misconfigured",
            reason="misconfigured",
            probed_at_monotonic=time.monotonic(),
            last_error=f"invalid base_url: {exc!s}",
        )
    if url.scheme not in ("http", "https") or not url.host:
        return UpstreamHealth(
            state="misconfigured",
            reason="misconfigured",
            probed_at_monotonic=time.monotonic(),
            last_error=(
                f"invalid base_url scheme/host: scheme={url.scheme!r}, "
                f"host={url.host!r}"
            ),
        )

    try:
        async with httpx.AsyncClient(transport=transport, timeout=timeout_secs) as client:
            resp = await client.get(url, headers=headers)
    except (httpx.UnsupportedProtocol, httpx.InvalidURL) as exc:
        return UpstreamHealth(
            state="misconfigured",
            reason="misconfigured",
            probed_at_monotonic=time.monotonic(),
            last_error=f"{type(exc).__name__}: {exc!s}",
        )
    except (httpx.ConnectError, httpx.ReadError, httpx.WriteError) as exc:
        # DNS unresolvable looks like ConnectError("nodename nor servname
        # provided"); accept the conservative classification rather than
        # parsing strerror. ConnectError is "could not reach the host"
        # → down (best-effort). Truly malformed URLs are caught above by
        # the scheme/host gate or UnsupportedProtocol.
        return UpstreamHealth(
            state="down",
            reason="down",
            probed_at_monotonic=time.monotonic(),
            last_error=f"{type(exc).__name__}: {exc!s}",
        )
    except httpx.TimeoutException as exc:
        return UpstreamHealth(
            state="down",
            reason="down",
            probed_at_monotonic=time.monotonic(),
            last_error=f"timeout: {exc!s}",
        )
    except Exception as exc:  # noqa: BLE001 — defense in depth, never crash the loop
        logger.warning("health probe %r unexpected error: %r", upstream.model_id, exc)
        return UpstreamHealth(
            state="down",
            reason="down",
            probed_at_monotonic=time.monotonic(),
            last_error=f"{type(exc).__name__}: {exc!s}",
        )

    now = time.monotonic()
    if resp.status_code in (401, 403):
        return UpstreamHealth(
            state="auth_failed",
            reason="auth_failed",
            probed_at_monotonic=now,
            last_error=f"http {resp.status_code}",
        )
    if 500 <= resp.status_code < 600:
        return UpstreamHealth(
            state="down",
            reason="down",
            probed_at_monotonic=now,
            last_error=f"http {resp.status_code}",
        )
    if not (200 <= resp.status_code < 300):
        # Other 4xx (404, 405, 400, 422, etc.) — endpoint reachable but
        # rejected the probe. Most often: wrong path on base_url. Operator
        # config issue.
        return UpstreamHealth(
            state="misconfigured",
            reason="misconfigured",
            probed_at_monotonic=now,
            last_error=f"http {resp.status_code}",
        )

    # 2xx — verify the model we route to is actually loaded.
    # Review r1 F5 LOW: keep the actual status code in `last_error`
    # so 204/206-via-proxy diagnostics aren't misleadingly stamped
    # as "http 200".
    sc = resp.status_code
    try:
        payload = resp.json()
    except ValueError:
        return UpstreamHealth(
            state="misconfigured",
            reason="misconfigured",
            probed_at_monotonic=now,
            last_error=f"http {sc} but body was not JSON",
        )
    if not isinstance(payload, dict):
        return UpstreamHealth(
            state="misconfigured",
            reason="misconfigured",
            probed_at_monotonic=now,
            last_error=f"http {sc} but body was not a JSON object",
        )
    data = payload.get("data")
    if not isinstance(data, list):
        return UpstreamHealth(
            state="misconfigured",
            reason="misconfigured",
            probed_at_monotonic=now,
            last_error=f"http {sc} but `data` field was missing or not a list",
        )
    advertised_ids = {
        entry.get("id")
        for entry in data
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    if upstream.model_id not in advertised_ids:
        return UpstreamHealth(
            state="misconfigured",
            reason="misconfigured",
            probed_at_monotonic=now,
            last_error=(
                f"model {upstream.model_id!r} not in catalog "
                f"(advertised: {sorted(s for s in advertised_ids if isinstance(s, str))!r})"
            ),
        )
    return UpstreamHealth(state="up", reason="up", probed_at_monotonic=now)


@dataclass
class HealthMonitor:
    """Owns probe state for one server instance.

    Lifecycle: `start()` launches the background loop, `stop()`
    cancels it cleanly. Both are idempotent.

    Probes are coalesced per-upstream: at most one `asyncio.Task`
    per upstream is in flight at any moment. Background ticks AND
    request-driven stale-cache fallbacks share the same `_inflight`
    dict, so they cannot fire duplicate probes against each other.
    """

    upstreams: dict[str, UpstreamConfig]
    interval_sec: float = 15.0
    timeout_sec: float = 5.0
    transport_provider: object | None = None
    """Callable returning an `httpx.AsyncBaseTransport | None` per
    probe. Tests use this to inject `httpx.MockTransport`. None
    means "use httpx default" (production)."""

    _cache: dict[str, UpstreamHealth] = field(default_factory=dict)
    _inflight: dict[str, asyncio.Task] = field(default_factory=dict)
    _loop_task: asyncio.Task | None = None
    _stop_event: asyncio.Event = field(default_factory=asyncio.Event)

    def is_running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    async def start(self) -> None:
        """Idempotent. Background loop runs until `stop()` or process exit."""
        if self.is_running():
            return
        self._stop_event.clear()
        self._loop_task = asyncio.create_task(
            self._run_loop(), name="meta-model.health-loop"
        )

    async def stop(self) -> None:
        """Cancel the background loop and any in-flight probes. Idempotent."""
        self._stop_event.set()
        loop_task = self._loop_task
        self._loop_task = None
        if loop_task is not None and not loop_task.done():
            loop_task.cancel()
            try:
                await loop_task
            except (asyncio.CancelledError, Exception):
                pass
        # Cancel any straggler probes.
        for task in list(self._inflight.values()):
            task.cancel()
        for task in list(self._inflight.values()):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._inflight.clear()

    async def aggregate(self) -> dict[str, object]:
        """Build the `/v1/health` response body.

        Triggers stale-cache probes via `_ensure_fresh` so a
        late-arriving request doesn't miss a recovery (or a fresh
        failure) by `interval_sec`. Concurrent calls share probes
        per-upstream via `_inflight`.

        Empty upstreams catalog → `status="unhealthy"` with a
        top-level ``reason="no_upstreams_configured"``. Readiness
        for a server with no upstreams cannot serve client requests
        (every `/v1/chat/completions` would 404 on model lookup or
        return an empty model list); reporting "ok" would lie.
        Review r1 F5 finding HIGH.
        """
        version = __import__("meta_model").__version__
        if not self.upstreams:
            return {
                "status": "unhealthy",
                "reason": "no_upstreams_configured",
                "unhealthy_upstreams": [],
                "version": version,
            }
        names = sorted(self.upstreams.keys())
        results = await asyncio.gather(
            *(self._ensure_fresh(name) for name in names),
            return_exceptions=False,
        )
        unhealthy = [
            {"name": name, "reason": r.reason}
            for name, r in zip(names, results, strict=True)
            if r.state != "up"
        ]
        return {
            "status": "ok" if not unhealthy else "unhealthy",
            "unhealthy_upstreams": unhealthy,
            "version": version,
        }

    async def _ensure_fresh(self, name: str) -> UpstreamHealth:
        cached = self._cache.get(name)
        if cached is not None and self._is_fresh(cached):
            return cached
        existing = self._inflight.get(name)
        if existing is not None:
            return await existing
        task = asyncio.create_task(
            self._probe_with_cleanup(name),
            name=f"meta-model.health-probe.{name}",
        )
        self._inflight[name] = task
        return await task

    def _is_fresh(self, entry: UpstreamHealth) -> bool:
        return (time.monotonic() - entry.probed_at_monotonic) < self.interval_sec

    async def _probe_with_cleanup(self, name: str) -> UpstreamHealth:
        try:
            transport = self._get_transport()
            result = await probe_upstream(
                self.upstreams[name],
                timeout_secs=self.timeout_sec,
                transport=transport,
            )
            # Review r1 F5 MED: surface non-up probe results in the
            # daemon log so operators can see WHY a probe classified
            # the way it did. /v1/health response body only carries
            # the typed reason; details (model_id missing, non-JSON
            # body, http status code, exception class) live in
            # `last_error` and would otherwise be invisible.
            # Log on every non-up outcome AND on transitions (so a
            # recovery from down → up announces itself), but quiet
            # on the steady-up case.
            prior = self._cache.get(name)
            if result.state != "up":
                logger.warning(
                    "health probe %s state=%s reason=%s detail=%s",
                    name,
                    result.state,
                    result.reason,
                    result.last_error or "-",
                )
            elif prior is not None and prior.state != "up":
                logger.info(
                    "health probe %s recovered: %s → up", name, prior.state
                )
            self._cache[name] = result
            return result
        finally:
            self._inflight.pop(name, None)

    def _get_transport(self) -> httpx.AsyncBaseTransport | None:
        if self.transport_provider is None:
            return None
        return self.transport_provider()  # type: ignore[operator]

    async def _run_loop(self) -> None:
        """Tick every `interval_sec`, kick off a probe per upstream.

        Coalesces with request-driven probes via `_inflight`. If a
        previous tick's probe is still running (slow upstream), the
        new tick skips it — the slow probe's result will land in
        cache when it finishes.
        """
        while not self._stop_event.is_set():
            for name in self.upstreams.keys():
                if self._stop_event.is_set():
                    return
                if name in self._inflight:
                    continue
                task = asyncio.create_task(
                    self._probe_with_cleanup(name),
                    name=f"meta-model.health-tick.{name}",
                )
                self._inflight[name] = task
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.interval_sec
                )
            except asyncio.TimeoutError:
                pass


__all__ = [
    "HealthMonitor",
    "HealthState",
    "UpstreamHealth",
    "probe_upstream",
]
