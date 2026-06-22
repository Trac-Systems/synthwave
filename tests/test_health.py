"""F5 — tests for upstream health probe + monitor + /v1/health 503."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from meta_model.config import UpstreamConfig
from meta_model.health import HealthMonitor, probe_upstream


# ── Helpers ──────────────────────────────────────────────────────────


def _upstream(
    *,
    base_url: str = "http://up.local/v1",
    api_key: str | None = None,
) -> UpstreamConfig:
    return UpstreamConfig(
        model_id="m",
        base_url=base_url,
        context=8192,
        max_output=512,
        api_key=api_key,
    )


def _ok_handler(req: httpx.Request) -> httpx.Response:
    """OpenAI-spec /v1/models response with our configured model_id
    in the catalog. Probe should classify ``up``."""
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {"id": "m", "object": "model", "created": 0, "owned_by": "me"},
            ],
        },
    )


def _models_handler_without_target(req: httpx.Request) -> httpx.Response:
    """200 OK from /models but the target model_id is not in the
    catalog — upstream is up but serving the wrong model. Probe
    should classify ``misconfigured`` so the operator notices the
    config mismatch instead of seeing it as a transient outage."""
    return httpx.Response(
        200,
        json={
            "object": "list",
            "data": [
                {"id": "different-model", "object": "model"},
            ],
        },
    )


def _models_handler_non_json(req: httpx.Request) -> httpx.Response:
    """200 OK with a non-JSON body. Some misconfigured reverse
    proxies fall back to HTML on /models — we should classify as
    `misconfigured` rather than `up`."""
    return httpx.Response(200, text="<html>...</html>")


def _models_handler_wrong_shape(req: httpx.Request) -> httpx.Response:
    """200 OK with parseable JSON but no `data` array. Same shape
    as ``misconfigured`` — the body shape doesn't match the OpenAI
    catalog contract."""
    return httpx.Response(200, json={"models": ["m"]})


def _status_handler(status: int):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "."}})

    return handler


def _connect_error_handler(req: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=req)


def _timeout_handler(req: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=req)


# ── Direct probe classifications ─────────────────────────────────────


@pytest.mark.asyncio
async def test_probe_classifies_2xx_as_up() -> None:
    transport = httpx.MockTransport(_ok_handler)
    h = await probe_upstream(_upstream(), timeout_secs=2.0, transport=transport)
    assert h.state == "up"
    assert h.reason == "up"


@pytest.mark.asyncio
async def test_probe_uses_get_to_models_path() -> None:
    """Readiness probe is decoupled from the inference queue: GET
    /v1/models, never POST /v1/chat/completions. A busy upstream
    should not look slow to the prober (root cause of false-down
    classification observed live 2026-05-04)."""
    seen: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["method"] = req.method
        seen["path"] = req.url.path
        return _ok_handler(req)

    transport = httpx.MockTransport(handler)
    h = await probe_upstream(_upstream(), timeout_secs=2.0, transport=transport)
    assert h.state == "up"
    assert seen["method"] == "GET"
    assert seen["path"].endswith("/models")


@pytest.mark.asyncio
async def test_probe_misconfigured_when_target_model_missing() -> None:
    """200 from /models but the configured model_id is not in the
    catalog → upstream is up but serving a different model. We
    classify as misconfigured so /v1/health surfaces the mismatch
    instead of optimistically reporting up."""
    transport = httpx.MockTransport(_models_handler_without_target)
    h = await probe_upstream(_upstream(), timeout_secs=2.0, transport=transport)
    assert h.state == "misconfigured"
    assert h.reason == "misconfigured"
    # Detail should name the missing model so logs are actionable.
    assert "not in catalog" in (h.last_error or "")


@pytest.mark.asyncio
async def test_probe_misconfigured_when_models_body_not_json() -> None:
    """200 with a non-JSON body. Some reverse proxies fall back to
    HTML on a wrong path — classify as misconfigured rather than
    up so the operator notices."""
    transport = httpx.MockTransport(_models_handler_non_json)
    h = await probe_upstream(_upstream(), timeout_secs=2.0, transport=transport)
    assert h.state == "misconfigured"


@pytest.mark.asyncio
async def test_probe_misconfigured_when_models_body_wrong_shape() -> None:
    """200 with parseable JSON but no `data` array (i.e., not the
    OpenAI catalog shape). Classify as misconfigured."""
    transport = httpx.MockTransport(_models_handler_wrong_shape)
    h = await probe_upstream(_upstream(), timeout_secs=2.0, transport=transport)
    assert h.state == "misconfigured"


@pytest.mark.asyncio
async def test_probe_misconfigured_when_auth_resolution_raises(monkeypatch) -> None:
    """Review r1 F5 HIGH: auth-header resolution can raise (e.g.,
    `api_key_env` configured but the env var is unset at runtime).
    The probe must honor its "Never raises" contract and classify
    that as `misconfigured` rather than letting it bubble up to the
    health endpoint or the background task supervisor."""
    from meta_model import health as health_mod

    def _boom(_upstream):
        raise RuntimeError("env var FAKE_KEY_ENV unset")

    monkeypatch.setattr(health_mod, "_build_auth_headers", _boom)

    transport = httpx.MockTransport(_ok_handler)
    h = await probe_upstream(_upstream(), timeout_secs=2.0, transport=transport)
    assert h.state == "misconfigured"
    assert "auth header resolution failed" in (h.last_error or "")
    assert "FAKE_KEY_ENV" in (h.last_error or "")


@pytest.mark.asyncio
async def test_probe_classifies_5xx_as_down() -> None:
    transport = httpx.MockTransport(_status_handler(503))
    h = await probe_upstream(_upstream(), timeout_secs=2.0, transport=transport)
    assert h.state == "down"
    assert h.reason == "down"


@pytest.mark.asyncio
async def test_probe_classifies_401_as_auth_failed() -> None:
    transport = httpx.MockTransport(_status_handler(401))
    h = await probe_upstream(_upstream(), timeout_secs=2.0, transport=transport)
    assert h.state == "auth_failed"
    assert h.reason == "auth_failed"


@pytest.mark.asyncio
async def test_probe_classifies_403_as_auth_failed() -> None:
    transport = httpx.MockTransport(_status_handler(403))
    h = await probe_upstream(_upstream(), timeout_secs=2.0, transport=transport)
    assert h.state == "auth_failed"


@pytest.mark.asyncio
async def test_probe_classifies_404_as_misconfigured() -> None:
    """Endpoint reachable but rejected the probe — most often a path
    typo on base_url, or model_id not loaded. Operator config issue."""
    transport = httpx.MockTransport(_status_handler(404))
    h = await probe_upstream(_upstream(), timeout_secs=2.0, transport=transport)
    assert h.state == "misconfigured"
    assert h.reason == "misconfigured"


@pytest.mark.asyncio
async def test_probe_classifies_connect_error_as_down() -> None:
    transport = httpx.MockTransport(_connect_error_handler)
    h = await probe_upstream(_upstream(), timeout_secs=2.0, transport=transport)
    assert h.state == "down"
    assert "ConnectError" in (h.last_error or "")


@pytest.mark.asyncio
async def test_probe_classifies_timeout_as_down() -> None:
    transport = httpx.MockTransport(_timeout_handler)
    h = await probe_upstream(_upstream(), timeout_secs=2.0, transport=transport)
    assert h.state == "down"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_url",
    [
        "::not-a-url",
        "/v1",  # scheme-less + relative
        "http://",  # scheme but no host
        "localhost:8000/v1",  # missing scheme — httpx parses but post fails
    ],
)
async def test_probe_classifies_malformed_url_as_misconfigured(bad_url: str) -> None:
    """``base_url`` shapes that lack scheme/host are operator typos.
    Review r1 F5 finding MED: explicit scheme+host gate ensures these
    classify as `misconfigured` rather than falling into the
    catch-all `down` branch."""
    h = await probe_upstream(
        _upstream(base_url=bad_url),
        timeout_secs=2.0,
        transport=None,
    )
    assert h.state == "misconfigured", f"{bad_url!r} → {h.state!r}: {h.last_error!r}"


# ── Monitor aggregation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_aggregate_all_up_returns_ok() -> None:
    monitor = HealthMonitor(
        upstreams={"a": _upstream(), "b": _upstream()},
        interval_sec=10.0,
        transport_provider=lambda: httpx.MockTransport(_ok_handler),
    )
    body = await monitor.aggregate()
    assert body["status"] == "ok"
    assert body["unhealthy_upstreams"] == []


@pytest.mark.asyncio
async def test_aggregate_one_down_marks_unhealthy() -> None:
    transports = {
        "a": httpx.MockTransport(_ok_handler),
        "b": httpx.MockTransport(_status_handler(503)),
    }
    # transport_provider is called per-probe and must select per
    # upstream — use a closure over a stack.
    current = {"name": "a"}

    async def aggregate_with_per_upstream_transports() -> dict:
        async def fake_probe(name: str):
            from meta_model.health import probe_upstream as real_probe

            return await real_probe(
                monitor.upstreams[name],
                timeout_secs=monitor.timeout_sec,
                transport=transports[name],
            )

        # Force ordered probing via _ensure_fresh per name.
        monitor._cache.clear()
        a = await fake_probe("a")
        b = await fake_probe("b")
        monitor._cache["a"] = a
        monitor._cache["b"] = b
        return await monitor.aggregate()

    monitor = HealthMonitor(
        upstreams={"a": _upstream(), "b": _upstream()},
        interval_sec=60.0,
    )
    body = await aggregate_with_per_upstream_transports()
    assert body["status"] == "unhealthy"
    names = {u["name"] for u in body["unhealthy_upstreams"]}
    assert names == {"b"}
    assert body["unhealthy_upstreams"][0]["reason"] == "down"


@pytest.mark.asyncio
async def test_aggregate_distinguishes_reasons() -> None:
    monitor = HealthMonitor(
        upstreams={
            "down_one": _upstream(),
            "auth_one": _upstream(),
            "config_one": _upstream(),
        },
        interval_sec=60.0,
    )
    # Pre-populate cache with three different states so aggregate
    # returns them as configured.
    now = time.monotonic()
    from meta_model.health import UpstreamHealth

    monitor._cache["down_one"] = UpstreamHealth("down", "down", now)
    monitor._cache["auth_one"] = UpstreamHealth("auth_failed", "auth_failed", now)
    monitor._cache["config_one"] = UpstreamHealth(
        "misconfigured", "misconfigured", now
    )
    body = await monitor.aggregate()
    assert body["status"] == "unhealthy"
    by_name = {u["name"]: u["reason"] for u in body["unhealthy_upstreams"]}
    assert by_name == {
        "down_one": "down",
        "auth_one": "auth_failed",
        "config_one": "misconfigured",
    }


@pytest.mark.asyncio
async def test_aggregate_empty_upstreams_is_unhealthy() -> None:
    """No upstreams configured → unhealthy. Review r1 F5 finding HIGH:
    a server with no upstreams cannot serve `/v1/chat/completions`,
    so reporting "ok" would lie."""
    monitor = HealthMonitor(upstreams={}, interval_sec=10.0)
    body = await monitor.aggregate()
    assert body["status"] == "unhealthy"
    assert body["reason"] == "no_upstreams_configured"
    assert body["unhealthy_upstreams"] == []


# ── Cache + coalescing ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cache_serves_subsequent_calls_within_interval() -> None:
    """Cache hit during the configured interval must NOT reprobe."""
    counter = {"n": 0}

    def counting_handler(req: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return _ok_handler(req)

    monitor = HealthMonitor(
        upstreams={"a": _upstream()},
        interval_sec=60.0,  # generous TTL
        transport_provider=lambda: httpx.MockTransport(counting_handler),
    )
    await monitor.aggregate()
    await monitor.aggregate()
    await monitor.aggregate()
    assert counter["n"] == 1, "cache should serve after the first probe"


@pytest.mark.asyncio
async def test_stale_cache_triggers_out_of_band_probe() -> None:
    """Cache older than `interval_sec` → next aggregate triggers a
    fresh probe via the same coalescing primitive (no background
    loop running in this test)."""
    counter = {"n": 0}

    def counting_handler(req: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        return _ok_handler(req)

    monitor = HealthMonitor(
        upstreams={"a": _upstream()},
        interval_sec=0.05,
        transport_provider=lambda: httpx.MockTransport(counting_handler),
    )
    await monitor.aggregate()
    assert counter["n"] == 1
    # Force the cache entry to look stale by rewinding probed_at.
    monitor._cache["a"].probed_at_monotonic -= 100.0
    await monitor.aggregate()
    assert counter["n"] == 2


@pytest.mark.asyncio
async def test_inflight_coalesces_concurrent_callers() -> None:
    """Multiple concurrent stale-cache requests must share one
    probe via _inflight."""
    counter = {"n": 0}
    release = asyncio.Event()

    async def slow_handler(req: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        await release.wait()
        return _ok_handler(req)

    transport = httpx.MockTransport(slow_handler)
    monitor = HealthMonitor(
        upstreams={"a": _upstream()},
        interval_sec=60.0,
        transport_provider=lambda: transport,
    )
    # Three concurrent aggregates all see empty cache → all enter
    # _ensure_fresh → first creates a probe Task, others await it.
    pending = [asyncio.create_task(monitor.aggregate()) for _ in range(3)]
    await asyncio.sleep(0.05)  # let the probe fire and others queue up
    release.set()
    results = await asyncio.gather(*pending)
    assert counter["n"] == 1, "concurrent stale-cache calls must share one probe"
    for body in results:
        assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_recovery_flips_status_when_probe_succeeds() -> None:
    """Failed → recovered upstream: cache entry replaced on next probe."""
    state = {"healthy": False}

    def toggling_handler(req: httpx.Request) -> httpx.Response:
        if state["healthy"]:
            return _ok_handler(req)
        return _status_handler(503)(req)

    monitor = HealthMonitor(
        upstreams={"a": _upstream()},
        interval_sec=0.01,
        transport_provider=lambda: httpx.MockTransport(toggling_handler),
    )
    body = await monitor.aggregate()
    assert body["status"] == "unhealthy"
    state["healthy"] = True
    monitor._cache["a"].probed_at_monotonic -= 100.0  # force stale
    body = await monitor.aggregate()
    assert body["status"] == "ok"


# ── Background loop ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_background_loop_probes_each_interval() -> None:
    """Loop fires once per upstream per interval. Multiple ticks with
    a short interval must produce >1 probes; a single tick produces
    exactly N probes for N upstreams."""
    counter = {"n": 0}
    cond = asyncio.Event()

    def counting_handler(req: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        if counter["n"] >= 4:
            cond.set()
        return _ok_handler(req)

    monitor = HealthMonitor(
        upstreams={"a": _upstream(), "b": _upstream()},
        interval_sec=0.02,
        transport_provider=lambda: httpx.MockTransport(counting_handler),
    )
    await monitor.start()
    try:
        await asyncio.wait_for(cond.wait(), timeout=2.0)
    finally:
        await monitor.stop()
    assert counter["n"] >= 4, "loop should have ticked at least twice (2 upstreams × 2 ticks)"


@pytest.mark.asyncio
async def test_stop_cancels_inflight_probes() -> None:
    """stop() must cancel any in-flight probes so process shutdown
    isn't blocked by a slow upstream."""
    release = asyncio.Event()

    async def slow_handler(req: httpx.Request) -> httpx.Response:
        await release.wait()
        return _ok_handler(req)

    monitor = HealthMonitor(
        upstreams={"a": _upstream()},
        interval_sec=0.01,
        transport_provider=lambda: httpx.MockTransport(slow_handler),
    )
    await monitor.start()
    await asyncio.sleep(0.05)  # let the loop kick off a probe
    await asyncio.wait_for(monitor.stop(), timeout=2.0)
    # If we reached here without timeout, stop() handled cancellation.
    assert not monitor.is_running()


# ── /v1/health endpoint integration ─────────────────────────────────


def test_endpoint_returns_503_when_no_monitor_configured() -> None:
    """F5 review r1 HIGH: with no HealthMonitor installed (lifespan
    not run, or empty config), readiness must report unhealthy.
    Reporting "ok" would lie about the server's ability to serve
    chat completions."""
    from meta_model import __version__
    from meta_model.server import app, set_config

    if hasattr(app.state, "upstream_health"):
        app.state.upstream_health = None
    set_config(None)
    client = TestClient(app)
    r = client.get("/v1/health")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unhealthy"
    assert body["reason"] == "no_monitor_configured"
    assert body["unhealthy_upstreams"] == []
    assert body["version"] == __version__


def test_endpoint_returns_503_when_any_upstream_unhealthy() -> None:
    from meta_model import __version__
    from meta_model.server import app

    # Build a monitor with a pre-populated unhealthy cache entry.
    monitor = HealthMonitor(upstreams={"a": _upstream()}, interval_sec=60.0)
    from meta_model.health import UpstreamHealth

    monitor._cache["a"] = UpstreamHealth(
        "down", "down", time.monotonic()
    )
    app.state.upstream_health = monitor
    try:
        client = TestClient(app)
        r = client.get("/v1/health")
        assert r.status_code == 503
        body = r.json()
        assert body["status"] == "unhealthy"
        assert body["unhealthy_upstreams"] == [{"name": "a", "reason": "down"}]
        assert body["version"] == __version__
    finally:
        app.state.upstream_health = None


def test_endpoint_returns_200_when_all_upstreams_up() -> None:
    from meta_model.server import app

    monitor = HealthMonitor(upstreams={"a": _upstream()}, interval_sec=60.0)
    from meta_model.health import UpstreamHealth

    monitor._cache["a"] = UpstreamHealth("up", "up", time.monotonic())
    app.state.upstream_health = monitor
    try:
        client = TestClient(app)
        r = client.get("/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["unhealthy_upstreams"] == []
    finally:
        app.state.upstream_health = None


def test_endpoint_post_method_returns_405() -> None:
    """Method-not-allowed regression check (matches existing test_server.py:156
    expectation). Health is GET-only."""
    from meta_model.server import app

    if hasattr(app.state, "upstream_health"):
        app.state.upstream_health = None
    client = TestClient(app)
    r = client.post("/v1/health")
    assert r.status_code == 405
