"""D.2.1 tests — generator fan-out primitive."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from meta_model.config import UpstreamConfig
from meta_model.moa.fanout import (
    GeneratorFailure,
    GeneratorSuccess,
    failures,
    fan_out,
    quorum_threshold,
    successes,
)


def _up(name: str, port: int) -> UpstreamConfig:
    return UpstreamConfig(
        model_id=f"{name}-model",
        base_url=f"http://upstream-{name}:{port}/v1",
        context=8192,
        max_output=2048,
    )


# ── Happy path ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fan_out_all_succeed() -> None:
    handler_calls: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        handler_calls[host] = handler_calls.get(host, 0) + 1
        return httpx.Response(
            200,
            json={"id": f"ok-{host}", "choices": [{"message": {"content": host}}]},
        )

    transport = httpx.MockTransport(handler)
    outcomes = await fan_out(
        [("a", _up("a", 9000)), ("b", _up("b", 9001)), ("c", _up("c", 9002))],
        {"messages": []},
        per_upstream_timeout_secs=2.0,
        transport=transport,
    )

    assert len(outcomes) == 3
    assert all(isinstance(o, GeneratorSuccess) for o in outcomes)
    # Order preserved
    assert [o.upstream_name for o in outcomes] == ["a", "b", "c"]
    # Each upstream was contacted exactly once
    assert handler_calls == {"upstream-a": 1, "upstream-b": 1, "upstream-c": 1}


@pytest.mark.asyncio
async def test_fan_out_empty_list_returns_empty() -> None:
    outcomes = await fan_out([], {"messages": []}, per_upstream_timeout_secs=1.0)
    assert outcomes == []


# ── Failure modes ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fan_out_one_5xx_recorded_as_non_2xx_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "upstream-b":
            return httpx.Response(503, json={"error": "down"})
        return httpx.Response(200, json={"choices": []})

    transport = httpx.MockTransport(handler)
    outcomes = await fan_out(
        [("a", _up("a", 9000)), ("b", _up("b", 9001)), ("c", _up("c", 9002))],
        {"messages": []},
        per_upstream_timeout_secs=2.0,
        transport=transport,
    )

    assert len(successes(outcomes)) == 2
    fails = failures(outcomes)
    assert len(fails) == 1
    assert fails[0].upstream_name == "b"
    assert fails[0].reason == "non_2xx"
    assert fails[0].status == 503


@pytest.mark.asyncio
async def test_fan_out_non_2xx_captures_body_in_detail() -> None:
    """Without the upstream's response body it's impossible to tell why
    a non_2xx happened (vLLM's `BadRequestError` envelope only lives in
    the body, not the status code). The body snippet must flow into
    `GeneratorFailure.detail` so dispatch + relay paths can surface it.
    """
    err_body = (
        '{"object":"error","message":"messages must end with user role",'
        '"type":"BadRequestError","param":"messages","code":400}'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=err_body)

    transport = httpx.MockTransport(handler)
    outcomes = await fan_out(
        [("a", _up("a", 9000))],
        {"messages": []},
        per_upstream_timeout_secs=2.0,
        transport=transport,
    )

    fails = failures(outcomes)
    assert len(fails) == 1
    assert fails[0].reason == "non_2xx"
    assert fails[0].status == 400
    # The full body fits inside the 512-char detail cap.
    assert err_body in fails[0].detail
    assert fails[0].detail.startswith("upstream returned HTTP 400: ")


@pytest.mark.asyncio
async def test_fan_out_non_2xx_detail_truncates_long_body() -> None:
    """Misbehaving upstreams returning megabytes of HTML must not
    dominate telemetry. Detail caps at 512 chars + a single `…` marker.
    """
    long_body = "X" * 4000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=long_body)

    transport = httpx.MockTransport(handler)
    outcomes = await fan_out(
        [("a", _up("a", 9000))],
        {"messages": []},
        per_upstream_timeout_secs=2.0,
        transport=transport,
    )

    fails = failures(outcomes)
    assert len(fails) == 1
    # Detail = "upstream returned HTTP 400: " (28 chars) + 512 X's + "…"
    assert fails[0].detail.endswith("…")
    assert fails[0].detail.count("X") == 512


@pytest.mark.asyncio
async def test_fan_out_non_json_recorded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "upstream-a":
            return httpx.Response(200, content=b"<html>oops</html>")
        return httpx.Response(200, json={"choices": []})

    transport = httpx.MockTransport(handler)
    outcomes = await fan_out(
        [("a", _up("a", 9000)), ("b", _up("b", 9001))],
        {"messages": []},
        per_upstream_timeout_secs=2.0,
        transport=transport,
    )

    fails = failures(outcomes)
    assert len(fails) == 1
    assert fails[0].reason == "non_json"
    assert isinstance(outcomes[1], GeneratorSuccess)


@pytest.mark.asyncio
async def test_fan_out_transport_error_recorded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "upstream-b":
            raise httpx.ConnectError("dial failed", request=request)
        return httpx.Response(200, json={"choices": []})

    transport = httpx.MockTransport(handler)
    outcomes = await fan_out(
        [("a", _up("a", 9000)), ("b", _up("b", 9001)), ("c", _up("c", 9002))],
        {"messages": []},
        per_upstream_timeout_secs=2.0,
        transport=transport,
    )

    fails = failures(outcomes)
    assert len(fails) == 1
    assert fails[0].upstream_name == "b"
    assert fails[0].reason == "transport"
    assert "ConnectError" in fails[0].detail


@pytest.mark.asyncio
async def test_fan_out_timeout_recorded() -> None:
    """Slow upstream past per-upstream timeout → timeout failure,
    siblings unaffected."""

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "upstream-slow":
            await asyncio.sleep(2.0)
        return httpx.Response(200, json={"choices": []})

    transport = httpx.MockTransport(slow_handler)
    outcomes = await fan_out(
        [("fast", _up("fast", 9000)), ("slow", _up("slow", 9001))],
        {"messages": []},
        per_upstream_timeout_secs=0.2,
        transport=transport,
    )

    assert isinstance(outcomes[0], GeneratorSuccess)
    assert isinstance(outcomes[1], GeneratorFailure)
    assert outcomes[1].reason == "timeout"
    # Hard cap held — outcome elapsed should be ≤ ~600ms (timeout 200 + slack)
    assert outcomes[1].elapsed_ms < 600


# ── Concurrency proof ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fan_out_runs_in_parallel() -> None:
    """All upstreams should run concurrently. We prove this via an
    overlap counter inside the handler rather than a wall-clock bound
    (review r11: timing-based proof flakes under loaded CI)."""
    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def overlap_handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        async with lock:
            active += 1
            max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        async with lock:
            active -= 1
        return httpx.Response(200, json={"choices": []})

    transport = httpx.MockTransport(overlap_handler)
    outcomes = await fan_out(
        [("a", _up("a", 9000)), ("b", _up("b", 9001)), ("c", _up("c", 9002))],
        {"messages": []},
        per_upstream_timeout_secs=2.0,
        transport=transport,
    )

    assert len(successes(outcomes)) == 3
    # If sequential, max_active would be 1. Parallel = 3.
    assert max_active == 3, f"upstreams ran sequentially (max_active={max_active})"


# ── JSON shape contract ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fan_out_non_object_json_rejected_as_non_json() -> None:
    """200 with a JSON array/scalar is a protocol failure — D.2.2
    expects choices[0] from a dict (review r11)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "upstream-a":
            return httpx.Response(200, json=["unexpected", "array"])
        return httpx.Response(200, json={"choices": []})

    transport = httpx.MockTransport(handler)
    outcomes = await fan_out(
        [("a", _up("a", 9000)), ("b", _up("b", 9001))],
        {"messages": []},
        per_upstream_timeout_secs=2.0,
        transport=transport,
    )

    fails = failures(outcomes)
    assert len(fails) == 1
    assert fails[0].upstream_name == "a"
    assert fails[0].reason == "non_json"
    assert "non-object" in fails[0].detail.lower()


# ── Quorum-based early exit ────────────────────────────────────────


def test_quorum_threshold_formula() -> None:
    """Quorum is `ceil(n * 2/3)`, floored at 1, capped at n."""
    assert quorum_threshold(0) == 0  # defensive — caller shouldn't invoke with empty
    assert quorum_threshold(1) == 1
    assert quorum_threshold(2) == 2  # n<=2: no early-exit benefit
    assert quorum_threshold(3) == 2  # 3 gens → 2 needed (load-bearing for our setup)
    assert quorum_threshold(4) == 3
    assert quorum_threshold(5) == 4
    assert quorum_threshold(6) == 4


@pytest.mark.asyncio
async def test_fan_out_cancels_pending_once_quorum_met() -> None:
    """Once `quorum` successes land, pending generators are cancelled
    and reported as `GeneratorFailure(reason='cancelled')`. The fast
    pair return immediately; the slow third hangs and gets cut off
    after fanout returns.
    """
    cancel_log: dict[str, bool] = {}

    async def slow_handler_factory():
        async def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            try:
                if host == "upstream-slow":
                    # Hang ~2s — long enough for fast generators to finish
                    # and quorum cutoff to fire.
                    await asyncio.sleep(2.0)
                    return httpx.Response(200, json={"choices": [{"message": {"content": host}}]})
                # Fast generators
                return httpx.Response(200, json={"choices": [{"message": {"content": host}}]})
            except asyncio.CancelledError:
                cancel_log[host] = True
                raise

        return handler

    handler = await slow_handler_factory()
    transport = httpx.MockTransport(handler)

    outcomes = await fan_out(
        [
            ("a", _up("a", 9000)),
            ("b", _up("b", 9001)),
            ("slow", _up("slow", 9002)),
        ],
        {"messages": []},
        per_upstream_timeout_secs=10.0,  # generous so timeout doesn't fire instead
        transport=transport,
        quorum=2,
        grace_secs=0.0,  # skip grace window — cancel immediately on quorum
    )

    assert len(outcomes) == 3
    # Order preserved
    assert [o.upstream_name for o in outcomes] == ["a", "b", "slow"]
    # Two fast successes
    assert isinstance(outcomes[0], GeneratorSuccess)
    assert isinstance(outcomes[1], GeneratorSuccess)
    # Third was cancelled
    assert isinstance(outcomes[2], GeneratorFailure)
    assert outcomes[2].reason == "cancelled"
    assert "quorum" in outcomes[2].detail.lower()


@pytest.mark.asyncio
async def test_fan_out_no_quorum_cutoff_when_quorum_zero() -> None:
    """Passing `quorum=0` (or N+1) waits for all generators (legacy
    behavior). Useful for dispatch sites that haven't been migrated
    yet, and as a safety release."""

    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "upstream-slow":
            await asyncio.sleep(0.05)
        return httpx.Response(200, json={"choices": [{"message": {"content": host}}]})

    transport = httpx.MockTransport(handler)
    outcomes = await fan_out(
        [("a", _up("a", 9000)), ("slow", _up("slow", 9001))],
        {"messages": []},
        per_upstream_timeout_secs=2.0,
        transport=transport,
        quorum=0,
    )

    # Both succeeded — no cancellations
    assert len(outcomes) == 2
    assert all(isinstance(o, GeneratorSuccess) for o in outcomes)


@pytest.mark.asyncio
async def test_fan_out_proceeds_when_quorum_unreachable() -> None:
    """If only N-K generators can succeed (rest fail), fanout still
    returns once it's clear quorum can't be met. All outcomes are
    reported (no cancellation), since cancellation would lose info
    that callers need to diagnose the degradation."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host in ("upstream-fail-a", "upstream-fail-b"):
            return httpx.Response(500, json={"error": "boom"})
        return httpx.Response(200, json={"choices": [{"message": {"content": host}}]})

    transport = httpx.MockTransport(handler)
    outcomes = await fan_out(
        [
            ("ok", _up("ok", 9000)),
            ("fail-a", _up("fail-a", 9001)),
            ("fail-b", _up("fail-b", 9002)),
        ],
        {"messages": []},
        per_upstream_timeout_secs=2.0,
        transport=transport,
        quorum=2,  # only 1 will succeed; quorum unreachable
    )

    assert len(outcomes) == 3
    # All real outcomes (no cancelled — quorum couldn't be reached, so
    # waiting for all was the right move)
    s = successes(outcomes)
    f = failures(outcomes)
    assert len(s) == 1
    assert len(f) == 2
    assert all(o.reason == "non_2xx" for o in f)


@pytest.mark.asyncio
async def test_fan_out_default_quorum_is_two_thirds() -> None:
    """When `quorum` is omitted, fan_out uses `quorum_threshold(N)`.
    For 3 generators that's 2 — so a hanging third (slower than the
    grace window) gets cancelled."""

    cancel_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal cancel_count
        host = request.url.host
        if host == "upstream-slow":
            try:
                # Sleep longer than DEFAULT_QUORUM_GRACE_SECS (5s) so
                # the grace window expires and cancellation fires.
                await asyncio.sleep(10.0)
            except asyncio.CancelledError:
                cancel_count += 1
                raise
        return httpx.Response(200, json={"choices": [{"message": {"content": host}}]})

    transport = httpx.MockTransport(handler)
    outcomes = await fan_out(
        [
            ("a", _up("a", 9000)),
            ("b", _up("b", 9001)),
            ("slow", _up("slow", 9002)),
        ],
        {"messages": []},
        per_upstream_timeout_secs=20.0,
        transport=transport,
        # quorum omitted — defaults to 2 for 3 generators
        grace_secs=0.1,  # tighten grace so test runs fast
    )

    # Two fast successes + one cancelled
    assert isinstance(outcomes[0], GeneratorSuccess)
    assert isinstance(outcomes[1], GeneratorSuccess)
    assert isinstance(outcomes[2], GeneratorFailure)
    assert outcomes[2].reason == "cancelled"


@pytest.mark.asyncio
async def test_fan_out_grace_window_lets_almost_done_finish() -> None:
    """If a slower generator finishes within `grace_secs` of quorum,
    its real outcome is preserved instead of being cancelled. This
    catches the review r1 C-MED concern: a 200ms-slower diversity
    contributor shouldn't be killed when synth has plenty of budget.
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "upstream-medium":
            # Slower than fast pair, but well within 1s grace
            await asyncio.sleep(0.05)
        return httpx.Response(200, json={"choices": [{"message": {"content": host}}]})

    transport = httpx.MockTransport(handler)
    outcomes = await fan_out(
        [
            ("a", _up("a", 9000)),
            ("b", _up("b", 9001)),
            ("medium", _up("medium", 9002)),
        ],
        {"messages": []},
        per_upstream_timeout_secs=10.0,
        transport=transport,
        quorum=2,
        grace_secs=1.0,  # 1s grace covers the 50ms slow path
    )

    # All three should succeed — grace window absorbed the slow one
    assert all(isinstance(o, GeneratorSuccess) for o in outcomes)


@pytest.mark.asyncio
async def test_fan_out_grace_zero_cancels_immediately() -> None:
    """`grace_secs=0` is the legacy quorum-only behavior: cancel
    pending immediately on quorum, regardless of how close they are."""

    async def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "upstream-medium":
            await asyncio.sleep(0.5)  # 500ms — would survive any reasonable grace
        return httpx.Response(200, json={"choices": [{"message": {"content": host}}]})

    transport = httpx.MockTransport(handler)
    outcomes = await fan_out(
        [
            ("a", _up("a", 9000)),
            ("b", _up("b", 9001)),
            ("medium", _up("medium", 9002)),
        ],
        {"messages": []},
        per_upstream_timeout_secs=10.0,
        transport=transport,
        quorum=2,
        grace_secs=0.0,
    )

    # Medium should be cancelled — no grace allowed
    assert isinstance(outcomes[2], GeneratorFailure)
    assert outcomes[2].reason == "cancelled"
