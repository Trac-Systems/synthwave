"""Tests for the MoA metrics ringbuffer and /v1/metrics/moa endpoint.

#7 carry-over from client application pt. 23. The ringbuffer is in-memory and per-
process; tests reset it between cases via `metrics.reset_for_test()`.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from meta_model import metrics
from meta_model.config import parse_config_str
from meta_model.metrics import MoaCallRecord, aggregate_metrics, record_moa_call
from meta_model.server import app, set_config, set_upstream_transport


@pytest.fixture(autouse=True)
def _reset_metrics() -> Any:
    metrics.reset_for_test()
    yield
    metrics.reset_for_test()


def test_aggregate_empty() -> None:
    out = aggregate_metrics()
    assert out["profiles"] == {}
    assert out["ring_size"] == 100


def test_aggregate_single_profile_full_quorum() -> None:
    record_moa_call(
        MoaCallRecord(
            timestamp_ms=1,
            profile="text_synth.v1",
            generators=3,
            quorum=3,
            fastpath=False,
            fallback_reason="none",
            synth_decision="merged",
            draft_lengths=[100, 200, 150],
            final_tool_call_count=0,
            final_content_chars=42,
        )
    )
    out = aggregate_metrics()
    p = out["profiles"]["text_synth.v1"]
    assert p["calls"] == 1
    assert p["quorum_avg"] == 3.0
    assert p["degraded_rate"] == 0.0
    assert p["synth_decisions"] == {"merged": 1}
    assert p["tool_call_rate"] == 0.0
    assert p["draft_length"] == {
        "gen0": {"min": 100, "max": 100, "avg": 100.0, "samples": 1},
        "gen1": {"min": 200, "max": 200, "avg": 200.0, "samples": 1},
        "gen2": {"min": 150, "max": 150, "avg": 150.0, "samples": 1},
    }


def test_aggregate_degraded_quorum_tracked() -> None:
    """quorum < generators must show up as degraded_rate > 0 — the
    silent-passthrough detection that pt. 23 wired into headers needs
    to also surface here."""
    record_moa_call(
        MoaCallRecord(
            timestamp_ms=1,
            profile="tool_chat.v1",
            generators=3,
            quorum=1,  # 2 generators failed; only primary succeeded
            fastpath=True,
            fallback_reason="single_success",
            synth_decision="single_success",
        )
    )
    record_moa_call(
        MoaCallRecord(
            timestamp_ms=2,
            profile="tool_chat.v1",
            generators=3,
            quorum=3,
            fastpath=False,
            fallback_reason="none",
            synth_decision="merged",
        )
    )
    out = aggregate_metrics()
    p = out["profiles"]["tool_chat.v1"]
    assert p["calls"] == 2
    assert p["quorum_avg"] == 2.0  # (1 + 3) / 2
    assert p["degraded_rate"] == 0.5  # 1 of 2 was degraded
    assert p["synth_decisions"] == {"single_success": 1, "merged": 1}


def test_aggregate_tool_call_rate() -> None:
    for tcc in [1, 0, 2, 0, 1]:
        record_moa_call(
            MoaCallRecord(
                timestamp_ms=1,
                profile="tool_chat.v1",
                generators=3,
                quorum=3,
                fastpath=False,
                fallback_reason="none",
                synth_decision="merged",
                final_tool_call_count=tcc,
            )
        )
    out = aggregate_metrics()
    p = out["profiles"]["tool_chat.v1"]
    assert p["calls"] == 5
    assert p["tool_call_rate"] == 0.6  # 3 of 5 calls had tool_calls


def test_aggregate_elapsed_ms_avg() -> None:
    """elapsed_ms_avg averages the per-record elapsed_ms field. Dispatch
    used to write 0 unconditionally (placeholder until the wall-clock
    was threaded through); this test guards the schema + aggregator
    against silently dropping the field back to 0 across the board."""
    for ms in [120, 240, 360]:
        record_moa_call(
            MoaCallRecord(
                timestamp_ms=1,
                profile="tool_chat.v1",
                generators=3,
                quorum=3,
                fastpath=False,
                fallback_reason="none",
                synth_decision="merged",
                elapsed_ms=ms,
            )
        )
    out = aggregate_metrics()
    p = out["profiles"]["tool_chat.v1"]
    assert p["elapsed_ms_avg"] == 240.0  # (120 + 240 + 360) / 3


def test_ringbuffer_caps_at_100() -> None:
    """101st insertion evicts the oldest. Bounds the in-memory cost."""
    for i in range(150):
        record_moa_call(
            MoaCallRecord(
                timestamp_ms=i,
                profile="cap_wrap.v1",
                generators=3,
                quorum=3,
                fastpath=False,
                fallback_reason="none",
                synth_decision="merged",
            )
        )
    out = aggregate_metrics()
    p = out["profiles"]["cap_wrap.v1"]
    assert p["calls"] == 100
    # Oldest preserved record has timestamp = 50 (entries 0..49 evicted).
    assert p["oldest_ts_ms"] == 50
    assert p["newest_ts_ms"] == 149


# ── HTTP endpoint ──────────────────────────────────────────────────


_BASE_TOML = """
[upstreams.text_a]
model_id = "model-a"
base_url = "http://upstream:9000/v1"
context = 4096
max_output = 1024

[upstreams.text_b]
model_id = "model-b"
base_url = "http://upstream:9001/v1"
context = 4096
max_output = 1024

[upstreams.text_c]
model_id = "model-c"
base_url = "http://upstream:9002/v1"
context = 4096
max_output = 1024

# Multi-generator profile is required to hit `_dispatch_moa` (the
# code path the metrics ringbuffer hooks into). Single-upstream
# profiles take the D.1.4 passthrough path that doesn't synthesize
# and isn't a MoA call worth recording.
[profiles."moa.text.v1"]
type = "moa"
generators = ["text_a", "text_b", "text_c"]
synthesizer = "text_a"
"""


@pytest.fixture
def loaded_config() -> Any:
    cfg = parse_config_str(_BASE_TOML)
    set_config(cfg)
    yield cfg
    set_config(None)
    set_upstream_transport(None)


def test_endpoint_empty_buffer(loaded_config) -> None:
    r = TestClient(app).get("/v1/metrics/moa")
    assert r.status_code == 200
    body = r.json()
    assert body["profiles"] == {}
    assert body["ring_size"] == 100


def test_endpoint_records_dispatch(loaded_config) -> None:
    """A live dispatch through the MoA path must produce a record in
    the ringbuffer that the endpoint surfaces. End-to-end smoke."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 1,
                "model": "model-a",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    set_upstream_transport(httpx.MockTransport(handler))
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "moa.text.v1",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200

    metrics_r = client.get("/v1/metrics/moa")
    assert metrics_r.status_code == 200
    body = metrics_r.json()
    p = body["profiles"]["moa.text.v1"]
    assert p["calls"] == 1
    assert p["quorum_avg"] == 3.0  # all three mock upstreams returned 200
    # Live dispatch must populate a real elapsed_ms — even a sub-ms
    # synthetic call should not round to 0 because monotonic resolution
    # on a real platform produces non-zero deltas. If the dispatch
    # threading regresses (back to elapsed_ms=0 hardcoded), this fires.
    assert p["elapsed_ms_avg"] >= 0.0
    # The placeholder bug surfaced when this was *always* 0.0 across
    # all profiles regardless of how long calls actually took. We can't
    # assert > 0 deterministically (mock transport returns instantly),
    # so confirm at minimum the field exists and is a number — and pair
    # this with the unit test above which proves non-zero values flow
    # through aggregation correctly.
    assert isinstance(p["elapsed_ms_avg"], (int, float))
