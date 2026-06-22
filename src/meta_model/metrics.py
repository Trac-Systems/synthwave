"""Per-profile MoA metrics ringbuffer.

#7 carry-over from client application pt. 23: prove each MoA generator earns its
compute per surface. Currently invisible whether reasoning vs primary
vs fast each pull their weight, beyond per-call DEBUG `moa.draft` log
lines.

Records the last `N` dispatch outcomes per profile in an in-memory
ringbuffer (Python `collections.deque(maxlen=N)`). The
`/v1/metrics/moa` endpoint (server.py) summarizes the buffer into
per-profile aggregates:

- `calls`: number of records in the buffer
- `quorum_avg`: average successful generator count
- `degraded_rate`: fraction of calls where quorum < generators
- `synth_decisions`: histogram of `synth_decision` labels (merged,
  single_success, fastpath_consensus, tool_preserved, etc.)
- `tool_call_rate`: fraction of calls where the final response had at
  least one tool_call
- `draft_length`: per-generator-position average / min / max draft
  byte-length (positional, not by upstream name — N generators per
  profile so position N maps to the same upstream every call)

Bounded memory: `_RING_PER_PROFILE * N_PROFILES * record_size`. With
N=100 and record_size ~200 bytes this is ~6KB even with 30 profiles.

Threading: FastAPI runs all dispatch on the same asyncio event loop,
so a `deque` is safe without an explicit lock for the dispatcher path.
The metrics endpoint reads via `list(deque)` which is also atomic
under the GIL for deque snapshots.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from threading import Lock
from typing import Any

# Per-profile ring depth. 100 calls is enough to compute meaningful
# rates without memory pressure.
_RING_PER_PROFILE = 100


@dataclass(frozen=True)
class FailureRecord:
    """Per-generator failure entry for a single MoA call.

    Reason is drawn from a closed vocabulary so /v1/metrics/moa
    `failure_breakdown` cardinality stays bounded:
    - `timeout`        — fanout-level deadline
    - `transport`      — TCP/TLS error before HTTP response
    - `non_2xx`        — upstream returned 4xx/5xx
    - `non_json`       — upstream 2xx body wasn't valid JSON
    - `undeclared_tool`     — candidate emitted a tool name not in
                              request's declared `tools[*].function.name`
    - `dual_shape_response` — candidate carried BOTH `tool_calls` and
                              legacy `function_call` (malformed)

    The first four come from `fanout.GeneratorFailure.reason` (existing
    wire labels — unchanged). The latter two are produced by dispatch's
    constraint-validation step.
    """

    upstream_name: str
    reason: str


@dataclass
class MoaCallRecord:
    """One dispatch outcome. Pure-data — never includes user content."""

    timestamp_ms: int
    profile: str
    generators: int
    quorum: int
    fastpath: bool
    fallback_reason: str
    synth_decision: str
    draft_lengths: list[int] = field(default_factory=list)
    final_tool_call_count: int = 0
    final_content_chars: int = 0
    elapsed_ms: int = 0
    # Per-call failure records. Empty when all generators succeeded and
    # no constraint demotions fired. Populated by `_dispatch_moa` when
    # candidates fail (timeout, non_2xx, transport, non_json) OR when
    # they violate the declared-tool contract (`undeclared_tool`,
    # `dual_shape_response`). Reasons are drawn from a closed bounded
    # vocabulary — see dispatch.py demotion path. The actual offending
    # tool name is NOT in this field (cardinality protection); it
    # appears only in the per-call WARN log.
    failures: tuple["FailureRecord", ...] = field(default_factory=tuple)


# `defaultdict(deque)` factory needs a closure binding the maxlen.
def _new_deque() -> deque[MoaCallRecord]:
    return deque(maxlen=_RING_PER_PROFILE)


_buffers: dict[str, deque[MoaCallRecord]] = defaultdict(_new_deque)
_buffers_lock = Lock()


def record_moa_call(record: MoaCallRecord) -> None:
    """Append one dispatch outcome to the per-profile ringbuffer.

    Called from the MoA dispatch path. Defensively cheap: a single
    `deque.append` under the GIL.
    """
    with _buffers_lock:
        _buffers[record.profile].append(record)


def reset_for_test() -> None:
    """Clear all ringbuffers. Test-only."""
    with _buffers_lock:
        _buffers.clear()


def aggregate_metrics() -> dict[str, Any]:
    """Render the `/v1/metrics/moa` payload from the ringbuffers."""
    with _buffers_lock:
        snapshot = {p: list(buf) for p, buf in _buffers.items()}

    profiles_out: dict[str, dict[str, Any]] = {}
    for profile_name, records in snapshot.items():
        if not records:
            continue
        n = len(records)
        quorum_sum = sum(r.quorum for r in records)
        quorum_avg = quorum_sum / n
        degraded = sum(1 for r in records if r.quorum < r.generators)
        degraded_rate = degraded / n

        synth_decisions: dict[str, int] = defaultdict(int)
        for r in records:
            label = r.synth_decision or "unknown"
            synth_decisions[label] += 1

        tool_calls_n = sum(1 for r in records if r.final_tool_call_count > 0)
        tool_call_rate = tool_calls_n / n

        # Per-position draft length stats. Profile generators are stable
        # in their declared order, so position 0 maps to the same upstream
        # across calls (modulo failures, which leave the slot empty in
        # `draft_lengths` per dispatch's record).
        per_position: dict[int, list[int]] = defaultdict(list)
        for r in records:
            for i, length in enumerate(r.draft_lengths):
                per_position[i].append(length)
        draft_length: dict[str, dict[str, int | float]] = {}
        for i, lens in per_position.items():
            if not lens:
                continue
            draft_length[f"gen{i}"] = {
                "min": min(lens),
                "max": max(lens),
                "avg": round(sum(lens) / len(lens), 2),
                "samples": len(lens),
            }

        elapsed_total = sum(r.elapsed_ms for r in records)
        elapsed_avg = round(elapsed_total / n, 2)

        # Per-reason failure histogram across the ringbuffer. Surfaces
        # constraint-violation drift (`undeclared_tool`,
        # `dual_shape_response`) alongside transport/timeout/non_2xx so
        # operators can spot which class of failure dominates without
        # scraping per-call logs.
        failure_breakdown: dict[str, int] = defaultdict(int)
        for r in records:
            for f in r.failures:
                failure_breakdown[f.reason] += 1

        profiles_out[profile_name] = {
            "calls": n,
            "quorum_avg": round(quorum_avg, 3),
            "degraded_rate": round(degraded_rate, 3),
            "synth_decisions": dict(synth_decisions),
            "tool_call_rate": round(tool_call_rate, 3),
            "elapsed_ms_avg": elapsed_avg,
            "draft_length": draft_length,
            "failure_breakdown": dict(failure_breakdown),
            "newest_ts_ms": max(r.timestamp_ms for r in records),
            "oldest_ts_ms": min(r.timestamp_ms for r in records),
        }

    return {
        "ring_size": _RING_PER_PROFILE,
        "profiles": profiles_out,
    }


def now_ms() -> int:
    return int(time.time() * 1000)


def record_dict(record: MoaCallRecord) -> dict[str, Any]:
    """Hook for tests / future raw-dump endpoint. Not used by aggregate."""
    return asdict(record)
