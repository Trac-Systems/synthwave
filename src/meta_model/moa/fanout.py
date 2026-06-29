"""Generator fan-out — parallel HTTP to N upstreams.

D.2.1 primitive. Calls `forward_chat_completion` against each upstream
concurrently, applies a per-upstream timeout, and reports a per-upstream
outcome. Quorum decisions can short-circuit the wait via
`_gather_with_quorum`: once `quorum` successes have landed, pending
generators are cancelled so a single slow upstream cannot drain the
caller's wall-clock budget (a 599s fanout used to leave synthesis with
<1s of headroom — see plans/moa-quorum-and-harmony-strip-2026-05-04.md).

Invariants:
- Outcomes are returned in the same order as the input upstream list.
  Synthesizer can index by position when merging.
- A failure on one upstream never aborts the others. Synthesizer
  decides whether N-1 (or fewer) is enough.
- Timeout is per-upstream wall-clock, applied via `asyncio.wait_for`
  on top of httpx's own timeout. Doubled-up because httpx's timeout
  controls connect/read/write granularity; the outer wrapper is the
  hard wall-clock cap.
- When quorum-based early exit fires, pending tasks are cancelled and
  awaited (httpx connection pool drains cleanly), then mapped to a
  `GeneratorFailure(reason="cancelled")` so the outcome list shape
  stays uniform for downstream consumers.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from ..config import UpstreamConfig
from ..upstream import forward_chat_completion

_log = logging.getLogger("meta_model.fanout")

# Truncation cap (Unicode characters, not bytes) for non_2xx response
# bodies surfaced into `GeneratorFailure.detail`. Big enough to fit
# vLLM's `{"object":"error","message":"...","type":"BadRequestError",
# "param":null,"code":400}` envelopes, small enough that a misbehaving
# upstream returning megabytes of HTML can't hose telemetry.
_NON_2XX_DETAIL_CHARS = 512

# Tighter cap for the WARN log line. vLLM's `message` field can echo
# parts of the request (e.g. "messages must end with user role, got
# '...'"), so we keep the journalctl exposure narrow even though this
# is an operator-only deployment.
_NON_2XX_WARN_CHARS = 200


def _snippet_from_response(resp: httpx.Response, *, max_chars: int = _NON_2XX_DETAIL_CHARS) -> str:
    """Pull a UTF-8 snippet of the response body, truncated for logging.
    Best-effort: any read error degrades to an empty string so we never
    promote a logging detail into a second failure mode.
    """
    try:
        text = resp.text
    except Exception:
        return ""
    if not text:
        return ""
    snippet = text.strip()
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "…"
    return snippet


@dataclass
class GeneratorSuccess:
    upstream_name: str
    response: dict[str, Any]
    elapsed_ms: int


@dataclass
class GeneratorFailure:
    upstream_name: str
    reason: str  # "timeout" | "transport" | "non_2xx" | "non_json" | "cancelled"
    detail: str
    status: int | None
    elapsed_ms: int


GeneratorOutcome = GeneratorSuccess | GeneratorFailure


def quorum_threshold(n: int) -> int:
    """Successes required before fanout can short-circuit.

    `ceil(n * 2/3)` — strict majority for n=3 (need 2), all-required for
    n<=2 (no benefit from cancellation), 2/3 majority above (n=4→3,
    n=5→4, n=6→4, etc.).

    Always >= 1 and <= n. Defensive on n=0 (returns 0; caller should
    never invoke fanout with empty upstream list).
    """
    if n <= 0:
        return 0
    return max(1, math.ceil(n * 2 / 3))


# Default grace window: after quorum lands, wait up to this many
# additional seconds for pending generators to finish naturally. A
# pure quorum cutoff (grace=0) over-cancels — a third generator that
# is 5-15s slower than the median pair would be killed even when
# there's plenty of synth budget remaining, costing MoA diversity.
# Empirical observation 2026-05-04: with grace=5s, ~every MoA
# tool_chat.v1 request under real opencode traffic ended at quorum=2
# because reasoning-model-20b on .30 reasoning routinely takes 5-15s longer
# than the primary synthesizer on heavier prompts. Bumping to 30s catches
# the typical spread (p99 of cluster generator-latency variance) while
# still cutting off the truly-stuck (the original 599s pathology — at
# worst we now waste 30s instead of 5s before falling back).
# Review r1 C-MED finding: gating must be deadline/grace based, not
# pure count, AND the grace must reflect actual cluster latency
# distribution, not aspirational fairness.
DEFAULT_QUORUM_GRACE_SECS = float(os.environ.get("TK_MOA_QUORUM_GRACE_SECS") or 30.0)


async def _gather_with_quorum(
    factories: list[tuple[str, Callable[[], Awaitable[GeneratorOutcome]]]],
    *,
    quorum: int,
    grace_secs: float = DEFAULT_QUORUM_GRACE_SECS,
) -> list[GeneratorOutcome]:
    """Run all coroutine factories in parallel; cancel pending once
    `quorum` successes have landed AND `grace_secs` of extra wait
    have elapsed without all tasks finishing. Outcomes are returned
    in input order, with cancelled tasks reported as
    `GeneratorFailure(reason="cancelled")`.

    `factories` is a list of (name, zero-arg async factory) tuples.
    Factories are used (not pre-built coroutines) so we control task
    creation and can name each task for debugging.

    `quorum <= 0` or `quorum > len(factories)` disables cancellation
    (waits for all). `grace_secs <= 0` cancels immediately on quorum
    (legacy quorum-only behavior, retained for tests).

    Cancellation policy is await-after-cancel so httpx releases its
    connection back to the pool. Cancellation race: a task may
    complete (success or failure) AFTER quorum + grace expire but
    BEFORE its `cancel()` lands. That outcome is preserved as-is —
    we do not synthesise a `cancelled` failure on top of a real
    result. Only tasks that raise `CancelledError` from `await t`
    become `cancelled` outcomes.

    Caller cancellation: if the surrounding generator/task is itself
    cancelled while this function is running (e.g., client disconnect
    propagated through Starlette), `asyncio.current_task().cancelling()`
    becomes >0 and we re-raise `CancelledError` from the drain loop to
    propagate the outer cancellation. Review r1 I-MED finding.
    """
    if not factories:
        return []
    if quorum < 1 or quorum > len(factories):
        # No quorum cutoff requested; classic gather.
        return list(await asyncio.gather(*[f() for _, f in factories]))

    loop = asyncio.get_running_loop()
    fanout_t0 = loop.time()
    tasks: list[asyncio.Task[GeneratorOutcome]] = [
        asyncio.create_task(f(), name=f"fanout/{n}") for n, f in factories
    ]
    name_for_task: dict[asyncio.Task, str] = {t: factories[i][0] for i, t in enumerate(tasks)}
    outcomes: dict[asyncio.Task, GeneratorOutcome] = {}
    pending: set[asyncio.Task] = set(tasks)
    successes = 0
    quorum_landed_at: float | None = None

    def _absorb(d: asyncio.Task) -> None:
        nonlocal successes
        try:
            outcome = d.result()
        except asyncio.CancelledError:
            # Task itself was cancelled mid-flight (rare in this branch
            # since cancel() only runs in the finally below). Synthesize
            # a cancelled outcome so the loop stays well-formed.
            outcome = GeneratorFailure(
                upstream_name=name_for_task[d],
                reason="cancelled",
                detail="task cancelled before completion",
                status=None,
                elapsed_ms=int((loop.time() - fanout_t0) * 1000),
            )
        except Exception as e:
            # Defensive: a factory should never raise — it must always
            # return a typed outcome. Synthesize a transport failure so
            # the loop stays well-formed.
            outcome = GeneratorFailure(
                upstream_name=name_for_task[d],
                reason="transport",
                detail=f"unexpected exception: {type(e).__name__}: {e}",
                status=None,
                elapsed_ms=int((loop.time() - fanout_t0) * 1000),
            )
        outcomes[d] = outcome
        if isinstance(outcome, GeneratorSuccess):
            successes += 1

    try:
        # Phase 1: wait for FIRST_COMPLETED until quorum is met or all done.
        while pending and successes < quorum:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for d in done:
                _absorb(d)

        # Phase 2: quorum is met (or pending is empty). If grace_secs > 0
        # and tasks remain pending, wait UP TO grace_secs for them to
        # finish naturally before cancelling. This catches the "almost
        # done" common case (slow generator 200ms behind median) without
        # blocking on the truly-stuck one (599s pathology).
        if pending and successes >= quorum and grace_secs > 0:
            quorum_landed_at = loop.time()
            try:
                done, pending = await asyncio.wait(pending, timeout=grace_secs)
                for d in done:
                    _absorb(d)
            except asyncio.CancelledError:
                # Caller cancelled us during grace wait. Pending tasks
                # will be cancelled in `finally` below, then we re-raise.
                raise
    finally:
        if pending:
            for t in pending:
                t.cancel()
            cancel_started_ms = int((loop.time() - fanout_t0) * 1000)
            # Drain each cancelled task so httpx releases its connection.
            # If the OUTER caller cancelled us, awaiting cancelled tasks
            # can re-raise CancelledError; we must distinguish "caller
            # cancelled us" from "we cancelled t" so the outer cancel
            # propagates. Python 3.11+: `current_task().cancelling()`
            # is non-zero when the runtime has scheduled cancellation
            # for us. Only true if outer requested it.
            current = asyncio.current_task()
            for t in pending:
                try:
                    real_outcome = await t
                except asyncio.CancelledError:
                    if current is not None and current.cancelling() > 0:
                        # Outer caller cancelled us. Re-raise so it
                        # propagates correctly. (Best-effort drain of
                        # remaining tasks is sacrificed; httpx will
                        # close their connections when their async
                        # contexts unwind regardless.)
                        raise
                    real_outcome = None
                except Exception as e:
                    real_outcome = GeneratorFailure(
                        upstream_name=name_for_task[t],
                        reason="transport",
                        detail=f"unexpected exception during cancel drain: {type(e).__name__}: {e}",
                        status=None,
                        elapsed_ms=int((loop.time() - fanout_t0) * 1000),
                    )
                cancel_total_ms = int((loop.time() - fanout_t0) * 1000)
                if real_outcome is not None:
                    outcomes[t] = real_outcome
                else:
                    outcomes[t] = GeneratorFailure(
                        upstream_name=name_for_task[t],
                        reason="cancelled",
                        detail=(
                            f"quorum {quorum} reached, generator cancelled "
                            f"after {grace_secs:.1f}s grace"
                        ),
                        status=None,
                        elapsed_ms=cancel_total_ms,
                    )

    # Preserve input order
    return [outcomes[t] for t in tasks]


async def _call_one(
    name: str,
    upstream: UpstreamConfig,
    body: dict[str, Any],
    per_upstream_timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None,
) -> GeneratorOutcome:
    loop = asyncio.get_running_loop()
    start = loop.time()
    try:
        resp = await asyncio.wait_for(
            forward_chat_completion(
                upstream,
                body,
                timeout_secs=per_upstream_timeout_secs,
                transport=transport,
            ),
            timeout=per_upstream_timeout_secs,
        )
    except (TimeoutError, httpx.TimeoutException):
        return GeneratorFailure(
            upstream_name=name,
            reason="timeout",
            detail=f"per-upstream timeout {per_upstream_timeout_secs}s exceeded",
            status=None,
            elapsed_ms=int((loop.time() - start) * 1000),
        )
    except httpx.RequestError as e:
        return GeneratorFailure(
            upstream_name=name,
            reason="transport",
            detail=f"{type(e).__name__}: {e}",
            status=None,
            elapsed_ms=int((loop.time() - start) * 1000),
        )

    elapsed_ms = int((loop.time() - start) * 1000)

    if resp.status_code < 200 or resp.status_code >= 300:
        detail_snippet = _snippet_from_response(resp, max_chars=_NON_2XX_DETAIL_CHARS)
        warn_snippet = _snippet_from_response(resp, max_chars=_NON_2XX_WARN_CHARS)
        detail = f"upstream returned HTTP {resp.status_code}"
        if detail_snippet:
            detail = f"{detail}: {detail_snippet}"
        _log.warning(
            "non_2xx upstream=%s status=%s elapsed_ms=%s body=%r",
            name,
            resp.status_code,
            elapsed_ms,
            warn_snippet,
        )
        return GeneratorFailure(
            upstream_name=name,
            reason="non_2xx",
            detail=detail,
            status=resp.status_code,
            elapsed_ms=elapsed_ms,
        )

    try:
        payload = resp.json()
    except ValueError:
        return GeneratorFailure(
            upstream_name=name,
            reason="non_json",
            detail="upstream returned non-JSON content",
            status=resp.status_code,
            elapsed_ms=elapsed_ms,
        )

    # OpenAI Chat Completions responses are always JSON objects. A
    # syntactically-valid JSON array/scalar would surprise D.2.2's
    # synthesizer (which expects to read `choices[0]`). Classify
    # non-object JSON as a protocol failure.
    if not isinstance(payload, dict):
        return GeneratorFailure(
            upstream_name=name,
            reason="non_json",
            detail="upstream returned non-object JSON (expected dict)",
            status=resp.status_code,
            elapsed_ms=elapsed_ms,
        )

    return GeneratorSuccess(
        upstream_name=name,
        response=payload,
        elapsed_ms=elapsed_ms,
    )


async def fan_out(
    upstreams: list[tuple[str, UpstreamConfig]],
    body: dict[str, Any],
    *,
    per_upstream_timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None = None,
    quorum: int | None = None,
    grace_secs: float = DEFAULT_QUORUM_GRACE_SECS,
) -> list[GeneratorOutcome]:
    """Call N upstreams in parallel, return per-upstream outcomes.

    Outcomes preserve input order. Failures don't abort siblings.

    `quorum` (default `quorum_threshold(len(upstreams))`): once that
    many `GeneratorSuccess` outcomes have landed, the gather waits up
    to `grace_secs` more before cancelling stragglers. Cancelled tasks
    surface as `GeneratorFailure(reason="cancelled")`.
    `quorum=0` (or any value < 1 / > len(upstreams)) waits for all
    generators (legacy behavior, no cutoff).
    """
    if not upstreams:
        return []
    effective_quorum = quorum if quorum is not None else quorum_threshold(len(upstreams))
    factories: list[tuple[str, Callable[[], Awaitable[GeneratorOutcome]]]] = [
        (
            n,
            (lambda n=n, u=u: _call_one(n, u, body, per_upstream_timeout_secs, transport)),
        )
        for n, u in upstreams
    ]
    return await _gather_with_quorum(
        factories, quorum=effective_quorum, grace_secs=grace_secs
    )


def successes(outcomes: list[GeneratorOutcome]) -> list[GeneratorSuccess]:
    return [o for o in outcomes if isinstance(o, GeneratorSuccess)]


def failures(outcomes: list[GeneratorOutcome]) -> list[GeneratorFailure]:
    return [o for o in outcomes if isinstance(o, GeneratorFailure)]
