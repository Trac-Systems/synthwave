"""Profile dispatch — wire fanout + compaction + synthesizer per profile.

D.2.4 primitive. Replaces D.1.4's degenerate-profile passthrough with
a real multi-upstream routing path. The chat-completions endpoint
selects a profile (via `model` field or `x_meta_model.profile`) and
this module dispatches according to the profile's ensemble type:

- **MoA**: shared-tail compact → fan out to N generators → synthesize
  per profile's `synthesis_mode` (merge | best-of). 1-success and
  agreement fast-paths short-circuit the synthesizer.
  D.3.2 enforces per-profile multimodal policy when the request
  contains image_url parts: `image_tool_policy` filters generators
  to vision-capable (intersected with `supports_function_calling`
  when tools are active), `max_images` caps the input, and the
  config-load validator catches the strict-quorum trap. Profiles
  without a `[multimodal]` block continue to passthrough multimodal
  requests unfiltered (back-compat).
- **Cascade**: try upstreams in priority order; first 2xx response
  wins. On full-cascade failure, the profile's `on_all_fail` policy
  decides between bubbling the last error or returning a structured
  502.
- **Voting**: parallel YES/NO across all upstreams; aggregator picks
  per profile (currently `any_yes` only). Requires
  `[features].voting = true` to be callable. Returns a synthetic
  ChatCompletion with `content = "yes" | "no"`.

The endpoint passes the original raw request body (so client-set
fields like `tools`, `response_format`, `temperature` flow through
without pydantic-default leakage). The dispatcher mutates only what
the profile demands (per-upstream model id, server-wins
chat_template_kwargs).

Returns ``DispatchResult`` carrying the response payload, status
code, and a metadata dict the caller turns into ``X-MetaModel-*``
headers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import (
    CascadeProfile,
    MetaModelConfig,
    MoaProfile,
    Profile,
    UpstreamConfig,
    VotingProfile,
)
from ..reasoning import has_reasoning_rescue_text
from ..upstream import forward_chat_completion, prepare_upstream_body
from .compaction import compact_with_shared_tail, estimate_messages_tokens
from .fanout import GeneratorFailure, GeneratorOutcome, GeneratorSuccess, successes
from .multimodal import (
    MessageModality,
    detect_message_modality,
)
from .synthesizer import synthesize
from .tools import (
    ToolNormalizationError,
    candidate_violates_declared,
    normalize_tool_request,
    raw_emitted_call_names,
    resolve_tool_constraint,
)

log = logging.getLogger(__name__)


# Compaction reserves — match client-core surface defaults.
DEFAULT_RESPONSE_RESERVE = 2000
DEFAULT_SAFETY_MARGIN = 512


@dataclass
class DispatchResult:
    """Outcome of a profile dispatch.

    `payload` is the OpenAI-shape Chat Completion body to return to the
    client. `status_code` is the HTTP status. `headers` carries
    `X-MetaModel-*` informational fields the caller emits verbatim.
    `error` is set when the dispatcher decides to return an error
    envelope (caller wraps in `error_envelope()`); on success it stays
    None.

    `error_type` (review r1 F6 MED): optional override for the OpenAI
    error envelope's `type` field. Default mapping by HTTP status
    (`error_type_for_status` in `errors.py`) treats 501 as
    `api_error`, but F6's modality_not_supported (501) is a client
    request shape problem and the spec calls for
    `invalid_request_error`. When set, the server uses this verbatim
    instead of the status-derived default.
    """

    payload: dict[str, Any] | None
    status_code: int
    headers: dict[str, str] = field(default_factory=dict)
    error: tuple[str, str | None] | None = None  # (message, code)
    error_type: str | None = None


def _err(
    status: int,
    message: str,
    *,
    code: str | None = None,
    headers: dict[str, str] | None = None,
    error_type: str | None = None,
) -> DispatchResult:
    return DispatchResult(
        payload=None,
        status_code=status,
        headers=headers or {},
        error=(message, code),
        error_type=error_type,
    )


# ── Profile resolution ──────────────────────────────────────────────


def resolve_profile(
    cfg: MetaModelConfig, model: str, ext_profile: str | None
) -> tuple[Profile | None, str | None, str | None]:
    """Resolve a request's target profile.

    `x_meta_model.profile` (if set) wins over the top-level `model`
    field — it's the explicit, type-aware path. The `model` field
    stays available as a fallback for raw OpenAI-SDK clients that
    don't know about extension fields.

    Lookup order:
    1. exact profile name (case-sensitive — TOML-keyed)
    2. alias map (case-insensitive — F4-A)
    3. server brand (case-insensitive — F10 ``[server] model_name``)
    4. raw upstream key (case-sensitive — synthesized 1-element MoA)
    5. not found

    Returns ``(profile, profile_name, error_code)``. Error codes:
    - "model_not_found": neither `model` nor `x_meta_model.profile`
      maps to anything (or maps to a voting profile when voting is
      disabled).
    - "feature_disabled": maps to a voting profile but voting feature
      is off — caller should 400.

    F10: when a request matches ``cfg.server.model_name`` (case-
    insensitive), it routes to the first callable non-voting profile
    in config order. The CANONICAL profile name is returned (not the
    brand) so downstream metadata stays internally consistent;
    response-side rewriting to the brand happens at the endpoint
    handler. Voting profiles aren't a valid brand target — voting
    needs an explicit `model="voting.v1"` choice.
    """
    name = ext_profile or model
    if name in cfg.profiles:
        prof = cfg.profiles[name]
        if isinstance(prof, VotingProfile) and not cfg.features.voting:
            return None, name, "feature_disabled"
        return prof, name, None

    # F4-A: case-insensitive alias map. Resolves to the canonical
    # profile name so downstream metadata (X-MetaModel-Profile,
    # response.model, /v1/models id correlation) reports the canonical,
    # not the alias the client typed. Voting + features.voting=false
    # still 400s via the same path.
    alias_target = cfg.alias_map().get(name.lower()) if name else None
    if alias_target is not None:
        prof = cfg.profiles[alias_target]
        if isinstance(prof, VotingProfile) and not cfg.features.voting:
            return None, alias_target, "feature_disabled"
        return prof, alias_target, None

    # F10: server brand resolves to the first callable non-voting
    # profile in config (TOML insertion) order. ``callable_profiles()``
    # already drops voting when features.voting=false; we additionally
    # skip MoA-or-Cascade-only because voting has no ``model_name``
    # routing semantic.
    brand = (cfg.server.model_name or "").strip()
    if brand and name and name.lower() == brand.lower():
        for pname, prof in cfg.callable_profiles().items():
            if isinstance(prof, VotingProfile):
                continue
            return prof, pname, None
        # No suitable target — surface the typed not-found rather
        # than silently bailing into raw-upstream resolution.
        return None, brand, "model_not_found"

    # Direct upstream key: synthesize a 1-element MoA profile so the
    # MoA path stays uniform. Single-upstream passthrough behavior is
    # preserved (1 success → return verbatim, no synthesizer call).
    upstream = cfg.upstreams.get(name)
    if upstream is not None:
        synthetic = MoaProfile(
            type="moa",
            generators=[name],
            synthesizer=name,
        )
        return synthetic, name, None

    return None, name, "model_not_found"


# ── MoA dispatch ────────────────────────────────────────────────────


async def _dispatch_moa(
    profile: MoaProfile,
    profile_name: str,
    cfg: MetaModelConfig,
    request_body: dict[str, Any],
    *,
    modality: MessageModality,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None,
) -> DispatchResult:
    # Capture wall-clock at MoA dispatch entry. Threaded down to
    # record_moa_call so /v1/metrics/moa can report real elapsed_ms_avg
    # per profile instead of the 0-placeholder it carried before this
    # observability pass landed.
    _moa_t0 = time.monotonic()

    generators_with_cfg: list[tuple[str, UpstreamConfig]] = [
        (name, cfg.upstreams[name]) for name in profile.generators if name in cfg.upstreams
    ]
    if not generators_with_cfg:
        return _err(503, "no generators available", code="no_upstream")

    synth_upstream = cfg.upstreams.get(profile.synthesizer)
    if synth_upstream is None:
        return _err(503, f"synthesizer {profile.synthesizer!r} not configured", code="no_upstream")

    # Profile-level tool stripping. Used by `recovery_synthesis.v1` to
    # structurally guarantee no tool_calls in the output (callers asking
    # for a synthesized final answer after a tool-loop, not another
    # tool attempt). Strips at the boundary so every downstream branch
    # — single-upstream passthrough, fan-out, synth — sees a no-tools
    # request. Defensive: caller's body is left untouched.
    if profile.strip_tools:
        request_body = dict(request_body)
        for k in ("tools", "tool_choice", "parallel_tool_calls"):
            request_body.pop(k, None)

    # F6: any image/video/audio content was already short-circuited
    # to the server-level multimodal cascade in `dispatch()` before
    # this MoA path runs. So `modality.has_images/videos/audios` is
    # always False here. The multimodal_max_active legacy parameter
    # below stays at 1 since it's only consulted by compaction's
    # image-tail logic, which is now unreachable in MoA.
    multimodal_max_active = 1
    multimodal_headers: dict[str, str] = {}

    # Single-upstream MoA (raw upstream addressing, OR a profile that
    # degenerates to one upstream + one synthesizer-as-same-upstream)
    # is a direct passthrough: forward the body, relay status + body
    # verbatim. Skips fan-out + synthesizer overhead, and preserves
    # the D.1.4 contract that non_2xx responses are relayed (not
    # wrapped in our error envelope).
    active_names = {name for name, _ in generators_with_cfg}
    if len(active_names) == 1 and profile.synthesizer in active_names:
        only = generators_with_cfg[0]
        return await _passthrough_single(
            only[0],
            only[1],
            profile_name,
            request_body,
            timeout_secs=timeout_secs,
            transport=transport,
            extra_headers=multimodal_headers,
        )

    # Shared-tail compaction so every generator sees the same recent
    # reality. Tools schema cost is folded in if `tools` is present.
    messages = list(request_body.get("messages") or [])
    tools_token_estimate = _estimate_tools_tokens(request_body.get("tools"))
    tokens_in = estimate_messages_tokens(messages)
    layout = compact_with_shared_tail(
        messages,
        generators_with_cfg,
        response_reserve=DEFAULT_RESPONSE_RESERVE,
        tools_token_estimate=tools_token_estimate,
        safety_margin=DEFAULT_SAFETY_MARGIN,
        image_max_active=multimodal_max_active,
    )

    if not layout.per_generator:
        # Irreducible input — every generator's budget is non-positive
        # before the tail can even be assembled. Phase D plan §253:
        # 413 only when no generator can fit; never silent truncation.
        return _err(
            413,
            "no generator can fit the prompt within its context budget",
            code="context_length_exceeded",
            headers={
                "X-MetaModel-Profile": profile_name,
                "X-MetaModel-Prompt-Tokens-In": str(tokens_in),
                "X-MetaModel-Tools-Tokens": str(tools_token_estimate),
                **multimodal_headers,
            },
        )

    # Compaction stats — report from the SMALLEST per-generator payload
    # (the most aggressively compressed view). That's the honest
    # worst-case the client should reason about: "the worst-budget
    # generator saw N tokens and lost M chunks from the input."
    # Compacted-N comes from the primitive's tracking, not a message-
    # length delta — compaction inserts a `[N earlier groups…]`
    # sentinel that shifts message count without reflecting drops
    # (review r21 finding 1).
    payload_tokens = [estimate_messages_tokens(gp.messages) for gp in layout.per_generator]
    tokens_out_min = min(payload_tokens) if payload_tokens else 0
    compacted_n = (
        max(gp.compacted_chunks for gp in layout.per_generator) if layout.per_generator else 0
    )

    # Build per-generator request bodies. Each gets its own messages
    # list (head + shared_tail); everything else is identical.
    per_gen_bodies: list[tuple[str, UpstreamConfig, dict[str, Any]]] = []
    for gp in layout.per_generator:
        gen_cfg = cfg.upstreams[gp.upstream_name]
        body = dict(request_body)
        body["messages"] = gp.messages
        if profile.generator_temperature is not None:
            body["temperature"] = profile.generator_temperature
        per_gen_bodies.append((gp.upstream_name, gen_cfg, prepare_upstream_body(body, gen_cfg)))

    # Fan out. Note fan_out's body is shared; we instead invoke each
    # call with its own pre-prepared body via a local wrapper.
    # `dispatch_start` captures the dispatch entry time so the synth
    # call later sees the REMAINING budget, not the original
    # timeout_secs. Without this, a slow fan-out (>60s) makes the
    # synthesizer's static 60s margin (synthesizer.py) ineffective —
    # the synth deadline can land after the caller's deadline. Review
    # r1b finding (HIGH).
    dispatch_start = time.monotonic()
    outcomes = await _fan_out_per_body(
        per_gen_bodies,
        timeout_secs=timeout_secs,
        transport=transport,
    )

    # Prompt-Tokens-In/Out cover MESSAGES ONLY. Tool schema overhead is
    # reserved separately in the compaction budget and exposed via
    # X-MetaModel-Tools-Tokens (review r21 finding 2 — folding tools
    # into Prompt-Tokens-In would inflate the In/Out comparison; a
    # separate header keeps the message-vs-overhead distinction).
    compaction_headers = {
        "X-MetaModel-Compacted-N": str(compacted_n),
        "X-MetaModel-Prompt-Tokens-In": str(tokens_in),
        "X-MetaModel-Prompt-Tokens-Out": str(tokens_out_min),
        "X-MetaModel-Tools-Tokens": str(tools_token_estimate),
        **multimodal_headers,
    }

    succ = successes(outcomes)
    if not succ:
        # Single-upstream profiles: surface the specific failure mode
        # so clients see the same status/code as a direct passthrough
        # would have produced (timeout→504, non_2xx→relay status,
        # non_json→502/upstream_protocol_error). For multi-generator
        # profiles, the per-generator detail is in the upstream logs;
        # client gets a single all_generators_failed signal.
        if len(layout.per_generator) == 1:
            failure = _single_upstream_failure(outcomes[0])
            failure.headers.update(compaction_headers)
            return failure
        return _err(
            502,
            "all generators failed",
            code="all_generators_failed",
            headers=compaction_headers,
        )

    # ── Per-candidate constraint validation (review r2 HIGH) ────────────
    # Demote candidates that emit undeclared tool names or carry malformed
    # dual-shape responses (both `tool_calls` and `function_call` set).
    # Treated as candidate-level failures: same degradation path as 5xx /
    # timeout / non-2xx so the existing degraded-headers + metrics
    # machinery surfaces them naturally. Only the actual tool name leaks
    # to the WARN log; the public failure-reason string is bounded.
    constraint = resolve_tool_constraint(request_body)
    constraint_demotions: list[tuple[str, str]] = []
    if len(layout.per_generator) > 1:  # only meaningful for MoA fan-out
        new_outcomes: list[GeneratorOutcome] = []
        for o in outcomes:
            if not isinstance(o, GeneratorSuccess):
                new_outcomes.append(o)
                continue
            try:
                msg = (
                    o.response.get("choices", [{}])[0].get("message")
                    if isinstance(o.response.get("choices"), list)
                    and o.response["choices"]
                    else None
                )
            except (AttributeError, IndexError, TypeError):
                msg = None
            if not isinstance(msg, dict):
                # Synth handles malformed-shape candidates downstream.
                new_outcomes.append(o)
                continue
            reason = candidate_violates_declared(msg, constraint)
            if reason is None:
                new_outcomes.append(o)
                continue
            # Demote: replace the success with a synthetic failure so
            # synthesize() never sees this candidate.
            log.warning(
                "candidate demoted: profile=%s upstream=%s reason=%s names=%s",
                profile_name,
                o.upstream_name,
                reason,
                raw_emitted_call_names(msg),
            )
            constraint_demotions.append((o.upstream_name, reason))
            new_outcomes.append(
                GeneratorFailure(
                    upstream_name=o.upstream_name,
                    reason=reason,
                    detail="candidate violates declared tool contract",
                    status=200,
                    elapsed_ms=o.elapsed_ms,
                )
            )
        outcomes = new_outcomes
        succ = successes(outcomes)

        # All candidates demoted → distinct status from "all upstreams
        # failed" because the upstreams ARE alive; they just emitted
        # contract-violating output. 503 (upstream available but
        # cannot satisfy) signals "model-discipline issue", which
        # operators should treat differently from 502 backend dead.
        if not succ:
            fail_str = ",".join(
                f"{name}:{reason}" for name, reason in constraint_demotions
            )
            no_quorum_headers = {
                "X-MetaModel-Profile": profile_name,
                "X-MetaModel-Generators": str(len(layout.per_generator)),
                "X-MetaModel-Quorum": "0",
                "X-MetaModel-Degraded": "true",
                "X-MetaModel-Failed-Generators": fail_str,
                **compaction_headers,
            }
            log.info(
                "moa.no_quorum_after_constraint profile=%s failed=%s",
                profile_name,
                fail_str,
            )
            # Record the demotion in the per-profile metrics so operators
            # see the failure breakdown via /v1/metrics/moa.
            try:
                from ..metrics import (
                    FailureRecord,
                    MoaCallRecord,
                    now_ms,
                    record_moa_call,
                )

                record_moa_call(
                    MoaCallRecord(
                        timestamp_ms=now_ms(),
                        profile=profile_name,
                        generators=len(layout.per_generator),
                        quorum=0,
                        fastpath=False,
                        fallback_reason="no_quorum_after_constraint",
                        synth_decision="no_quorum_after_constraint",
                        draft_lengths=[],
                        final_tool_call_count=0,
                        final_content_chars=0,
                        elapsed_ms=int((time.monotonic() - _moa_t0) * 1000),
                        failures=tuple(
                            FailureRecord(upstream_name=n, reason=r)
                            for n, r in constraint_demotions
                        ),
                    )
                )
            except Exception as exc:  # pragma: no cover
                log.warning("metrics.record_moa_call failed: %s", exc)
            return _err(
                503,
                "no candidate satisfied the request's tool contract",
                code="no_quorum_after_constraint",
                headers=no_quorum_headers,
            )

    # Synthesize. The synth call uses the original (untruncated) request
    # body's response-contract fields; the synthesizer module handles
    # propagating the right subset.
    # Pass REMAINING budget, not original timeout_secs — synthesizer.py
    # subtracts a margin from this for the upstream call so the fallback
    # path has time to return before the inbound httpx ReadTimeout fires.
    fanout_elapsed = time.monotonic() - dispatch_start
    remaining_budget = max(timeout_secs - fanout_elapsed, 1.0)
    synth_result = await synthesize(
        profile,
        outcomes,
        synth_upstream,
        request_body,
        timeout_secs=remaining_budget,
        transport=transport,
        profile_name=profile_name,
    )

    # Synthesizer can return SynthesisFailure when every 2xx generator
    # body is malformed (no choices[0].message). Surface as a typed 502
    # rather than a 500; client sees structured error shape.
    from .synthesizer import SynthesisFailure

    if isinstance(synth_result, SynthesisFailure):
        return _err(
            502,
            f"synthesis failed: {synth_result.reason}: {synth_result.detail}",
            code="synthesis_failed",
            headers={
                "X-MetaModel-Profile": profile_name,
                "X-MetaModel-Generators": str(len(layout.per_generator)),
                "X-MetaModel-Quorum": str(len(succ)),
                **compaction_headers,
            },
        )

    # D.x.observability — per-draft telemetry. Lengths and per-call
    # salted hashes expose generator divergence at the wire level.
    # Equality within a single call → degenerate (passthrough) MoA.
    # Cross-call equality conveys nothing because the salt is fresh
    # per dispatch (see synthesizer._compute_draft_stats).
    #
    # Synth-decision label is set authoritatively at the wrap call site
    # by the path that produced the response (review r103 finding —
    # earlier derivation in dispatch from fastpath+fallback_reason
    # mislabeled single-success / best-of / synth-fail-then-tool-repair).
    draft_stats = synth_result.draft_stats
    draft_headers: dict[str, str] = {}
    if draft_stats.lengths:
        draft_headers["X-MetaModel-Draft-Lengths"] = ",".join(
            str(n) for n in draft_stats.lengths
        )
        draft_headers["X-MetaModel-Draft-Hashes"] = ",".join(draft_stats.hashes)
        if synth_result.synth_decision:
            draft_headers["X-MetaModel-Synth-Decision"] = synth_result.synth_decision

    # Degraded-mode signal. MoA dispatch can succeed at quorum=1/3
    # (single_success / fast-paths) when 2 generators fail — currently
    # this looks identical to a healthy 3/3 success at the HTTP layer,
    # so observability can't distinguish "MoA running normally" from
    # "MoA degenerated to passthrough because 2 upstreams are broken".
    # Surface it as an explicit header + INFO log so operators see it
    # without scraping per-draft DEBUG telemetry. The corresponding
    # WARN log lives in fanout.py with the actual upstream error body;
    # this is the dispatch-level summary.
    fails = [o for o in outcomes if isinstance(o, GeneratorFailure)]
    degraded = bool(fails)
    degraded_headers: dict[str, str] = {}
    if degraded:
        degraded_headers["X-MetaModel-Degraded"] = "true"
        # `name:reason` per failed generator. Reason is the typed
        # GeneratorFailure.reason (timeout|transport|non_2xx|non_json),
        # not the body — that's still in the WARN log emitted by
        # fanout.py / dispatch._one. Keeps the header bounded.
        degraded_headers["X-MetaModel-Failed-Generators"] = ",".join(
            f"{f.upstream_name}:{f.reason}" for f in fails
        )
        # INFO level — the per-failure body has already been logged at
        # WARN inside fanout. This is the dispatch-level summary.
        log.info(
            "moa.degraded profile=%s quorum=%d/%d failed=%s",
            profile_name,
            len(succ),
            len(layout.per_generator),
            ",".join(f"{f.upstream_name}:{f.reason}" for f in fails),
        )

    headers = {
        "X-MetaModel-Profile": profile_name,
        "X-MetaModel-Generators": str(len(layout.per_generator)),
        "X-MetaModel-Quorum": str(len(succ)),
        "X-MetaModel-Fallback-Reason": synth_result.fallback_reason,
        "X-MetaModel-Fastpath": "true" if synth_result.fastpath else "false",
        **compaction_headers,
        **draft_headers,
        **degraded_headers,
    }

    # #7: per-profile metrics ringbuffer. Pure observation — no behavior
    # change. Records the dispatch outcome for /v1/metrics/moa to
    # aggregate later. Defensive: any failure in metrics recording is
    # logged and swallowed so it can never break dispatch.
    try:
        from ..metrics import FailureRecord, MoaCallRecord, now_ms, record_moa_call

        # Pull the final response's tool_call count + content length out
        # of the synthesized payload for the rate metrics.
        final_tool_call_count = 0
        final_content_chars = 0
        try:
            choices = synth_result.response.get("choices") or []
            if choices and isinstance(choices[0], dict):
                msg = choices[0].get("message") or {}
                tcs = msg.get("tool_calls")
                if isinstance(tcs, list):
                    final_tool_call_count = len(tcs)
                content = msg.get("content")
                if isinstance(content, str):
                    final_content_chars = len(content)
        except Exception:
            pass

        # Per-failure records for /v1/metrics/moa.failure_breakdown.
        # Includes both fanout-level failures (timeout, transport,
        # non_2xx, non_json) AND constraint-violation demotions
        # (undeclared_tool, dual_shape_response) since both flavors
        # were converted into GeneratorFailure entries by this point.
        failure_records = tuple(
            FailureRecord(upstream_name=f.upstream_name, reason=f.reason)
            for f in fails
        )

        record_moa_call(
            MoaCallRecord(
                timestamp_ms=now_ms(),
                profile=profile_name,
                generators=len(layout.per_generator),
                quorum=len(succ),
                fastpath=bool(synth_result.fastpath),
                fallback_reason=synth_result.fallback_reason,
                synth_decision=synth_result.synth_decision or "",
                draft_lengths=list(draft_stats.lengths),
                final_tool_call_count=final_tool_call_count,
                final_content_chars=final_content_chars,
                elapsed_ms=int((time.monotonic() - _moa_t0) * 1000),
                failures=failure_records,
            )
        )
    except Exception as exc:  # pragma: no cover - metrics must not break dispatch
        log.warning("metrics.record_moa_call failed: %s", exc)

    return DispatchResult(
        payload=synth_result.response,
        status_code=200,
        headers=headers,
    )


async def _fan_out_per_body(
    per_gen_bodies: list[tuple[str, UpstreamConfig, dict[str, Any]]],
    *,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None,
    quorum: int | None = None,
    grace_secs: float | None = None,
) -> list:
    """Fan out with a different body per upstream.

    `fanout.fan_out` was designed for a single shared body across all
    upstreams. With shared-tail compaction we have a different
    `messages` list per generator, so we duplicate the per-upstream
    HTTP logic here but share the quorum-aware gather primitive
    (`_gather_with_quorum`) so a single slow generator can't drain
    the wall-clock budget.

    `quorum` (default `quorum_threshold(len(per_gen_bodies))`) +
    `grace_secs` (default `DEFAULT_QUORUM_GRACE_SECS`): see
    `fanout.fan_out` for semantics.
    """
    import asyncio

    from .fanout import (
        _NON_2XX_DETAIL_CHARS,
        _NON_2XX_WARN_CHARS,
        DEFAULT_QUORUM_GRACE_SECS,
        GeneratorFailure,
        GeneratorSuccess,
        _gather_with_quorum,
        _snippet_from_response,
        quorum_threshold,
    )

    async def _one(name: str, up: UpstreamConfig, body: dict[str, Any]) -> Any:
        loop = asyncio.get_running_loop()
        start = loop.time()
        try:
            resp = await asyncio.wait_for(
                forward_chat_completion(
                    up,
                    body,
                    timeout_secs=timeout_secs,
                    transport=transport,
                ),
                timeout=timeout_secs,
            )
        except (TimeoutError, httpx.TimeoutException):
            return GeneratorFailure(
                upstream_name=name,
                reason="timeout",
                detail=f"per-upstream timeout {timeout_secs}s exceeded",
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
            logging.getLogger("meta_model.fanout").warning(
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
        if not isinstance(payload, dict):
            return GeneratorFailure(
                upstream_name=name,
                reason="non_json",
                detail="upstream returned non-object JSON",
                status=resp.status_code,
                elapsed_ms=elapsed_ms,
            )
        return GeneratorSuccess(
            upstream_name=name,
            response=payload,
            elapsed_ms=elapsed_ms,
        )

    effective_quorum = (
        quorum if quorum is not None else quorum_threshold(len(per_gen_bodies))
    )
    effective_grace = (
        grace_secs if grace_secs is not None else DEFAULT_QUORUM_GRACE_SECS
    )
    factories = [
        (name, (lambda name=name, up=up, body=body: _one(name, up, body)))
        for name, up, body in per_gen_bodies
    ]
    return await _gather_with_quorum(
        factories, quorum=effective_quorum, grace_secs=effective_grace
    )


async def _passthrough_single(
    name: str,
    upstream: UpstreamConfig,
    profile_name: str,
    request_body: dict[str, Any],
    *,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None,
    extra_headers: dict[str, str] | None = None,
) -> DispatchResult:
    """Forward to a single upstream and relay status + body verbatim.

    Preserves the D.1.4 single-upstream passthrough contract: non_2xx
    responses come through with the upstream's body intact (not
    wrapped in our error envelope), so clients addressing one upstream
    directly see exactly what that upstream emitted. Non-JSON
    responses still 502 with `upstream_protocol_error` since dressing
    arbitrary bytes as a JSON body would mislead clients.
    """
    # Single-upstream passthrough doesn't run shared-tail compaction
    # (one upstream sees the body verbatim); emit Tokens-In = Tokens-Out
    # and Compacted-N = 0 so the header set stays consistent across
    # dispatch surfaces. Tools-Tokens reflects the tool schema overhead
    # the upstream will see — surface it for parity with MoA dispatch
    # observability.
    msgs = request_body.get("messages") or []
    tokens = estimate_messages_tokens(list(msgs)) if isinstance(msgs, list) else 0
    tools_tokens = _estimate_tools_tokens(request_body.get("tools"))
    headers = {
        "X-MetaModel-Profile": profile_name,
        "X-MetaModel-Generators": "1",
        "X-MetaModel-Quorum": "1",
        "X-MetaModel-Fallback-Reason": "none",
        "X-MetaModel-Compacted-N": "0",
        "X-MetaModel-Prompt-Tokens-In": str(tokens),
        "X-MetaModel-Prompt-Tokens-Out": str(tokens),
        "X-MetaModel-Tools-Tokens": str(tools_tokens),
        **(extra_headers or {}),
    }
    try:
        resp = await forward_chat_completion(
            upstream,
            request_body,
            timeout_secs=timeout_secs,
            transport=transport,
        )
    except (TimeoutError, httpx.TimeoutException):
        return _err(504, "upstream timed out", code="upstream_timeout", headers=headers)
    except httpx.RequestError as e:
        return _err(502, f"upstream transport error: {e}", code="upstream_error", headers=headers)
    try:
        payload = resp.json()
    except ValueError:
        return _err(
            502,
            "upstream returned non-JSON response",
            code="upstream_protocol_error",
            headers=headers,
        )
    if not isinstance(payload, dict):
        return _err(
            502,
            "upstream returned non-object JSON",
            code="upstream_protocol_error",
            headers=headers,
        )
    return DispatchResult(payload=payload, status_code=resp.status_code, headers=headers)


def _single_upstream_failure(outcome) -> DispatchResult:
    """Translate a single GeneratorFailure to a passthrough-equivalent error.

    Preserves the D.1.4 single-upstream semantics for clients that
    address an upstream directly: timeout → 504, transport → 502 with
    upstream_error, non_2xx → relay status with upstream_error,
    non_json → 502 with upstream_protocol_error.
    """
    from .fanout import GeneratorFailure

    if not isinstance(outcome, GeneratorFailure):
        return _err(502, "all generators failed", code="all_generators_failed")
    if outcome.reason == "timeout":
        return _err(504, "upstream timed out", code="upstream_timeout")
    if outcome.reason == "transport":
        return _err(502, f"upstream transport error: {outcome.detail}", code="upstream_error")
    if outcome.reason == "non_2xx":
        return _err(
            outcome.status or 502,
            f"upstream returned HTTP {outcome.status}",
            code="upstream_error",
        )
    if outcome.reason == "non_json":
        return _err(502, "upstream returned non-JSON response", code="upstream_protocol_error")
    if outcome.reason == "cancelled":
        # Reached only if a `cancelled` outcome ever flows into the
        # single-upstream error mapper. Today `_passthrough_single`
        # doesn't go through `_fan_out_per_body`, so this branch is
        # not exercised in current code — but keeping a defensive
        # 502 mapping protects future callers.
        return _err(502, "upstream call cancelled", code="upstream_error")
    return _err(502, "upstream call failed", code="upstream_error")


def _estimate_tools_tokens(tools: Any) -> int:
    """Rough estimate of the JSON tool-schema tokens vLLM prepends to a prompt.

    Mirrors client-core's `tools_token_estimate = bytes(json) // 3 + margin`.
    Returns 0 when no tools are present.
    """
    if not tools:
        return 0
    import json

    try:
        text = json.dumps(tools)
    except (TypeError, ValueError):
        return 0
    # bytes / 3 + small margin for the chat-template scaffolding around
    # each tool. Matches client-core's chat_with_tools_synth formula.
    return len(text) // 3 + 64


# ── Cascade dispatch ────────────────────────────────────────────────


def _is_valid_chat_completion_for_cascade(
    payload: dict[str, Any],
) -> tuple[bool, str | None]:
    """Cascade-success validator. NOT a generic OpenAI shape check —
    rejects 2xx responses that look successful at the HTTP layer but
    carry nothing usable (empty/missing content with no tool_calls).
    Without this, a 200 with `{"choices":[]}` or `{"choices":[{
    "message":{"content":""}}]}` short-circuits the cascade and
    forces the client to fail open instead of trying the next
    upstream. Generic "OpenAI shape" would not reject those — but
    cascading does. No verdict semantics; the meta-model never
    parses VERDICT/OK/FAKE/etc.

    Review r2 HIGH: guard `choices[0]` is a dict before .get().
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, "missing or empty choices"
    first = choices[0]
    if not isinstance(first, dict):
        return False, "choices[0] is not an object"
    msg = first.get("message")
    if not isinstance(msg, dict):
        return False, "choices[0].message missing or non-object"
    content = msg.get("content")
    tool_calls = msg.get("tool_calls")
    has_text = isinstance(content, str) and content.strip()
    has_tool_calls = isinstance(tool_calls, list) and tool_calls
    # F3: thinking-model upstreams may return content empty with the
    # actual answer in `reasoning_content`/`reasoning`. The F3
    # sanitizer rescues the text into `content` downstream, so cascade
    # validation must accept that shape as a valid success — otherwise
    # the cascade falls through to the next upstream on a perfectly
    # good reasoning-only response.
    has_reasoning_rescue = has_reasoning_rescue_text(msg)
    if not has_text and not has_tool_calls and not has_reasoning_rescue:
        return False, "empty content and no tool_calls"
    return True, None


async def _dispatch_cascade(
    profile: CascadeProfile,
    profile_name: str,
    cfg: MetaModelConfig,
    request_body: dict[str, Any],
    *,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None,
) -> DispatchResult:
    tried: list[str] = []
    last_status: int | None = None
    last_detail: str | None = None
    # Cascade forwards body verbatim per upstream → no compaction. Emit
    # the standard observability header set with zero deltas so clients
    # see one stable schema across MoA / cascade / voting / passthrough
    # (review r22 — header parity).
    msgs = request_body.get("messages") or []
    cascade_tokens = estimate_messages_tokens(list(msgs)) if isinstance(msgs, list) else 0
    cascade_tools_tokens = _estimate_tools_tokens(request_body.get("tools"))
    parity_headers = {
        "X-MetaModel-Compacted-N": "0",
        "X-MetaModel-Prompt-Tokens-In": str(cascade_tokens),
        "X-MetaModel-Prompt-Tokens-Out": str(cascade_tokens),
        "X-MetaModel-Tools-Tokens": str(cascade_tools_tokens),
    }
    # Cascade-wide deadline. Review r2 HIGH: per-upstream
    # `timeout_secs` × N upstreams was the historic ceiling, so two
    # hung judges could take ~2× the configured budget. Track a
    # monotonic deadline and pass the remaining budget to each
    # attempt; once exhausted, emit cascade_exhausted with last
    # detail intact.
    cascade_deadline = time.monotonic() + max(timeout_secs, 0.0)
    for name in profile.upstreams:
        up = cfg.upstreams.get(name)
        if up is None:
            continue
        remaining = cascade_deadline - time.monotonic()
        if remaining <= 0.0:
            last_status = 504
            last_detail = "cascade deadline exceeded before next attempt"
            break
        tried.append(name)
        try:
            resp = await forward_chat_completion(
                up,
                request_body,
                timeout_secs=remaining,
                transport=transport,
            )
        except (TimeoutError, httpx.TimeoutException):
            last_status = 504
            last_detail = f"upstream {name!r} timed out"
            continue
        except httpx.RequestError as e:
            last_status = 502
            last_detail = f"upstream {name!r} transport error: {e}"
            continue
        if resp.status_code < 200 or resp.status_code >= 300:
            last_status = resp.status_code
            last_detail = f"upstream {name!r} returned HTTP {resp.status_code}"
            continue
        try:
            payload = resp.json()
        except ValueError:
            last_status = 502
            last_detail = f"upstream {name!r} returned non-JSON"
            continue
        if not isinstance(payload, dict):
            last_status = 502
            last_detail = f"upstream {name!r} returned non-object JSON"
            continue
        ok, shape_reason = _is_valid_chat_completion_for_cascade(payload)
        if not ok:
            last_status = 502
            last_detail = f"upstream {name!r} returned malformed ChatCompletion: {shape_reason}"
            continue
        headers = {
            "X-MetaModel-Profile": profile_name,
            "X-MetaModel-Cascade-Tried": ",".join(tried),
            "X-MetaModel-Cascade-Winner": name,
            **parity_headers,
        }
        return DispatchResult(payload=payload, status_code=200, headers=headers)

    # All upstreams exhausted.
    headers = {
        "X-MetaModel-Profile": profile_name,
        "X-MetaModel-Cascade-Tried": ",".join(tried) or "",
        "X-MetaModel-Cascade-Exhausted": str(len(tried)),
        # Review r2 MED: surface last_detail on structured_502 so the
        # client can reproduce today's per-attempt log lines without
        # the server inventing a verdict-shape failure code.
        "X-MetaModel-Cascade-Last-Detail": last_detail or "all cascade upstreams failed",
        "X-MetaModel-Cascade-Last-Status": str(last_status) if last_status else "",
        **parity_headers,
    }
    if profile.on_all_fail == "bubble_last_error":
        return _err(
            last_status or 502,
            last_detail or "all cascade upstreams failed",
            code="cascade_exhausted",
            headers=headers,
        )
    # structured_502: emit canonical message with the last detail
    # appended so an ops dashboard or client log line can recover the
    # last upstream's failure without fishing through headers.
    structured_message = "all cascade upstreams failed"
    if last_detail:
        structured_message = f"{structured_message}: {last_detail}"
    return _err(
        502,
        structured_message,
        code="cascade_exhausted",
        headers=headers,
    )


# ── Voting dispatch ─────────────────────────────────────────────────


async def _dispatch_voting(
    profile: VotingProfile,
    profile_name: str,
    cfg: MetaModelConfig,
    request_body: dict[str, Any],
    *,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None,
) -> DispatchResult:
    upstreams_with_cfg: list[tuple[str, UpstreamConfig]] = [
        (name, cfg.upstreams[name]) for name in profile.upstreams if name in cfg.upstreams
    ]
    if len(upstreams_with_cfg) < 2:
        return _err(503, "voting profile needs ≥2 upstreams", code="no_upstream")

    body = dict(request_body)
    body["temperature"] = profile.temperature
    body["max_tokens"] = profile.max_tokens
    # Strip max_completion_tokens too — when the client sent it
    # (legal alone, only conflicts with max_tokens), keeping it
    # alongside the profile-owned max_tokens forms an upstream-rejecting
    # combo (review r18 finding 2). Profile wins.
    body.pop("max_completion_tokens", None)
    # Strip incompatible knobs that confuse YES/NO classifiers. (D.3.1
    # adds parallel_tool_calls to the strip list — even though tools
    # are gone, leaving the flag would propagate a pointless field.)
    body.pop("tools", None)
    body.pop("tool_choice", None)
    body.pop("parallel_tool_calls", None)
    body.pop("response_format", None)

    per_voter_bodies: list[tuple[str, UpstreamConfig, dict[str, Any]]] = [
        (name, up, prepare_upstream_body(body, up)) for name, up in upstreams_with_cfg
    ]

    outcomes = await _fan_out_per_body(
        per_voter_bodies,
        timeout_secs=timeout_secs,
        transport=transport,
    )

    votes: list[str] = []
    failures = 0
    for o in outcomes:
        from .fanout import GeneratorFailure

        if isinstance(o, GeneratorFailure):
            failures += 1
            votes.append(profile.failure_vote)
            continue
        # Extract first content token, lowercase, classify yes/no.
        text = _extract_first_text(o.response)
        votes.append(_classify_vote(text, default=profile.failure_vote))

    if profile.aggregation == "any_yes":
        verdict = "yes" if any(v == "yes" for v in votes) else "no"
    else:  # pragma: no cover - schema only allows any_yes today
        verdict = "no"

    payload = _voting_response(profile_name, verdict, votes)
    # Voting strips `tools` before forwarding, so Tools-Tokens=0 even
    # if the client sent some. Compaction doesn't run; In=Out per
    # voter (each saw the same body). Header parity with MoA / cascade
    # / passthrough (review r22).
    msgs = request_body.get("messages") or []
    vote_tokens = estimate_messages_tokens(list(msgs)) if isinstance(msgs, list) else 0
    headers = {
        "X-MetaModel-Profile": profile_name,
        "X-MetaModel-Voters": str(len(upstreams_with_cfg)),
        "X-MetaModel-Vote-Failures": str(failures),
        "X-MetaModel-Verdict": verdict,
        "X-MetaModel-Compacted-N": "0",
        "X-MetaModel-Prompt-Tokens-In": str(vote_tokens),
        "X-MetaModel-Prompt-Tokens-Out": str(vote_tokens),
        "X-MetaModel-Tools-Tokens": "0",
    }
    return DispatchResult(payload=payload, status_code=200, headers=headers)


def _extract_first_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                t = part.get("text")
                if isinstance(t, str):
                    return t
    return ""


def _classify_vote(text: str, *, default: str) -> str:
    """Classify a YES/NO vote from free text.

    Picks the first standalone yes/no token. Conservative on
    ambiguity: returns the configured `failure_vote`. Designed for the
    short-output (max_tokens=5) probe surface, not free-form prose.
    """
    if not text:
        return default
    lower = text.strip().lower()
    if lower.startswith("yes"):
        return "yes"
    if lower.startswith("no"):
        return "no"
    # Fallback: scan first 32 chars for whichever appears first.
    head = lower[:32]
    yes_pos = head.find("yes")
    no_pos = head.find("no")
    if yes_pos == -1 and no_pos == -1:
        return default
    if yes_pos == -1:
        return "no"
    if no_pos == -1:
        return "yes"
    return "yes" if yes_pos < no_pos else "no"


def _voting_response(profile_name: str, verdict: str, votes: list[str]) -> dict[str, Any]:
    """Build a synthetic ChatCompletion carrying the verdict.

    Voting profiles return an OpenAI-shape response so generic clients
    can consume them — `content` is the verdict, and per-voter detail
    is logged in the X-MetaModel-* headers.
    """
    import time
    import uuid

    return {
        "id": "metamodel-vote-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": profile_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": verdict},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1},
        "x_meta_model": {"votes": votes, "verdict": verdict},
    }


# ── F6: server-level multimodal cascade ─────────────────────────────


def _missing_capability_err(kind: str) -> DispatchResult:
    """Standard envelope when a multimodal request hits a server with
    no `[<kind>].endpoints` configured. Review r1 MED: 501 maps to
    `api_error` by default but the spec wants `invalid_request_error`
    for "this server doesn't support this modality" — explicitly
    override.
    """
    return _err(
        501,
        f"{kind} input is not supported by this server "
        f"(no upstreams configured in [{kind}].endpoints)",
        code=f"{kind}_not_supported",
        error_type="invalid_request_error",
    )


async def _multimodal_cascade(
    endpoint_names: list[str],
    cfg: MetaModelConfig,
    request_body: dict[str, Any],
    *,
    kind: str,
    profile_name: str,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None,
) -> DispatchResult:
    """Try each upstream in `endpoint_names` order. First 2xx wins.

    On 4xx/5xx or transport error: log + record + try next. The
    cascade enforces a **wall-clock deadline** of `timeout_secs` (review
    r1 F6 HIGH): each attempt gets the remaining budget, never the
    full per-call timeout, so N hung upstreams cannot multiply the
    configured ceiling. Mirrors the deadline logic in
    `_dispatch_cascade`.

    On exhaustion, returns whichever failure was last recorded —
    HTTP response if any non-2xx came back, else a structured 502 if
    every attempt was a transport exception (review r1 F6 LOW: track
    last failure explicitly to preserve "last error" semantics).

    The whole request body forwards verbatim — tools, response_format,
    temperature all flow through. The `model` field gets rewritten
    per-upstream by `prepare_upstream_body`.
    """
    attempts: list[str] = []  # "<name>:<status>" / "<name>:transport_error"
    last_response: httpx.Response | None = None
    last_transport_err: Exception | None = None
    last_failure_was_transport = False  # review r1 LOW: track which fired most recently
    cascade_deadline = time.monotonic() + max(timeout_secs, 0.0)

    for name in endpoint_names:
        upstream = cfg.upstreams.get(name)
        if upstream is None:
            # Cross-validator should have caught this at config load.
            attempts.append(f"{name}:not_configured")
            continue
        remaining = cascade_deadline - time.monotonic()
        if remaining <= 0.0:
            # Deadline already exceeded — record + bail.
            attempts.append(f"{name}:deadline_exceeded")
            last_transport_err = TimeoutError(
                f"cascade deadline exceeded before {kind} upstream {name!r}"
            )
            last_failure_was_transport = True
            break
        try:
            resp = await forward_chat_completion(
                upstream, request_body, timeout_secs=remaining, transport=transport
            )
        except Exception as exc:  # transport / DNS / timeout
            log.warning("[%s] cascade upstream %s transport error: %s", kind, name, exc)
            attempts.append(f"{name}:transport_error")
            last_transport_err = exc
            last_failure_was_transport = True
            continue
        attempts.append(f"{name}:{resp.status_code}")
        if 200 <= resp.status_code < 300:
            try:
                payload = resp.json()
            except Exception:
                # Upstream returned 2xx with non-JSON body — treat as
                # malformed and continue cascade.
                log.warning("[%s] cascade upstream %s 2xx but non-JSON body", kind, name)
                last_response = resp
                last_failure_was_transport = False
                continue
            # Review r2 F6 MED: preserve single-model transparency by
            # rewriting the response's `model` field to the canonical
            # profile/alias the client requested, and by emitting the
            # standard `X-MetaModel-Profile` header so downstream
            # treats the cascade response identically to a profile
            # response.
            if isinstance(payload, dict):
                payload["model"] = profile_name
            return DispatchResult(
                payload=payload,
                status_code=resp.status_code,
                headers={
                    "X-MetaModel-Profile": profile_name,
                    "X-MetaModel-Multimodal-Path": kind,
                    "X-MetaModel-Multimodal-Upstream": name,
                    "X-MetaModel-Multimodal-Attempts": ",".join(attempts),
                },
            )
        # Non-2xx: log + remember + continue.
        log.info("[%s] cascade upstream %s returned %d", kind, name, resp.status_code)
        last_response = resp
        last_failure_was_transport = False

    # Cascade exhausted. Surface whichever failure was actually last.
    # Review r1 LOW: prefer the temporally-last attempt, not just
    # "any HTTP response we ever saw".
    headers = {
        "X-MetaModel-Profile": profile_name,
        "X-MetaModel-Multimodal-Path": kind,
        "X-MetaModel-Multimodal-Attempts": ",".join(attempts),
    }
    if last_failure_was_transport or last_response is None:
        return _err(
            502,
            f"all {kind} cascade upstreams unreachable: {last_transport_err}",
            code=f"{kind}_cascade_unreachable",
            headers=headers,
        )
    try:
        payload = last_response.json()
    except Exception:
        payload = None
    if payload is not None:
        return DispatchResult(
            payload=payload,
            status_code=last_response.status_code,
            headers=headers,
        )
    return _err(
        last_response.status_code,
        f"all {kind} cascade upstreams failed; last upstream returned non-JSON body",
        code=f"{kind}_cascade_exhausted",
        headers=headers,
    )


# ── Top-level entry ─────────────────────────────────────────────────


async def dispatch(
    cfg: MetaModelConfig,
    request_body: dict[str, Any],
    *,
    model: str,
    ext_profile: str | None,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> DispatchResult:
    """Resolve `model` / `x_meta_model.profile` and dispatch.

    Returns a ``DispatchResult`` carrying the OpenAI-shape response
    body (or error tuple), HTTP status, and X-MetaModel-* headers.
    """
    # D.3.1: normalize legacy `functions` / `function_call` onto modern
    # `tools` / `tool_choice` and reject out-of-scope shapes (custom
    # tools, allowed_tools, novel tool_choice payloads). Runs once at
    # entry so every dispatch path sees the canonical shape.
    try:
        request_body = normalize_tool_request(request_body)
    except ToolNormalizationError as e:
        return _err(e.status, e.message, code=e.code)

    # Server-owned system prompt injection. When `[server] system_prompt`
    # is set in config, prepend a leading system message to every
    # request before any dispatch branch runs. Caller's own system
    # messages follow ours, preserving their task framing while
    # ensuring the operator's identity / branding contract is the FIRST
    # thing every upstream sees. Empty / None → no behavior change.
    # No heuristic gating on message content: every chat request gets
    # the same injection unconditionally, every upstream sees the same
    # leading system message, full passthrough on the body shape (we
    # don't deep-copy or rewrite anything else).
    sp = (cfg.server.system_prompt or "").strip()
    if sp:
        msgs = request_body.get("messages")
        new_msgs: list[Any] = [{"role": "system", "content": sp}]
        if isinstance(msgs, list):
            new_msgs.extend(msgs)
        request_body = dict(request_body)
        request_body["messages"] = new_msgs

    # D.3.2 + F6: detect message modality + reject malformed parts
    # globally. F6 also recognizes video/audio at detection time;
    # whether the server can serve them is decided by the multimodal
    # cascade short-circuit further down.
    modality = detect_message_modality(request_body.get("messages") or [])
    if modality.unsupported_parts:
        return _err(
            400,
            "request contains unsupported content part type(s): "
            + ", ".join(modality.unsupported_parts),
            code="unsupported_content_part",
        )

    # Resolve the profile FIRST — even for multimodal requests. Review
    # r1 F6 MED: short-circuiting before resolution would let
    # `model="does-not-exist"` succeed via the multimodal cascade.
    # We still bypass profile-based routing for multimodal (the
    # cascade is server-level), but the model field has to name a
    # real, callable profile / upstream / alias.
    profile, profile_name, code = resolve_profile(cfg, model, ext_profile)
    if profile is None:
        if code == "feature_disabled":
            return _err(
                400,
                f"profile {profile_name!r} requires [features].voting = true",
                code="feature_disabled",
            )
        target = profile_name or model
        return _err(404, f"model {target!r} is not configured", code="model_not_found")

    # F6: server-level multimodal cascade. Bypasses profile dispatch
    # entirely — single-model transparency. The selected profile is
    # irrelevant for ROUTING once the request carries non-text content;
    # validation already happened above. Empty array → standard OpenAI
    # "modality not supported" envelope.
    resolved_name = profile_name or model
    if modality.has_images:
        if not cfg.vision.endpoints:
            return _missing_capability_err("vision")
        return await _multimodal_cascade(
            cfg.vision.endpoints, cfg, request_body,
            kind="vision", profile_name=resolved_name,
            timeout_secs=timeout_secs, transport=transport,
        )
    if modality.has_videos:
        if not cfg.video.endpoints:
            return _missing_capability_err("video")
        return await _multimodal_cascade(
            cfg.video.endpoints, cfg, request_body,
            kind="video", profile_name=resolved_name,
            timeout_secs=timeout_secs, transport=transport,
        )
    if modality.has_audios:
        if not cfg.audio.endpoints:
            return _missing_capability_err("audio")
        return await _multimodal_cascade(
            cfg.audio.endpoints, cfg, request_body,
            kind="audio", profile_name=resolved_name,
            timeout_secs=timeout_secs, transport=transport,
        )

    if isinstance(profile, MoaProfile):
        return await _dispatch_moa(
            profile,
            profile_name or model,
            cfg,
            request_body,
            modality=modality,
            timeout_secs=timeout_secs,
            transport=transport,
        )
    if isinstance(profile, CascadeProfile):
        return await _dispatch_cascade(
            profile,
            profile_name or model,
            cfg,
            request_body,
            timeout_secs=timeout_secs,
            transport=transport,
        )
    if isinstance(profile, VotingProfile):
        return await _dispatch_voting(
            profile,
            profile_name or model,
            cfg,
            request_body,
            timeout_secs=timeout_secs,
            transport=transport,
        )
    return _err(500, f"unknown profile type for {profile_name!r}", code="internal_error")


__all__ = [
    "DispatchResult",
    "dispatch",
    "resolve_profile",
]
