"""D.3.1 — tool-calling helpers (legacy normalization + deterministic finalize).

This module owns three concerns that touch every dispatch path:

1. **Request-shape normalization** (`normalize_tool_request`). Maps the
   legacy OpenAI fields `functions` / `function_call` onto the modern
   `tools` / `tool_choice` shape, drops the legacy fields, and rejects
   shapes outside D.3.1's function-tools-only scope (custom tools,
   `allowed_tools`, novel `tool_choice` payloads).

2. **Constraint resolution** (`resolve_tool_constraint`). Returns a
   typed `ToolConstraint` describing the active `tool_choice` and
   `parallel_tool_calls` for a request, with OpenAI's defaults applied:
   absent `tool_choice` ∧ `tools` present → `auto`; absent ∧ no tools →
   `none`. `parallel_tool_calls` defaults to `True`.

3. **Response finalization** (`finalize_response`). Every return path
   in the synthesizer flows through this finalizer so constraint
   violations cannot bypass enforcement on fast paths. Order:
     a. parallel_tool_calls=false → trim retained calls to one
     b. tool_choice="none" → strip tool_calls; recompute finish_reason
     c. tool_choice="required" or specific function → if violated,
        deterministic fallback to the most-frequent satisfying
        candidate signature (ties broken by generator order). If no
        candidate satisfies, surface a typed error so the caller can
        return a 502.
     d. finish_reason coherence (always last).

   Fallback signature comparison ignores tool-call IDs (per F-fallback
   rule) and runs AFTER deterministic parallel trimming so a candidate
   that satisfied the constraint with `parallel=false` after trim is
   considered. The chosen candidate's actual response (with its
   original IDs/shape) is returned — we don't fabricate a synthetic
   message.

D.3.1 scope: function tools only. Custom tools, `allowed_tools`, and
exotic `tool_choice` shapes 400 with `feature_not_supported_in_v1` so
clients fail loud. Mixed legacy + modern fields in the same request
also 400 (intentional divergence from OpenAI's silent-modern-wins
behavior — a deterministic proxy must reject ambiguous input).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

# ── Public types ────────────────────────────────────────────────────


class ToolNormalizationError(Exception):
    """Raised by `normalize_tool_request` for unsupported / mixed shapes.

    `code` follows the OpenAI error envelope's `code` field. `status`
    is the HTTP status the dispatcher should surface.
    """

    def __init__(self, message: str, *, code: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


ToolChoiceMode = Literal["none", "auto", "required", "specific"]


@dataclass(frozen=True)
class ToolConstraint:
    """Resolved tool-call constraints for a single request."""

    mode: ToolChoiceMode
    # Set only when mode == "specific"
    forced_function_name: str | None
    parallel_tool_calls: bool
    has_tools: bool
    # Frozenset of declared tool names from `request.tools[*].function.name`.
    # Empty when `tools` is absent or empty. Every candidate's emitted call
    # name MUST be a member; otherwise the candidate is treated as a
    # constraint-violation failure (ViolationReason.undeclared_tool).
    declared_tool_names: frozenset[str] = frozenset()

    @property
    def tool_arbitration_needed(self) -> bool:
        """Conservative request-side classification: True when the
        constraint *could* require tool reasoning. Includes any mode
        beyond `none`. The synthesizer applies a tighter
        candidate-aware rule (`_is_tool_arbitration_needed`) so the
        `auto + tools + all-candidates-text-only` case keeps the
        existing text best-of LLM judge per review r25.
        """
        return self.mode in ("required", "specific", "auto")


# Closed vocabulary for constraint-violation reasons. Header / metric
# carry these strings unchanged; the actual offending tool name is
# logged at WARN, not surfaced through public observability (cardinality
# protection). Keep this list small and stable — operators grep on
# these values.
ViolationReason = Literal["undeclared_tool", "dual_shape_response"]


@dataclass(frozen=True)
class FinalizeError:
    code: str
    message: str
    fallback_reason: str  # also surfaced as the header value


@dataclass
class FinalizeOutcome:
    """Result of running a candidate (or synth) message through the
    constraint finalizer."""

    msg: dict[str, Any]
    fallback_reason: str | None  # None when no transform/fallback applied
    error: FinalizeError | None  # set when constraint cannot be satisfied
    # When finalize selected a fallback candidate, this is its index
    # in the `candidates` list. The caller wraps `msg` with that
    # candidate's full response (id / model / usage / choice
    # finish_reason) — splicing into the synth/primary wrapper would
    # mix metadata across responses (review r26 P1).
    fallback_idx: int | None = None


# ── Canonicalization (used by finalize + signatures) ───────────────


def canonical_args(args: Any) -> str:
    """Stable JSON form of a tool-call arguments field.

    OpenAI emits tool-call arguments as a JSON-encoded string. Sort
    keys + json.dumps so structurally-equal args canonicalize equal
    despite key-order differences. Non-JSON falls back to the raw
    string; non-string non-dict/list falls back to repr.
    """
    if args is None:
        return ""
    if isinstance(args, str):
        try:
            return json.dumps(json.loads(args), sort_keys=True)
        except (ValueError, TypeError):
            return args
    try:
        return json.dumps(args, sort_keys=True)
    except (ValueError, TypeError):
        return repr(args)


def normalize_tool_calls_signature(tcs: Any) -> tuple[tuple[str, str], ...]:
    """Comparable form for a tool_calls list.

    Each call → `(function_name, canonical_args)`. ID is excluded so
    different generators producing structurally-equal calls compare
    equal regardless of generated id.
    """
    if not isinstance(tcs, list):
        return ()
    out: list[tuple[str, str]] = []
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = str(fn.get("name", ""))
        args = fn.get("arguments")
        out.append((name, canonical_args(args)))
    return tuple(out)


# ── Legacy normalization ───────────────────────────────────────────


def _classify_tool_def(td: Any) -> str:
    """Three-way classification of a tools[*] entry:

    - "ok"             — `{type: "function", function: {...}}` shape
    - "unsupported"    — explicit non-function `type` (custom, novel)
    - "malformed"      — wrong outer shape, MISSING `type`, or
                         missing/null function body

    Review r26 P2: keep `feature_not_supported_in_v1` for actually-
    unsupported features, and surface malformed function-typed entries
    as `invalid_request_error` so clients can fix their request.
    Review r27: missing `type` is request-shape malformation, not
    unsupported feature — distinguish via `"type" in td`.
    """
    if not isinstance(td, dict):
        return "malformed"
    if "type" not in td:
        return "malformed"
    t = td.get("type")
    if t != "function":
        return "unsupported"
    fn = td.get("function")
    if not isinstance(fn, dict):
        return "malformed"
    return "ok"


def _classify_tool_choice(tc: Any) -> str:
    """Three-way classification of a tool_choice dict:

    - "ok"             — `{type: "function", function: {name: <str>}}`
    - "unsupported"    — explicit non-function `type`
    - "malformed"      — wrong outer shape, MISSING `type`, or
                         function-typed missing/empty name body

    Review r27: missing `type` is malformed, not unsupported.
    """
    if not isinstance(tc, dict):
        return "malformed"
    if "type" not in tc:
        return "malformed"
    t = tc.get("type")
    if t != "function":
        return "unsupported"
    fn = tc.get("function")
    if not isinstance(fn, dict):
        return "malformed"
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        return "malformed"
    return "ok"


def normalize_tool_request(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy `functions` / `function_call` fields onto modern
    `tools` / `tool_choice`, validate the request stays inside the
    D.3.1 function-tools scope, and return a SHALLOW-COPIED body with
    only the modern fields populated.

    Raises `ToolNormalizationError` on:
    - mixed legacy + modern (`functions` AND `tools`, `function_call`
      AND `tool_choice`): intentional 400, deterministic-proxy rule.
    - `tools` containing non-`function` types (custom tools).
    - `allowed_tools` (any value).
    - `tool_choice` dict shape that isn't `{type:"function",
      function:{name:str}}`.

    Empty `tools=[]` / `functions=[]` are no-ops (consistent with
    OpenAI's permissive treatment).
    """
    out = dict(body)

    has_functions = "functions" in out and out["functions"] is not None
    has_tools = "tools" in out and out["tools"] is not None
    has_function_call = "function_call" in out and out["function_call"] is not None
    has_tool_choice = "tool_choice" in out and out["tool_choice"] is not None
    has_allowed_tools = out.get("allowed_tools") is not None

    if has_allowed_tools:
        raise ToolNormalizationError(
            "`allowed_tools` is not supported in this version (D.3.1 covers "
            "function tools + tool_choice only)",
            code="feature_not_supported_in_v1",
        )

    # Treat empty-array shapes as absent.
    if isinstance(out.get("functions"), list) and not out["functions"]:
        out.pop("functions", None)
        has_functions = False
    if isinstance(out.get("tools"), list) and not out["tools"]:
        out.pop("tools", None)
        has_tools = False

    if has_functions and has_tools:
        raise ToolNormalizationError(
            "request mixes legacy `functions` with modern `tools`; pick one",
            code="invalid_request_error",
        )
    if has_function_call and has_tool_choice:
        raise ToolNormalizationError(
            "request mixes legacy `function_call` with modern `tool_choice`; pick one",
            code="invalid_request_error",
        )

    # Promote legacy functions → tools.
    if has_functions:
        funcs = out.pop("functions")
        if not isinstance(funcs, list):
            raise ToolNormalizationError("`functions` must be a list", code="invalid_request_error")
        out["tools"] = [{"type": "function", "function": fn} for fn in funcs]
        has_tools = True

    # Promote legacy function_call → tool_choice.
    if has_function_call:
        fc = out.pop("function_call")
        if isinstance(fc, str):
            if fc in ("none", "auto"):
                out["tool_choice"] = fc
            else:
                raise ToolNormalizationError(
                    f"unsupported `function_call` string {fc!r}",
                    code="invalid_request_error",
                )
        elif isinstance(fc, dict):
            name = fc.get("name")
            if not isinstance(name, str) or not name:
                raise ToolNormalizationError(
                    "legacy `function_call` dict must carry a non-empty name",
                    code="invalid_request_error",
                )
            out["tool_choice"] = {"type": "function", "function": {"name": name}}
        else:
            raise ToolNormalizationError(
                "`function_call` must be a string or object",
                code="invalid_request_error",
            )

    # Validate tools shape (post-normalization). Review r26 P2: split
    # malformed-shape (invalid_request) from unsupported-feature (custom
    # tools) so clients see the right error class.
    if has_tools:
        tools = out.get("tools")
        if not isinstance(tools, list):
            raise ToolNormalizationError("`tools` must be a list", code="invalid_request_error")
        for td in tools:
            kind = _classify_tool_def(td)
            if kind == "unsupported":
                raise ToolNormalizationError(
                    "this server only supports function tools "
                    '(`{type:"function", function:{...}}`); custom tools '
                    "are not supported in D.3.1",
                    code="feature_not_supported_in_v1",
                )
            if kind == "malformed":
                raise ToolNormalizationError(
                    'malformed `tools` entry: must match `{type:"function", function:{...}}`',
                    code="invalid_request_error",
                )

    # Validate tool_choice shape (post-normalization). Same split.
    tc = out.get("tool_choice")
    if tc is not None and not isinstance(tc, str):
        kind = _classify_tool_choice(tc)
        if kind == "unsupported":
            raise ToolNormalizationError(
                '`tool_choice` object must use `type="function"`; other '
                "tool_choice types are not supported in D.3.1",
                code="feature_not_supported_in_v1",
            )
        if kind == "malformed":
            raise ToolNormalizationError(
                "malformed `tool_choice` object: must match "
                '`{type:"function", function:{name: <str>}}`',
                code="invalid_request_error",
            )

    if isinstance(tc, str) and tc not in ("none", "auto", "required"):
        raise ToolNormalizationError(
            f"`tool_choice` string must be 'none' | 'auto' | 'required', got {tc!r}",
            code="invalid_request_error",
        )

    # Forced-name × declared-tools coherence (review r2-HIGH): a request
    # forcing `tool_choice={type:"function", function:{name:X}}` is only
    # satisfiable if X is in `tools[*].function.name`. Reject impossible
    # contracts up-front rather than letting the inference path freelance
    # and trip the candidate-level declared-name check downstream. Ditto
    # if `tools` is absent and a name is forced — there's nothing to
    # force the model toward.
    if isinstance(tc, dict):
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        forced_name = fn.get("name") if isinstance(fn, dict) else None
        if isinstance(forced_name, str) and forced_name:
            declared = _declared_tool_names_from_body(out)
            if not declared:
                raise ToolNormalizationError(
                    f"`tool_choice` forces function {forced_name!r} but no `tools` are declared",
                    code="invalid_request_error",
                )
            if forced_name not in declared:
                raise ToolNormalizationError(
                    f"`tool_choice` forces function {forced_name!r} which is not in declared `tools`",
                    code="invalid_request_error",
                )

    return out


def _declared_tool_names_from_body(body: dict[str, Any]) -> frozenset[str]:
    """Extract the set of declared tool names from a (post-normalization)
    request body. Empty when `tools` is absent / empty / malformed.
    Operates on the post-normalize shape so legacy `functions` are
    already promoted to `tools`."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return frozenset()
    names: set[str] = set()
    for td in tools:
        if not isinstance(td, dict):
            continue
        fn = td.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return frozenset(names)


# ── Constraint resolution ──────────────────────────────────────────


def resolve_tool_constraint(body: dict[str, Any]) -> ToolConstraint:
    """Compute the active tool constraint with OpenAI defaults.

    Defaults (mirroring OpenAI):
    - tool_choice absent + tools present → "auto"
    - tool_choice absent + no tools → "none"
    - parallel_tool_calls absent → True
    """
    tools = body.get("tools")
    has_tools = isinstance(tools, list) and len(tools) > 0
    tc = body.get("tool_choice")

    forced: str | None = None
    if isinstance(tc, str):
        if tc == "specific":
            # No legitimate string form; reserved literal in our type
            mode: ToolChoiceMode = "auto" if has_tools else "none"
        else:
            mode = tc  # type: ignore[assignment]
    elif isinstance(tc, dict):
        # normalize_tool_request guarantees this is the function-named form
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        forced = str(fn.get("name", ""))
        mode = "specific"
    else:
        mode = "auto" if has_tools else "none"

    parallel = bool(body.get("parallel_tool_calls", True))
    declared = _declared_tool_names_from_body(body)
    return ToolConstraint(
        mode=mode,
        forced_function_name=forced,
        parallel_tool_calls=parallel,
        has_tools=has_tools,
        declared_tool_names=declared,
    )


# ── Response finalization ──────────────────────────────────────────


def _extract_tool_calls(msg: dict[str, Any]) -> list[dict[str, Any]]:
    tcs = msg.get("tool_calls")
    return list(tcs) if isinstance(tcs, list) else []


# Tokenizer control-token marker pattern.
#
# Matches `<|...|>` boundary markers that some chat templates / decoders
# leak into structured response fields. Empirically observed:
#   - openai-harmony (reasoning-model):  `exec<|channel|>commentary` in
#     `function.name` under malformed-JSON fallback paths in vLLM's
#     auto tool-call parser (manager-confirmed 2026-05-04).
#   - chatml / im-style / llama specials: `<|im_end|>`, `<|end|>`, etc.
#
# OpenAI's tool-name spec restricts names to `[a-zA-Z0-9_-]{1,64}` so
# a `<|...|>` substring is structurally illegal in a conformant name.
# Stripping is therefore zero-false-positive on conformant inputs and
# load-bearing for non-conformant ones.
#
# The regex matches: `<|`, then any chars except `>`, then `|>` (or a
# bare `<|...|` followed by trailing tail until the next `<|` boundary
# or end of string — captures the reasoning-model `<|channel|>commentary`
# shape where the closing `|>` is followed by a payload word).
_CONTROL_TOKEN_RE = re.compile(
    r"<\|[^|>]*\|>[a-zA-Z_]*"  # `<|name|>tail` shape — strips name + trailing tail word
    r"|<\|[^|>]*\|"            # bare `<|name|` (defensive — no closing `>`)
    r"|<\|[^|>]*"              # bare `<|name` (defensive — no closing `|`)
)


def _sanitize_tool_name(name: str) -> str:
    """Strip tokenizer control-token markers from a tool name.

    See `_CONTROL_TOKEN_RE` for the matched shapes. Returns the cleaned
    name (whitespace-stripped); returns `""` if the entire input was
    control tokens (no real name to recover — caller should treat as
    no name emitted).

    Generic on `<|...|>` boundaries — not reasoning-model-specific. Sister of
    `client-llm/src/client.rs::sanitize_tool_name` which performs the
    same cleanup on client application's response-parse path; this is the server-side
    counterpart so any client (SDK clients, browser clients, raw curl) benefits.
    """
    return _CONTROL_TOKEN_RE.sub("", name).strip()


def raw_emitted_call_names(msg: dict[str, Any]) -> list[str]:
    """Names every tool call this message proposes to invoke, drawn
    from BOTH modern `tool_calls` and legacy `function_call` shapes.

    **Mutates `msg` in place** to canonicalise tool names — strips
    tokenizer control markers (`<|channel|>commentary` and friends)
    that some upstream decoders leak. Mutation is load-bearing: every
    downstream finalizer step (signature checks, parallel-trim,
    repair, response shaping) needs to see the cleaned name; otherwise
    the declared-set check would pass but the response would still
    carry the polluted name to the caller. Review r1 confirmed.

    Used by:
      - candidate_violates_declared (declared-name check)
      - dispatch's per-candidate constraint validation
      - finalize_response (via signature derivation)

    Review r2 HIGH: legacy `function_call` on the response side is
    deprecated but supported by OpenAI. A backend emitting only
    `function_call: {name: f}` (no modern tool_calls) must be treated
    as having proposed `f`, otherwise the declared-set check leaks.
    """
    names: list[str] = []
    tcs = msg.get("tool_calls")
    if isinstance(tcs, list):
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = fn.get("name")
            if isinstance(name, str) and name:
                cleaned = _sanitize_tool_name(name)
                if cleaned:
                    if cleaned != name:
                        fn["name"] = cleaned
                    names.append(cleaned)
    fc = msg.get("function_call")
    if isinstance(fc, dict):
        name = fc.get("name")
        if isinstance(name, str) and name:
            cleaned = _sanitize_tool_name(name)
            if cleaned:
                if cleaned != name:
                    fc["name"] = cleaned
                names.append(cleaned)
    return names


def candidate_violates_declared(
    msg: dict[str, Any],
    constraint: ToolConstraint,
) -> ViolationReason | None:
    """Constraint-violation check: returns the typed reason if the
    candidate's emitted calls violate the request contract, or None
    on satisfaction.

    Two failure modes:
      - `dual_shape_response`: candidate carries BOTH `tool_calls`
        AND legacy `function_call` simultaneously. Malformed dual
        shape; OpenAI's spec mandates one or the other on response.
      - `undeclared_tool`: candidate emits a call name that isn't
        in the request's declared tool set. Includes the case where
        `tools` was absent (declared set empty) but the candidate
        emits any call at all.

    Constraint-mode violations (e.g. parallel=false with multiple
    calls, mode="none" with calls present) are NOT covered here —
    those are handled by `finalize_response` because they're
    repair-shaped (trim or strip) rather than candidate-failure.
    """
    has_tcs = isinstance(msg.get("tool_calls"), list) and len(msg["tool_calls"]) > 0
    has_fc = isinstance(msg.get("function_call"), dict)
    if has_tcs and has_fc:
        return "dual_shape_response"

    emitted = raw_emitted_call_names(msg)
    if not emitted:
        return None

    declared = constraint.declared_tool_names
    for name in emitted:
        if name not in declared:
            return "undeclared_tool"
    return None


def _normalize_response_shape(msg: dict[str, Any]) -> None:
    """Promote legacy `function_call` to modern `tool_calls` on a
    candidate message in place. Idempotent.

    After this runs, downstream finalizer steps (`_strip_tool_calls`,
    `_signature_satisfies`, parallel trim) can treat the message as
    modern-shape only. Review r2 HIGH: without this, a candidate that
    only emits legacy `function_call` would slip through `mode="none"`
    stripping (we only popped `tool_calls`) and through forced-name
    validation (we only checked `tool_calls` signature).

    The promotion synthesizes a tool_call entry with a stable id so
    downstream code that needs an id has one. The legacy field is
    dropped.
    """
    fc = msg.get("function_call")
    if not isinstance(fc, dict):
        return
    name = fc.get("name")
    args = fc.get("arguments")
    if not isinstance(name, str) or not name:
        msg.pop("function_call", None)
        return
    existing_tcs = msg.get("tool_calls")
    if not isinstance(existing_tcs, list):
        existing_tcs = []
    promoted = {
        "id": fc.get("id") or "call_legacy_promoted",
        "type": "function",
        "function": {"name": name, "arguments": args if isinstance(args, str) else ""},
    }
    msg["tool_calls"] = [*existing_tcs, promoted]
    msg.pop("function_call", None)


def _signature_satisfies(
    sig: tuple[tuple[str, str], ...],
    constraint: ToolConstraint,
) -> bool:
    """Does a tool-calls signature satisfy the constraint?

    Applied AFTER deterministic parallel trimming — so a [X, Y]
    signature under parallel=false is evaluated as [X] (the trimmed
    form). Caller is responsible for invoking with the post-trim
    signature.
    """
    if constraint.mode == "none":
        return len(sig) == 0
    if constraint.mode == "auto":
        return True
    if constraint.mode == "required":
        return len(sig) >= 1
    if constraint.mode == "specific":
        if len(sig) == 0:
            return False
        # ALL retained calls must match the forced name (per review r24
        # F4 — `tool_choice={name:X}` with parallel=true and [X, Y]
        # does not satisfy unless parallel-trim already collapsed it).
        return all(name == constraint.forced_function_name for name, _ in sig)
    return False


def _trimmed_signature(
    msg: dict[str, Any], constraint: ToolConstraint
) -> tuple[tuple[str, str], ...]:
    """Signature for a candidate AFTER deterministic parallel trimming.

    Used for fallback ranking so a candidate that becomes valid under
    parallel=false trim is considered for selection. Legacy
    `function_call` shape is normalized to modern `tool_calls` on a
    shallow copy so a candidate whose only call is in legacy form
    contributes its signature (review r2 HIGH — emitted-call abstraction
    must be uniform across every constraint check).
    """
    if isinstance(msg.get("function_call"), dict):
        msg = dict(msg)
        _normalize_response_shape(msg)
    tcs = _extract_tool_calls(msg)
    if not constraint.parallel_tool_calls and len(tcs) > 1:
        tcs = tcs[:1]
    return normalize_tool_calls_signature(tcs)


def pick_fallback_candidate(
    candidates: list[dict[str, Any]],
    constraint: ToolConstraint,
) -> tuple[int, dict[str, Any]] | None:
    """Pick a candidate satisfying `constraint` per the F-fallback rule:
    most-frequent satisfying signature, ties broken by generator order.

    Returns `(generator_index, candidate_msg)` or `None` if no candidate
    satisfies.
    """
    # Build (idx, signature) pairs over post-trim signatures so candidates
    # that become valid under parallel=false trimming participate.
    tagged: list[tuple[int, tuple[tuple[str, str], ...]]] = []
    for idx, c in enumerate(candidates):
        sig = _trimmed_signature(c, constraint)
        if _signature_satisfies(sig, constraint):
            tagged.append((idx, sig))
    if not tagged:
        return None
    # Frequency by signature, preserving first-occurrence order for ties.
    counts: dict[tuple[tuple[str, str], ...], int] = {}
    first_idx: dict[tuple[tuple[str, str], ...], int] = {}
    for idx, sig in tagged:
        counts[sig] = counts.get(sig, 0) + 1
        if sig not in first_idx:
            first_idx[sig] = idx
    # Sort: highest count desc, then first generator index asc.
    best_sig = sorted(counts.items(), key=lambda kv: (-kv[1], first_idx[kv[0]]))[0][0]
    chosen_idx = first_idx[best_sig]
    return chosen_idx, candidates[chosen_idx]


def _coerce_finish_reason(msg: dict[str, Any]) -> None:
    """Coerce finish_reason both directions to match the message's
    current emitted-call shape (modern tool_calls OR legacy
    function_call).

    OpenAI's spec: `tool_calls` finish_reason is only valid when at
    least one call remains; conversely a message that DOES carry a
    call should not report `stop`. Both stale states arise here:
    stripping calls leaves `tool_calls` stale, and post-synth fallback
    splicing can wrap a stop-reason wrapper around a chosen candidate
    that emits calls. Coerce in both directions (review r26 P2).
    Legacy `function_call` shape is treated as a present call
    (review r2 HIGH).
    """
    has_call = bool(_extract_tool_calls(msg)) or isinstance(
        msg.get("function_call"), dict
    )
    fr = msg.get("finish_reason")
    if has_call:
        if fr == "stop":
            msg["finish_reason"] = "tool_calls"
    else:
        if fr == "tool_calls":
            msg["finish_reason"] = "stop"


def _strip_tool_calls(msg: dict[str, Any]) -> None:
    """Drop tool_calls AND legacy function_call; ensure content is at
    least an empty string so the response is syntactically valid
    (OpenAI allows content=null only WITH tool_calls).

    Legacy function_call removal (review r2 HIGH): without it, a
    candidate that only emits legacy shape would survive
    `tool_choice="none"` stripping and leak a call to the caller.
    """
    msg.pop("tool_calls", None)
    msg.pop("function_call", None)
    if msg.get("content") is None:
        msg["content"] = ""
    _coerce_finish_reason(msg)


def _trim_to_first_call(msg: dict[str, Any]) -> bool:
    """Trim retained tool_calls to one. Returns True if something
    was trimmed (caller uses this to pick a fallback_reason label)."""
    tcs = _extract_tool_calls(msg)
    if len(tcs) <= 1:
        return False
    msg["tool_calls"] = [tcs[0]]
    return True


def finalize_response(
    msg: dict[str, Any],
    candidates: list[dict[str, Any]],
    constraint: ToolConstraint,
    *,
    synth_ran: bool,
) -> FinalizeOutcome:
    """Run a candidate / synth message through the constraint finalizer.

    `msg` is mutated in place — caller passes a freshly cloned dict to
    avoid leaking changes back into the original candidate list.
    `candidates` is the list of all generator candidate messages (used
    for fallback selection).

    `synth_ran` distinguishes "synth had its repair chance" from
    fast-path returns. When False AND a constraint is violated, the
    caller may choose to invoke synth-repair instead of immediately
    falling back. When True, no further repair is attempted and we
    fall back deterministically.
    """
    fallback_reason: str | None = None

    # Step 0: normalize legacy `function_call` to modern `tool_calls`
    # so every downstream step operates on a single canonical shape.
    # Without this, mode="none" stripping and parallel trimming would
    # ignore legacy-shape responses and let undeclared/forbidden calls
    # leak through.
    _normalize_response_shape(msg)

    # Step a: parallel_tool_calls=false — trim to one call.
    if not constraint.parallel_tool_calls:
        if _trim_to_first_call(msg):
            fallback_reason = "parallel_violation_trimmed"
            _coerce_finish_reason(msg)

    # Step b: tool_choice="none" — strip tool_calls.
    if constraint.mode == "none":
        had_calls = bool(_extract_tool_calls(msg))
        if had_calls:
            _strip_tool_calls(msg)
            fallback_reason = "tool_choice_none_stripped"
        else:
            _coerce_finish_reason(msg)
        # No further constraint to enforce in "none" mode.
        return FinalizeOutcome(msg=msg, fallback_reason=fallback_reason, error=None)

    # Step c: tool_choice="required" / "specific" — verify; fall back if violated.
    sig = normalize_tool_calls_signature(_extract_tool_calls(msg))
    if not _signature_satisfies(sig, constraint):
        # Fast-path constraint violation: synth has not run yet → caller
        # may invoke synth-to-repair before falling back. Signal that
        # via a sentinel error code the caller recognizes.
        if not synth_ran:
            return FinalizeOutcome(
                msg=msg,
                fallback_reason=None,
                error=FinalizeError(
                    code="constraint_violated_pre_synth",
                    message=(
                        "fast-path candidate violates tool_choice constraint; synth-repair required"
                    ),
                    fallback_reason="constraint_violated_pre_synth",
                ),
            )
        # Synth has already run — deterministic fallback pick.
        picked = pick_fallback_candidate(candidates, constraint)
        if picked is None:
            code = "tool_required_unmet" if constraint.mode == "required" else "tool_choice_unmet"
            return FinalizeOutcome(
                msg=msg,
                fallback_reason=None,
                error=FinalizeError(
                    code=code,
                    message=(
                        "no candidate or synth output satisfied the active "
                        f"tool_choice constraint ({constraint.mode})"
                    ),
                    fallback_reason=code,
                ),
            )
        chosen_idx, chosen = picked
        chosen_msg = json.loads(json.dumps(chosen))  # defensive deep-copy
        # Apply parallel trim to the chosen candidate too (signature
        # was computed post-trim; output must match).
        if not constraint.parallel_tool_calls:
            _trim_to_first_call(chosen_msg)
        # The chosen candidate's signature already satisfies; mark the
        # repair label depending on which class of repair this is.
        repair_label = (
            "tool_required_repaired" if constraint.mode == "required" else "tool_choice_repaired"
        )
        _coerce_finish_reason(chosen_msg)
        # Surface the chosen index so caller can wrap with the candidate's
        # full response (id / model / usage / choice.finish_reason) rather
        # than splicing into the synth/primary wrapper (review r26 P1).
        return FinalizeOutcome(
            msg=chosen_msg,
            fallback_reason=repair_label,
            error=None,
            fallback_idx=chosen_idx,
        )

    # Step d: declared-name post-check (review r2 HIGH).
    # Modes "auto" and "required" return True from _signature_satisfies
    # for any tool-shaped output (auto: even with empty calls; required:
    # any call). Neither distinguishes "calls a declared name" from
    # "calls an undeclared name". The synth model (or a fast-path
    # candidate that wasn't validated at dispatch level — single-success
    # fast path) can land here with an undeclared name; without a check,
    # it leaks through. Run the membership predicate explicitly.
    #
    # Skipped when `tools` is absent (declared_tool_names empty) because
    # there's no contract to enforce; the request would accept any
    # freelance call OpenAI-style. Also skipped under "specific" mode
    # because _signature_satisfies already requires the forced name.
    if (
        constraint.declared_tool_names
        and constraint.mode in ("auto", "required")
        and raw_emitted_call_names(msg)
    ):
        violation = candidate_violates_declared(msg, constraint)
        if violation is not None:
            picked = pick_fallback_candidate(candidates, constraint)
            if picked is None:
                return FinalizeOutcome(
                    msg=msg,
                    fallback_reason=None,
                    error=FinalizeError(
                        code="undeclared_tool_unrepairable",
                        message=(
                            f"output emits {violation} and no candidate satisfies the "
                            "declared-tool contract"
                        ),
                        fallback_reason="undeclared_tool_unrepairable",
                    ),
                )
            chosen_idx, chosen = picked
            chosen_msg = json.loads(json.dumps(chosen))  # defensive deep-copy
            _normalize_response_shape(chosen_msg)
            if not constraint.parallel_tool_calls:
                _trim_to_first_call(chosen_msg)
            _coerce_finish_reason(chosen_msg)
            return FinalizeOutcome(
                msg=chosen_msg,
                fallback_reason="undeclared_tool_repaired",
                error=None,
                fallback_idx=chosen_idx,
            )

    # Constraint already satisfied — finish_reason coherence is the only
    # remaining concern.
    _coerce_finish_reason(msg)
    return FinalizeOutcome(msg=msg, fallback_reason=fallback_reason, error=None)


__all__ = [
    "FinalizeError",
    "FinalizeOutcome",
    "ToolConstraint",
    "ToolNormalizationError",
    "ViolationReason",
    "_sanitize_tool_name",  # exported for tests
    "candidate_violates_declared",
    "canonical_args",
    "finalize_response",
    "normalize_tool_calls_signature",
    "normalize_tool_request",
    "pick_fallback_candidate",
    "raw_emitted_call_names",
    "resolve_tool_constraint",
]
