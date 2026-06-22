"""Synthesizer merge — combine N candidate responses into one.

D.2.2 primitive, extended in D.3.1 with tool-aware behavior. Takes the
per-upstream successes from `fanout` plus the profile's synthesis
configuration, returns a single OpenAI-shaped ChatCompletion response.

Two modes (per profile):
- `merge`  — call the synthesizer upstream with all candidates; it
             produces a new answer that combines the strongest
             insights from each.
- `best-of` — call the synthesizer upstream with all candidates and
              ask it to pick the single best candidate verbatim.

Tool-aware behavior (D.3.1):
- When `tool_arbitration_needed` (any candidate emitted tool_calls,
  or `tool_choice` ∈ {required, specific, auto+tools}), the synth
  body keeps `tools` / `tool_choice` / `parallel_tool_calls` so the
  synth model can re-emit a tool_call. A tool-aware system prompt
  replaces `_SYSTEM_MERGE`.
- `best-of` with tool arbitration switches to a deterministic
  candidate pick (no LLM call) — most-frequent satisfying signature,
  ties by generator order. Per review r24 F5: passing tools to a
  synth that emits an integer index is unsafe; deterministic pick
  preserves the "judge, not author" invariant.
- Every return path runs through `tools.finalize_response` so
  fast-path candidates that violate `tool_choice` cannot bypass
  enforcement. When a fast-path violation is detected pre-synth,
  the synthesizer is invoked for a repair attempt rather than
  immediately falling back.

Fast-paths (no synthesizer call) — gated by finalize:
- 1 success → return that candidate directly (post-finalize).
- All candidates byte-identical (content + tool_calls, whitespace-
  normalized) → return the first (`fastpath_on_agreement` profile
  flag must be set; otherwise the synth runs anyway).

The synthesizer's own response is wrapped to look like a normal
upstream Chat Completion: choices[0].message + usage. Caller (the
chat endpoint) layers X-MetaModel-* headers based on metadata.

Synthesizer prompts are intentionally generic for D.2.2/D.3.1.
Surface-specific prompts (write_synth, tool_chat, etc.) attach via
the profile in D.4 once the surface→profile mapping lands.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import MoaProfile, UpstreamConfig
from ..reasoning import (
    coerce_reasoning_into_content as _coerce_reasoning_into_content,
)
from ..reasoning import (
    is_visible_content_missing as _is_visible_content_missing,
)
from ..reasoning import (
    reasoning_fallback_text as _reasoning_fallback_text,
)
from ..upstream import forward_chat_completion
from .fanout import GeneratorSuccess, successes
from .tools import (
    ToolConstraint,
    canonical_args,
    finalize_response,
    normalize_tool_calls_signature,
    resolve_tool_constraint,
)

log = logging.getLogger(__name__)

# ── Synthesizer prompts (generic) ──────────────────────────────────


_SYSTEM_MERGE = (
    "You are synthesizing several candidate responses from independent "
    "assistants into one final answer. The candidates were given the "
    "same conversation and tool outputs.\n\n"
    "Rules:\n"
    "- Merge complementary insights from the candidates.\n"
    "- On disagreement, pick the more grounded option (prefer exact "
    "lines from tool results / file content / source data).\n"
    "- Do NOT mention the candidates or that synthesis is happening.\n"
    "- Output only the final answer, no preamble or meta-commentary."
)


_SYSTEM_MERGE_WITH_TOOLS = (
    "You are synthesizing several candidate responses from independent "
    "assistants into one final answer. The candidates were given the "
    "same conversation, the same tools, and the same tool_choice "
    "constraint.\n\n"
    "Rules:\n"
    "- If the candidates emitted tool_calls, evaluate which call is "
    "most grounded in the conversation/tool outputs. Pick the most "
    "defensible call, or emit your own if every candidate is wrong.\n"
    "- Respect the active tool_choice constraint exactly:\n"
    "  - 'none' → text-only response, no tool_calls.\n"
    "  - 'auto' → tool_call optional.\n"
    "  - 'required' → you MUST emit at least one tool_call.\n"
    "  - {type:'function',function:{name:X}} → you MUST call X.\n"
    "- Honor parallel_tool_calls=false by emitting at most one "
    "tool_call.\n"
    "- Prefer exact lines from tool results / source data over "
    "summaries.\n"
    "- Do NOT mention the candidates or that synthesis is happening."
)


_SYSTEM_BEST_OF = (
    "You are picking the single best response from several independent "
    "candidate answers to the same conversation. Read every candidate, "
    "then choose the one that is most accurate, complete, and grounded "
    "in the tool outputs / source data.\n\n"
    "Output ONLY a single integer (1, 2, 3, ...) on the first line — "
    "the index of the candidate you chose. The server returns that "
    "candidate verbatim; you do not write the answer yourself. No "
    "explanation, no markdown, just the index."
)


_CANDIDATE_HEADER = "--- candidate {idx} ---"


# ── Outcome types ──────────────────────────────────────────────────


@dataclass
class DraftStats:
    """Per-call telemetry on the candidate drafts received from generators.

    `lengths` is the byte-length of each candidate's stable serialization
    (content + tool_calls + legacy function_call concatenated with NUL
    separators). `hashes` are 16-hex (64-bit) sha256 prefixes salted
    with a per-call random salt — equality within a single call
    indicates degenerate (passthrough) MoA, divergence indicates genuine
    fan-out diversity. Cross-call hash equality conveys nothing because
    the salt is fresh each call. 64 bits is safe for the small N=3..5
    draft case and acceptable should we ever aggregate.
    """

    lengths: list[int] = field(default_factory=list)
    hashes: list[str] = field(default_factory=list)


# Synth decision labels — emitted as `X-MetaModel-Synth-Decision`.
# These are authoritative (set at the wrap call site by the path that
# actually produced the response) instead of derived from `fastpath`
# + `fallback_reason`, which conflated several distinct paths (review
# r103 finding).
SYNTH_DECISION_SINGLE_SUCCESS = "single_success"  # 1 candidate, no synth ran
SYNTH_DECISION_FASTPATH_CONSENSUS = "fastpath_consensus"  # ≥2 candidates agreed
SYNTH_DECISION_BEST_OF_PICKED = "best_of_picked"  # best-of selected a candidate
SYNTH_DECISION_MERGED = "merged"  # synth LLM ran; output adopted as-is
SYNTH_DECISION_MERGED_WITH_REPAIR = "merged_with_repair"  # synth ran, finalize repaired
SYNTH_DECISION_FALLBACK_PRIMARY = "fallback_primary"  # synth failed; primary used
# Tool-preservation fallback: under tool_choice=auto, the synth merged
# multiple drafts into a text-only response while at least one
# pre-synth candidate had emitted a tool_call. We prefer the candidate
# that called a tool over the synth's refusal text — generator intent
# wins. Structural rule (signal is `tool_calls` list non-emptiness),
# never a prose-shape detector. Bug observed 2026-05-02 with
# `what's the world news?`: primary→web_search, reasoning-model-20b→refusal,
# synth merged into refusal, dropped the tool call.
SYNTH_DECISION_TOOL_PRESERVED = "tool_preserved"


@dataclass
class SynthesizedResponse:
    """Final response in OpenAI-shaped dict + meta about how it got built."""

    response: dict[str, Any]
    fastpath: bool
    # "none" — full synth ran (or agreement fast-path with full quorum)
    # "single_success" — only one generator succeeded (reduced quorum)
    # "reduced_quorum" — multiple successes but caller had >M generators
    # "synth_failed_picked_primary" — synth call errored; first candidate used
    # D.3.1 tool-aware fallback labels:
    # "parallel_violation_trimmed" — finalize trimmed >1 calls to one
    # "tool_choice_none_stripped" — finalize stripped tool_calls
    # "tool_required_repaired" — synth produced calls when generators didn't
    # "tool_choice_repaired" — synth produced/picked the right named call
    fallback_reason: str
    n_candidates: int
    quorum: int  # how many succeeded
    # D.x.observability — per-draft stats for telemetry surfacing.
    # Empty for SynthesisFailure paths (no draft survived).
    draft_stats: DraftStats = field(default_factory=DraftStats)
    # Authoritative label for the path that produced this response.
    # Set by the wrap call site, NOT derived in dispatch (review r103).
    # One of SYNTH_DECISION_* constants below; "" only when the
    # response is constructed before the constants are introduced
    # (defensive default for dataclass shape compat).
    synth_decision: str = ""


@dataclass
class SynthesisFailure:
    reason: str
    detail: str


SynthesisOutcome = SynthesizedResponse | SynthesisFailure


# ── Helpers ────────────────────────────────────────────────────────


_WS = re.compile(r"\s+")


def _extract_assistant_message(resp: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the first choice's assistant message from a Chat Completion
    response. Returns None if the shape is unexpected."""
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        return None
    return msg


def _normalize_function_call(fc: Any) -> tuple[str, str] | None:
    """Legacy `function_call` field normalized like a single tool_call."""
    if not isinstance(fc, dict):
        return None
    return (str(fc.get("name", "")), canonical_args(fc.get("arguments")))


# Reasoning-rescue primitives moved to ``meta_model.reasoning`` (F3).
# The underscore-prefixed names are imported as aliases at the top of
# this file so all in-module callers continue working unchanged.


def _candidate_signature(msg: dict[str, Any]) -> tuple:
    """Structural comparable form for fast-path agreement check.

    Combines whitespace-normalized content + canonical tool_calls +
    canonical legacy function_call. Tuples + ignored IDs make the
    comparison resistant to trivial divergence (id values, key order
    in arguments) without folding semantically-different calls
    together (review r12)."""
    content = msg.get("content")
    # Order: missing-content check FIRST so empty-string `""` falls into
    # the reasoning-fallback path alongside `None` (sibling of the same
    # gate widening in `_draft_signature`). `isinstance(str)` would match
    # `""` first and short-circuit rescue.
    if _is_visible_content_missing(msg):
        # Thinking-model fallback (review r115 + empty-string sibling):
        # reasoning text is the actual draft when content is missing and
        # no tool calls fired.
        rc = _reasoning_fallback_text(msg)
        content_sig = _WS.sub(" ", rc.strip()) if rc else ""
    elif isinstance(content, str):
        content_sig = _WS.sub(" ", content.strip())
    else:
        # Multimodal list: serialize stably so different image_urls etc.
        # don't collapse together.
        try:
            content_sig = json.dumps(content, sort_keys=True)
        except (ValueError, TypeError):
            content_sig = repr(content)
    return (
        content_sig,
        normalize_tool_calls_signature(msg.get("tool_calls")),
        _normalize_function_call(msg.get("function_call")),
    )


def _render_candidate(msg: dict[str, Any]) -> str:
    """Render a single candidate message for inclusion in the synth
    prompt. Always shows tool_calls when present — empty-string
    content with tool_calls is a real OpenAI shape (review r12).

    Reasoning-rescue gate uses ``_is_visible_content_missing`` so that
    ``content: ""`` (and whitespace-only / empty list) take the same
    fallback path as ``content: None``. Without this, a candidate with
    ``content=""`` and a populated ``reasoning_content`` rendered as
    ``(empty)`` while ``_draft_signature`` / ``_candidate_signature``
    saw the rescued reasoning text — the synth merge prompt got the
    inconsistent view and the candidate effectively did not contribute
    to merging.
    """
    lines: list[str] = []
    content = msg.get("content")
    if _is_visible_content_missing(msg):
        # Thinking-model fallback (review r115 + empty-string sibling):
        # reasoning_content is the actual draft when visible content is
        # missing and no tool calls fired.
        fallback = _reasoning_fallback_text(msg)
        if fallback:
            lines.append(fallback)
    elif isinstance(content, str):
        lines.append(content)
    elif isinstance(content, list):
        # Multimodal: flatten text parts
        texts = [
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        ]
        flat = " ".join(t for t in texts if t)
        if flat:
            lines.append(flat)
    if msg.get("tool_calls"):
        lines.append(f"[tool_calls: {json.dumps(msg['tool_calls'], sort_keys=True)}]")
    if msg.get("function_call"):  # legacy
        lines.append(f"[function_call: {json.dumps(msg['function_call'], sort_keys=True)}]")
    if not lines:
        lines.append("(empty)")
    return "\n".join(lines)


def _extract_authority_context(
    original_body: dict[str, Any], *, max_chars: int = 16000
) -> str:
    """Pull the leading authority block from the original request: every
    system / developer message at the head of `messages`, before the
    first user/assistant/tool turn. These hold per-call context the
    caller wants the model to obey (date, persona, caller-supplied mode
    rules, tool budgets, vision warnings, etc.). The synthesizer must
    see them or it operates blind on every request-level constraint
    and can pick a candidate that contradicts them (review r106 — root
    cause of the wrong-year regression in a long-form generation case).

    `max_chars` is a soft cap on the joined block. If exceeded, we keep
    the head and tail and elide the middle. Long client prompts can be
    600+ lines; passing them verbatim into every synth call would blow
    context budgets without much extra value beyond the head (date,
    persona) and tail (most-recent constraints).
    """
    messages = original_body.get("messages")
    if not isinstance(messages, list):
        return ""
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            break
        role = msg.get("role")
        if role not in ("system", "developer"):
            break
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # Multimodal authority blocks are unusual but possible —
            # flatten text parts only.
            texts = [
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            if texts:
                parts.append("\n".join(t for t in texts if t))
    block = "\n\n".join(p for p in parts if p)
    if len(block) <= max_chars:
        return block
    # Soft cap: keep head + tail, mark elision in the middle.
    half = max_chars // 2
    return (
        block[:half]
        + f"\n\n[... {len(block) - max_chars} chars elided from authority context ...]\n\n"
        + block[-half:]
    )


def _build_synth_messages(
    system_prompt: str,
    user_input: str,
    candidates: list[dict[str, Any]],
    *,
    authority_context: str = "",
    recent_tool_chain: str = "",
    has_tool_evidence: bool = False,
) -> list[dict[str, Any]]:
    """Synthesizer prompt: composite system + (user + candidates).

    The composite system is the request's authority context (date,
    persona, mode-specific rules) followed by a clearly-marked
    SYNTHESIS TASK section that holds the merge instructions. The
    authority context is named "primary" and the synth instructions
    are explicitly subordinate (review r106 — synth merge must not
    override per-call constraints set by the caller).

    `has_tool_evidence` is a structural signal: True when the request
    carries at least one tool-role message anywhere in the conversation
    (computed by `_has_visible_tool_evidence`). It governs which
    conflict-handling rule the synth sees:
    - **True**: tool-grounding carve-out applies — don't discard
      candidate content for post-cutoff dates; treat convergent claims
      as evidence the tools surfaced real data.
    - **False**: stricter rule — convergence is draft agreement only;
      outcome claims without supporting evidence (tool results OR
      user-supplied source/log text in the prompt) are fabricated and
      must be excluded.

    `recent_tool_chain` is a text block (formatted post-user transcript)
    that gets included in the user message so the synth can read the
    actual tool output. It is independent of `has_tool_evidence` —
    the gate signals that evidence exists somewhere; the chain text
    is only the post-user portion (review r-fab-1-H1: a follow-up like
    "summarize what you scraped" has prior evidence with empty chain).
    """
    if authority_context:
        # The conflict-handling rule has two flavors. Both share the
        # explicit-constraint discard core. The tool-grounding carve-out
        # (cutoff, convergence-as-evidence) only makes sense when the
        # request actually carries tool-result evidence; without it,
        # "treat convergence as evidence the tools surfaced real data"
        # is false on its face and pulls the synth toward honoring
        # fabricated consensus (auto-mode planning regression: three
        # generators converge on "✓ 1 passed" with no test ever run).
        # When no tool evidence exists, convergence is draft agreement
        # only and outcome claims without supporting evidence must be
        # excluded. Review r-fab-1-H1 MED: "supporting evidence" includes
        # user-pasted logs/source/measurements visible in the prompt —
        # not only tool-role messages — so legitimate user-supplied
        # ground truth is not falsely classified as fabrication.
        if has_tool_evidence:
            # Review r-fab-1-H1 r2 MED: `has_tool_evidence` is broad —
            # any prior tool message selects this branch, even ones
            # unrelated to this turn's claims. In tool-heavy histories
            # (e.g. auto-mode after a web_search ran several phases
            # ago), three candidates could converge on a fabricated
            # outcome ("tests pass", "✓ 1 passed") with no test ever
            # run, and the cutoff carve-out alone would tell the synth
            # to honor it. Keep the cutoff/grounded framing for the
            # legitimate case AND add an unsupported-outcome exclusion
            # so convergence-as-evidence requires evidence visible in
            # the prompt, not just any tool-history presence.
            conflict_rule = (
                "Use the candidate responses below as drafts only. If a "
                "candidate states facts that conflict with EXPLICIT "
                "constraints in the authoritative context above (e.g. "
                "the user's stated persona, mode-specific rules, dates "
                "the conversation has explicitly fixed), discard the "
                "conflicting content. Do NOT discard candidate content "
                "because it post-dates your own training cutoff — tool "
                "results in the preceding conversation are the ground "
                "truth for this turn, and the candidates have already "
                "incorporated them. Treat convergent fact-claims across "
                "candidates as evidence the tools surfaced real data, "
                "not as evidence of fabrication. However, candidate "
                "convergence is only evidence when the claim is "
                "actually supported by evidence visible in this prompt — "
                "tool-result content, user-supplied source/log text in "
                "the conversation, or the authoritative context. If "
                "candidates assert outcomes (test results, "
                "measurements, file contents, fetched facts) that are "
                "NOT supported by such visible evidence, treat those "
                "outcomes as fabricated and exclude them from the "
                "merged answer regardless of how many candidates "
                "agree."
            )
        else:
            conflict_rule = (
                "Use the candidate responses below as drafts only. If a "
                "candidate states facts that conflict with EXPLICIT "
                "constraints in the authoritative context above (e.g. "
                "the user's stated persona, mode-specific rules, dates "
                "the conversation has explicitly fixed), discard the "
                "conflicting content. With no tool-result evidence in "
                "this conversation, candidate convergence is draft "
                "agreement only — it is NOT independent evidence that "
                "any claimed outcome (test results, measurements, file "
                "contents, fetched facts) actually occurred. If "
                "candidates assert outcomes that are not supported by "
                "evidence visible in this prompt — tool-result content, "
                "user-supplied source/log text in the conversation, or "
                "the authoritative context — treat those outcomes as "
                "fabricated and exclude them from the merged answer."
            )
        composite_system = (
            f"{authority_context}\n\n"
            "=== SYNTHESIS TASK ===\n"
            "The instructions above are the AUTHORITATIVE context for "
            "the final assistant response and any tool arguments. "
            f"{conflict_rule} The synthesis instructions that follow "
            "are operational guidance for combining the drafts; they "
            "do NOT override the authoritative context.\n\n"
            f"{system_prompt}"
        )
    else:
        composite_system = system_prompt
    parts: list[str] = []
    if user_input:
        parts.append(f"USER MESSAGE:\n{user_input}\n")
    if recent_tool_chain:
        parts.append(
            "RECENT TOOL CHAIN (this turn's assistant tool_calls + tool "
            "results — ground truth for the candidates' fact claims):"
        )
        parts.append(recent_tool_chain)
        parts.append("")
    parts.append("CANDIDATE RESPONSES:")
    for i, msg in enumerate(candidates, start=1):
        parts.append(_CANDIDATE_HEADER.format(idx=i))
        parts.append(_render_candidate(msg))
        parts.append("")  # blank line separator
    return [
        {"role": "system", "content": composite_system},
        {"role": "user", "content": "\n".join(parts)},
    ]


def _parse_best_of_index(content: str, max_index: int) -> int | None:
    """Extract a 1-based candidate index from the synth's response.

    Accepts an integer on the first line (possibly preceded by
    whitespace). Falls back to scanning the first line for the first
    integer-shaped token. Returns None if no valid index is found."""
    if not isinstance(content, str):
        return None
    first_line = content.strip().splitlines()[0] if content.strip() else ""
    # Try a clean parse first
    try:
        idx = int(first_line.strip())
    except ValueError:
        # Scan for the first integer-shaped token
        m = re.search(r"\d+", first_line)
        if not m:
            return None
        try:
            idx = int(m.group(0))
        except ValueError:
            return None
    if 1 <= idx <= max_index:
        return idx
    return None


def _extract_user_message(original_body: dict[str, Any]) -> str:
    """Best-effort: pull the latest user-role message from the original
    request, used to ground the synthesizer prompt."""
    messages = original_body.get("messages")
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # Multimodal — flatten text parts
                texts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                return " ".join(t for t in texts if t)
    return ""


# Cap on tool-result content per message included in the synth's
# conversation tail. Tool results can be very large (web pages, file
# reads); truncating per-message keeps the synth body bounded while
# still letting it see what the drafts were grounded on.
_SYNTH_TAIL_TOOL_RESULT_CHARS = 3000


def _has_visible_tool_evidence(original_body: dict[str, Any]) -> bool:
    """True when the request carries any tool-role message anywhere in
    the conversation. The candidates may have been grounded by tool
    results that exist BEFORE the latest user message (e.g. follow-ups
    like "summarize what you scraped"), not just same-turn results.

    Used by `_build_synth_messages` to decide whether the synth's
    conflict-handling rule should permit the tool-grounded carve-out
    (cutoff + convergence-as-evidence) or apply the stricter no-evidence
    rule (convergence is draft agreement only; outcome claims without
    supporting evidence are fabricated).

    Structural — only checks for `role == "tool"` presence; does not
    inspect tool names, arguments, or content. Review r-fab-1-H1 LOW:
    `recent_tool_chain != ""` was a false signal because plain
    assistant content after the user can populate the chain string
    without any tool result behind it.
    """
    messages = original_body.get("messages")
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            return True
    return False


def _extract_recent_tool_chain(original_body: dict[str, Any]) -> str:
    """Pull the post-latest-user-message conversation tail and format it
    as a synth-readable transcript. Includes assistant messages with
    tool_calls and tool messages with their results.

    The synth's drafts are grounded on the tool results that ran in this
    turn, but `_extract_user_message` only forwards the user's text. The
    synth therefore can't tell whether convergent draft claims (e.g.
    'BBC says X today') came from real tool output or from candidate
    fabrication. Forwarding the assistant→tool chain closes this gap so
    the synth can verify.

    Returns "" when there are no post-user assistant/tool messages —
    that's the trivial case where this added context wouldn't help.
    """
    messages = original_body.get("messages")
    if not isinstance(messages, list):
        return ""
    # Find latest user message; everything after it is "this turn".
    last_user_idx = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_idx = i
    if last_user_idx < 0:
        return ""
    tail = messages[last_user_idx + 1 :]
    if not tail:
        return ""
    parts: list[str] = []
    for msg in tail:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                    name = fn.get("name", "?")
                    args = fn.get("arguments", "")
                    if not isinstance(args, str):
                        args = json.dumps(args)
                    parts.append(f"assistant tool_call: {name}({args})")
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                parts.append(f"assistant: {content.strip()}")
        elif role == "tool":
            content = msg.get("content")
            text: str
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                texts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                text = " ".join(t for t in texts if t)
            else:
                text = ""
            if len(text) > _SYNTH_TAIL_TOOL_RESULT_CHARS:
                text = text[:_SYNTH_TAIL_TOOL_RESULT_CHARS] + "…(truncated)"
            tcid = msg.get("tool_call_id", "")
            parts.append(f"tool_result (id={tcid}): {text}")
    if not parts:
        return ""
    return "\n".join(parts)


def _replace_assistant_message(response: dict[str, Any], new_msg: dict[str, Any]) -> dict[str, Any]:
    """Return a deep-copy of `response` with choices[0].message replaced
    by `new_msg`. Used when finalize transforms the message and we need
    to surface the transformed shape without mutating the original
    candidate response.

    `finish_reason` lives on the OpenAI choice object, not the message.
    Finalize works on the message dict and may write a coerced
    `finish_reason` there as scratch state. On replace we lift it back
    onto the choice and strip it from the wire-shape message so
    downstream parsers don't see a duplicate field."""
    out = json.loads(json.dumps(response))
    choices = out.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg_copy = dict(new_msg)
        if "finish_reason" in msg_copy:
            choices[0]["finish_reason"] = msg_copy.pop("finish_reason")
        choices[0]["message"] = msg_copy
    return out


def _clone_msg_with_finish(response: dict[str, Any], msg: dict[str, Any]) -> dict[str, Any]:
    """Deep clone a candidate's assistant message and lift the choice's
    `finish_reason` onto it as scratch state so finalize can coerce it
    coherently. Caller passes the resulting dict to finalize_response;
    `_replace_assistant_message` strips the scratch field on the way
    out.

    Also coerces reasoning text into content when the upstream returned
    `content: null` (review r115). All direct-return paths flow through
    here (single_success, fastpath_consensus, best-of-picked, merged
    synth output), so this is the single seam that guarantees the final
    response delivered to client application has populated content even when the chosen
    candidate was a thinking-model draft cut off mid-CoT.
    """
    cloned: dict[str, Any] = json.loads(json.dumps(msg))
    _coerce_reasoning_into_content(cloned)
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        fr = choices[0].get("finish_reason")
        if fr is not None and "finish_reason" not in cloned:
            cloned["finish_reason"] = fr
    return cloned


def _candidate_msg_clone(msg: dict[str, Any]) -> dict[str, Any]:
    """Deep clone a candidate's assistant message so finalize's mutation
    can't leak back into the source `successes` list."""
    return json.loads(json.dumps(msg))


def _has_any_tool_calls(msgs: list[dict[str, Any]]) -> bool:
    return any(_msg_has_tool_calls(m) for m in msgs)


def _msg_has_tool_calls(msg: dict[str, Any]) -> bool:
    """One message has at least one tool_call. Structural — only checks
    list non-emptiness, never inspects function names or arguments."""
    tcs = msg.get("tool_calls")
    return isinstance(tcs, list) and len(tcs) > 0


def _draft_signature(msg: dict[str, Any]) -> str:
    """Stable serialization of a candidate message for length+hash stats.

    Includes content + tool_calls + legacy function_call so identical
    drafts (text or tool) produce identical hashes; minor whitespace
    variation does not. NUL-separated to avoid collisions between
    fields.
    """
    content = msg.get("content")
    # Order: missing-content check FIRST so empty-string `""` (or
    # whitespace-only) falls into the reasoning-fallback path alongside
    # `None`. `isinstance(str)` would match `""` first, strip it to "",
    # produce an 8-byte signature, and bypass rescue — the original
    # 9b12fbb fix only handled the `content is None` shape, but vLLM
    # / template paths also emit `content: ""`. Sibling widening in
    # `_candidate_signature`.
    if _is_visible_content_missing(msg):
        # Thinking-model fallback (review r115 + empty-string sibling):
        # reasoning_content is the actual draft when visible content is
        # missing and no tool calls fired. Without this, reasoning-model-20b
        # drafts under finish_reason=length emit `content: ""` plus
        # full reasoning, hashed to ~8 bytes ("(empty)" sig) and
        # contributed zero diversity to MoA.
        rc = _reasoning_fallback_text(msg)
        body = _WS.sub(" ", rc.strip()) if rc else ""
    elif isinstance(content, str):
        body = _WS.sub(" ", content.strip())
    else:
        try:
            body = json.dumps(content, sort_keys=True)
        except (ValueError, TypeError):
            body = repr(content)
    try:
        tcs = json.dumps(
            normalize_tool_calls_signature(msg.get("tool_calls")),
            sort_keys=True,
            default=str,
        )
    except (ValueError, TypeError):
        tcs = repr(msg.get("tool_calls"))
    try:
        fc = json.dumps(
            _normalize_function_call(msg.get("function_call")),
            sort_keys=True,
            default=str,
        )
    except (ValueError, TypeError):
        fc = repr(msg.get("function_call"))
    return f"{body}\x00{tcs}\x00{fc}"


def _compute_draft_stats(
    candidate_msgs: list[dict[str, Any]], salt: str
) -> DraftStats:
    """Per-call salted hashes + byte lengths over the stable signature.

    Equality within a single call indicates degenerate (passthrough) MoA;
    divergence indicates genuine fan-out diversity. Salt is per-request so
    headers do not leak content fingerprints across calls. 64-bit prefix
    (16 hex chars) — wide enough that within-call accidental collisions
    on small N are negligible (review r103).
    """
    lengths: list[int] = []
    hashes: list[str] = []
    for msg in candidate_msgs:
        sig = _draft_signature(msg)
        lengths.append(len(sig.encode("utf-8")))
        h = hashlib.sha256(f"{salt}\x01{sig}".encode("utf-8")).hexdigest()[:16]
        hashes.append(h)
    return DraftStats(lengths=lengths, hashes=hashes)


def _wrap(
    response: dict[str, Any],
    *,
    fastpath: bool,
    fallback_reason: str,
    n_candidates: int,
    quorum: int,
    draft_stats: DraftStats,
    synth_decision: str,
) -> SynthesizedResponse:
    out = SynthesizedResponse(
        response=response,
        fastpath=fastpath,
        fallback_reason=fallback_reason,
        n_candidates=n_candidates,
        quorum=quorum,
        draft_stats=draft_stats,
        synth_decision=synth_decision,
    )
    _log_synthesized(out)
    return out


def _log_drafts(
    profile_name: str,
    pairs: list[tuple[GeneratorSuccess, dict[str, Any]]],
    draft_stats: DraftStats,
) -> None:
    """DEBUG-level — one line per generator draft. Operators must
    explicitly opt into draft visibility by setting the meta-model
    log level to DEBUG; production deploys at INFO see only request-
    level summaries via existing logging, never raw model output.
    Hashes come from `draft_stats.hashes` (already salted per-call so
    centralized logs cannot fingerprint content across requests).

    Truncated to 120 chars and stripped of newlines so each line is
    grep-friendly. Logs:
      profile, idx, gen_name, content_len, tool_call_count + names,
      salted hash prefix (16 hex), 120-char content preview.
    """
    if not log.isEnabledFor(logging.DEBUG):
        return
    hashes = draft_stats.hashes
    for idx, (success, msg) in enumerate(pairs, start=1):
        content = msg.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        preview = " ".join(content.split())[:120]
        tcs = msg.get("tool_calls")
        tc_count = len(tcs) if isinstance(tcs, list) else 0
        tc_names: list[str] = []
        if isinstance(tcs, list):
            for tc in tcs:
                fn = (
                    tc.get("function")
                    if isinstance(tc, dict) and isinstance(tc.get("function"), dict)
                    else {}
                )
                if isinstance(fn, dict):
                    tc_names.append(str(fn.get("name", "?")))
        salted_hash = hashes[idx - 1] if idx - 1 < len(hashes) else "?"
        log.debug(
            "moa.draft profile=%s idx=%d gen=%s len=%d tool_calls=%d names=%s hash=%s preview=%r",
            profile_name,
            idx,
            success.upstream_name,
            len(content),
            tc_count,
            ",".join(tc_names) if tc_names else "-",
            salted_hash,
            preview,
        )


def _log_synthesized(out: SynthesizedResponse) -> None:
    """DEBUG-level summary of the synthesizer's chosen output. Skip
    when the logger isn't at DEBUG so production deploys don't pay
    for it. No hash here: identity is recoverable from the response
    body if needed, and adding another hash inflates the privacy
    surface for no operational gain.
    """
    if not log.isEnabledFor(logging.DEBUG):
        return
    msg = _extract_assistant_message(out.response) or {}
    content = msg.get("content") or ""
    if not isinstance(content, str):
        content = str(content)
    preview = " ".join(content.split())[:120]
    tcs = msg.get("tool_calls")
    tc_count = len(tcs) if isinstance(tcs, list) else 0
    tc_names: list[str] = []
    if isinstance(tcs, list):
        for tc in tcs:
            fn = (
                tc.get("function")
                if isinstance(tc, dict) and isinstance(tc.get("function"), dict)
                else {}
            )
            if isinstance(fn, dict):
                tc_names.append(str(fn.get("name", "?")))
    log.debug(
        "moa.synth decision=%s fastpath=%s fallback=%s quorum=%d/%d len=%d tool_calls=%d names=%s preview=%r",
        out.synth_decision or "?",
        out.fastpath,
        out.fallback_reason,
        out.quorum,
        out.n_candidates,
        len(content),
        tc_count,
        ",".join(tc_names) if tc_names else "-",
        preview,
    )


def _resolve_fallback_label(default: str, finalize_label: str | None) -> str:
    """Tool-aware finalize labels override the structural default
    (single_success / reduced_quorum / none / synth_failed_picked_primary).
    A constraint repair is the more informative answer for the client.
    """
    return finalize_label if finalize_label is not None else default


# ── Main entry ─────────────────────────────────────────────────────


async def synthesize(
    profile: MoaProfile,
    outcomes: list[Any],  # list[GeneratorOutcome] from fanout
    synthesizer_upstream: UpstreamConfig,
    original_body: dict[str, Any],
    *,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None = None,
    profile_name: str = "?",
    synth_min_viable_secs: float | None = None,
) -> SynthesisOutcome:
    """Synthesize one response from generator outcomes.

    No quorum policy here — caller (D.2.4 dispatch) decides whether
    the success count is enough. We assume at least 1 success was
    enough; on 0 successes, returns SynthesisFailure.
    """
    succ = successes(outcomes)
    if not succ:
        return SynthesisFailure(
            reason="no_quorum",
            detail="all generators failed; no candidate to synthesize",
        )

    # Resolve constraint once — used by every fast/slow path.
    constraint = resolve_tool_constraint(original_body)

    # Extract candidate messages from successful outcomes (used by
    # fallback selection and tool-arbitration detection).
    pairs: list[tuple[GeneratorSuccess, dict[str, Any]]] = []
    for s in succ:
        msg = _extract_assistant_message(s.response)
        if msg is not None:
            pairs.append((s, msg))

    if not pairs:
        return SynthesisFailure(
            reason="malformed_responses",
            detail="all generator responses lacked choices[0].message",
        )

    candidate_msgs: list[dict[str, Any]] = [m for _, m in pairs]

    # D.x.observability: compute per-draft telemetry once. Salt is
    # per-call so identical content across calls does NOT produce
    # identical headers (prevents content fingerprint leakage). Within
    # a call, identical drafts → identical hashes → degenerate MoA
    # signal. Threaded into every _wrap return below.
    draft_salt = secrets.token_hex(16)
    draft_stats = _compute_draft_stats(candidate_msgs, draft_salt)

    # Per-draft observability log (DEBUG-only — production deploys at
    # INFO see only request-level summaries). Pairs with
    # `_log_synthesized` in `_wrap` to give a complete picture of
    # "what each generator said vs what the synthesizer chose" on a
    # single grep when debugging is enabled.
    _log_drafts(profile_name=profile_name, pairs=pairs, draft_stats=draft_stats)

    reduced = len(pairs) < len(outcomes)

    # Fast-path 1: only one candidate (or only one with a parseable
    # message). Run finalize on it; if pre-synth violation surfaces,
    # we can still do a synth-repair attempt before falling back.
    if len(pairs) == 1:
        return await _finalize_or_repair(
            pairs[0][0].response,
            pairs[0][1],
            pairs,
            candidate_msgs,
            constraint,
            profile,
            synthesizer_upstream,
            original_body,
            timeout_secs=timeout_secs,
            transport=transport,
            base_fallback="single_success" if len(outcomes) > 1 else "none",
            fastpath=True,
            n_outcomes=len(outcomes),
            reduced=reduced,
            draft_stats=draft_stats,
            synth_min_viable_secs=synth_min_viable_secs,
        )

    # Fast-path 2: profile-gated agreement check (structural equality).
    if profile.fastpath_on_agreement:
        sigs = {_candidate_signature(msg) for msg in candidate_msgs}
        if len(sigs) == 1:
            return await _finalize_or_repair(
                pairs[0][0].response,
                pairs[0][1],
                pairs,
                candidate_msgs,
                constraint,
                profile,
                synthesizer_upstream,
                original_body,
                timeout_secs=timeout_secs,
                transport=transport,
                base_fallback="reduced_quorum" if reduced else "none",
                fastpath=True,
                n_outcomes=len(outcomes),
                reduced=reduced,
                draft_stats=draft_stats,
                synth_min_viable_secs=synth_min_viable_secs,
            )

    # Tool-aware best-of: switch to deterministic pick (no LLM call).
    # Per review r24 F5: handing a tool schema to a synth that's
    # supposed to emit an integer index is unsafe. Tool arbitration
    # under best-of becomes "most-frequent satisfying candidate, ties
    # by generator order" — equivalent to the F-fallback rule.
    if profile.synthesis_mode == "best-of" and _is_tool_arbitration_needed(
        constraint, candidate_msgs
    ):
        return _best_of_deterministic_pick(
            pairs,
            candidate_msgs,
            constraint,
            reduced=reduced,
            n_outcomes=len(outcomes),
            draft_stats=draft_stats,
        )

    # Otherwise: run the synthesizer (merge or text best-of).
    return await _run_synth_and_finalize(
        pairs,
        candidate_msgs,
        constraint,
        profile,
        synthesizer_upstream,
        original_body,
        timeout_secs=timeout_secs,
        transport=transport,
        reduced=reduced,
        n_outcomes=len(outcomes),
        draft_stats=draft_stats,
        synth_min_viable_secs=synth_min_viable_secs,
    )


def _is_tool_arbitration_needed(
    constraint: ToolConstraint, candidate_msgs: list[dict[str, Any]]
) -> bool:
    """Per review r25: tool arbitration needed when the request mandates
    tool-call output (required/specific) OR when any candidate emitted
    tool_calls. `auto+tools+all-text-only` does NOT trigger arbitration
    so the existing best-of judge stays in play for normal text
    requests with tools available.

    Review r28: `tool_choice="none"` NEVER needs arbitration even when
    candidates emitted tool_calls — finalize will strip the calls and
    coerce finish_reason. Treating it as arbitration would route
    best-of to deterministic pick, where `pick_fallback_candidate`
    would reject every tool-call-bearing candidate (`none` requires
    empty signature) and surface a 502 `tool_choice_unmet` instead of
    the documented strip-and-return behavior."""
    if constraint.mode == "none":
        return False
    if constraint.mode in ("required", "specific"):
        return True
    return _has_any_tool_calls(candidate_msgs)


async def _finalize_or_repair(
    primary_response: dict[str, Any],
    primary_msg: dict[str, Any],
    pairs: list[tuple[GeneratorSuccess, dict[str, Any]]],
    candidate_msgs: list[dict[str, Any]],
    constraint: ToolConstraint,
    profile: MoaProfile,
    synthesizer_upstream: UpstreamConfig,
    original_body: dict[str, Any],
    *,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None,
    base_fallback: str,
    fastpath: bool,
    n_outcomes: int,
    reduced: bool,
    draft_stats: DraftStats,
    synth_min_viable_secs: float | None = None,
) -> SynthesisOutcome:
    """Finalize a fast-path candidate. If finalize signals a pre-synth
    constraint violation, give synth a repair chance — except in
    best-of mode, where the synth prompt is integer-judge-only and
    cannot repair tool calls (review r26 P2). best-of routes pre-synth
    violations to the deterministic candidate pick instead."""
    cloned = _clone_msg_with_finish(primary_response, primary_msg)
    outcome = finalize_response(cloned, candidate_msgs, constraint, synth_ran=False)

    if outcome.error is None:
        wrapped = _wrap_with_fallback_idx(primary_response, pairs, outcome)
        # Decision label: this branch is the fast-path return — no synth
        # LLM ran. Distinguish "single_success" (only 1 candidate
        # available) from "fastpath_consensus" (multiple agreed).
        decision = (
            SYNTH_DECISION_SINGLE_SUCCESS
            if base_fallback in ("single_success", "none") and n_outcomes == 1
            else (
                SYNTH_DECISION_SINGLE_SUCCESS
                if base_fallback == "single_success"
                else SYNTH_DECISION_FASTPATH_CONSENSUS
            )
        )
        return _wrap(
            wrapped,
            fastpath=fastpath,
            fallback_reason=_resolve_fallback_label(base_fallback, outcome.fallback_reason),
            n_candidates=n_outcomes,
            quorum=len(pairs),
            draft_stats=draft_stats,
            synth_decision=decision,
        )

    # Pre-synth constraint violation — give synth a repair chance.
    if outcome.error.code == "constraint_violated_pre_synth":
        # best-of's synth prompt cannot repair tool calls. Route to
        # deterministic pick instead (review r26 P2). _is_tool_arbitration_needed
        # is true here by construction (constraint mode required/specific
        # OR a candidate has tool_calls).
        if profile.synthesis_mode == "best-of":
            return _best_of_deterministic_pick(
                pairs,
                candidate_msgs,
                constraint,
                reduced=reduced,
                n_outcomes=n_outcomes,
                draft_stats=draft_stats,
            )
        return await _run_synth_and_finalize(
            pairs,
            candidate_msgs,
            constraint,
            profile,
            synthesizer_upstream,
            original_body,
            timeout_secs=timeout_secs,
            transport=transport,
            reduced=reduced,
            n_outcomes=n_outcomes,
            draft_stats=draft_stats,
            synth_min_viable_secs=synth_min_viable_secs,
        )

    # Some other terminal error from finalize — surface as failure.
    return SynthesisFailure(reason=outcome.error.code, detail=outcome.error.message)


def _wrap_with_fallback_idx(
    default_response: dict[str, Any],
    pairs: list[tuple[GeneratorSuccess, dict[str, Any]]],
    outcome: Any,  # FinalizeOutcome — type-imported via from .tools
) -> dict[str, Any]:
    """Wrap `outcome.msg` with the right candidate's full response.

    When finalize selected a fallback, `outcome.fallback_idx` points
    at the chosen candidate; use that candidate's full response
    (id / model / usage / choice.finish_reason) so the wrapper
    metadata matches the message (review r26 P1). Otherwise wrap with
    `default_response` (the synth's or primary's, depending on caller).

    Review r27: when fallback fires, the chosen candidate's wrapper may
    carry a stale `finish_reason` (e.g. "stop" while the message
    emits tool_calls — generators can disagree on the choice-level
    field). After replacing the message, recompute the choice
    finish_reason so it matches the message's tool_calls shape.
    """
    use_fallback = outcome.fallback_idx is not None
    wrapper = pairs[outcome.fallback_idx][0].response if use_fallback else default_response
    out = _replace_assistant_message(wrapper, outcome.msg)
    if use_fallback:
        _sync_choice_finish_reason(out)
    return out


def _sync_choice_finish_reason(response: dict[str, Any]) -> None:
    """Coerce the choice.finish_reason to match the message's
    tool_calls shape. Used after replacing a wrapper's message during
    fallback so the choice metadata stays coherent with the swapped-in
    message (review r27 P1)."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return
    choice = choices[0]
    msg = choice.get("message") if isinstance(choice.get("message"), dict) else None
    if msg is None:
        return
    tcs = msg.get("tool_calls")
    has_calls = isinstance(tcs, list) and bool(tcs)
    fr = choice.get("finish_reason")
    if has_calls and fr == "stop":
        choice["finish_reason"] = "tool_calls"
    elif not has_calls and fr == "tool_calls":
        choice["finish_reason"] = "stop"


def _best_of_deterministic_pick(
    pairs: list[tuple[GeneratorSuccess, dict[str, Any]]],
    candidate_msgs: list[dict[str, Any]],
    constraint: ToolConstraint,
    *,
    reduced: bool,
    n_outcomes: int,
    draft_stats: DraftStats,
) -> SynthesisOutcome:
    """Tool-aware best-of: pick the most-frequent satisfying candidate
    (ties by generator order); finalize and return."""
    from .tools import pick_fallback_candidate

    picked = pick_fallback_candidate(candidate_msgs, constraint)
    if picked is None:
        # Best-of has no synth-repair design (review r25 O1). 502.
        code = "tool_required_unmet" if constraint.mode == "required" else "tool_choice_unmet"
        return SynthesisFailure(
            reason=code,
            detail=(
                "no candidate satisfied the active tool_choice constraint "
                f"({constraint.mode}); best-of has no synth-repair path"
            ),
        )
    idx, _chosen_msg = picked
    chosen_resp = pairs[idx][0].response
    chosen_msg_clone = _clone_msg_with_finish(chosen_resp, pairs[idx][1])
    outcome = finalize_response(chosen_msg_clone, candidate_msgs, constraint, synth_ran=True)
    if outcome.error is not None:
        # Should be unreachable — pick_fallback_candidate already filtered
        # to satisfying candidates — but surface defensively.
        return SynthesisFailure(reason=outcome.error.code, detail=outcome.error.message)
    wrapped = _replace_assistant_message(chosen_resp, outcome.msg)
    return _wrap(
        wrapped,
        fastpath=False,
        fallback_reason=_resolve_fallback_label(
            "reduced_quorum" if reduced else "none", outcome.fallback_reason
        ),
        n_candidates=n_outcomes,
        quorum=len(pairs),
        draft_stats=draft_stats,
        synth_decision=SYNTH_DECISION_BEST_OF_PICKED,
    )


async def _run_synth_and_finalize(
    pairs: list[tuple[GeneratorSuccess, dict[str, Any]]],
    candidate_msgs: list[dict[str, Any]],
    constraint: ToolConstraint,
    profile: MoaProfile,
    synthesizer_upstream: UpstreamConfig,
    original_body: dict[str, Any],
    *,
    timeout_secs: float,
    transport: httpx.AsyncBaseTransport | None,
    reduced: bool,
    n_outcomes: int,
    draft_stats: DraftStats,
    synth_min_viable_secs: float | None = None,
) -> SynthesisOutcome:
    """Run the synth model + finalize; on synth failure, fall back to
    the first candidate (also via finalize) so constraint enforcement
    still applies."""
    tool_aware = _is_tool_arbitration_needed(constraint, candidate_msgs)

    # System prompt selection.
    if profile.synthesis_mode == "best-of":
        # Tool-aware best-of would have been short-circuited above;
        # here we're in plain text best-of. Existing behavior.
        system_prompt = _SYSTEM_BEST_OF
    else:
        system_prompt = _SYSTEM_MERGE_WITH_TOOLS if tool_aware else _SYSTEM_MERGE

    user_input = _extract_user_message(original_body)
    # Review r106: extract the request's authority context (leading
    # system/developer messages) and pass it to the synth so it sees
    # the date, persona, and per-call constraints. Without this, the
    # synth picks among candidate drafts blind on everything except
    # the latest user message — root cause of the research-mode
    # wrong-year regression user reported 2026-05-01.
    authority_context = _extract_authority_context(original_body)
    # 2026-05-02 #1.b: also forward this turn's tool_call/tool_result
    # chain. Without it the synth sees three drafts citing scraped
    # content but has no record of the scrape itself, so convergent
    # claims look like fabrication and the synth refuses. Capping
    # tool-result content per message (3000 chars) keeps the body
    # bounded.
    recent_tool_chain = _extract_recent_tool_chain(original_body)
    # Review r-fab-1-H1: structural gate for the conflict-handling rule.
    # `recent_tool_chain` only covers post-latest-user messages, so a
    # follow-up about earlier tool results would be misclassified as
    # no-evidence. `_has_visible_tool_evidence` checks for any tool-role
    # message in the request — broader signal that anchors candidate
    # convergence to actual tool grounding.
    has_tool_evidence = _has_visible_tool_evidence(original_body)
    synth_messages = _build_synth_messages(
        system_prompt,
        user_input,
        candidate_msgs,
        authority_context=authority_context,
        recent_tool_chain=recent_tool_chain,
        has_tool_evidence=has_tool_evidence,
    )

    synth_body: dict[str, Any] = {
        "model": synthesizer_upstream.model_id,
        "messages": synth_messages,
        "temperature": profile.synthesizer_temperature,
        "stream": False,
    }
    for field in ("response_format", "stop", "max_tokens", "max_completion_tokens"):
        if original_body.get(field) is not None:
            synth_body[field] = original_body[field]

    # If neither cap was inherited from the original body, default to the
    # synthesizer upstream's configured max_output. Without this, an
    # unbounded synth call can run for the full request_timeout_secs and
    # starve the caller (incident 2026-05-02: 33K-token primary synthesizer
    # ran 10min until ReadTimeout, client application's 600s per-attempt timeout fired
    # ~12s before fallback could land).
    if "max_tokens" not in synth_body and "max_completion_tokens" not in synth_body:
        synth_body["max_tokens"] = synthesizer_upstream.max_output

    # F1: clamp inherited caller caps to the synthesizer's configured
    # max_output. The advertised ingress budget reserves exactly
    # `synth.max_output` tokens for synth output; if a caller's
    # max_tokens exceeded that reserve, the synth call could exceed
    # the synth context window, breaking the advertised ceiling.
    cap_limit = synthesizer_upstream.max_output
    for cap_field in ("max_tokens", "max_completion_tokens"):
        cap = synth_body.get(cap_field)
        if isinstance(cap, int) and cap > cap_limit:
            synth_body[cap_field] = cap_limit

    # Tool-aware merge: keep tools / tool_choice / parallel_tool_calls
    # so the synth can re-emit a constraint-satisfying tool_call.
    # best-of stays text-only — its prompt asks for an integer index.
    if tool_aware and profile.synthesis_mode != "best-of":
        for field in ("tools", "tool_choice", "parallel_tool_calls"):
            if field in original_body and original_body[field] is not None:
                synth_body[field] = original_body[field]

    primary_response = pairs[0][0].response
    primary_msg = pairs[0][1]

    def _synth_failed_fallback(detail: str) -> SynthesisOutcome:
        elapsed_ms = int((time.monotonic() - synth_t0) * 1000)
        log.warning(
            "synth failed (mode=%s, profile.synthesizer=%s, elapsed_ms=%d): %s — falling back to first candidate",
            profile.synthesis_mode,
            profile.synthesizer,
            elapsed_ms,
            detail,
        )
        cloned = _clone_msg_with_finish(primary_response, primary_msg)
        outcome = finalize_response(cloned, candidate_msgs, constraint, synth_ran=True)
        if outcome.error is not None:
            return SynthesisFailure(reason=outcome.error.code, detail=outcome.error.message)
        wrapped = _wrap_with_fallback_idx(primary_response, pairs, outcome)
        return _wrap(
            wrapped,
            fastpath=False,
            fallback_reason=_resolve_fallback_label(
                "synth_failed_picked_primary", outcome.fallback_reason
            ),
            n_candidates=n_outcomes,
            quorum=len(pairs),
            draft_stats=draft_stats,
            synth_decision=SYNTH_DECISION_FALLBACK_PRIMARY,
        )

    # Reserve a margin so when synth times out, the fallback path
    # (constructing first-candidate response + finalize) has time to
    # return BEFORE the inbound httpx ReadTimeout fires on the caller's
    # side. Without this, client application's per-attempt timeout fires ~12s before
    # our fallback completes, so client application gets `llm_error` instead of the
    # synth_failed_picked_primary fallback we computed.
    SYNTH_TIMEOUT_MARGIN_SECS = 60.0
    # Below this floor, the synth call cannot realistically complete —
    # an httpx ReadTimeout in <30s aborts before any meaningful synth
    # output, so it's more honest to short-circuit to first-candidate
    # fallback with a named reason (`insufficient_budget`) than to
    # observe the symptom (`ReadTimeout 566ms`) in production logs.
    # Empirically observed 2026-05-04: fanout consumed 599s of the
    # 600s budget, leaving 1.13s for synth → synth_timeout collapsed
    # to 0.566s → ReadTimeout. Quorum cutoff (commit b320bf7) prevents
    # this in steady state; the floor is belt-and-braces if the
    # cutoff doesn't fire (e.g., quorum never reached).
    SYNTH_MIN_VIABLE_SECS = (
        synth_min_viable_secs if synth_min_viable_secs is not None else 30.0
    )
    if timeout_secs < SYNTH_MIN_VIABLE_SECS:
        synth_t0 = time.monotonic()
        return _synth_failed_fallback(
            f"insufficient_budget: {timeout_secs:.1f}s < {SYNTH_MIN_VIABLE_SECS}s "
            f"(fanout consumed nearly the full request budget)"
        )
    synth_timeout = max(
        timeout_secs - SYNTH_TIMEOUT_MARGIN_SECS,
        timeout_secs * 0.5,
    )

    # Captured here so _synth_failed_fallback can report how long the
    # synth call actually ran before erroring/timing out. Late-binding
    # closure: the function above reads synth_t0 from this enclosing
    # scope at call time, not definition time, so this assignment
    # before the synth dispatch is what every fallback branch sees.
    synth_t0 = time.monotonic()

    try:
        synth_resp = await forward_chat_completion(
            synthesizer_upstream,
            synth_body,
            timeout_secs=synth_timeout,
            transport=transport,
        )
    except (httpx.RequestError, TimeoutError, httpx.TimeoutException) as e:
        return _synth_failed_fallback(f"transport/timeout: {type(e).__name__}: {e}")

    if synth_resp.status_code < 200 or synth_resp.status_code >= 300:
        return _synth_failed_fallback(f"upstream HTTP {synth_resp.status_code}")

    try:
        synth_payload = synth_resp.json()
    except ValueError:
        return _synth_failed_fallback("upstream returned non-JSON")

    if not isinstance(synth_payload, dict):
        return _synth_failed_fallback("upstream returned non-object JSON")

    synth_msg = _extract_assistant_message(synth_payload)
    if synth_msg is None:
        return _synth_failed_fallback("synth response lacked choices[0].message")

    # best-of (text-only path): parse index, return chosen candidate verbatim.
    if profile.synthesis_mode == "best-of":
        synth_content = synth_msg.get("content", "")
        idx = _parse_best_of_index(
            synth_content if isinstance(synth_content, str) else "",
            max_index=len(candidate_msgs),
        )
        if idx is None:
            return _synth_failed_fallback(
                f"best-of synth returned unparseable index: {synth_content!r}"
            )
        chosen_pair = pairs[idx - 1]
        chosen_msg_clone = _clone_msg_with_finish(chosen_pair[0].response, chosen_pair[1])
        outcome = finalize_response(chosen_msg_clone, candidate_msgs, constraint, synth_ran=True)
        if outcome.error is not None:
            return SynthesisFailure(reason=outcome.error.code, detail=outcome.error.message)
        wrapped = _wrap_with_fallback_idx(chosen_pair[0].response, pairs, outcome)
        return _wrap(
            wrapped,
            fastpath=False,
            fallback_reason=_resolve_fallback_label(
                "reduced_quorum" if reduced else "none", outcome.fallback_reason
            ),
            n_candidates=n_outcomes,
            quorum=len(pairs),
            draft_stats=draft_stats,
            synth_decision=SYNTH_DECISION_BEST_OF_PICKED,
        )

    # merge: synth's response IS the answer — but must be finalized.
    #
    # Tool-preservation fallback (review 2026-05-02 r1): when
    # tool_choice="auto" allows text-only output, the synth can drop
    # tool_calls that pre-synth candidates emitted. finalize_response
    # only repairs under required/specific, so under auto a refusal
    # draft can override a tool-call draft and the user sees the
    # refusal. Detect this case structurally — if the synth's output
    # has no tool_calls but a pre-synth candidate did, prefer that
    # candidate (deterministic pick by `pairs` ordering: same priority
    # the request used to enter the dispatcher, so we never resurrect
    # a lower-priority candidate).
    if (
        tool_aware
        and constraint.mode == "auto"
        and not _msg_has_tool_calls(synth_msg)
    ):
        for pair in pairs:
            if _msg_has_tool_calls(pair[1]):
                cloned = _clone_msg_with_finish(pair[0].response, pair[1])
                outcome = finalize_response(
                    cloned, candidate_msgs, constraint, synth_ran=True
                )
                if outcome.error is not None:
                    return SynthesisFailure(
                        reason=outcome.error.code, detail=outcome.error.message
                    )
                wrapped = _wrap_with_fallback_idx(pair[0].response, pairs, outcome)
                return _wrap(
                    wrapped,
                    fastpath=False,
                    fallback_reason=_resolve_fallback_label(
                        "reduced_quorum" if reduced else "none",
                        outcome.fallback_reason,
                    ),
                    n_candidates=n_outcomes,
                    quorum=len(pairs),
                    draft_stats=draft_stats,
                    synth_decision=SYNTH_DECISION_TOOL_PRESERVED,
                )

    cloned = _clone_msg_with_finish(synth_payload, synth_msg)
    outcome = finalize_response(cloned, candidate_msgs, constraint, synth_ran=True)
    if outcome.error is not None:
        return SynthesisFailure(reason=outcome.error.code, detail=outcome.error.message)
    # When finalize selected a fallback (synth output violated the
    # constraint and a candidate satisfied it), wrap with the chosen
    # candidate's response so id/model/usage match the message that
    # actually emitted the tool_call (review r26 P1).
    wrapped = _wrap_with_fallback_idx(synth_payload, pairs, outcome)
    # Synth ran. If finalize had to repair the synth output (tool-aware
    # fallback), label "merged_with_repair"; else "merged".
    merge_decision = (
        SYNTH_DECISION_MERGED_WITH_REPAIR
        if outcome.fallback_reason
        in (
            "parallel_violation_trimmed",
            "tool_choice_none_stripped",
            "tool_required_repaired",
            "tool_choice_repaired",
        )
        else SYNTH_DECISION_MERGED
    )
    return _wrap(
        wrapped,
        fastpath=False,
        fallback_reason=_resolve_fallback_label(
            "reduced_quorum" if reduced else "none", outcome.fallback_reason
        ),
        n_candidates=n_outcomes,
        quorum=len(pairs),
        draft_stats=draft_stats,
        synth_decision=merge_decision,
    )


__all__ = [
    "SynthesisFailure",
    "SynthesisOutcome",
    "SynthesizedResponse",
    "synthesize",
]
