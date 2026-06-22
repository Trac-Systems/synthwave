"""Shared-tail compaction — per-generator payloads with identical recent context.

When fan-out queries N generators with different context windows, naive
per-generator compaction makes independent keep/drop decisions about
*recent* messages — generators end up disagreeing about what just
happened. The synthesizer then has to reconcile drafts derived from
divergent realities.

This module guarantees every generator sees the SAME recent context
(the `shared_tail`); divergence is pushed to OLDER history, where
larger-context generators retain more and smaller-context generators
retain less (or none).

## Algorithm

1. Multimodal hygiene — collapse stale `image_url` parts so older
   images don't bloat the tail (only the most recent image survives
   across all generators).
2. `tail_budget = min over generators of
   (context - response_reserve - tools_token_estimate - safety_margin)`.
3. Group messages into atomic chunks: assistant+tool_calls together
   with following tool_result messages stay glued; system messages
   are isolated; everything else is solo.
4. Walk chunks from END toward START, accumulating into the shared
   tail until the next chunk would exceed `tail_budget`. Stop at any
   system chunk (system lives in the head). Always include at least
   one chunk.
5. Per generator, compact the OLDER head to that generator's head
   budget (`context - tail_tokens - reserve - tools - margin`), then
   concat `head + shared_tail`. Strip any trailing plain assistant
   message (vLLM template constraint).

Edge cases:
- A generator whose `tail_budget <= 0` is dropped from the layout
  with a logged warning. Caller can decide whether to fall back to
  fewer generators or skip MoA entirely.
- If no generator has a usable budget, the layout returns the
  collapsed messages as the "tail" with `per_generator = []`.

## Inputs

Messages are OpenAI chat-shape dicts: ``{"role": str, "content": str
| list[dict], "tool_calls": [...] | None, "tool_call_id": str | None}``.
Generators are ``[(name, UpstreamConfig)]`` matching `fanout.fan_out`.

Token budgets (`response_reserve`, `tools_token_estimate`,
`safety_margin`) are per-call constants supplied by the caller; this
module makes no assumptions about their values.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

from ..config import UpstreamConfig

log = logging.getLogger(__name__)

Message = dict


# ── Token estimation ───────────────────────────────────────────────


def estimate_tokens(text: str) -> int:
    """Estimate tokens for a string. ``bytes * 10 // 28 ≈ bytes / 2.8``.

    Conservative token estimator. Must over-count, never under-count —
    vLLM rejects requests whose actual prompt exceeds the computed
    max_tokens budget, and a low estimate would 400 the request with
    negative max_tokens.
    """
    n = len(text.encode("utf-8"))
    return (n * 10 + 27) // 28


def _content_text(content: object) -> str:
    """Flatten OpenAI content (string or parts list) to a string.

    Image parts are ignored for token estimation (image bytes don't
    tokenize as text).
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                t = part.get("text")
                if isinstance(t, str):
                    out.append(t)
        return "\n".join(out)
    return ""


def estimate_message_tokens(msg: Message) -> int:
    """Estimate tokens for a single OpenAI-shape message."""
    role = str(msg.get("role", ""))
    text = _content_text(msg.get("content"))
    total = estimate_tokens(text) + estimate_tokens(role) + 4
    tcs = msg.get("tool_calls")
    if isinstance(tcs, list):
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                fn = {}
            name = fn.get("name")
            args = fn.get("arguments")
            total += estimate_tokens(str(name) if name is not None else "")
            total += estimate_tokens(str(args) if args is not None else "")
            total += 10
    return total


def estimate_messages_tokens(messages: list[Message]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


# ── Multimodal hygiene ─────────────────────────────────────────────


def collapse_stale_images(messages: list[Message], max_active: int = 1) -> list[Message]:
    """Cap the number of ``image_url`` parts across the conversation.

    Most vision-capable backends accept at most 1 image per prompt.
    When the history contains multiple multimodal turns, every LLM
    call after the second vision call hits "At most 1 image(s)".
    Strip the older image_url parts, preserve the text parts of their
    containing messages, keep only the ``max_active`` most recent.

    Collapse stale image_url parts so older images don't bloat the tail.
    """
    positions: list[tuple[int, int]] = []
    for mi, m in enumerate(messages):
        content = m.get("content")
        if isinstance(content, list):
            for pi, part in enumerate(content):
                if isinstance(part, dict) and part.get("type") == "image_url":
                    positions.append((mi, pi))
    if len(positions) <= max_active:
        return [dict(m) for m in messages]

    keep_start = len(positions) - max_active
    drop_set = set(positions[:keep_start])

    out: list[Message] = []
    for mi, m in enumerate(messages):
        content = m.get("content")
        if not isinstance(content, list):
            out.append(dict(m))
            continue
        if not any(idx == mi for idx, _ in drop_set):
            out.append(dict(m))
            continue

        kept_parts: list[dict] = []
        for pi, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            is_image = part.get("type") == "image_url"
            if is_image and (mi, pi) in drop_set:
                continue
            kept_parts.append(part)

        any_image = any(p.get("type") == "image_url" for p in kept_parts)
        cloned = dict(m)
        if any_image:
            cloned["content"] = kept_parts
        else:
            text_concat = "\n".join(
                p.get("text", "") for p in kept_parts if p.get("type") == "text"
            )
            note = "\n\n[image content removed for context budget]"
            cloned["content"] = (text_concat + note) if text_concat else note.strip()
        out.append(cloned)
    return out


# ── Chunk grouping ─────────────────────────────────────────────────


class ChunkKind(Enum):
    ESSENTIAL_SYSTEM = "essential_system"
    SYNTHETIC_SYSTEM = "synthetic_system"
    REGULAR = "regular"


@dataclass
class _Chunk:
    messages: list[Message]
    tokens: int
    kind: ChunkKind


_SYSTEM_ROLES: frozenset[str] = frozenset({"system", "developer"})


def _group_into_chunks(messages: list[Message]) -> list[_Chunk]:
    """Group messages into atomic units.

    - First system/developer message → ESSENTIAL_SYSTEM (always preserved).
    - Subsequent system/developer messages → SYNTHETIC_SYSTEM (droppable).
    - Assistant with tool_calls + following tool messages whose
      ``tool_call_id`` matches one of the assistant's ``tool_calls[].id``
      → REGULAR (kept atomic).
    - Everything else solo → REGULAR.

    Tool messages without a matching ``tool_call_id`` (or following a
    non-tool-calling assistant) are NOT glued — they become their own
    REGULAR chunks and `_sanitize_tool_pairs` will strip them later.
    Group atomic chunks. tool_call_id is matched explicitly because
    Python message dicts are looser than typed structs.
    """
    chunks: list[_Chunk] = []
    saw_essential = False
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role")
        if role in _SYSTEM_ROLES:
            kind = ChunkKind.ESSENTIAL_SYSTEM if not saw_essential else ChunkKind.SYNTHETIC_SYSTEM
            saw_essential = True
            chunks.append(
                _Chunk(
                    messages=[msg],
                    tokens=estimate_message_tokens(msg),
                    kind=kind,
                )
            )
            i += 1
            continue

        tool_calls = msg.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list) and len(tool_calls) > 0:
            call_ids: set[str] = set()
            for tc in tool_calls:
                if isinstance(tc, dict) and isinstance(tc.get("id"), str):
                    call_ids.add(tc["id"])
            chunk_msgs = [msg]
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                tcid = messages[j].get("tool_call_id")
                if not isinstance(tcid, str) or tcid not in call_ids:
                    break
                chunk_msgs.append(messages[j])
                j += 1
            chunks.append(
                _Chunk(
                    messages=chunk_msgs,
                    tokens=sum(estimate_message_tokens(m) for m in chunk_msgs),
                    kind=ChunkKind.REGULAR,
                )
            )
            i = j
        else:
            chunks.append(
                _Chunk(
                    messages=[msg],
                    tokens=estimate_message_tokens(msg),
                    kind=ChunkKind.REGULAR,
                )
            )
            i += 1
    return chunks


def _strip_trailing_plain_assistant(messages: list[Message]) -> list[Message]:
    """Drop trailing assistant messages that have no tool_calls.

    vLLM's chat template rejects a payload that ends on a bare
    assistant message — the next-token slot is gone.
    """
    out = list(messages)
    while out:
        last = out[-1]
        if last.get("role") != "assistant":
            break
        tcs = last.get("tool_calls")
        if isinstance(tcs, list) and len(tcs) > 0:
            break
        out.pop()
    return out


def _sanitize_tool_pairs(messages: list[Message]) -> list[Message]:
    """Ensure every assistant tool_call has a matching tool result.

    After dropping older chunks, an assistant tool_call may survive
    while its tool result was dropped (or vice versa). Insert a
    placeholder for missing results; remove orphan tool results.

    Sanitize orphan tool_call/tool_result pairs introduced by chunk drops.
    """
    valid_call_ids: set[str] = set()
    for m in messages:
        tcs = m.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                if isinstance(tc, dict) and isinstance(tc.get("id"), str):
                    valid_call_ids.add(tc["id"])

    result_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            if isinstance(tcid, str):
                result_ids.add(tcid)

    out: list[Message] = []
    for m in messages:
        if m.get("role") == "tool":
            tcid = m.get("tool_call_id")
            # Drop unless tool_call_id is a string AND maps to an
            # assistant tool_call in this slice. Catches missing/null/
            # non-string tool_call_ids too (review r14 finding 2).
            if not (isinstance(tcid, str) and tcid in valid_call_ids):
                continue
        out.append(m)

        tcs = m.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                tcid = tc.get("id")
                if not isinstance(tcid, str) or tcid in result_ids:
                    continue
                fn = tc.get("function") or {}
                name = fn.get("name") if isinstance(fn, dict) else "tool"
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tcid,
                        "name": name or "tool",
                        "content": "(result unavailable — context was compacted)",
                    }
                )
    return out


# ── Head compaction ────────────────────────────────────────────────


def _compact_head_for_budget(
    messages: list[Message], budget_tokens: int
) -> tuple[list[Message], int]:
    """Trim head messages to fit within ``budget_tokens``.

    Returns ``(compacted_messages, dropped_chunk_count)``. The dropped
    count is the truth-source for ``GeneratorPayload.compacted_chunks``
    (review r21 finding 1 — deriving from payload length is unsound
    because of the sentinel message).

    Compact a message list to fit a token budget. Algorithm:

    1. Group into atomic chunks; keep ESSENTIAL_SYSTEM by default.
    2. Walk chunks from end → start, keeping any that fit (most
       recent old history retained first).
    3. Try to also keep up to two earliest REGULAR chunks for task
       framing (matches Rust's head_candidates pass at context.rs:265).
    4. Flatten with a `[N earlier groups…]` sentinel; sanitize tool
       pairs (which can ADD placeholder tool results).
    5. **Guard loop:** if the flattened result still exceeds budget
       (because of the sentinel + sanitize placeholders + estimator
       drift), drop the oldest droppable chunk and re-flatten. If no
       chunks remain droppable but the budget is still busted, truncate
       the largest non-system message; iterate up to 48 times. This
       guarantees a fit even when the essential system message is
       enormous.
    """
    if budget_tokens <= 0 or not messages:
        return [], 0

    chunks = _group_into_chunks(messages)
    if not chunks:
        return [], 0

    kept = [False] * len(chunks)
    used = 0
    for idx, chunk in enumerate(chunks):
        if chunk.kind == ChunkKind.ESSENTIAL_SYSTEM:
            kept[idx] = True
            used += chunk.tokens

    head_candidates: list[int] = []
    for idx, chunk in enumerate(chunks):
        if chunk.kind == ChunkKind.REGULAR:
            head_candidates.append(idx)
            if len(head_candidates) == 2:
                break

    # Walk back from end, keeping chunks that fit.
    for idx in range(len(chunks) - 1, -1, -1):
        if kept[idx]:
            continue
        if used + chunks[idx].tokens <= budget_tokens:
            kept[idx] = True
            used += chunks[idx].tokens

    # Try to keep early task-framing chunks if there's room.
    for idx in head_candidates:
        if kept[idx]:
            continue
        if used + chunks[idx].tokens <= budget_tokens:
            kept[idx] = True
            used += chunks[idx].tokens

    dropped = sum(1 for k in kept if not k)
    result = _flatten_chunks(chunks, kept, dropped)

    # Guard loop: sentinel + sanitize placeholders can push us over
    # budget; drop the oldest droppable chunk and re-flatten.
    guard = 0
    while estimate_messages_tokens(result) > budget_tokens and guard < 48:
        if _drop_oldest_droppable(chunks, kept):
            dropped = sum(1 for k in kept if not k)
            result = _flatten_chunks(chunks, kept, dropped)
            guard += 1
            continue
        # No more droppable chunks. Truncate the largest non-system
        # message text. If even that fails, give up — caller surfaces
        # 413 via the budget check downstream.
        overshoot = estimate_messages_tokens(result) - budget_tokens
        if not _truncate_largest_message(result, overshoot):
            break
        guard += 1
    return result, dropped


def _drop_oldest_droppable(chunks: list[_Chunk], kept: list[bool]) -> bool:
    """Drop the oldest non-essential kept chunk. Returns True if dropped.

    Drop the oldest droppable chunk: SYNTHETIC_SYSTEM first, then the
    oldest REGULAR. Never
    drops below one kept non-essential chunk so the conversation
    has something.
    """
    kept_nonessential = [
        idx for idx, c in enumerate(chunks) if kept[idx] and c.kind != ChunkKind.ESSENTIAL_SYSTEM
    ]
    if len(kept_nonessential) <= 1:
        return False
    for idx in kept_nonessential:
        if chunks[idx].kind == ChunkKind.SYNTHETIC_SYSTEM:
            kept[idx] = False
            return True
    kept[kept_nonessential[0]] = False
    return True


def _truncate_text(value: str, overshoot_tokens: int) -> tuple[str, int] | None:
    """Trim ``value`` to claw back roughly ``overshoot_tokens`` tokens.

    Returns ``(trimmed, dropped_bytes)`` or ``None`` if no useful
    truncation was possible (already short, target ≥ original).
    Respects UTF-8 char boundaries.
    """
    if len(value) <= 256:
        return None
    overshoot_bytes = overshoot_tokens * 6 + len(value) // 8 + 128
    target_bytes = max(256, len(value) - overshoot_bytes)
    if target_bytes >= len(value):
        return None
    end = target_bytes
    encoded = value.encode("utf-8")
    while end > 0 and (encoded[end] & 0xC0) == 0x80:
        end -= 1
    trimmed = encoded[:end].decode("utf-8", errors="ignore")
    if len(trimmed) >= len(value):
        return None
    dropped_bytes = len(encoded) - len(trimmed.encode("utf-8"))
    return trimmed, dropped_bytes


def _truncate_largest_message(messages: list[Message], overshoot_tokens: int) -> bool:
    """Truncate the largest truncatable text source to claw back overshoot.

    Considers three text shapes (review r15 finding):
    - String ``content`` field
    - ``content`` list entries with ``type == "text"``
    - ``tool_calls[*].function.arguments`` strings

    Replaces the target dict (and its nested list/dict if needed) with
    SHALLOW COPIES carrying trimmed values. Never mutates the caller's
    original dicts or nested list/dict references — different generators
    share message references via head/tail concat, and a small
    generator's truncation must not retroactively shrink a larger
    generator's payload (review r14 finding 1; r15 finding extends to
    nested shapes).

    Truncate the largest message in place. Tries non-system messages
    first; falls back to any. Returns False
    if nothing was truncated (caller breaks the guard loop).
    """

    @dataclass
    class _Target:
        msg_idx: int
        kind: str  # "content_str" | "content_part" | "tool_arg"
        part_idx: int  # for content_part / tool_arg
        text: str

    candidates: list[_Target] = []
    for mi, msg in enumerate(messages):
        content = msg.get("content")
        if isinstance(content, str):
            candidates.append(_Target(mi, "content_str", 0, content))
        elif isinstance(content, list):
            for pi, part in enumerate(content):
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str):
                        candidates.append(_Target(mi, "content_part", pi, t))
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list):
            for ti, tc in enumerate(tcs):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    continue
                args = fn.get("arguments")
                # OpenAI spec says arguments is a JSON STRING, but
                # callers sometimes pass dicts/lists. Normalize to a
                # string for comparison so the estimator's `str(args)`
                # path stays in sync with truncation. The cloned
                # write-back below replaces with the truncated string
                # — which the upstream expects anyway.
                if isinstance(args, str):
                    candidates.append(_Target(mi, "tool_arg", ti, args))
                elif args is not None:
                    try:
                        serialized = json.dumps(args, separators=(",", ":"))
                    except (TypeError, ValueError):
                        serialized = str(args)
                    candidates.append(_Target(mi, "tool_arg", ti, serialized))

    if not candidates:
        return False

    def _is_system_target(t: _Target) -> bool:
        return messages[t.msg_idx].get("role") in _SYSTEM_ROLES

    nonsystem = [t for t in candidates if not _is_system_target(t)]
    pool = nonsystem or candidates
    target = max(pool, key=lambda t: len(t.text))

    trimmed = _truncate_text(target.text, overshoot_tokens)
    if trimmed is None:
        return False
    new_text, dropped_bytes = trimmed
    suffix = f"\n\n[... {dropped_bytes} bytes truncated to fit context]"

    cloned_msg = dict(messages[target.msg_idx])
    if target.kind == "content_str":
        cloned_msg["content"] = new_text + suffix
    elif target.kind == "content_part":
        # Clone the parts list and the affected part dict.
        original_parts = cloned_msg.get("content")
        if not isinstance(original_parts, list):
            return False
        new_parts = list(original_parts)
        new_part = dict(new_parts[target.part_idx])
        new_part["text"] = new_text + suffix
        new_parts[target.part_idx] = new_part
        cloned_msg["content"] = new_parts
    else:  # "tool_arg"
        original_tcs = cloned_msg.get("tool_calls")
        if not isinstance(original_tcs, list):
            return False
        new_tcs = list(original_tcs)
        new_tc = dict(new_tcs[target.part_idx])
        new_fn = dict(new_tc.get("function") or {})
        new_fn["arguments"] = new_text + suffix
        new_tc["function"] = new_fn
        new_tcs[target.part_idx] = new_tc
        cloned_msg["tool_calls"] = new_tcs

    messages[target.msg_idx] = cloned_msg
    return True


def _flatten_chunks(
    chunks: list[_Chunk],
    kept: list[bool],
    dropped: int,
) -> list[Message]:
    """Reassemble kept chunks in order, with a `[N earlier groups…]` sentinel."""
    out: list[Message] = []
    # Essential system first
    for idx, chunk in enumerate(chunks):
        if kept[idx] and chunk.kind == ChunkKind.ESSENTIAL_SYSTEM:
            out.extend(chunk.messages)
    if dropped > 0:
        out.append(
            {
                "role": "system",
                "content": f"[{dropped} earlier message groups compacted to save context]",
            }
        )
    # Then everything else in order
    for idx, chunk in enumerate(chunks):
        if not kept[idx] or chunk.kind == ChunkKind.ESSENTIAL_SYSTEM:
            continue
        out.extend(chunk.messages)
    return _sanitize_tool_pairs(_strip_trailing_plain_assistant(out))


# ── Layout ─────────────────────────────────────────────────────────


@dataclass
class GeneratorPayload:
    """Per-generator outcome from shared-tail compaction.

    `compacted_chunks` is the count of input atomic chunks (tool chains
    or solo messages) that were dropped from the head when building
    THIS generator's payload. It's the truth-source for observability
    headers — derive-from-payload-length is unsound because compaction
    inserts a `[N earlier message groups compacted]` sentinel that
    changes the message count without reflecting the real drop count.
    """

    upstream_name: str
    messages: list[Message]
    compacted_chunks: int = 0


@dataclass
class SharedTailLayout:
    """Per-generator payloads with byte-identical recent suffix.

    `shared_tail` is the list of recent messages every generator's
    payload ends with verbatim. `per_generator` is the full payload
    list (head + shared_tail) for each generator that fit the budget.
    Generators with insufficient context are dropped; callers can
    decide whether to fall back.
    """

    shared_tail: list[Message]
    per_generator: list[GeneratorPayload]


def compact_with_shared_tail(
    messages: list[Message],
    generators: list[tuple[str, UpstreamConfig]],
    *,
    response_reserve: int,
    tools_token_estimate: int = 0,
    safety_margin: int = 512,
    image_max_active: int = 1,
) -> SharedTailLayout:
    """Build per-generator payloads sharing a byte-identical recent tail.

    See module docstring for algorithm. ``response_reserve`` is the
    output token budget reserved on every upstream (caller-uniform per
    surface). ``tools_token_estimate`` reserves space for the tool-
    schema prefix the upstream prepends; pass 0 for non-tool calls.
    ``safety_margin`` covers tokenizer drift between our estimator and
    the upstream's actual tokenizer; 512 is a typical caller value.

    ``image_max_active`` (D.3.2) — max number of recent images kept
    across the message list. Defaults to 1 (legacy behavior). Dispatch
    plumbs the active profile's `multimodal.max_images` here so a
    request with N≤max_images images doesn't get silently collapsed
    to one (review r30 P2).
    """
    # Multimodal hygiene first, then strip trailing plain assistants
    # BEFORE chunk grouping. If we strip after the tail walk, the most-
    # recent assistant chunk may be the only thing in the tail; stripping
    # it would erase the shared tail entirely and let head compaction
    # diverge across generators (review r13 finding 2).
    collapsed = collapse_stale_images(messages, max_active=image_max_active)
    collapsed = _strip_trailing_plain_assistant(collapsed)

    if not generators:
        return SharedTailLayout(shared_tail=collapsed, per_generator=[])

    # Filter generators with usable budget.
    usable: list[tuple[str, UpstreamConfig]] = []
    for name, up in generators:
        budget = up.context - response_reserve - tools_token_estimate - safety_margin
        if budget <= 0:
            log.warning(
                "compact_with_shared_tail: dropping generator %r — "
                "context=%d cannot fit response_reserve=%d + tools=%d + margin=%d",
                name,
                up.context,
                response_reserve,
                tools_token_estimate,
                safety_margin,
            )
            continue
        usable.append((name, up))

    if not usable:
        return SharedTailLayout(shared_tail=collapsed, per_generator=[])

    tail_budget = min(
        up.context - response_reserve - tools_token_estimate - safety_margin for _, up in usable
    )

    # Group + walk back from end to build the tail.
    chunks = _group_into_chunks(collapsed)
    tail_tokens = 0
    tail_chunk_count = 0
    for chunk in reversed(chunks):
        if chunk.kind in (ChunkKind.ESSENTIAL_SYSTEM, ChunkKind.SYNTHETIC_SYSTEM):
            # System chunks live in the head.
            break
        next_total = tail_tokens + chunk.tokens
        if next_total > tail_budget and tail_chunk_count > 0:
            break
        tail_tokens = next_total
        tail_chunk_count += 1
        # If the very first chunk we add already exceeds the budget,
        # we still include it (something must be in the tail).

    split_point = len(chunks) - tail_chunk_count
    raw_tail: list[Message] = []
    for c in chunks[split_point:]:
        raw_tail.extend(c.messages)

    # Sanitize the tail itself BEFORE freezing it (review r13 finding 3).
    # Tool messages whose tool_call_id doesn't match a tool_call in the
    # tail are removed; missing results get placeholders. Otherwise the
    # tail can carry orphaned tool messages (e.g., when group_into_chunks
    # placed a tool message as its own chunk because the assistant call
    # was dropped to the head and the head/tail boundary split the pair).
    raw_tail = _sanitize_tool_pairs(raw_tail)
    shared_tail = _strip_trailing_plain_assistant(raw_tail)

    # If the tail (typically just a single oversized newest chunk) still
    # exceeds tail_budget, truncate the largest message in a CLONED tail
    # so all generators see the same bounded tail. Without this guard a
    # 50k-char user message with a 4k-context generator returns a payload
    # the upstream will reject (review r14 finding 3). Clone before
    # mutating so caller's input dicts are never touched.
    tail_tokens = estimate_messages_tokens(shared_tail)
    if tail_tokens > tail_budget and shared_tail:
        cloned_tail: list[Message] = [dict(m) for m in shared_tail]
        guard = 0
        while estimate_messages_tokens(cloned_tail) > tail_budget and guard < 48:
            overshoot = estimate_messages_tokens(cloned_tail) - tail_budget
            if not _truncate_largest_message(cloned_tail, overshoot):
                break
            guard += 1
        shared_tail = cloned_tail
        tail_tokens = estimate_messages_tokens(shared_tail)

    head_messages: list[Message] = []
    for c in chunks[:split_point]:
        head_messages.extend(c.messages)
    head_chunk_count = split_point  # number of head chunks before per-gen compaction

    log.debug(
        "shared-tail compaction: tail_budget=%d tail_chunks=%d (%d tokens) "
        "head_msgs=%d generators=%d",
        tail_budget,
        tail_chunk_count,
        tail_tokens,
        len(head_messages),
        len(usable),
    )

    per_generator: list[GeneratorPayload] = []
    for name, up in usable:
        head_budget = (
            up.context - tail_tokens - response_reserve - tools_token_estimate - safety_margin
        )
        if head_budget <= 0 or not head_messages:
            compacted_head: list[Message] = []
            # All head chunks dropped — observable as the entire head's
            # chunk count (review r21 finding 1: report truth, not
            # message-length deltas which are confounded by the
            # `[N earlier message groups…]` sentinel).
            head_dropped = head_chunk_count
        else:
            compacted_head, head_dropped = _compact_head_for_budget(head_messages, head_budget)
        payload = compacted_head + [dict(m) for m in shared_tail]
        # Final sanitize: head/tail concat can re-orphan tool pairs if
        # the head dropped an assistant whose tool result is in the tail
        # (or vice versa). Run once more after concat. Then strip any
        # trailing plain assistant the head may have ended on.
        payload = _sanitize_tool_pairs(payload)
        payload = _strip_trailing_plain_assistant(payload)
        per_generator.append(
            GeneratorPayload(
                upstream_name=name,
                messages=payload,
                compacted_chunks=head_dropped,
            )
        )

    return SharedTailLayout(shared_tail=shared_tail, per_generator=per_generator)


__all__ = [
    "GeneratorPayload",
    "SharedTailLayout",
    "collapse_stale_images",
    "compact_with_shared_tail",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    "estimate_tokens",
]
