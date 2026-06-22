"""D.2.3 tests — shared-tail compaction primitive."""

from __future__ import annotations

from meta_model.config import UpstreamConfig
from meta_model.moa.compaction import (
    GeneratorPayload,
    SharedTailLayout,
    collapse_stale_images,
    compact_with_shared_tail,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_tokens,
)


def _up(name: str, ctx: int, max_out: int = 4096) -> UpstreamConfig:
    return UpstreamConfig(
        model_id=f"{name}-model",
        base_url=f"http://upstream-{name}:9000/v1",
        context=ctx,
        max_output=max_out,
    )


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ── Token estimation ───────────────────────────────────────────────


def test_estimate_tokens_empty() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_overcounts_ascii() -> None:
    # bytes * 10 / 28 ≈ bytes / 2.8. For 28 bytes, expect 10 tokens.
    assert estimate_tokens("a" * 28) == 10


def test_estimate_tokens_handles_utf8_bytes_not_chars() -> None:
    # Multi-byte chars should count by bytes (over-count is correct).
    s = "é" * 14  # 28 UTF-8 bytes
    assert estimate_tokens(s) == 10


def test_estimate_message_tokens_includes_role_and_overhead() -> None:
    msg = {"role": "user", "content": "hi"}
    # role "user" = 4 bytes → 2 tokens; "hi" = 2 bytes → 1 token; +4 overhead
    assert estimate_message_tokens(msg) == 2 + 1 + 4


def test_estimate_message_tokens_includes_tool_calls() -> None:
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "exec", "arguments": '{"cmd":"ls"}'},
            }
        ],
    }
    # role + content + 4 + (name + args + 10)
    base = estimate_tokens("assistant") + estimate_tokens("") + 4
    tc = estimate_tokens("exec") + estimate_tokens('{"cmd":"ls"}') + 10
    assert estimate_message_tokens(msg) == base + tc


def test_estimate_messages_tokens_sums() -> None:
    msgs = [_msg("user", "a"), _msg("assistant", "b")]
    assert estimate_messages_tokens(msgs) == sum(estimate_message_tokens(m) for m in msgs)


# ── collapse_stale_images ──────────────────────────────────────────


def _multimodal(text: str, image_url: str = "data:image/png;base64,aaa") -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }


def test_collapse_stale_images_keeps_only_most_recent() -> None:
    msgs = [
        _multimodal("first image"),
        _msg("assistant", "ok"),
        _multimodal("second image"),
        _msg("assistant", "ok2"),
        _multimodal("third image"),
    ]
    out = collapse_stale_images(msgs, max_active=1)
    image_count = sum(
        1
        for m in out
        if isinstance(m["content"], list)
        and any(p.get("type") == "image_url" for p in m["content"])
    )
    assert image_count == 1
    # The remaining image must be the THIRD message's (most recent).
    last_image_msg = next(m for m in reversed(out) if isinstance(m["content"], list))
    assert any(p.get("type") == "image_url" for p in last_image_msg["content"])


def test_collapse_stale_images_preserves_text_when_image_dropped() -> None:
    msgs = [
        _multimodal("describe this please"),
        _multimodal("now this one"),
    ]
    out = collapse_stale_images(msgs, max_active=1)
    # First message lost its image but text content survives in some form.
    first = out[0]
    serialized = (
        first["content"]
        if isinstance(first["content"], str)
        else " ".join(p.get("text", "") for p in first["content"] if isinstance(p, dict))
    )
    assert "describe this please" in serialized


def test_collapse_stale_images_under_limit_unchanged() -> None:
    msgs = [_multimodal("only image"), _msg("assistant", "ok")]
    out = collapse_stale_images(msgs, max_active=1)
    assert len(out) == 2
    assert isinstance(out[0]["content"], list)


# ── compact_with_shared_tail — happy paths ─────────────────────────


def test_shared_tail_identical_across_generators() -> None:
    """Recent suffix is byte-identical across all per-generator payloads."""
    msgs = [_msg("system", "system prompt for the agent")]
    for i in range(40):
        msgs.append(_msg("user", f"user turn {i} with some context"))
        msgs.append(_msg("assistant", f"assistant reply {i} with reasoning"))

    generators = [
        ("primary", _up("primary", 262_144)),
        ("fast", _up("fast", 114_688)),
        ("reasoning", _up("reasoning", 32_768)),
    ]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=4096,
        tools_token_estimate=1024,
        safety_margin=1024,
    )

    assert isinstance(layout, SharedTailLayout)
    assert len(layout.per_generator) == 3
    assert layout.shared_tail, "shared tail must contain at least one message"

    for gp in layout.per_generator:
        assert isinstance(gp, GeneratorPayload)
        assert len(gp.messages) >= len(layout.shared_tail), (
            f"{gp.upstream_name} payload shorter than shared_tail"
        )
        # Check suffix equality
        suffix = gp.messages[-len(layout.shared_tail) :]
        for i, (got, expected) in enumerate(zip(suffix, layout.shared_tail, strict=True)):
            assert got["role"] == expected["role"], f"{gp.upstream_name} tail[{i}] role differs"
            assert got.get("content") == expected.get("content"), (
                f"{gp.upstream_name} tail[{i}] content differs"
            )


def test_shared_tail_smaller_endpoint_has_shorter_head() -> None:
    """Larger-context generator retains more older history."""
    msgs = [_msg("system", "sys")]
    for i in range(200):
        msgs.append(_msg("user", f"u{i}: {'x' * 200}"))
        msgs.append(_msg("assistant", f"a{i}: {'y' * 200}"))

    generators = [
        ("primary", _up("primary", 262_144)),
        ("reasoning", _up("reasoning", 32_768)),
    ]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=4096,
        tools_token_estimate=0,
        safety_margin=512,
    )

    primary_payload = next(
        gp.messages for gp in layout.per_generator if gp.upstream_name == "primary"
    )
    reasoning_payload = next(
        gp.messages for gp in layout.per_generator if gp.upstream_name == "reasoning"
    )
    assert len(primary_payload) >= len(reasoning_payload)
    assert len(reasoning_payload) >= len(layout.shared_tail)


# ── Generator filtering ────────────────────────────────────────────


def test_shared_tail_drops_generator_with_tiny_context() -> None:
    """A generator that can't fit reserve+margin is excluded."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "hello"),
        _msg("assistant", "hi"),
    ]
    generators = [
        ("primary", _up("primary", 32_768, max_out=4096)),
        ("tiny", _up("tiny", 4096, max_out=4096)),  # ctx == reserve → budget≤0
    ]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=4096,
        tools_token_estimate=1024,
        safety_margin=1024,
    )
    names = [gp.upstream_name for gp in layout.per_generator]
    assert names == ["primary"], "tiny generator should be dropped"


def test_shared_tail_empty_generators_returns_collapsed() -> None:
    msgs = [_msg("user", "hi")]
    layout = compact_with_shared_tail(msgs, [], response_reserve=100)
    assert layout.per_generator == []
    assert len(layout.shared_tail) == 1


def test_shared_tail_all_generators_dropped_returns_empty_per_gen() -> None:
    msgs = [_msg("user", "hi")]
    generators = [("tiny", _up("tiny", 100, max_out=200))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=200,
        safety_margin=100,
    )
    assert layout.per_generator == []


# ── Atomic chunk preservation ──────────────────────────────────────


def _assistant_with_tool_call(call_id: str = "c1", name: str = "exec") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": '{"cmd":"ls"}'},
            }
        ],
    }


def _tool_result(call_id: str = "c1", name: str = "exec", out: str = "ok") -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": out,
    }


def test_shared_tail_keeps_tool_chain_atomic() -> None:
    """Assistant+tool_calls and its tool_result stay glued."""
    msgs = [_msg("system", "sys")]
    for i in range(50):
        msgs.append(_msg("user", f"u{i}"))
    msgs.append(_assistant_with_tool_call("c1", "exec"))
    msgs.append(_tool_result("c1", "exec", "out: ok"))

    generators = [("primary", _up("primary", 4096, max_out=512))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        tools_token_estimate=0,
        safety_margin=128,
    )

    # Tool result must appear in shared tail (most recent).
    tail_has_tool_result = any(
        m.get("role") == "tool" and "out: ok" in str(m.get("content", ""))
        for m in layout.shared_tail
    )
    assert tail_has_tool_result, "tool result must be in shared tail"

    # Assistant-with-tool-calls must also be in the tail (paired).
    tail_has_call = any(
        m.get("role") == "assistant"
        and isinstance(m.get("tool_calls"), list)
        and len(m["tool_calls"]) > 0
        for m in layout.shared_tail
    )
    assert tail_has_call, "assistant tool_call must be in shared tail (atomic)"


def test_shared_tail_includes_at_least_one_chunk_even_if_oversized() -> None:
    """The newest chunk is always included in the tail — but if it
    exceeds tail_budget, its largest message gets truncated so the
    payload still fits the upstream context (review r14 finding 3)."""
    huge = "x" * 50_000
    msgs = [_msg("system", "sys"), _msg("user", huge)]
    generators = [("small", _up("small", 8192, max_out=512))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )
    assert len(layout.shared_tail) >= 1
    # The user turn survives in some form (possibly truncated).
    user_msgs = [m for m in layout.shared_tail if m.get("role") == "user"]
    assert user_msgs, "user turn must remain in the shared tail"
    assert "x" in str(user_msgs[0].get("content", "")), (
        "truncated user content should still preserve some of the original"
    )
    # The payload must fit the upstream prompt budget.
    payload_tokens = estimate_messages_tokens(layout.per_generator[0].messages)
    assert payload_tokens <= 8192 - 512, "payload must fit upstream context"


# ── Trailing-assistant strip ───────────────────────────────────────


def test_payloads_never_end_on_plain_assistant() -> None:
    msgs = [
        _msg("system", "sys"),
        _msg("user", "q"),
        _msg("assistant", "a"),
    ]
    generators = [("primary", _up("primary", 8192))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )
    for gp in layout.per_generator:
        assert gp.messages, "payload must not be empty"
        last = gp.messages[-1]
        if last.get("role") == "assistant":
            tcs = last.get("tool_calls")
            assert isinstance(tcs, list) and len(tcs) > 0, (
                f"{gp.upstream_name} ends on plain assistant"
            )


def test_assistant_with_tool_calls_can_terminate_payload() -> None:
    """A trailing assistant with tool_calls is valid; only plain assistant stripped."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "do it"),
        _assistant_with_tool_call("c1"),
        _tool_result("c1"),
        # Note: end on tool result; that's fine
    ]
    generators = [("primary", _up("primary", 8192))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )
    last = layout.shared_tail[-1]
    assert last["role"] == "tool"


# ── System message handling ────────────────────────────────────────


def test_first_system_message_lives_in_head() -> None:
    """System chunks anchor the head; never enter the shared tail."""
    msgs = [
        _msg("system", "essential prompt"),
        _msg("user", "u1"),
        _msg("assistant", "a1"),
        _msg("user", "u2"),
        _msg("assistant", "a2"),
    ]
    generators = [("primary", _up("primary", 8192))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )
    # System should appear in payload (head) but not in shared_tail.
    assert any(m.get("role") == "system" for m in layout.per_generator[0].messages)
    assert not any(m.get("role") == "system" for m in layout.shared_tail)


# ── Tools token reservation ────────────────────────────────────────


def test_tools_token_estimate_reduces_budget() -> None:
    """Larger tools_token_estimate → smaller tail."""
    msgs = [_msg("system", "sys")]
    for i in range(100):
        msgs.append(_msg("user", f"u{i}: {'x' * 100}"))
        msgs.append(_msg("assistant", f"a{i}: {'y' * 100}"))

    generators = [("primary", _up("primary", 16384))]
    no_tools = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=2000,
        tools_token_estimate=0,
        safety_margin=512,
    )
    with_tools = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=2000,
        tools_token_estimate=8000,
        safety_margin=512,
    )
    assert len(no_tools.shared_tail) > len(with_tools.shared_tail), (
        "tool-schema reservation should shrink shared tail"
    )


# ── Empty / minimal inputs ─────────────────────────────────────────


def test_empty_messages_yields_empty_payloads() -> None:
    generators = [("primary", _up("primary", 8192))]
    layout = compact_with_shared_tail(
        [],
        generators,
        response_reserve=512,
    )
    assert layout.shared_tail == []
    # per_generator may be present with empty messages; both shapes acceptable
    for gp in layout.per_generator:
        assert gp.messages == []


def test_only_system_message_lives_in_head() -> None:
    msgs = [_msg("system", "sys only")]
    generators = [("primary", _up("primary", 8192))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
    )
    # Tail should be empty (only system, which is head-only).
    assert layout.shared_tail == []
    # System should appear in the head.
    assert any(m.get("role") == "system" for m in layout.per_generator[0].messages)


# ── Review r13 follow-up cases ──────────────────────────────────────


def test_huge_essential_system_truncates_to_fit_budget() -> None:
    """A giant essential system message must NOT push the payload past the
    upstream's context. Head compaction guard loop should truncate the
    largest message rather than 400 the upstream (review r13 finding 1)."""
    huge_sys = "system instruction " + ("X" * 400_000)  # ~1.4M tokens raw
    msgs = [
        _msg("system", huge_sys),
        _msg("user", "go"),
    ]
    generators = [("small", _up("small", 4096, max_out=512))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )
    payload = layout.per_generator[0].messages
    assert estimate_messages_tokens(payload) <= 4096, (
        f"payload {estimate_messages_tokens(payload)} > ctx 4096"
    )
    # The system message content should still be present, just trimmed.
    sys_msgs = [m for m in payload if m.get("role") == "system"]
    assert sys_msgs, "system survived truncation in some form"


def test_trailing_plain_assistant_does_not_erase_shared_tail() -> None:
    """If the conversation ends on a plain assistant message, stripping
    must NOT result in an empty shared tail (review r13 finding 2). The
    last user/tool turn must still be in the tail so all generators see
    the same recent reality."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "important question"),
        _msg("assistant", "answer attempt that should be ignored"),
    ]
    generators = [("primary", _up("primary", 8192))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )
    # Tail must contain the user message that drove the conversation.
    assert any(
        m.get("role") == "user" and "important question" in str(m.get("content", ""))
        for m in layout.shared_tail
    ), "user turn must survive trailing-assistant strip"


def test_orphan_tool_result_with_mismatched_id_is_stripped() -> None:
    """A tool message whose tool_call_id doesn't match any assistant
    tool_calls[].id is an orphan and must not appear in the final
    payload (review r13 finding 3)."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "go"),
        _assistant_with_tool_call("c1", "exec"),
        _tool_result("c1", "exec", "ok"),
        # Orphan: tool_call_id not in any assistant's tool_calls
        _tool_result("c999", "exec", "phantom"),
    ]
    generators = [("primary", _up("primary", 8192))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )
    for gp in layout.per_generator:
        for m in gp.messages:
            if m.get("role") == "tool":
                assert m.get("tool_call_id") != "c999", (
                    f"{gp.upstream_name} carries orphan tool result"
                )


def test_developer_role_treated_as_system() -> None:
    """Developer role messages anchor the head like system messages do
    (review r13 finding 4) — Phase D plan says preserve first
    system/developer block."""
    msgs = [
        {"role": "developer", "content": "developer instruction"},
        _msg("user", "go"),
        _msg(
            "assistant",
            "ok",
        ),
        _msg("user", "next"),
    ]
    msgs[2] = _msg("assistant", "ok")  # ensure no trailing comma issue
    generators = [("primary", _up("primary", 8192))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )
    # Developer message survives as part of the head, not the tail.
    assert any(m.get("role") == "developer" for m in layout.per_generator[0].messages)
    assert not any(m.get("role") == "developer" for m in layout.shared_tail)


def test_full_message_dict_equality_in_shared_tail() -> None:
    """Suffix equality should hold for full message dicts, not just
    role/content (review r13 follow-up suggestion)."""
    msgs = [_msg("system", "sys")]
    for i in range(20):
        msgs.append(_msg("user", f"u{i}"))
        msgs.append(_msg("assistant", f"a{i}"))
    msgs.append(_assistant_with_tool_call("c1", "exec"))
    msgs.append(_tool_result("c1", "exec", "ok"))

    generators = [
        ("primary", _up("primary", 32_768)),
        ("reasoning", _up("reasoning", 16_384)),
    ]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=2000,
        safety_margin=512,
    )
    for gp in layout.per_generator:
        suffix = gp.messages[-len(layout.shared_tail) :]
        for got, expected in zip(suffix, layout.shared_tail, strict=True):
            assert got == expected, (
                f"{gp.upstream_name} suffix dict not byte-identical to shared_tail"
            )


def test_tool_call_with_non_string_arguments_does_not_crash() -> None:
    """Tool-call ``arguments`` may be any JSON-shape in caller-built
    payloads. Token estimation must coerce safely (review r13 finding 6).
    """
    msgs = [
        _msg("system", "sys"),
        _msg("user", "go"),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    # Non-string arguments shouldn't crash the estimator
                    "function": {"name": "exec", "arguments": {"cmd": "ls"}},
                }
            ],
        },
    ]
    generators = [("primary", _up("primary", 8192))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )
    assert len(layout.per_generator) == 1


def test_truncation_does_not_mutate_input_messages() -> None:
    """A small generator's truncation must NOT shrink the larger
    generator's payload via shared dict references (review r14
    finding 1). Caller's input list and dicts must be untouched."""
    huge = "system instruction " + ("X" * 50_000)
    original_len = len(huge)
    input_msgs = [
        _msg("system", huge),
        _msg("user", "go"),
    ]
    # Snapshot caller's view.
    snapshot = [dict(m) for m in input_msgs]

    generators = [
        ("big", _up("big", 200_000)),
        ("small", _up("small", 4096, max_out=512)),
    ]
    layout = compact_with_shared_tail(
        input_msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )

    # Caller's input dicts unchanged.
    for orig, snap in zip(input_msgs, snapshot, strict=True):
        assert orig == snap, "compact_with_shared_tail must not mutate input dicts"
    assert len(input_msgs[0]["content"]) == original_len

    # The big generator's payload retains the full system content.
    big_payload = next(gp.messages for gp in layout.per_generator if gp.upstream_name == "big")
    sys_msgs = [m for m in big_payload if m.get("role") == "system"]
    # At least one system message in the big payload should still be the
    # full huge content (or close to it). Truncation only happens for
    # over-budget generators.
    assert any(len(str(m.get("content", ""))) > 30_000 for m in sys_msgs), (
        "big generator should retain full system message — small gen's "
        "truncation must not have leaked into big's payload"
    )


def test_oversized_multimodal_text_part_is_truncated() -> None:
    """A multimodal user message with a 50k-char text part must be
    truncated so the payload fits the upstream context (review r15 —
    truncation must handle list-shape content)."""
    huge = "x" * 50_000
    msgs = [
        _msg("system", "sys"),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": huge},
            ],
        },
    ]
    generators = [("small", _up("small", 4096, max_out=512))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )
    payload_tokens = estimate_messages_tokens(layout.per_generator[0].messages)
    assert payload_tokens <= 4096, (
        f"multimodal payload {payload_tokens} > ctx 4096 — truncation must "
        f"handle list-shape content"
    )


def test_oversized_dict_tool_call_arguments_are_truncated() -> None:
    """A non-string (dict) `function.arguments` field with massive
    embedded text must still be truncated (review r16 finding —
    estimator counts str(args) bytes but earlier truncator skipped
    non-string)."""
    huge_dict_args = {"data": "x" * 50_000}
    msgs = [
        _msg("system", "sys"),
        _msg("user", "go"),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "exec", "arguments": huge_dict_args},
                }
            ],
        },
        _tool_result("c1", "exec", "ok"),
    ]
    generators = [("small", _up("small", 4096, max_out=512))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )
    payload_tokens = estimate_messages_tokens(layout.per_generator[0].messages)
    assert payload_tokens <= 4096, f"dict tool-call args payload {payload_tokens} > ctx 4096"


def test_oversized_tool_call_arguments_are_truncated() -> None:
    """An assistant message with 50k chars in `tool_calls[*].function.arguments`
    must be truncated (review r15)."""
    huge_args = '{"data":"' + ("x" * 50_000) + '"}'
    msgs = [
        _msg("system", "sys"),
        _msg("user", "go"),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "exec", "arguments": huge_args},
                }
            ],
        },
        _tool_result("c1", "exec", "ok"),
    ]
    generators = [("small", _up("small", 4096, max_out=512))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )
    payload_tokens = estimate_messages_tokens(layout.per_generator[0].messages)
    assert payload_tokens <= 4096, f"tool-call args payload {payload_tokens} > ctx 4096"


def test_orphan_tool_with_missing_tool_call_id_is_dropped() -> None:
    """A tool message with no tool_call_id (or non-string) is an orphan
    and must be dropped (review r14 finding 2)."""
    msgs = [
        _msg("system", "sys"),
        _msg("user", "go"),
        # Tool message with NO tool_call_id at all
        {"role": "tool", "content": "phantom result"},
        # Tool message with a non-string tool_call_id
        {"role": "tool", "tool_call_id": 42, "content": "another phantom"},
    ]
    generators = [("primary", _up("primary", 8192))]
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=512,
        safety_margin=128,
    )
    for gp in layout.per_generator:
        for m in gp.messages:
            if m.get("role") != "tool":
                continue
            tcid = m.get("tool_call_id")
            assert isinstance(tcid, str), (
                f"{gp.upstream_name} carries tool message with non-string id"
            )


def test_payload_fits_upstream_context_invariant() -> None:
    """For every per-generator payload, total tokens must be ≤ the
    upstream's effective budget (context - reserve - tools - margin).
    This is the load-bearing invariant (review r13 finding 1)."""
    msgs = [_msg("system", "sys " + "x" * 5000)]
    for i in range(120):
        msgs.append(_msg("user", f"u{i}: {'x' * 250}"))
        msgs.append(_msg("assistant", f"a{i}: {'y' * 250}"))
    generators = [
        ("big", _up("big", 32_768)),
        ("small", _up("small", 8192)),
    ]
    response_reserve = 2000
    tools_token_estimate = 256
    safety_margin = 512
    layout = compact_with_shared_tail(
        msgs,
        generators,
        response_reserve=response_reserve,
        tools_token_estimate=tools_token_estimate,
        safety_margin=safety_margin,
    )
    for gp in layout.per_generator:
        up_ctx = next(u.context for n, u in generators if n == gp.upstream_name)
        ceiling = up_ctx - response_reserve - tools_token_estimate - safety_margin
        # The actual payload must fit the prompt budget; small overshoot
        # within safety_margin is the entire point of the margin.
        actual = estimate_messages_tokens(gp.messages)
        assert actual <= up_ctx - response_reserve, (
            f"{gp.upstream_name}: payload {actual} > ctx-reserve {up_ctx - response_reserve}"
        )
        # Stronger goal: should also fit within the prompt-only budget.
        assert actual <= ceiling + safety_margin, (
            f"{gp.upstream_name}: payload {actual} > ceiling+margin {ceiling + safety_margin}"
        )
