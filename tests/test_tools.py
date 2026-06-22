"""D.3.1 — `meta_model.moa.tools` unit tests.

Covers legacy normalization, constraint resolution, fallback selection,
and the `finalize_response` enforcement flow. Synthesizer integration
lives in `test_synthesizer.py`; dispatch-level entry tests live in
`test_dispatch.py`.
"""

from __future__ import annotations

from typing import Any

import pytest

from meta_model.moa.tools import (
    FinalizeOutcome,
    ToolConstraint,
    ToolNormalizationError,
    _sanitize_tool_name,
    candidate_violates_declared,
    canonical_args,
    finalize_response,
    normalize_tool_calls_signature,
    normalize_tool_request,
    pick_fallback_candidate,
    raw_emitted_call_names,
    resolve_tool_constraint,
)

# ── canonical_args / normalize_tool_calls_signature ─────────────────


def test_canonical_args_orders_keys() -> None:
    """JSON-serialized args canonicalize key order so structurally
    equal call args compare equal across generators."""
    assert canonical_args('{"b": 1, "a": 2}') == canonical_args('{"a":2,"b":1}')


def test_canonical_args_handles_dict_input() -> None:
    """Some upstreams return arguments as a dict instead of a JSON
    string. Canonicalization handles both."""
    assert canonical_args({"a": 1, "b": 2}) == canonical_args('{"b": 2, "a": 1}')


def test_normalize_tool_calls_signature_ignores_id() -> None:
    a = [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    b = [{"id": "9999", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    assert normalize_tool_calls_signature(a) == normalize_tool_calls_signature(b)


def test_normalize_tool_calls_signature_distinguishes_by_args() -> None:
    a = [{"type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}]
    b = [{"type": "function", "function": {"name": "f", "arguments": '{"x": 2}'}}]
    assert normalize_tool_calls_signature(a) != normalize_tool_calls_signature(b)


# ── normalize_tool_request — legacy promotion ──────────────────────


def test_normalize_promotes_functions_to_tools() -> None:
    body = {
        "messages": [],
        "functions": [{"name": "f", "parameters": {}}],
    }
    out = normalize_tool_request(body)
    assert "functions" not in out
    assert out["tools"] == [{"type": "function", "function": {"name": "f", "parameters": {}}}]


def test_normalize_promotes_function_call_string_to_tool_choice() -> None:
    body = {"messages": [], "function_call": "auto"}
    out = normalize_tool_request(body)
    assert "function_call" not in out
    assert out["tool_choice"] == "auto"


def test_normalize_promotes_function_call_dict_to_tool_choice() -> None:
    # Forced-name promotion now requires that the name exists in declared
    # `tools` — review r2 HIGH closes the impossible-contract escape valve
    # where a forced name was silently accepted with no tools to force.
    body = {
        "messages": [],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "function_call": {"name": "f"},
    }
    out = normalize_tool_request(body)
    assert out["tool_choice"] == {"type": "function", "function": {"name": "f"}}


def test_normalize_rejects_forced_name_with_no_tools() -> None:
    body = {"messages": [], "function_call": {"name": "f"}}
    with pytest.raises(ToolNormalizationError) as ei:
        normalize_tool_request(body)
    assert ei.value.code == "invalid_request_error"
    assert "no `tools` are declared" in ei.value.message


def test_normalize_rejects_forced_name_not_in_declared_tools() -> None:
    body = {
        "messages": [],
        "tools": [{"type": "function", "function": {"name": "g"}}],
        "tool_choice": {"type": "function", "function": {"name": "f"}},
    }
    with pytest.raises(ToolNormalizationError) as ei:
        normalize_tool_request(body)
    assert ei.value.code == "invalid_request_error"
    assert "not in declared `tools`" in ei.value.message


def test_normalize_empty_arrays_are_noop() -> None:
    body = {"messages": [], "functions": [], "tools": []}
    out = normalize_tool_request(body)
    assert "functions" not in out
    assert "tools" not in out


# ── normalize_tool_request — rejections ────────────────────────────


def test_normalize_rejects_mixed_functions_and_tools() -> None:
    body = {
        "messages": [],
        "functions": [{"name": "a"}],
        "tools": [{"type": "function", "function": {"name": "b"}}],
    }
    with pytest.raises(ToolNormalizationError) as exc:
        normalize_tool_request(body)
    assert exc.value.code == "invalid_request_error"


def test_normalize_rejects_mixed_function_call_and_tool_choice() -> None:
    body = {
        "messages": [],
        "function_call": "auto",
        "tool_choice": "auto",
    }
    with pytest.raises(ToolNormalizationError) as exc:
        normalize_tool_request(body)
    assert exc.value.code == "invalid_request_error"


def test_normalize_rejects_custom_tool_type() -> None:
    body = {
        "messages": [],
        "tools": [{"type": "custom", "custom": {"name": "x"}}],
    }
    with pytest.raises(ToolNormalizationError) as exc:
        normalize_tool_request(body)
    assert exc.value.code == "feature_not_supported_in_v1"


def test_normalize_rejects_allowed_tools() -> None:
    body = {"messages": [], "allowed_tools": ["f"]}
    with pytest.raises(ToolNormalizationError) as exc:
        normalize_tool_request(body)
    assert exc.value.code == "feature_not_supported_in_v1"


def test_normalize_rejects_exotic_tool_choice_dict() -> None:
    body = {
        "messages": [],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": {"type": "custom", "custom": {"name": "f"}},
    }
    with pytest.raises(ToolNormalizationError) as exc:
        normalize_tool_request(body)
    assert exc.value.code == "feature_not_supported_in_v1"


def test_normalize_rejects_invalid_tool_choice_string() -> None:
    body = {"messages": [], "tool_choice": "must_call"}
    with pytest.raises(ToolNormalizationError) as exc:
        normalize_tool_request(body)
    assert exc.value.code == "invalid_request_error"


def test_normalize_does_not_mutate_input() -> None:
    body = {
        "messages": [],
        "functions": [{"name": "f"}],
    }
    snapshot = {"messages": [], "functions": [{"name": "f"}]}
    normalize_tool_request(body)
    assert body == snapshot


# ── resolve_tool_constraint — defaults ─────────────────────────────


def test_constraint_defaults_to_auto_when_tools_present() -> None:
    body = {"messages": [], "tools": [{"type": "function", "function": {"name": "f"}}]}
    c = resolve_tool_constraint(body)
    assert c.mode == "auto"
    assert c.has_tools is True


def test_constraint_defaults_to_none_without_tools() -> None:
    body = {"messages": []}
    c = resolve_tool_constraint(body)
    assert c.mode == "none"
    assert c.has_tools is False


def test_constraint_specific_extracts_forced_name() -> None:
    body = {
        "messages": [],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": {"type": "function", "function": {"name": "f"}},
    }
    c = resolve_tool_constraint(body)
    assert c.mode == "specific"
    assert c.forced_function_name == "f"


def test_constraint_parallel_default_true() -> None:
    body = {"messages": []}
    c = resolve_tool_constraint(body)
    assert c.parallel_tool_calls is True


def test_constraint_parallel_false_passes_through() -> None:
    body = {"messages": [], "parallel_tool_calls": False}
    c = resolve_tool_constraint(body)
    assert c.parallel_tool_calls is False


def test_tool_arbitration_needed_required() -> None:
    body = {
        "messages": [],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": "required",
    }
    c = resolve_tool_constraint(body)
    assert c.tool_arbitration_needed is True


def test_tool_arbitration_needed_none() -> None:
    body = {"messages": []}
    c = resolve_tool_constraint(body)
    assert c.tool_arbitration_needed is False


# ── pick_fallback_candidate ────────────────────────────────────────


def _msg_with_calls(*names: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"id-{i}",
                "type": "function",
                "function": {"name": n, "arguments": "{}"},
            }
            for i, n in enumerate(names)
        ],
    }


def _text(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def test_pick_fallback_required_finds_any_call() -> None:
    candidates = [_text("just text"), _msg_with_calls("f")]
    constraint = ToolConstraint(
        mode="required", forced_function_name=None, parallel_tool_calls=True, has_tools=True
    )
    picked = pick_fallback_candidate(candidates, constraint)
    assert picked is not None
    idx, _ = picked
    assert idx == 1


def test_pick_fallback_specific_filters_by_name() -> None:
    candidates = [_msg_with_calls("wrong"), _msg_with_calls("right")]
    constraint = ToolConstraint(
        mode="specific", forced_function_name="right", parallel_tool_calls=True, has_tools=True
    )
    picked = pick_fallback_candidate(candidates, constraint)
    assert picked is not None
    idx, _ = picked
    assert idx == 1


def test_pick_fallback_most_frequent_signature() -> None:
    """Per F-fallback rule: most-frequent satisfying signature wins,
    ties by generator order. Two of three candidates agree on f({})
    while one differs — agreement wins."""
    a = _msg_with_calls("f")
    b = _msg_with_calls("f")
    c = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}],
    }
    # Order [c, a, b]: c's outlier signature has count 1; a/b's
    # signature has count 2 — agreement should win.
    candidates = [c, a, b]
    constraint = ToolConstraint(
        mode="required", forced_function_name=None, parallel_tool_calls=True, has_tools=True
    )
    picked = pick_fallback_candidate(candidates, constraint)
    assert picked is not None
    idx, _ = picked
    assert idx == 1  # `a` is the first occurrence of the winning signature


def test_pick_fallback_ties_by_generator_order() -> None:
    """Equal-frequency signatures break by first-occurrence index."""
    a = _msg_with_calls("alpha")
    b = _msg_with_calls("beta")
    candidates = [a, b]
    constraint = ToolConstraint(
        mode="required", forced_function_name=None, parallel_tool_calls=True, has_tools=True
    )
    picked = pick_fallback_candidate(candidates, constraint)
    assert picked is not None
    idx, _ = picked
    assert idx == 0  # `a` came first


def test_pick_fallback_post_parallel_trim_validates_specific() -> None:
    """A candidate emitting [right, wrong] under parallel=false must
    qualify because deterministic trim collapses to [right]."""
    candidates = [_msg_with_calls("right", "wrong")]
    constraint = ToolConstraint(
        mode="specific", forced_function_name="right", parallel_tool_calls=False, has_tools=True
    )
    picked = pick_fallback_candidate(candidates, constraint)
    assert picked is not None


def test_pick_fallback_no_satisfier_returns_none() -> None:
    candidates = [_text("text"), _text("more text")]
    constraint = ToolConstraint(
        mode="required", forced_function_name=None, parallel_tool_calls=True, has_tools=True
    )
    assert pick_fallback_candidate(candidates, constraint) is None


# ── finalize_response — ordering + transforms ──────────────────────


def _constraint(
    mode: str = "auto",
    forced: str | None = None,
    parallel: bool = True,
    has_tools: bool = True,
) -> ToolConstraint:
    return ToolConstraint(
        mode=mode,  # type: ignore[arg-type]
        forced_function_name=forced,
        parallel_tool_calls=parallel,
        has_tools=has_tools,
    )


def test_finalize_parallel_false_trims_to_one() -> None:
    msg = _msg_with_calls("a", "b", "c")
    msg["finish_reason"] = "tool_calls"
    out = finalize_response(msg, [msg], _constraint(parallel=False), synth_ran=True)
    assert out.error is None
    assert len(out.msg["tool_calls"]) == 1
    assert out.fallback_reason == "parallel_violation_trimmed"
    assert out.msg["finish_reason"] == "tool_calls"  # call still present → keeps tag


def test_finalize_parallel_false_single_call_no_trim() -> None:
    msg = _msg_with_calls("a")
    out = finalize_response(msg, [msg], _constraint(parallel=False), synth_ran=True)
    assert out.error is None
    assert out.fallback_reason is None


def test_finalize_tool_choice_none_strips_calls() -> None:
    msg = _msg_with_calls("f")
    msg["finish_reason"] = "tool_calls"
    out = finalize_response(msg, [msg], _constraint(mode="none", has_tools=False), synth_ran=True)
    assert out.error is None
    assert "tool_calls" not in out.msg
    assert out.msg["finish_reason"] == "stop"  # coerced
    assert out.msg["content"] == ""  # null-or-missing → ""
    assert out.fallback_reason == "tool_choice_none_stripped"


def test_finalize_tool_choice_none_text_unchanged() -> None:
    msg = _text("hello")
    out = finalize_response(msg, [msg], _constraint(mode="none", has_tools=False), synth_ran=True)
    assert out.error is None
    assert out.fallback_reason is None
    assert out.msg["content"] == "hello"


def test_finalize_required_violation_pre_synth_signals_repair() -> None:
    """Fast-path candidate with no tool_calls under required → returns
    pre-synth violation marker so the synthesizer can attempt repair."""
    msg = _text("nope")
    out = finalize_response(msg, [msg], _constraint(mode="required"), synth_ran=False)
    assert out.error is not None
    assert out.error.code == "constraint_violated_pre_synth"


def test_finalize_required_post_synth_uses_fallback() -> None:
    msg = _text("synth still failed")
    candidate = _msg_with_calls("f")
    out = finalize_response(msg, [msg, candidate], _constraint(mode="required"), synth_ran=True)
    assert out.error is None
    assert out.fallback_reason == "tool_required_repaired"
    assert out.msg.get("tool_calls")


def test_finalize_required_no_satisfier_502() -> None:
    msg = _text("nope")
    out = finalize_response(msg, [msg], _constraint(mode="required"), synth_ran=True)
    assert out.error is not None
    assert out.error.code == "tool_required_unmet"


def test_finalize_specific_wrong_function_repairs_post_synth() -> None:
    msg = _msg_with_calls("wrong")
    candidate = _msg_with_calls("right")
    out = finalize_response(
        msg,
        [msg, candidate],
        _constraint(mode="specific", forced="right"),
        synth_ran=True,
    )
    assert out.error is None
    assert out.fallback_reason == "tool_choice_repaired"
    assert out.msg["tool_calls"][0]["function"]["name"] == "right"


def test_finalize_specific_no_satisfier_unmet() -> None:
    msg = _msg_with_calls("wrong")
    out = finalize_response(
        msg, [msg], _constraint(mode="specific", forced="right"), synth_ran=True
    )
    assert out.error is not None
    assert out.error.code == "tool_choice_unmet"


def test_finalize_specific_with_extra_call_violates() -> None:
    """tool_choice={name:X} with parallel=true and synth emitted [X, Y]
    must NOT pass — ALL retained calls must be X (review r24 F4)."""
    msg = _msg_with_calls("right", "extra")
    candidate = _msg_with_calls("right")
    out = finalize_response(
        msg,
        [msg, candidate],
        _constraint(mode="specific", forced="right"),
        synth_ran=True,
    )
    assert out.error is None
    assert out.fallback_reason == "tool_choice_repaired"
    assert len(out.msg["tool_calls"]) == 1


def test_finalize_specific_under_parallel_false_trims_then_validates() -> None:
    """[right, wrong] under parallel=false trims to [right]; specific=right
    is then satisfied — no fallback."""
    msg = _msg_with_calls("right", "wrong")
    out = finalize_response(
        msg,
        [msg],
        _constraint(mode="specific", forced="right", parallel=False),
        synth_ran=True,
    )
    assert out.error is None
    assert out.fallback_reason == "parallel_violation_trimmed"
    assert len(out.msg["tool_calls"]) == 1
    assert out.msg["tool_calls"][0]["function"]["name"] == "right"


def test_finalize_auto_passes_through_text() -> None:
    msg = _text("plain answer")
    out = finalize_response(msg, [msg], _constraint(mode="auto"), synth_ran=True)
    assert out.error is None
    assert out.fallback_reason is None


def test_finalize_does_not_leak_changes_to_candidates() -> None:
    """Caller is expected to pass a clone, but verify the candidates
    list itself is never mutated by finalize."""
    cand = _msg_with_calls("a", "b")
    cand["finish_reason"] = "tool_calls"
    snapshot_calls = list(cand["tool_calls"])
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": list(cand["tool_calls"]),
        "finish_reason": "tool_calls",
    }
    finalize_response(msg, [cand], _constraint(parallel=False), synth_ran=True)
    assert cand["tool_calls"] == snapshot_calls
    assert cand["finish_reason"] == "tool_calls"


def test_finalize_outcome_dataclass_shape() -> None:
    """Smoke test on the outcome shape — type signature stability."""
    out = finalize_response(_text("ok"), [_text("ok")], _constraint(mode="auto"), synth_ran=True)
    assert isinstance(out, FinalizeOutcome)
    assert out.msg["content"] == "ok"


# ── review r26 follow-ups ───────────────────────────────────────────


def test_finalize_fallback_idx_set_on_repair() -> None:
    """When finalize falls back to a candidate, fallback_idx points
    at the chosen candidate's index (review r26 P1)."""
    msg = _msg_with_calls("wrong")
    candidate_right = _msg_with_calls("right")
    out = finalize_response(
        msg,
        [msg, candidate_right],
        _constraint(mode="specific", forced="right"),
        synth_ran=True,
    )
    assert out.error is None
    assert out.fallback_idx == 1


def test_finalize_no_fallback_idx_when_msg_passes() -> None:
    msg = _msg_with_calls("right")
    out = finalize_response(
        msg, [msg], _constraint(mode="specific", forced="right"), synth_ran=True
    )
    assert out.error is None
    assert out.fallback_idx is None


def test_finalize_finish_reason_stop_to_tool_calls() -> None:
    """Review r26 P2: a message with retained tool_calls that carries
    a stale `stop` reason must coerce to `tool_calls`."""
    msg = _msg_with_calls("f")
    msg["finish_reason"] = "stop"
    out = finalize_response(msg, [msg], _constraint(mode="auto"), synth_ran=True)
    assert out.error is None
    assert out.msg["finish_reason"] == "tool_calls"


def test_finalize_finish_reason_unchanged_when_coherent() -> None:
    """No coercion when msg + reason already match."""
    msg = _msg_with_calls("f")
    msg["finish_reason"] = "tool_calls"
    out = finalize_response(msg, [msg], _constraint(mode="auto"), synth_ran=True)
    assert out.error is None
    assert out.msg["finish_reason"] == "tool_calls"


def test_normalize_malformed_function_tool_400_invalid_request() -> None:
    """Review r26 P2: function-typed tool with missing function body =>
    invalid_request_error, not feature_not_supported_in_v1."""
    body = {"messages": [], "tools": [{"type": "function"}]}  # no `function` key
    with pytest.raises(ToolNormalizationError) as exc:
        normalize_tool_request(body)
    assert exc.value.code == "invalid_request_error"


def test_normalize_function_tool_with_null_body_invalid_request() -> None:
    body = {"messages": [], "tools": [{"type": "function", "function": None}]}
    with pytest.raises(ToolNormalizationError) as exc:
        normalize_tool_request(body)
    assert exc.value.code == "invalid_request_error"


def test_normalize_function_tool_choice_missing_name_invalid_request() -> None:
    body = {
        "messages": [],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": {"type": "function", "function": {}},
    }
    with pytest.raises(ToolNormalizationError) as exc:
        normalize_tool_request(body)
    assert exc.value.code == "invalid_request_error"


def test_normalize_missing_type_field_invalid_request() -> None:
    """Review r27: tools entry without `type` field is malformed shape,
    not unsupported feature."""
    body = {"messages": [], "tools": [{"function": {"name": "f"}}]}  # no type
    with pytest.raises(ToolNormalizationError) as exc:
        normalize_tool_request(body)
    assert exc.value.code == "invalid_request_error"


def test_normalize_tool_choice_missing_type_invalid_request() -> None:
    body = {
        "messages": [],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": {"function": {"name": "f"}},  # missing type
    }
    with pytest.raises(ToolNormalizationError) as exc:
        normalize_tool_request(body)
    assert exc.value.code == "invalid_request_error"


# ── declared-tool constraint enforcement (review r2 HIGH) ──────────────


def _msg_with_call(name: str, args: str = "{}") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": name, "arguments": args}}
        ],
    }


def _msg_with_legacy_call(name: str, args: str = "{}") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "function_call": {"name": name, "arguments": args},
    }


def test_raw_emitted_call_names_modern() -> None:
    msg = _msg_with_call("foo")
    assert raw_emitted_call_names(msg) == ["foo"]


def test_raw_emitted_call_names_legacy() -> None:
    msg = _msg_with_legacy_call("foo")
    assert raw_emitted_call_names(msg) == ["foo"]


def test_raw_emitted_call_names_dual_shape() -> None:
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "x", "type": "function", "function": {"name": "a"}}],
        "function_call": {"name": "b"},
    }
    assert raw_emitted_call_names(msg) == ["a", "b"]


def test_raw_emitted_call_names_no_calls() -> None:
    msg = {"role": "assistant", "content": "hi"}
    assert raw_emitted_call_names(msg) == []


# ── _sanitize_tool_name ─────────────────────────────────────────────


def test_sanitize_tool_name_pass_through_clean_name() -> None:
    """Conformant tool names are unchanged."""
    assert _sanitize_tool_name("get_weather") == "get_weather"
    assert _sanitize_tool_name("search-files") == "search-files"
    assert _sanitize_tool_name("fn_v2") == "fn_v2"


def test_sanitize_tool_name_strips_harmony_channel_marker() -> None:
    """reasoning-model Harmony leak: `exec<|channel|>commentary` → `exec`.
    Observed 3× in 2h on tool_chat.v1 (2026-05-04 prod logs).
    """
    assert _sanitize_tool_name("exec<|channel|>commentary") == "exec"
    assert _sanitize_tool_name("find?<|channel|>commentary") == "find?"
    assert _sanitize_tool_name("vision<|channel|>commentary") == "vision"


def test_sanitize_tool_name_strips_chatml_specials() -> None:
    """ChatML / IM-style / Llama specials also use the `<|...|>` boundary."""
    assert _sanitize_tool_name("get_weather<|im_end|>") == "get_weather"
    assert _sanitize_tool_name("foo<|end|>") == "foo"


def test_sanitize_tool_name_returns_empty_when_only_control_tokens() -> None:
    """If the entire name is control tokens, no real name to recover."""
    assert _sanitize_tool_name("<|channel|>commentary") == ""
    assert _sanitize_tool_name("<|im_end|>") == ""


def test_sanitize_tool_name_handles_whitespace_around_marker() -> None:
    """Trailing whitespace after marker is stripped."""
    assert _sanitize_tool_name("exec <|channel|>commentary") == "exec"


def test_raw_emitted_call_names_strips_harmony_marker_in_tool_calls() -> None:
    """Modern tool_calls shape: name carries Harmony bleed → cleaned in
    place. Mutation is load-bearing — downstream finalizer needs to
    see the cleaned name."""
    msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "exec<|channel|>commentary", "arguments": "{}"},
            }
        ],
    }
    assert raw_emitted_call_names(msg) == ["exec"]
    # Verify in-place mutation of the underlying message
    assert msg["tool_calls"][0]["function"]["name"] == "exec"


def test_raw_emitted_call_names_strips_harmony_marker_in_function_call() -> None:
    """Legacy function_call shape: also cleaned in place."""
    msg = {
        "role": "assistant",
        "function_call": {"name": "search<|channel|>commentary", "arguments": "{}"},
    }
    assert raw_emitted_call_names(msg) == ["search"]
    assert msg["function_call"]["name"] == "search"


def test_raw_emitted_call_names_drops_call_when_name_is_only_control_tokens() -> None:
    """If a tool_call's name is entirely control tokens, drop it from
    the returned list (treat as if no name was emitted)."""
    msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "<|channel|>commentary", "arguments": "{}"},
            }
        ],
    }
    assert raw_emitted_call_names(msg) == []


def test_raw_emitted_call_names_clean_name_not_mutated() -> None:
    """Conformant names should NOT trigger any mutation (preserves
    object identity for callers who care)."""
    fn = {"name": "get_weather", "arguments": "{}"}
    msg = {
        "role": "assistant",
        "tool_calls": [{"id": "1", "type": "function", "function": fn}],
    }
    raw_emitted_call_names(msg)
    # Name unchanged
    assert fn["name"] == "get_weather"


def test_undeclared_tool_check_passes_after_sanitization() -> None:
    """A candidate emitting `exec<|channel|>commentary` against a declared
    tool set of `["exec"]` must pass the declared-set check after
    raw_emitted_call_names sanitizes in place. This is the load-bearing
    integration: harmony bleed should NOT cause undeclared_tool demotion
    when the cleaned name is in the declared set.
    """
    msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "exec<|channel|>commentary", "arguments": "{}"},
            }
        ],
    }
    constraint = _constraint_with_declared(["exec"])
    # raw_emitted_call_names mutates msg in place
    raw_emitted_call_names(msg)
    assert candidate_violates_declared(msg, constraint) is None


def _constraint_with_declared(declared: list[str]) -> ToolConstraint:
    body = {
        "tools": [{"type": "function", "function": {"name": n}} for n in declared],
    }
    return resolve_tool_constraint(body)


def test_candidate_violates_declared_none_when_all_declared() -> None:
    constraint = _constraint_with_declared(["foo", "bar"])
    assert candidate_violates_declared(_msg_with_call("foo"), constraint) is None
    assert candidate_violates_declared(_msg_with_call("bar"), constraint) is None


def test_candidate_violates_declared_undeclared_modern() -> None:
    constraint = _constraint_with_declared(["foo"])
    assert candidate_violates_declared(_msg_with_call("list"), constraint) == "undeclared_tool"


def test_candidate_violates_declared_undeclared_legacy() -> None:
    """A backend emitting only legacy `function_call` to an undeclared
    name must be flagged just like a modern tool_calls violation."""
    constraint = _constraint_with_declared(["foo"])
    assert (
        candidate_violates_declared(_msg_with_legacy_call("list"), constraint)
        == "undeclared_tool"
    )


def test_candidate_violates_declared_dual_shape() -> None:
    """Both `tool_calls` AND `function_call` set is malformed."""
    constraint = _constraint_with_declared(["foo"])
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "x", "type": "function", "function": {"name": "foo"}}],
        "function_call": {"name": "foo"},
    }
    assert candidate_violates_declared(msg, constraint) == "dual_shape_response"


def test_candidate_violates_declared_text_only_no_violation() -> None:
    constraint = _constraint_with_declared(["foo"])
    text_msg = {"role": "assistant", "content": "hello"}
    assert candidate_violates_declared(text_msg, constraint) is None


def test_candidate_violates_declared_no_tools_with_calls_violates() -> None:
    """When `tools` is absent (declared set empty), any tool call is a
    violation per OpenAI's contract — the request didn't authorize any
    tool to be called."""
    constraint = resolve_tool_constraint({})  # no tools
    assert (
        candidate_violates_declared(_msg_with_call("anything"), constraint)
        == "undeclared_tool"
    )


def test_resolve_tool_constraint_populates_declared_names() -> None:
    body = {
        "tools": [
            {"type": "function", "function": {"name": "alpha"}},
            {"type": "function", "function": {"name": "beta"}},
        ],
    }
    c = resolve_tool_constraint(body)
    assert c.declared_tool_names == frozenset({"alpha", "beta"})


def test_resolve_tool_constraint_empty_when_tools_absent() -> None:
    c = resolve_tool_constraint({})
    assert c.declared_tool_names == frozenset()


# ── finalize_response: declared-name post-check ──────────────────────


def test_finalize_response_undeclared_in_auto_mode_falls_back_to_satisfying_candidate() -> None:
    """Synth output emitting an undeclared name under tool_choice='auto'
    must fall back to the lowest-index surviving candidate that
    satisfies the declared-tool contract — the post-check review r2
    HIGH demanded."""
    body = {
        "tools": [{"type": "function", "function": {"name": "good"}}],
        "messages": [],
    }
    constraint = resolve_tool_constraint(body)
    candidates = [_msg_with_call("good", '{"a":1}'), _msg_with_call("good", '{"a":2}')]
    # Synth fabricated a call to undeclared "rogue"
    synth_msg = _msg_with_call("rogue", '{"a":3}')
    out = finalize_response(synth_msg, candidates, constraint, synth_ran=True)
    assert out.error is None
    assert out.fallback_reason == "undeclared_tool_repaired"
    # Picked candidate calls a declared name
    picked_calls = out.msg.get("tool_calls") or []
    assert len(picked_calls) == 1
    assert picked_calls[0]["function"]["name"] == "good"


def test_finalize_response_undeclared_when_no_satisfying_candidate_returns_error() -> None:
    body = {
        "tools": [{"type": "function", "function": {"name": "good"}}],
        "messages": [],
    }
    constraint = resolve_tool_constraint(body)
    # All candidates ALSO emit the undeclared name (or text-only without calls)
    candidates = [_msg_with_call("rogue"), {"role": "assistant", "content": "hi"}]
    synth_msg = _msg_with_call("rogue")
    out = finalize_response(synth_msg, candidates, constraint, synth_ran=True)
    # Either repairs to text candidate (if accepted) or surfaces error.
    # Text candidate has no calls → satisfies "auto" trivially → picked.
    assert out.error is None or out.error.code == "undeclared_tool_unrepairable"


def test_finalize_response_strips_legacy_function_call_under_tool_choice_none() -> None:
    """Review r2 HIGH: a candidate emitting only legacy function_call
    must be stripped under tool_choice='none', not just modern
    tool_calls."""
    body = {
        "tools": [{"type": "function", "function": {"name": "foo"}}],
        "tool_choice": "none",
        "messages": [],
    }
    constraint = resolve_tool_constraint(body)
    msg = _msg_with_legacy_call("foo")
    out = finalize_response(msg, [msg], constraint, synth_ran=True)
    assert out.error is None
    assert out.fallback_reason == "tool_choice_none_stripped"
    assert "function_call" not in out.msg
    assert out.msg.get("tool_calls", []) == []


def test_finalize_response_normalizes_legacy_to_modern() -> None:
    """At entry, finalize_response promotes legacy function_call to
    modern tool_calls so all downstream steps see one canonical shape."""
    body = {
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "messages": [],
    }
    constraint = resolve_tool_constraint(body)
    msg = _msg_with_legacy_call("f", '{"x":1}')
    out = finalize_response(msg, [msg], constraint, synth_ran=True)
    assert out.error is None
    assert "function_call" not in out.msg
    tcs = out.msg.get("tool_calls") or []
    assert len(tcs) == 1
    assert tcs[0]["function"]["name"] == "f"
