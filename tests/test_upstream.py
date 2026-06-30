"""Unit tests for `meta_model.upstream`.

Most of `prepare_upstream_body` is covered indirectly via dispatch and
streaming tests. The new `requires_leading_system_only` branch is
non-trivial enough to warrant direct coverage: the demotion has to
preserve message position, handle string and multimodal content, and
leave the leading system message untouched.
"""

from __future__ import annotations

from meta_model.config import AnthropicOptions, OpenAIOptions, UpstreamConfig
from meta_model.upstream import _demote_non_leading_system_messages, prepare_upstream_body


def _up(*, requires_leading_system_only: bool = False) -> UpstreamConfig:
    return UpstreamConfig(
        model_id="x",
        base_url="http://x",
        context=4096,
        max_output=128,
        requires_leading_system_only=requires_leading_system_only,
    )


def test_demote_leaves_leading_system_alone() -> None:
    msgs = [
        {"role": "system", "content": "you are client application"},
        {"role": "user", "content": "hi"},
    ]
    out = _demote_non_leading_system_messages(msgs)
    assert out[0] == {"role": "system", "content": "you are client application"}
    assert out[1] == {"role": "user", "content": "hi"}


def test_demote_demotes_mid_stream_system() -> None:
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "ok"},
        {"role": "system", "content": "STOP — fix this first"},
        {"role": "user", "content": "next"},
    ]
    out = _demote_non_leading_system_messages(msgs)
    # Leading system unchanged, position preserved for the demotion.
    assert out[0]["role"] == "system"
    assert out[3]["role"] == "user"
    assert out[3]["content"] == "[SYSTEM]: STOP — fix this first"
    # Non-system messages flow through unchanged.
    assert out[1]["role"] == "user"
    assert out[2]["role"] == "assistant"
    assert out[4]["role"] == "user"


def test_demote_handles_no_leading_system() -> None:
    """If the conversation starts with a non-system role, every system
    message in it is non-leading — all of them get demoted."""
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "rule"},
    ]
    out = _demote_non_leading_system_messages(msgs)
    assert out[0]["role"] == "user"
    assert out[1]["role"] == "user"
    assert out[1]["content"] == "[SYSTEM]: rule"


def test_demote_handles_multimodal_content_with_text_part() -> None:
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hi"},
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        },
    ]
    out = _demote_non_leading_system_messages(msgs)
    assert out[2]["role"] == "user"
    assert out[2]["content"][0] == {"type": "text", "text": "[SYSTEM]: look at this"}
    assert out[2]["content"][1]["type"] == "image_url"


def test_demote_handles_multimodal_content_no_text_part() -> None:
    msgs = [
        {"role": "system", "content": "persona"},
        {
            "role": "system",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        },
    ]
    out = _demote_non_leading_system_messages(msgs)
    assert out[1]["role"] == "user"
    # Synthetic text part inserted at index 0.
    assert out[1]["content"][0] == {"type": "text", "text": "[SYSTEM]:"}
    assert out[1]["content"][1]["type"] == "image_url"


def test_demote_handles_empty_or_missing_content() -> None:
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "system"},  # malformed but not our place to reject
    ]
    out = _demote_non_leading_system_messages(msgs)
    assert out[1]["role"] == "user"
    assert out[1]["content"] == "[SYSTEM]:"


def test_prepare_upstream_body_demotes_when_flag_true() -> None:
    body = {
        "model": "client_addressed",
        "messages": [
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "nudge"},
        ],
    }
    out = prepare_upstream_body(body, _up(requires_leading_system_only=True))
    assert out["model"] == "x"
    assert out["messages"][0]["role"] == "system"
    assert out["messages"][2]["role"] == "user"
    assert out["messages"][2]["content"] == "[SYSTEM]: nudge"


def test_prepare_upstream_body_no_demotion_when_flag_false() -> None:
    body = {
        "model": "client_addressed",
        "messages": [
            {"role": "system", "content": "persona"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "nudge"},
        ],
    }
    out = prepare_upstream_body(body, _up(requires_leading_system_only=False))
    # Default upstream leaves the messages list untouched.
    assert out["messages"] == body["messages"]


def test_prepare_upstream_body_demotion_handles_no_messages() -> None:
    """No `messages` key (malformed but possible — e.g., a config call)
    must not crash the demotion step."""
    body = {"model": "x"}
    out = prepare_upstream_body(body, _up(requires_leading_system_only=True))
    assert "messages" not in out


def test_prepare_upstream_body_does_not_mutate_input() -> None:
    """Caller's body is sometimes used for retries — must stay clean."""
    body = {
        "model": "x",
        "messages": [
            {"role": "system", "content": "persona"},
            {"role": "system", "content": "nudge"},
        ],
    }
    snapshot = [dict(m) for m in body["messages"]]
    prepare_upstream_body(body, _up(requires_leading_system_only=True))
    assert body["messages"] == snapshot


# ── OpenAI reasoning-model normalization (opt-in [openai] block) ─────


def _oai_up(openai: OpenAIOptions) -> UpstreamConfig:
    return UpstreamConfig(
        model_id="gpt-5.5",
        base_url="https://api.openai.com/v1",
        context=400000,
        max_output=16000,
        openai=openai,
    )


def test_openai_reasoning_effort_injected() -> None:
    body = {"model": "client", "messages": [], "max_tokens": 100}
    out = prepare_upstream_body(body, _oai_up(OpenAIOptions(reasoning_effort="xhigh")))
    assert out["reasoning_effort"] == "xhigh"
    assert out["model"] == "gpt-5.5"


def test_openai_renames_max_tokens_to_max_completion_tokens() -> None:
    body = {"model": "client", "messages": [], "max_tokens": 256}
    out = prepare_upstream_body(
        body, _oai_up(OpenAIOptions(max_tokens_param="max_completion_tokens"))
    )
    assert "max_tokens" not in out
    assert out["max_completion_tokens"] == 256


def test_openai_drops_unsupported_sampling_params() -> None:
    body = {"model": "client", "messages": [], "temperature": 0.6, "top_p": 0.9}
    out = prepare_upstream_body(body, _oai_up(OpenAIOptions(drop_params=["temperature", "top_p"])))
    assert "temperature" not in out
    assert "top_p" not in out


def test_openai_block_absent_is_byte_identical_passthrough() -> None:
    """No [openai] block → behaves exactly like a vanilla upstream."""
    body = {"model": "client", "messages": [], "max_tokens": 100, "temperature": 0.5}
    out = prepare_upstream_body(body, _oai_up_none())
    assert out["max_tokens"] == 100
    assert out["temperature"] == 0.5
    assert "reasoning_effort" not in out


def _oai_up_none() -> UpstreamConfig:
    return UpstreamConfig(
        model_id="vllm-model",
        base_url="http://x",
        context=4096,
        max_output=128,
    )


def test_anthropic_protocol_skips_vllm_shims() -> None:
    """Anthropic-protocol upstreams keep their messages untouched here
    (the adapter owns system handling) and never get reasoning renames."""
    up = UpstreamConfig(
        model_id="claude-opus-4-8",
        base_url="https://api.anthropic.com/v1",
        context=200000,
        max_output=8192,
        protocol="anthropic",
        anthropic=AnthropicOptions(thinking="adaptive", effort="xhigh"),
        requires_leading_system_only=True,  # must be ignored on this path
    )
    body = {
        "model": "client",
        "max_tokens": 100,
        "messages": [
            {"role": "system", "content": "a"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "b"},
        ],
    }
    out = prepare_upstream_body(body, up)
    assert out["model"] == "claude-opus-4-8"
    # No demotion applied — the second system message is still a system role.
    assert out["messages"][2]["role"] == "system"
    assert out["max_tokens"] == 100  # untouched (adapter handles the rename)
