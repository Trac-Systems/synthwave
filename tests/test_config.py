"""D.1.3 tests — TOML config loader + validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from meta_model.config import (
    CascadeProfile,
    MetaModelConfig,
    MoaProfile,
    UpstreamConfig,
    VotingProfile,
    load_config,
    parse_config_str,
)

# Repo-root example. Loadable means the example is honest documentation.
EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "meta-model.toml.example"


# ── Minimal fixture builders ────────────────────────────────────────


_MINIMAL_UPSTREAMS = """
[upstreams.text_a]
model_id = "text-a"
base_url = "http://localhost:9000/v1"
context = 8192
max_output = 2048

[upstreams.text_b]
model_id = "text-b"
base_url = "http://localhost:9001/v1"
context = 8192
max_output = 2048

[upstreams.vision_a]
model_id = "vision-a"
base_url = "http://localhost:9002/v1"
context = 8192
max_output = 2048
modalities = ["text", "image"]
"""


def _build(extras: str) -> str:
    return _MINIMAL_UPSTREAMS + extras


# ── Loader ──────────────────────────────────────────────────────────


def test_example_config_loads_successfully() -> None:
    """The committed example must parse + validate. If it doesn't,
    the example is lying about what's allowed."""
    cfg = load_config(EXAMPLE_PATH)
    assert "primary" in cfg.upstreams
    assert "write_synth.v1" in cfg.profiles


def test_load_minimal_moa_profile() -> None:
    cfg = parse_config_str(
        _build(
            """
[profiles."test.simple.v1"]
type = "moa"
generators = ["text_a", "text_b"]
synthesizer = "text_a"
"""
        )
    )
    prof = cfg.profiles["test.simple.v1"]
    assert isinstance(prof, MoaProfile)
    assert prof.generators == ["text_a", "text_b"]
    assert prof.synthesis_mode == "merge"  # default


def test_load_cascade_profile() -> None:
    cfg = parse_config_str(
        _build(
            """
[profiles."test.cascade.v1"]
type = "cascade"
upstreams = ["vision_a", "text_a"]
on_all_fail = "structured_502"
"""
        )
    )
    prof = cfg.profiles["test.cascade.v1"]
    assert isinstance(prof, CascadeProfile)
    assert prof.on_all_fail == "structured_502"


def test_load_voting_profile() -> None:
    cfg = parse_config_str(
        _build(
            """
[features]
voting = true

[profiles."test.vote.v1"]
type = "voting"
upstreams = ["text_a", "text_b"]
"""
        )
    )
    prof = cfg.profiles["test.vote.v1"]
    assert isinstance(prof, VotingProfile)
    assert prof.aggregation == "any_yes"
    assert prof.failure_vote == "yes"


# ── Cross-validation: undefined upstream references ────────────────


def test_profile_referencing_missing_upstream_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        parse_config_str(
            _build(
                """
[profiles."test.bad.v1"]
type = "moa"
generators = ["text_a", "ghost"]
synthesizer = "text_a"
"""
            )
        )
    assert "ghost" in str(exc.value)


def test_cascade_profile_missing_upstream_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        parse_config_str(
            _build(
                """
[profiles."test.cascade.v1"]
type = "cascade"
upstreams = ["vision_a", "ghost"]
"""
            )
        )
    assert "ghost" in str(exc.value)


# ── Auth validation ────────────────────────────────────────────────


def test_upstream_api_key_and_basic_auth_mutually_exclusive() -> None:
    with pytest.raises(ValidationError):
        UpstreamConfig(
            model_id="x",
            base_url="http://x",
            context=1,
            max_output=1,
            api_key="abc",
            basic_auth_user="u",
            basic_auth_pass="p",
        )


def test_upstream_basic_auth_requires_both_user_and_pass() -> None:
    with pytest.raises(ValidationError):
        UpstreamConfig(
            model_id="x",
            base_url="http://x",
            context=1,
            max_output=1,
            basic_auth_user="u",  # missing pass
        )


def test_upstream_resolved_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOO_KEY", "secret-value")
    up = UpstreamConfig(
        model_id="x",
        base_url="http://x",
        context=1,
        max_output=1,
        api_key_env="FOO_KEY",
    )
    assert up.resolved_api_key() == "secret-value"


def test_upstream_resolved_api_key_missing_env_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FOO_KEY", raising=False)
    up = UpstreamConfig(
        model_id="x",
        base_url="http://x",
        context=1,
        max_output=1,
        api_key_env="FOO_KEY",
    )
    with pytest.raises(RuntimeError, match="FOO_KEY"):
        up.resolved_api_key()


def test_upstream_resolved_api_key_env_overrides_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env wins over the literal — operators can override at deploy
    time without rewriting the TOML.
    """
    monkeypatch.setenv("FOO_KEY", "from-env")
    up = UpstreamConfig(
        model_id="x",
        base_url="http://x",
        context=1,
        max_output=1,
        api_key="from-literal",
        api_key_env="FOO_KEY",
    )
    assert up.resolved_api_key() == "from-env"


def test_upstream_resolved_api_key_falls_back_to_literal_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If env_var is configured but unset at runtime, the literal acts
    as a default — the operator can leave the env unset for local dev
    and still get the literal."""
    monkeypatch.delenv("FOO_KEY", raising=False)
    up = UpstreamConfig(
        model_id="x",
        base_url="http://x",
        context=1,
        max_output=1,
        api_key="from-literal",
        api_key_env="FOO_KEY",
    )
    assert up.resolved_api_key() == "from-literal"


def test_upstream_basic_auth_pass_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BASIC_PASS", "shh")
    up = UpstreamConfig(
        model_id="x",
        base_url="http://x",
        context=1,
        max_output=1,
        basic_auth_user="u",
        basic_auth_pass_env="BASIC_PASS",
    )
    assert up.resolved_basic_auth() == ("u", "shh")


def test_upstream_basic_auth_pass_double_specified_rejected() -> None:
    with pytest.raises(ValidationError):
        UpstreamConfig(
            model_id="x",
            base_url="http://x",
            context=1,
            max_output=1,
            basic_auth_user="u",
            basic_auth_pass="literal",
            basic_auth_pass_env="BASIC_PASS",
        )


# ── Field bounds ────────────────────────────────────────────────────


def test_negative_context_rejected() -> None:
    with pytest.raises(ValidationError):
        UpstreamConfig(model_id="x", base_url="http://x", context=-1, max_output=1)


def test_zero_max_output_rejected() -> None:
    with pytest.raises(ValidationError):
        UpstreamConfig(model_id="x", base_url="http://x", context=1, max_output=0)


def test_port_out_of_range_rejected() -> None:
    from meta_model.config import ServerConfig

    with pytest.raises(ValidationError):
        ServerConfig(port=70_000)
    with pytest.raises(ValidationError):
        ServerConfig(port=0)


def test_temperature_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_config_str(
            _build(
                """
[profiles."test.bad.v1"]
type = "moa"
generators = ["text_a", "text_b"]
synthesizer = "text_a"
generator_temperature = 3.5
"""
            )
        )


# ── Voting ≥ 2 + dedup ─────────────────────────────────────────────


def test_voting_with_one_upstream_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_config_str(
            _build(
                """
[features]
voting = true

[profiles."solo.v1"]
type = "voting"
upstreams = ["text_a"]
"""
            )
        )


def test_moa_duplicate_generators_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        parse_config_str(
            _build(
                """
[profiles."dup.v1"]
type = "moa"
generators = ["text_a", "text_a", "text_b"]
synthesizer = "text_b"
"""
            )
        )
    assert "duplicate" in str(exc.value).lower()


def test_voting_duplicate_upstreams_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_config_str(
            _build(
                """
[features]
voting = true

[profiles."dupvote.v1"]
type = "voting"
upstreams = ["text_a", "text_a"]
"""
            )
        )


def test_cascade_duplicate_upstreams_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_config_str(
            _build(
                """
[profiles."dupcas.v1"]
type = "cascade"
upstreams = ["vision_a", "vision_a"]
"""
            )
        )


def test_moa_synthesizer_can_appear_in_generators() -> None:
    """Allowed (the example uses synthesizer = primary AND primary
    also in generators)."""
    cfg = parse_config_str(
        _build(
            """
[profiles."shared.v1"]
type = "moa"
generators = ["text_a", "text_b"]
synthesizer = "text_a"
"""
        )
    )
    assert cfg.profiles["shared.v1"].synthesizer == "text_a"


# ── Voting feature gating ──────────────────────────────────────────


def _two_profiles_one_voting(voting_enabled: bool) -> MetaModelConfig:
    flag = "true" if voting_enabled else "false"
    return parse_config_str(
        _build(
            f"""
[features]
voting = {flag}

[profiles."test.text.v1"]
type = "moa"
generators = ["text_a", "text_b"]
synthesizer = "text_a"

[profiles."test.vote.v1"]
type = "voting"
upstreams = ["text_a", "text_b"]
"""
        )
    )


def test_voting_profile_hidden_when_feature_disabled() -> None:
    cfg = _two_profiles_one_voting(voting_enabled=False)
    callable_ = cfg.callable_profiles()
    assert "test.text.v1" in callable_
    assert "test.vote.v1" not in callable_
    # Still parsed; just hidden from clients.
    assert "test.vote.v1" in cfg.profiles


def test_voting_profile_visible_when_feature_enabled() -> None:
    cfg = _two_profiles_one_voting(voting_enabled=True)
    callable_ = cfg.callable_profiles()
    assert "test.vote.v1" in callable_


# ── Unknown-key rejection ──────────────────────────────────────────


def test_unknown_upstream_field_rejected() -> None:
    """extra='forbid' on each block surfaces typos, not silently
    drops them."""
    with pytest.raises(ValidationError):
        parse_config_str(
            """
[upstreams.x]
model_id = "x"
base_url = "http://x"
context = 1
max_output = 1
unknown_field = "oops"
"""
        )


def test_unknown_top_level_section_rejected() -> None:
    with pytest.raises(ValidationError):
        parse_config_str(
            _build(
                """
[mystery]
foo = "bar"
"""
            )
        )


# ── Loader env var ────────────────────────────────────────────────


def test_load_config_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = tmp_path / "test.toml"
    cfg_path.write_text(_MINIMAL_UPSTREAMS)
    monkeypatch.setenv("META_MODEL_CONFIG", str(cfg_path))
    from meta_model.config import load_config_from_env

    cfg = load_config_from_env()
    assert "text_a" in cfg.upstreams


def test_load_config_missing_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("META_MODEL_CONFIG", str(tmp_path / "nope.toml"))
    from meta_model.config import load_config_from_env

    with pytest.raises(FileNotFoundError):
        load_config_from_env()


# ── F6: server-level multimodal cascade config ─────────────────────


_F6_BASE = """
[upstreams.text_a]
model_id = "ta"
base_url = "http://up.text_a/v1"
context = 8192
max_output = 1024
modalities = ["text"]

[upstreams.vision_a]
model_id = "va"
base_url = "http://up.vision_a/v1"
context = 8192
max_output = 1024
modalities = ["text", "image"]

[profiles."txt.v1"]
type = "moa"
generators = ["text_a"]
synthesizer = "text_a"
"""


def test_f6_vision_endpoints_default_empty() -> None:
    cfg = parse_config_str(_F6_BASE)
    assert cfg.vision.endpoints == []
    assert cfg.video.endpoints == []
    assert cfg.audio.endpoints == []


def test_f6_vision_endpoints_loaded() -> None:
    cfg = parse_config_str(_F6_BASE + """
[vision]
endpoints = ["vision_a"]
""")
    assert cfg.vision.endpoints == ["vision_a"]


def test_f6_vision_endpoint_undefined_upstream_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        parse_config_str(_F6_BASE + """
[vision]
endpoints = ["ghost"]
""")
    assert "[vision].endpoints[0] = 'ghost'" in str(exc.value)
    assert "not a defined upstream" in str(exc.value)


def test_f6_vision_endpoint_text_only_upstream_rejected() -> None:
    """Vision endpoint must declare 'image' modality. text-only
    upstream → reject at config load (would 400 every image request)."""
    with pytest.raises(ValidationError) as exc:
        parse_config_str(_F6_BASE + """
[vision]
endpoints = ["text_a"]
""")
    msg = str(exc.value)
    assert "[vision].endpoints[0] = 'text_a'" in msg
    assert "modality 'image'" in msg


def test_f6_vision_endpoints_duplicate_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        parse_config_str(_F6_BASE + """
[vision]
endpoints = ["vision_a", "vision_a"]
""")
    assert "duplicate" in str(exc.value)


def test_f6_video_endpoint_requires_video_modality() -> None:
    """Symmetric check for video: upstream must declare 'video'."""
    with pytest.raises(ValidationError) as exc:
        parse_config_str(_F6_BASE + """
[video]
endpoints = ["vision_a"]
""")
    msg = str(exc.value)
    assert "[video].endpoints[0] = 'vision_a'" in msg
    assert "modality 'video'" in msg


def test_f6_audio_endpoint_requires_audio_modality() -> None:
    with pytest.raises(ValidationError) as exc:
        parse_config_str(_F6_BASE + """
[audio]
endpoints = ["vision_a"]
""")
    msg = str(exc.value)
    assert "[audio].endpoints[0] = 'vision_a'" in msg
    assert "modality 'audio'" in msg


def test_f6_video_audio_endpoints_accepted_when_modalities_match() -> None:
    cfg = parse_config_str("""
[upstreams.text_a]
model_id = "ta"
base_url = "http://up.text_a/v1"
context = 8192
max_output = 1024
modalities = ["text"]

[upstreams.vid]
model_id = "vid"
base_url = "http://up.vid/v1"
context = 8192
max_output = 1024
modalities = ["text", "video"]

[upstreams.aud]
model_id = "aud"
base_url = "http://up.aud/v1"
context = 8192
max_output = 1024
modalities = ["text", "audio"]

[profiles."txt.v1"]
type = "moa"
generators = ["text_a"]
synthesizer = "text_a"

[video]
endpoints = ["vid"]

[audio]
endpoints = ["aud"]
""")
    assert cfg.video.endpoints == ["vid"]
    assert cfg.audio.endpoints == ["aud"]


def test_f6_per_profile_multimodal_block_rejected_after_removal() -> None:
    """F6 removed `MoaProfile.multimodal`. A config that still sets
    the per-profile block should fail to load (extra="forbid"
    catches it)."""
    with pytest.raises(ValidationError) as exc:
        parse_config_str(_F6_BASE + """
[profiles."txt.v1".multimodal]
image_tool_policy = "vision_only_voters"
""")
    # Pydantic raises "Extra inputs are not permitted" with the field
    # name in the loc path.
    msg = str(exc.value).lower()
    assert "multimodal" in msg


def test_f6_profile_capabilities_no_longer_carries_multimodal_fields() -> None:
    """ProfileCapabilities dropped vision/video/audio/effective_image_*
    /supports_image_tools — they're server-level now (composed at
    /v1/models response time)."""
    cfg = parse_config_str(_F6_BASE)
    caps = cfg.profiles["txt.v1"].capabilities(cfg.upstreams)
    # Per-profile fields remain.
    assert hasattr(caps, "max_model_len")
    assert hasattr(caps, "function_calling")
    assert hasattr(caps, "thinking")
    assert hasattr(caps, "reasoning_visible")
    # Multimodal fields removed.
    assert not hasattr(caps, "vision")
    assert not hasattr(caps, "video")
    assert not hasattr(caps, "audio")
    assert not hasattr(caps, "effective_image_capability")
    assert not hasattr(caps, "supports_image_tools")


