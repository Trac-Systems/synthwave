"""F4-core — shared profile resolver."""

from __future__ import annotations

import pytest

from meta_model.config import MetaModelConfig, parse_config_str
from meta_model.routing import (
    ProfileResolutionError,
    error_response,
    resolve_profile,
)


_FIXTURE = """
[upstreams.a]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[upstreams.b]
model_id = "mb"
base_url = "http://b/v1"
context = 8192
max_output = 512

[profiles."plain.v1"]
type = "moa"
generators = ["a", "b"]
synthesizer = "a"

[profiles."voting.v1"]
type = "voting"
upstreams = ["a", "b"]
"""


def _cfg(extra: str = "") -> MetaModelConfig:
    return parse_config_str(_FIXTURE + extra)


def test_resolves_exact_profile_name() -> None:
    cfg = _cfg()
    profile, name = resolve_profile(cfg, "plain.v1")
    assert name == "plain.v1"
    assert profile is cfg.profiles["plain.v1"]


def test_resolves_raw_upstream_to_synthetic_moa() -> None:
    cfg = _cfg()
    profile, name = resolve_profile(cfg, "a")
    assert name == "a"
    # synthetic single-element MoA — generator + synthesizer == upstream
    assert profile.type == "moa"  # type: ignore[attr-defined]


def test_extension_profile_overrides_model() -> None:
    cfg = _cfg()
    profile, name = resolve_profile(cfg, model="a", ext_profile="plain.v1")
    assert name == "plain.v1"


def test_unknown_model_raises_typed_404() -> None:
    cfg = _cfg()
    with pytest.raises(ProfileResolutionError) as exc_info:
        resolve_profile(cfg, "nope")
    err = exc_info.value
    assert err.status_code == 404
    assert err.code == "model_not_found"
    assert err.param == "model"
    assert err.type_ == "invalid_request_error"


def test_voting_profile_with_feature_disabled_raises_400() -> None:
    cfg = _cfg()  # features.voting defaults to false
    with pytest.raises(ProfileResolutionError) as exc_info:
        resolve_profile(cfg, "voting.v1")
    err = exc_info.value
    assert err.status_code == 400
    assert err.code == "feature_disabled"


def test_missing_model_raises_400() -> None:
    cfg = _cfg()
    with pytest.raises(ProfileResolutionError) as exc_info:
        resolve_profile(cfg, model=None)
    assert exc_info.value.code == "missing_model"


def test_error_response_has_openai_envelope_shape() -> None:
    err = ProfileResolutionError(
        404, "model_not_found", "model 'foo' is not configured"
    )
    resp = error_response(err)
    assert resp.status_code == 404
    import json

    body = json.loads(resp.body)
    assert body["error"]["code"] == "model_not_found"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "model"
    assert "foo" in body["error"]["message"]


# ── F4-A: alias resolution ──────────────────────────────────────────


_ALIAS_FIXTURE = """
[upstreams.a]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[upstreams.b]
model_id = "mb"
base_url = "http://b/v1"
context = 8192
max_output = 512

[profiles."plain.v1"]
type = "moa"
generators = ["a", "b"]
synthesizer = "a"
aliases = ["fast", "MODEL-4"]

[profiles."voting.v1"]
type = "voting"
upstreams = ["a", "b"]
aliases = ["consensus"]
"""


def _alias_cfg() -> MetaModelConfig:
    return parse_config_str(_ALIAS_FIXTURE)


def test_alias_resolves_to_canonical_profile() -> None:
    cfg = _alias_cfg()
    profile, name = resolve_profile(cfg, "fast")
    # Canonical name is reported, not the alias the client typed.
    assert name == "plain.v1"
    assert profile is cfg.profiles["plain.v1"]


def test_alias_lookup_is_case_insensitive() -> None:
    cfg = _alias_cfg()
    # Configured as "MODEL-4"; client requests in mixed/upper/lower forms.
    for form in ("MODEL-4", "model-4", "Model-4", "MODEL-4"):
        profile, name = resolve_profile(cfg, form)
        assert name == "plain.v1"
        assert profile.type == "moa"  # type: ignore[attr-defined]


def test_exact_profile_name_wins_over_alias_lookup_order() -> None:
    cfg = _alias_cfg()
    # Exact match for "plain.v1" hits the case-sensitive profile dict
    # path BEFORE the alias map is consulted. Both end up at the same
    # profile here because aliases pointing at "plain.v1" exist; the
    # test verifies the exact path is taken (canonical name returned
    # without lowercasing).
    profile, name = resolve_profile(cfg, "plain.v1")
    assert name == "plain.v1"
    assert profile is cfg.profiles["plain.v1"]


def test_alias_via_extension_profile_field() -> None:
    cfg = _alias_cfg()
    profile, name = resolve_profile(cfg, model="a", ext_profile="fast")
    # Alias on x_meta_model.profile resolves the same way.
    assert name == "plain.v1"


def test_alias_to_voting_respects_feature_flag() -> None:
    # voting.v1 alias "consensus" must hit the same feature-disabled
    # 400 path as the canonical name when [features].voting=false
    # (default). Otherwise aliases would smuggle past the gate.
    cfg = _alias_cfg()
    with pytest.raises(ProfileResolutionError) as exc_info:
        resolve_profile(cfg, "consensus")
    err = exc_info.value
    assert err.status_code == 400
    assert err.code == "feature_disabled"
    # Canonical name is reported in the message even when an alias was
    # requested — surfaces what's actually disabled.
    assert "voting.v1" in err.message


def test_unknown_alias_raises_typed_404() -> None:
    cfg = _alias_cfg()
    with pytest.raises(ProfileResolutionError) as exc_info:
        resolve_profile(cfg, "nonexistent-alias")
    assert exc_info.value.status_code == 404
    assert exc_info.value.code == "model_not_found"


def test_alias_does_not_shadow_raw_upstream_addressing() -> None:
    # Raw upstream key resolution still works alongside aliases.
    cfg = _alias_cfg()
    profile, name = resolve_profile(cfg, "a")
    assert name == "a"
    assert profile.type == "moa"  # type: ignore[attr-defined]


def test_alias_collision_with_profile_name_rejected() -> None:
    bad = """
[upstreams.a]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."foo"]
type = "moa"
generators = ["a"]
synthesizer = "a"

[profiles."bar"]
type = "moa"
generators = ["a"]
synthesizer = "a"
aliases = ["FOO"]
"""
    with pytest.raises(ValueError, match="alias 'FOO' collides with profile 'foo'"):
        parse_config_str(bad)


def test_alias_collision_with_other_profile_alias_rejected() -> None:
    bad = """
[upstreams.a]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."foo"]
type = "moa"
generators = ["a"]
synthesizer = "a"
aliases = ["fast"]

[profiles."bar"]
type = "moa"
generators = ["a"]
synthesizer = "a"
aliases = ["Fast"]
"""
    with pytest.raises(ValueError, match="alias 'Fast' collides"):
        parse_config_str(bad)


def test_alias_collision_with_raw_upstream_rejected() -> None:
    bad = """
[upstreams.upstream_one]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."foo"]
type = "moa"
generators = ["upstream_one"]
synthesizer = "upstream_one"
aliases = ["UPSTREAM_ONE"]
"""
    with pytest.raises(
        ValueError, match="alias 'UPSTREAM_ONE' collides with upstream 'upstream_one'"
    ):
        parse_config_str(bad)


def test_profile_vs_upstream_case_insensitive_collision_rejected() -> None:
    """Review r1 MED: F4-A's case-insensitive resolution paradigm
    requires every resolvable name to be unique case-insensitively,
    not just aliases vs. the rest. profile 'foo' + upstream 'FOO'
    would otherwise leave `model='Foo'` order-dependent."""
    bad = """
[upstreams.FOO]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."foo"]
type = "moa"
generators = ["FOO"]
synthesizer = "FOO"
"""
    with pytest.raises(ValueError, match="upstream 'FOO' collides with profile 'foo'"):
        parse_config_str(bad)


def test_upstream_vs_upstream_case_insensitive_collision_rejected() -> None:
    bad = """
[upstreams.foo]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[upstreams.FOO]
model_id = "mb"
base_url = "http://b/v1"
context = 8192
max_output = 512

[profiles."plain"]
type = "moa"
generators = ["foo"]
synthesizer = "foo"
"""
    with pytest.raises(ValueError, match="upstream 'FOO' collides with upstream 'foo'"):
        parse_config_str(bad)


def test_profile_vs_profile_case_insensitive_collision_rejected() -> None:
    # Note: TOML enforces uniqueness on dict keys at the string level,
    # so two profiles with literally the same key fail at TOML parse.
    # Case-only differences pass TOML parse but must be rejected here.
    bad = """
[upstreams.a]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."Foo"]
type = "moa"
generators = ["a"]
synthesizer = "a"

[profiles."FOO"]
type = "moa"
generators = ["a"]
synthesizer = "a"
"""
    with pytest.raises(ValueError, match="profile 'FOO' collides with profile 'Foo'"):
        parse_config_str(bad)


def test_alias_duplicate_within_profile_rejected() -> None:
    bad = """
[upstreams.a]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."foo"]
type = "moa"
generators = ["a"]
synthesizer = "a"
aliases = ["bar", "BAR"]
"""
    with pytest.raises(ValueError, match="duplicate alias 'BAR' within profile"):
        parse_config_str(bad)


def test_alias_empty_string_rejected() -> None:
    bad = """
[upstreams.a]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."foo"]
type = "moa"
generators = ["a"]
synthesizer = "a"
aliases = [""]
"""
    with pytest.raises(ValueError, match="non-empty"):
        parse_config_str(bad)


def test_alias_with_whitespace_rejected() -> None:
    bad = """
[upstreams.a]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."foo"]
type = "moa"
generators = ["a"]
synthesizer = "a"
aliases = [" leading"]
"""
    with pytest.raises(ValueError, match="whitespace"):
        parse_config_str(bad)


def test_alias_default_empty_when_omitted() -> None:
    # Profiles without an aliases key should not break — backwards-
    # compatible default is empty.
    cfg = parse_config_str(_FIXTURE)  # original fixture has no aliases
    assert cfg.alias_map() == {}
    assert cfg.alias_entries() == []


def test_alias_map_lowercases_keys() -> None:
    cfg = _alias_cfg()
    m = cfg.alias_map()
    # Keys lowercased; values are canonical profile names.
    assert m == {"fast": "plain.v1", "model-4": "plain.v1", "consensus": "voting.v1"}


def test_alias_entries_preserves_original_casing() -> None:
    cfg = _alias_cfg()
    entries = sorted(cfg.alias_entries())
    # Original casing intact for surfacing back to clients.
    assert entries == [("MODEL-4", "plain.v1"), ("consensus", "voting.v1"), ("fast", "plain.v1")]
