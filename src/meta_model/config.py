"""Typed configuration model + TOML loader.

Schema mirrors `meta-model.toml.example`. Validation runs at parse
time:
- every profile's referenced upstream exists
- upstream auth fields don't conflict (api_key / api_key_env vs basic)
- multimodal MoA profiles with `vision_only_voters` + `on_no_vision_
  generator = "400"` must list at least one multimodal-capable
  generator (otherwise every image request 400s — clearly a config
  bug, not the runtime intent)

Voting profiles parse normally regardless of the [features].voting
flag; the flag controls whether they're callable. /v1/models hides
them when disabled (so a client introspecting "what can I call"
doesn't see something it can't actually use).
"""

from __future__ import annotations

import os
import tomllib
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── Sub-blocks ──────────────────────────────────────────────────────


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8400, ge=1, le=65535)
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    request_timeout_secs: int = Field(default=600, gt=0)
    bearer_token: str | None = None
    # F5: single timing knob — both the background-probe interval AND
    # the cache TTL. Cache is fresh for the entire interval between
    # ticks; stale-cache reads trigger an out-of-band probe via the
    # same coalescing primitive. Lower = faster recovery surfacing
    # but more upstream traffic; higher = quieter but slower
    # propagation of failure/recovery state. 15s is a reasonable
    # default for a fleet of <10 upstreams.
    health_probe_interval_sec: float = Field(default=15.0, gt=0, le=300)
    # F5: per-probe HTTP timeout. Must be < interval to avoid probes
    # piling up on a slow upstream. Default 5s gives the upstream
    # ample time to answer a 1-token completion without holding the
    # interval slot.
    health_probe_timeout_sec: float = Field(default=5.0, gt=0, le=60)
    # F10: server-wide brand name returned as ``response.model`` on
    # every endpoint when set. Acts as a single-model facade across
    # the underlying profiles + upstreams. When set:
    # - all endpoints emit ``model: <model_name>`` in responses
    # - /v1/models exposes it as a callable entry
    # - clients can address it as the ``model`` request parameter
    #   (resolves to the first callable non-voting profile in config
    #   order)
    # When unset (None) or empty after strip → no behavior change
    # (profile names flow through verbatim, current contract).
    model_name: str | None = None
    # F11: CORS origins permitted to call the API from a browser.
    # Default ``["*"]`` matches "drop-in OpenAI" behavior — public
    # surface, any origin allowed. Operators can lock down to a list
    # of explicit origins (e.g. ``["https://app.example.com"]``).
    #
    # Note: when the list contains ``"*"``, ``Access-Control-Allow-
    # Credentials`` cannot be true (browsers reject the combination).
    # Bearer-token auth via the Authorization header doesn't require
    # credentials mode, so the wildcard default is safe for that.
    # Cookie-based clients need an explicit origin.
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    # Server-owned system prompt prepended to every chat-completions
    # request before fan-out. Operator-defined identity / behavior
    # contract that runs alongside the caller's own system messages.
    # Empty / None → no injection (current contract; bytes-identical
    # passthrough). When set, a leading ``{role: "system", content:
    # <this>}`` is inserted at index 0 of ``messages`` before MoA /
    # cascade / voting / multimodal-cascade dispatch — every upstream
    # sees the same injected leading system. Client system messages
    # follow ours, so client-supplied task framing still applies; our
    # text supplies the identity / branding the operator wants the
    # synthesized response to reflect.
    system_prompt: str | None = None


class FeaturesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voting: bool = False


Modality = Literal["text", "image", "video", "audio"]


class UpstreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    context: int = Field(gt=0)
    max_output: int = Field(gt=0)
    supports_thinking: bool = False
    chat_template_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Some upstreams ship chat
    # templates that 400 with "System message must be at the beginning"
    # whenever the request contains a system message at any position
    # other than index 0. When this flag is true, `prepare_upstream_body`
    # demotes every non-leading system message to role="user" with a
    # "[SYSTEM]: " content prefix, preserving its position in the
    # conversation. The leading system message stays untouched.
    #
    # This is a per-upstream compatibility shim, not a global rewrite —
    # multi-system-message inputs are valid OpenAI Chat Completions
    # shape and other upstreams accept them as-is. Default false.
    requires_leading_system_only: bool = False
    # Per-upstream request body overrides applied to every forwarded
    # request. Use sparingly: this is a deliberate hook for upstream-
    # specific quirks such as disabling auxiliary reasoning output or
    # selecting a server-side parser mode. Server-owned: clients cannot
    # disable. Applied BEFORE `model` and `chat_template_kwargs` handling
    # so those dedicated fields cannot be clobbered. Do NOT use to set
    # `max_tokens` — generation length is owned by the caller.
    request_overrides: dict[str, Any] = Field(default_factory=dict)
    # Modalities the upstream accepts on input. Default is text-only;
    # vision-capable backends declare ["text", "image"]. Future video
    # / audio backends extend the list. Replaces the prior boolean
    # `multimodal` field — split lets profiles expose video / audio
    # capabilities independently of image support, and lets clients
    # planning their request shape know exactly what's accepted.
    modalities: list[Modality] = Field(default_factory=lambda: ["text"])
    supports_function_calling: bool = True
    api_key: str | None = None
    api_key_env: str | None = None
    basic_auth_user: str | None = None
    basic_auth_pass: str | None = None
    basic_auth_pass_env: str | None = None

    @property
    def multimodal(self) -> bool:
        """True if any non-text modality is supported. Convenience for
        existing callers that only need a coarse vision/text split."""
        return any(m != "text" for m in self.modalities)

    def has_modality(self, m: Modality) -> bool:
        return m in self.modalities

    @model_validator(mode="after")
    def _validate_modalities(self) -> UpstreamConfig:
        if not self.modalities:
            raise ValueError("upstream modalities must list at least 'text'")
        if "text" not in self.modalities:
            raise ValueError("upstream modalities must include 'text'")
        seen: set[str] = set()
        for m in self.modalities:
            if m in seen:
                raise ValueError(f"duplicate modality {m!r}")
            seen.add(m)
        return self

    @model_validator(mode="after")
    def _validate_auth(self) -> UpstreamConfig:
        has_api = self.api_key is not None or self.api_key_env is not None
        has_basic_user = self.basic_auth_user is not None
        has_basic_pass_any = (
            self.basic_auth_pass is not None or self.basic_auth_pass_env is not None
        )
        if has_api and (has_basic_user or has_basic_pass_any):
            raise ValueError("upstream auth cannot combine api_key/api_key_env with basic_auth_*")
        if has_basic_user != has_basic_pass_any:
            raise ValueError(
                "upstream basic_auth requires both user and a password "
                "(basic_auth_pass or basic_auth_pass_env)"
            )
        if self.basic_auth_pass is not None and self.basic_auth_pass_env is not None:
            raise ValueError(
                "upstream basic_auth password specified twice "
                "(set basic_auth_pass OR basic_auth_pass_env, not both)"
            )
        return self

    def resolved_api_key(self) -> str | None:
        """Resolve API bearer key.

        Env override wins over the literal: operators can override at
        deploy time without rewriting the TOML. Literal-precedence
        was an ops footgun found during review.
        """
        if self.api_key_env:
            val = os.environ.get(self.api_key_env)
            if val:
                return val
            if not self.api_key:
                raise RuntimeError(
                    f"upstream env var {self.api_key_env!r} is unset and no "
                    f"api_key fallback configured"
                )
        return self.api_key

    def resolved_basic_auth(self) -> tuple[str, str] | None:
        """Resolve basic-auth (user, pass), reading password from env when configured."""
        if self.basic_auth_user is None:
            return None
        if self.basic_auth_pass is not None:
            return (self.basic_auth_user, self.basic_auth_pass)
        if self.basic_auth_pass_env is not None:
            val = os.environ.get(self.basic_auth_pass_env)
            if not val:
                raise RuntimeError(f"upstream env var {self.basic_auth_pass_env!r} is unset")
            return (self.basic_auth_user, val)
        return None


# ── F6: server-level multimodal cascades ───────────────────────────
#
# Three independent top-level blocks define ranked-priority cascades
# for multimodal inputs. On a multimodal request, dispatch cascades
# through the matching list (first 2xx wins). Empty list means the
# modality is disabled and the request returns the standard "modality
# not supported" OpenAI error envelope.
#
# Independent of every profile: profiles do not opt into multimodality;
# the server does. Single-model transparency means any profile can
# accept image/video/audio content when the matching cascade is configured.


class _ModalityEndpointsConfig(BaseModel):
    """Common shape for the three modality cascades."""

    model_config = ConfigDict(extra="forbid")

    endpoints: list[str] = Field(default_factory=list)


class VisionConfig(_ModalityEndpointsConfig):
    """Image-capable upstream cascade. Empty → image input disabled."""


class VideoConfig(_ModalityEndpointsConfig):
    """Video-capable upstream cascade. Empty → video input disabled."""


class AudioConfig(_ModalityEndpointsConfig):
    """Audio-capable upstream cascade. Empty → audio input disabled."""


# ── Profile types ───────────────────────────────────────────────────


def _no_duplicate_upstreams(values: list[str]) -> list[str]:
    seen: set[str] = set()
    for v in values:
        if v in seen:
            raise ValueError(f"duplicate upstream reference {v!r}")
        seen.add(v)
    return values


def _validate_aliases(values: list[str]) -> list[str]:
    """F4-A: per-profile alias hygiene.

    Aliases must be non-empty and unique within the profile (compared
    case-insensitively, since resolution lowercases). Cross-profile
    collisions are checked at MetaModelConfig load time.
    """
    seen: set[str] = set()
    for v in values:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("alias entries must be non-empty strings")
        if v != v.strip():
            raise ValueError(f"alias {v!r} has leading/trailing whitespace")
        key = v.lower()
        if key in seen:
            raise ValueError(f"duplicate alias {v!r} within profile (case-insensitive)")
        seen.add(key)
    return values


# F1: advertised_context budget formula reserves.
# `synth_prompt_overhead_tokens` covers the synth template's fixed
# scaffold (system prompt + candidate-list framing). Default is
# conservative for a typical merge-mode synth template; profiles
# with unusually elaborate prompts can raise it.
DEFAULT_SYNTH_PROMPT_OVERHEAD_TOKENS = 256
# `non_client_synth_reserve_tokens` covers the variable
# server-added portion of the synth payload — concatenated
# candidate drafts (~N×output_reserve), tool schema, recent tool
# tail, capped authority context. Default is sized to leave most
# of a small-context window for the client; production profiles
# with N≥3 generators of 4k+ output should raise this to ~8192 or
# tune per their actual draft sizes.
DEFAULT_NON_CLIENT_SYNTH_RESERVE_TOKENS = 1024


class ProfileCapabilities(BaseModel):
    """Derived capabilities reported via ``/v1/models``.

    Profile-level fields only. The multimodal capabilities
    (vision/video/audio + their derived flags) are SERVER-LEVEL post
    F6 — every profile reports the same multimodal flags, derived
    from the `[vision]/[video]/[audio]` cascade blocks. /v1/models
    composes the profile fields here with the server-level flags
    at response time.

    `max_model_len` is the **smallest** generator/upstream context — the
    honest input ceiling. Reporting the largest or primary's context
    misleads clients planning token budgets, since shared-tail
    compaction will truncate to the smallest anyway.

    `function_calling` follows per-ensemble rules: ANY for MoA /
    Cascade, ALL for Voting (consensus needs every voter to support
    tools).

    `thinking` is True only when both an upstream supports it AND
    the profile opts to surface reasoning content
    (`expose_reasoning=true`). Default profiles ship with
    `expose_reasoning=false` so the response shape carries no
    `reasoning` / `reasoning_content` keys.

    `reasoning_visible` mirrors `expose_reasoning` directly.
    """

    max_model_len: int
    function_calling: bool
    thinking: bool = False
    reasoning_visible: bool = False


def _agg_any(upstreams: list[UpstreamConfig], pred) -> bool:
    return any(pred(u) for u in upstreams)


def _agg_all(upstreams: list[UpstreamConfig], pred) -> bool:
    return all(pred(u) for u in upstreams)


def _min_context(upstreams: list[UpstreamConfig]) -> int:
    return min(u.context for u in upstreams) if upstreams else 0


class MoaProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["moa"]
    generators: list[str] = Field(min_length=1)
    synthesizer: str = Field(min_length=1)
    synthesis_mode: Literal["merge", "best-of"] = "merge"
    generator_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    synthesizer_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    fastpath_on_agreement: bool = False
    # When true, strip `tools`, `tool_choice`, and `parallel_tool_calls`
    # from the request body before forwarding to every upstream. Used by
    # loop-recovery profiles: when a client's loop detector trips and
    # the client wants a structural guarantee that the synth output
    # contains no tool_calls — the model must produce a final user-
    # facing answer, not another tool attempt. Stripping at the meta-
    # model boundary is the structural fix; relying on system-prompt
    # instructions ("don't call tools") is what the loop-detector cap
    # is already failing on.
    strip_tools: bool = False
    # When false (default), the response sanitizer drops `reasoning`
    # and `reasoning_content` from emitted bodies and streaming
    # deltas. If the upstream returned the actual answer in
    # `reasoning_content` and `content` is empty, the sanitizer
    # rescues it into `content` BEFORE stripping. F3 in
    # plans/drop-in-fixes-2026-05-04.md.
    expose_reasoning: bool = False
    # F1: per-profile context-budget overrides. `advertised_context`
    # forces a specific `max_model_len` regardless of the derivation.
    # `synth_prompt_overhead_tokens` and `non_client_synth_reserve_tokens`
    # control the formula for `effective_ingress_budget`. See
    # plans/drop-in-fixes-2026-05-04.md F1 for derivation.
    advertised_context: int | None = Field(default=None, gt=0)
    synth_prompt_overhead_tokens: int = Field(
        default=DEFAULT_SYNTH_PROMPT_OVERHEAD_TOKENS, ge=0
    )
    non_client_synth_reserve_tokens: int = Field(
        default=DEFAULT_NON_CLIENT_SYNTH_RESERVE_TOKENS, ge=0
    )
    # F4-core: which upstream's tokenizer answers `/tokenize` requests
    # for this profile. Optional — the resolver falls back to "single
    # generator" or "all generators share a model_id" before requiring
    # this, and 409s on heterogeneous fleets without an explicit pick.
    # Must reference an upstream listed in `generators` (cross-validated
    # at config load).
    tokenizer_upstream: str | None = None
    # F4-A: model-name aliases for this profile. Lookup is
    # case-insensitive (`alias_map` lowercases at build time); the
    # /v1/models entry preserves original casing. Collisions across
    # profile names, raw upstream keys, and other profiles' aliases are
    # rejected at config load.
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_generators(self) -> MoaProfile:
        _no_duplicate_upstreams(self.generators)
        _validate_aliases(self.aliases)
        return self

    def effective_ingress_budget(self, upstreams: dict[str, UpstreamConfig]) -> int:
        """F1: client-side ingress budget for this MoA profile.

        Formula:
            synth.context
              − synth_prompt_overhead_tokens
              − non_client_synth_reserve_tokens
              − synth.max_output

        When `advertised_context` is set, returns that override
        directly (clamped to a positive integer at config load).

        Returns 0 if any upstream is unresolved; the cross-validator
        rejects that case at config load, but defensive callers
        treat 0 as "not advertise-able".
        """
        if self.advertised_context is not None:
            return self.advertised_context
        synth = upstreams.get(self.synthesizer)
        if synth is None:
            return 0
        budget = (
            synth.context
            - self.synth_prompt_overhead_tokens
            - self.non_client_synth_reserve_tokens
            - synth.max_output
        )
        return max(budget, 0)

    def capabilities(self, upstreams: dict[str, UpstreamConfig]) -> ProfileCapabilities:
        gens = [upstreams[g] for g in self.generators if g in upstreams]
        # Include the synthesizer in the context floor — if synth has a
        # smaller window than every generator, the synthesizer call will
        # 400 even though all generators succeeded. Honest budget = MIN
        # across every upstream the profile actually contacts.
        synth = upstreams.get(self.synthesizer)
        all_upstreams = list(gens)
        if synth is not None and synth not in all_upstreams:
            all_upstreams.append(synth)
        # F3: thinking advertised only when an upstream supports it
        # AND the profile opts to surface it.
        any_thinking_upstream = any(u.supports_thinking for u in all_upstreams)
        # F1: advertise the effective client-side ingress budget
        # (synth.context − overhead − reserve − output_reserve), not
        # `min(upstream.context)`. The synthesizer's window bounds the
        # achievable ingress; generator-only limits apply only to
        # generators, which receive shared-tail-compacted views.
        return ProfileCapabilities(
            max_model_len=self.effective_ingress_budget(upstreams),
            # F1: advertise the effective client-side ingress budget
            # (synth.context − overhead − reserve − output_reserve), not
            # `min(upstream.context)`.
            function_calling=_agg_any(gens, lambda u: u.supports_function_calling),
            thinking=any_thinking_upstream and self.expose_reasoning,
            reasoning_visible=self.expose_reasoning,
        )


class CascadeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["cascade"]
    upstreams: list[str] = Field(min_length=1)
    on_all_fail: Literal["bubble_last_error", "structured_502"] = "bubble_last_error"
    # F3: see MoaProfile.expose_reasoning for semantics.
    expose_reasoning: bool = False
    # F4-core: tokenizer override for `/tokenize`. See MoaProfile for
    # semantics. Optional — auto-detect picks the first cascade entry
    # if every upstream shares a model_id.
    tokenizer_upstream: str | None = None
    # F4-A: see MoaProfile.aliases.
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_upstreams(self) -> CascadeProfile:
        _no_duplicate_upstreams(self.upstreams)
        _validate_aliases(self.aliases)
        return self

    def capabilities(self, upstreams: dict[str, UpstreamConfig]) -> ProfileCapabilities:
        ups = [upstreams[u] for u in self.upstreams if u in upstreams]
        any_thinking_upstream = any(u.supports_thinking for u in ups)
        return ProfileCapabilities(
            max_model_len=_min_context(ups),
            function_calling=_agg_any(ups, lambda u: u.supports_function_calling),
            thinking=any_thinking_upstream and self.expose_reasoning,
            reasoning_visible=self.expose_reasoning,
        )


class VotingProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["voting"]
    upstreams: list[str] = Field(min_length=2)  # voting with one voter is not voting
    aggregation: Literal["any_yes"] = "any_yes"
    failure_vote: Literal["yes", "no"] = "yes"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=5, gt=0)
    # F3: see MoaProfile.expose_reasoning for semantics. Voting
    # bodies are typically YES/NO consensus tokens, but the field
    # is kept for symmetry and forward compatibility.
    expose_reasoning: bool = False
    # F4-core: tokenizer override for `/tokenize`. See MoaProfile for
    # semantics.
    tokenizer_upstream: str | None = None
    # F4-A: see MoaProfile.aliases.
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_upstreams(self) -> VotingProfile:
        _no_duplicate_upstreams(self.upstreams)
        _validate_aliases(self.aliases)
        return self

    def capabilities(self, upstreams: dict[str, UpstreamConfig]) -> ProfileCapabilities:
        ups = [upstreams[u] for u in self.upstreams if u in upstreams]
        # Voting voters all see the input; thinking advertised only
        # when ALL voters support it (mirrors the modality rule).
        all_thinking = bool(ups) and all(u.supports_thinking for u in ups)
        return ProfileCapabilities(
            max_model_len=_min_context(ups),
            # Voting needs every voter to support tools (mirrors the
            # consensus discipline: parallel YES/NO across voters).
            function_calling=_agg_all(ups, lambda u: u.supports_function_calling),
            thinking=all_thinking and self.expose_reasoning,
            reasoning_visible=self.expose_reasoning,
        )


Profile = Annotated[
    MoaProfile | CascadeProfile | VotingProfile,
    Field(discriminator="type"),
]


# ── Top-level config ────────────────────────────────────────────────


class MetaModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    upstreams: dict[str, UpstreamConfig] = Field(default_factory=dict)
    profiles: dict[str, Profile] = Field(default_factory=dict)
    # F6: server-level multimodal cascades. Independent of profiles.
    vision: VisionConfig = Field(default_factory=VisionConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)

    @model_validator(mode="after")
    def _cross_validate(self) -> MetaModelConfig:
        # 1. Every profile's referenced upstream must exist.
        for pname, prof in self.profiles.items():
            refs = (
                [*prof.generators, prof.synthesizer]
                if isinstance(prof, MoaProfile)
                else list(prof.upstreams)
            )
            for ref in refs:
                if ref not in self.upstreams:
                    raise ValueError(f"profile {pname!r} references undefined upstream {ref!r}")

        # F4-core: `tokenizer_upstream` (when set) must reference an
        # upstream this profile actually contacts; otherwise it's a
        # silent typo that breaks `/tokenize` only on the routes that
        # exercise it.
        for pname, prof in self.profiles.items():
            tup = getattr(prof, "tokenizer_upstream", None)
            if tup is None:
                continue
            referenced = (
                set(prof.generators) | {prof.synthesizer}
                if isinstance(prof, MoaProfile)
                else set(prof.upstreams)
            )
            if tup not in referenced:
                raise ValueError(
                    f"profile {pname!r} tokenizer_upstream={tup!r} is not "
                    f"listed in this profile's "
                    f"{'generators+synthesizer' if isinstance(prof, MoaProfile) else 'upstreams'}"
                )

        # F1: MoA profiles must have a non-negative ingress budget.
        # If `synth.context − overhead − reserve − max_output ≤ 0`,
        # there's no room left for client tokens. Fail at config load
        # rather than letting every request 413 in production.
        for pname, prof in self.profiles.items():
            if not isinstance(prof, MoaProfile):
                continue
            if prof.advertised_context is not None:
                continue  # operator override skips the formula check
            synth = self.upstreams.get(prof.synthesizer)
            if synth is None:
                continue  # caught above
            budget = (
                synth.context
                - prof.synth_prompt_overhead_tokens
                - prof.non_client_synth_reserve_tokens
                - synth.max_output
            )
            if budget <= 0:
                raise ValueError(
                    f"profile {pname!r}: synth context {synth.context} "
                    f"− overhead {prof.synth_prompt_overhead_tokens} "
                    f"− reserve {prof.non_client_synth_reserve_tokens} "
                    f"− output_reserve {synth.max_output} = {budget} ≤ 0; "
                    f"reduce non_client_synth_reserve_tokens / "
                    f"synth_prompt_overhead_tokens, or move to a "
                    f"larger-context synthesizer"
                )

        # F6: server-level multimodal cascades must reference defined
        # upstreams that declare the corresponding modality. Empty
        # arrays are valid (the modality is disabled — request returns
        # standard "not supported" envelope at dispatch time).
        for kind, cfg_block, required_modality in (
            ("vision", self.vision, "image"),
            ("video", self.video, "video"),
            ("audio", self.audio, "audio"),
        ):
            seen: set[str] = set()
            for idx, name in enumerate(cfg_block.endpoints):
                if name in seen:
                    raise ValueError(
                        f"[{kind}].endpoints[{idx}] = {name!r} is a duplicate"
                    )
                seen.add(name)
                up = self.upstreams.get(name)
                if up is None:
                    raise ValueError(
                        f"[{kind}].endpoints[{idx}] = {name!r} is not a defined "
                        f"upstream"
                    )
                if not up.has_modality(required_modality):
                    raise ValueError(
                        f"[{kind}].endpoints[{idx}] = {name!r} does not declare "
                        f"modality {required_modality!r} in its modalities list "
                        f"({up.modalities})"
                    )

        # F4-A: every resolvable name (profile, upstream, alias) must
        # be unique case-insensitively. F4-A's alias path resolves
        # case-insensitively, so a client-supplied `model="foo"` can
        # land in any of the three buckets — if two of them claim the
        # same lowercase key, resolution becomes order-dependent.
        # Reject at config load and force the operator to disambiguate.
        # (Review r1 MED: extend the check across profiles+upstreams+
        # aliases uniformly, not just aliases vs. the rest.)
        seen_lower: dict[str, str] = {}  # lowercase key → "<kind> <name>"

        def _register(key: str, label: str) -> None:
            existing = seen_lower.get(key)
            if existing is not None:
                raise ValueError(
                    f"{label} collides with {existing} (case-insensitive); "
                    f"rename one"
                )
            seen_lower[key] = label

        for pname in self.profiles:
            _register(pname.lower(), f"profile {pname!r}")
        for uname in self.upstreams:
            _register(uname.lower(), f"upstream {uname!r}")
        for pname, prof in self.profiles.items():
            for alias in getattr(prof, "aliases", []):
                _register(alias.lower(), f"profile {pname!r} alias {alias!r}")
        # F10: server-wide model_name brand must not collide with any
        # other resolvable name. Same case-insensitive rule as aliases.
        # Empty / whitespace-only strings strip to "no override" so the
        # operator can leave the field blank without tripping the
        # validator.
        brand = (self.server.model_name or "").strip()
        if brand:
            _register(brand.lower(), f"server.model_name {brand!r}")
            # Surface the stripped form back so runtime callers can rely
            # on the canonical (non-whitespace) value without re-stripping.
            object.__setattr__(self.server, "model_name", brand)
        elif self.server.model_name is not None:
            # Whitespace-only configured → normalize to None so callers
            # don't have to handle "" vs None separately.
            object.__setattr__(self.server, "model_name", None)
        return self

    # ── Helpers used by the server ─────────────────────────────────

    def callable_profiles(self) -> dict[str, Profile]:
        """Profiles a client can actually call given current features.

        Voting profiles are hidden when [features].voting is false:
        introspection (/v1/models) shouldn't list a profile that the
        invocation path will refuse.
        """
        if self.features.voting:
            return dict(self.profiles)
        return {
            name: prof
            for name, prof in self.profiles.items()
            if not isinstance(prof, VotingProfile)
        }

    def alias_map(self) -> dict[str, str]:
        """F4-A: lowercase alias → canonical profile name.

        Used by the resolver for O(1) case-insensitive lookup. The
        cross-validator already guarantees collision-free entries, so
        callers can index this dict directly.
        """
        out: dict[str, str] = {}
        for pname, prof in self.profiles.items():
            for alias in getattr(prof, "aliases", []):
                out[alias.lower()] = pname
        return out

    def alias_entries(self) -> list[tuple[str, str]]:
        """F4-A: ``(original_alias, canonical_profile_name)`` pairs.

        Preserves casing as configured — used by ``/v1/models`` to
        list each alias as its own entry alongside the canonical
        profile id.
        """
        out: list[tuple[str, str]] = []
        for pname, prof in self.profiles.items():
            for alias in getattr(prof, "aliases", []):
                out.append((alias, pname))
        return out


# ── Loader ──────────────────────────────────────────────────────────


CONFIG_PATH_ENV = "META_MODEL_CONFIG"
DEFAULT_CONFIG_PATH = "meta-model.toml"


def load_config(path: str | os.PathLike[str]) -> MetaModelConfig:
    """Load + validate config from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return MetaModelConfig.model_validate(data)


def load_config_from_env() -> MetaModelConfig:
    """Locate config via $META_MODEL_CONFIG or ./meta-model.toml."""
    path = os.environ.get(CONFIG_PATH_ENV) or DEFAULT_CONFIG_PATH
    return load_config(path)


def parse_config_str(toml_text: str) -> MetaModelConfig:
    """In-memory parse for tests."""
    data = tomllib.loads(toml_text)
    return MetaModelConfig.model_validate(data)
