# Synthwave: Improving Prompt and Code Quality with Mixture-of-Agents

**Author:** Markus Bopp, Trac Systems.

**Abstract.** We present **Synthwave**, an OpenAI-compatible HTTP service that productionizes Mixture-of-Agents (MoA) orchestration as a drop-in replacement endpoint. Synthwave's first application — *Code MoA* — extends MoA from natural-language aggregation to program synthesis: each code-generation request is fanned out to *N* diverse LLMs in parallel, each producing a complete independent implementation from the same task specification, and a designated synthesizer model composes a final version that cherry-picks the best architecture, error handling, and algorithmic choices from each. We demonstrate empirically that the synthesized output is consistently equal to or better than any individual model's output, with no additional training. We then report on generalizing the same primitive across six production profiles — `tool_chat`, `write_synth_content`, `write_synth_planning`, `text_synth`, `cap_wrap`, and `recovery_synthesis` — each with its own generator/synthesizer pairing, prompt template, and policy (tool stripping, deterministic tool-call merging, vision-modality cascade). The system is configured entirely through TOML, exposes per-profile observability via a ringbuffer endpoint, and is currently serving live traffic from an autonomous coding agent and a public web product. We share findings from migration from an in-process Rust implementation to a centralized Python service, including a structurally-anchored anti-anchoring guarantee (the *B-schema*), a reasoning-content rescue path for harmony-format models with empty visible content, an inference-decoupled readiness probe, and operator-defined identity injection. The paper is both a research note and a system paper: the empirical validation of multi-model code synthesis remains intact, and the operational reality of serving it as a shared service surfaces design choices that the original in-agent prototype did not.

---

## 1. Introduction

Large language models have demonstrated remarkable ability in code generation (Chen et al., 2021; Li et al., 2022; Guo et al., 2024), yet individual models exhibit systematic blind spots: one model may write clean architecture but miss edge cases, another may handle errors robustly but produce verbose code, and a third may optimize for correctness at the expense of readability. These failure modes are model-specific and often uncorrelated — the errors one model makes are not the errors another makes.

This observation, well-established in classical ensemble learning (Breiman, 1996; Wolpert, 1992; Freund & Schapire, 1997), has been surprisingly underexplored in the context of LLM-based code generation. Prior work has pursued three related but distinct strategies:

1. **Single-model sampling** (AlphaCode, CodeT, self-consistency): Generate many candidates from one model, filter by test execution or majority voting (Li et al., 2022; Chen et al., 2022; Wang et al., 2023). This increases coverage within one model's distribution but cannot escape that distribution's systematic biases.

2. **Multi-agent role decomposition** (Meta Programming framework, ChatDev, AgentCoder): Assign different *roles* (architect, coder, reviewer) to different agents, typically backed by the same model (Hong et al., 2023; Qian et al., 2023; Huang et al., 2023). This improves process but not the underlying code generation diversity.

3. **Text-domain MoA** (Together AI): Query multiple LLMs in parallel and aggregate responses for natural language tasks (Wang et al., 2024). Demonstrated strong results on AlpacaEval but not applied to structured code generation.

None of these approaches combine **diverse models × same task × independent generation × intelligent synthesis** for code. *Code MoA* — the technique introduced in the original version of this paper — fills that gap. *Synthwave* — the system described here — productionizes Code MoA and generalizes the same primitive to other surfaces (tool-call merging, conversational chat, drive-loop recovery, write-synthesis with audit), behind a single OpenAI-compatible API.

### 1.1 Key Insight

Different LLMs, trained on different data mixtures with different architectures and optimization objectives, develop different "intuitions" about code. When given the same specification:

- A model optimized for instruction following (primary coder) may produce clean, minimal implementations with strong tool integration.
- A general-purpose model with reasoning training (reasoning generator) may add defensive programming patterns — DNS validation, flexible input parsing, comprehensive error handling — reflecting its broader training on production code.
- A fast inference model (fast generator) may prioritize feature completeness — file I/O options, configurable parameters, usage examples — reflecting its training on user-facing applications.

These are not random variations. They reflect genuine architectural *perspectives* that a synthesizer model can evaluate and compose into a superior implementation. The same observation generalizes beyond code: in tool-call settings the same diversity manifests as differences in argument naming, error-recovery strategy, and tool-choice heuristics, all of which the synthesizer can compare and merge.

### 1.2 Contributions

1. **Code MoA**: A method for multi-model code synthesis where *N* diverse LLMs independently generate implementations from the same specification, and a synthesizer model produces a final version combining the best elements.
2. **The B-schema**: A structural anti-anchoring guarantee. The agent's `write` tool emits only `{path}` (the *intent* to write to a location) — never a draft body. Generators see the same task specification and cannot anchor on each other's output, because no candidate "first draft" exists at request time.
3. **Synthwave**: An OpenAI-compatible HTTP service that hosts the MoA dispatch path. Multiple profiles share the same fan-out + synthesize primitive but encode different policies (tool stripping, deterministic tool-call merging, multimodal cascade routing, max-token clamping per upstream).
4. **Reasoning-content rescue**: A robustness fix for harmony-format models (e.g., reasoning generator) which can return responses with empty visible content but populated reasoning. Without rescue, such responses appear to the synthesizer as 8-byte tombstones and degrade the candidate pool. The rescue lifts visible-empty/reasoning-populated candidates back into the merge.
5. **Operator-defined identity injection**: A server-level `system_prompt` config field that prepends an operator's branding/identity contract to every dispatch, ahead of any cascade or MoA branch. Used in production to surface Synthwave's identity ("ensemble of multiple models") consistently across profiles.
6. **Decoupled readiness probe**: A `/v1/health` implementation that probes upstreams via `GET /v1/models` (catalog presence check) rather than `POST /v1/chat/completions max_tokens=1`. The latter queues behind real inference and false-flags loaded upstreams as `down`; the former is structurally inference-decoupled.
7. **Per-profile observability**: A ringbuffer-backed `/v1/metrics/moa` endpoint that records, per call: quorum, fastpath/fallback decision, draft length per generator slot, final tool-call count, final content chars, and (since the most recent observability pass) wall-clock elapsed_ms. Operators see exactly which generator contributed what, without instrumenting consumers.
8. **Empirical evaluation**: Side-by-side comparison of individual model outputs and synthesized results across multiple coding tasks under the original prototype, plus production telemetry from the Synthwave deployment under live agent and web traffic.
9. **Open-source implementation**: The Synthwave service (Python FastAPI, ~10K LOC, 641 unit tests at the time of writing) and a thin Rust client (`MetaModelClient` in `agent-llm`) replacing the deleted in-process MoA code (`agent-core::moa`). All available for reproduction and extension.

---

## 2. Related Work

### 2.1 Ensemble Methods in Machine Learning

Ensemble learning is among the most reliable techniques for improving prediction quality. Bagging (Breiman, 1996) reduces variance by training on bootstrap samples. Boosting (Freund & Schapire, 1997) reduces bias by sequentially re-weighting examples. Stacking (Wolpert, 1992) trains a meta-learner on base model outputs. Random Forests (Breiman, 2001) combine bagging with feature randomization for decorrelated ensembles.

Code MoA is most analogous to **stacking**: base learners (generator LLMs) produce candidate outputs, and a meta-learner (synthesizer LLM) combines them. However, unlike classical stacking where the meta-learner sees numeric predictions, the synthesizer sees full program text and can reason about structural relationships between implementations.

### 2.2 Multi-Sample Code Generation

**AlphaCode** (Li et al., 2022) generates up to 1 million candidate programs from a single model and filters them via test execution and semantic clustering, achieving median-competitor performance on Codeforces. **AlphaCode 2** (large-model follow-up work, 2023) extends this with frontier-scale models to the 85th percentile. While extremely effective for competitive programming, this approach requires massive computational budgets (millions of samples) and relies on having test cases for filtering — unavailable for most real-world coding tasks.

**CodeT** (Chen et al., 2022) generates both code solutions and test cases, using dual execution agreement for ranking. This is elegant but still single-model: all candidates share the same model's biases and blind spots.

**Self-consistency** (Wang et al., 2023) samples multiple reasoning paths and selects by majority vote. Applied to code, this becomes pass@k with majority voting on outputs. Again, single-model — diverse reasoning paths but not diverse architectural perspectives.

**Best-of-N sampling** with learned verifiers (Cobbe et al., 2021; Lightman et al., 2023; Snell et al., 2024) uses reward models to select the highest-quality sample from N candidates. This has proven highly effective for math and reasoning but requires training a verifier, which is expensive and domain-specific.

The key limitation shared by all single-model approaches: **sampling more from one distribution cannot compensate for systematic gaps in that distribution.** If a model never learned flexible port range parsing (e.g., `"1-1024,8080,443"`), no amount of sampling will produce it.

### 2.3 Multi-Agent Software Development

**Meta Programming framework** (Hong et al., 2023) encodes SOPs (Standardized Operating Procedures) into multi-agent workflows with roles like product manager, architect, and engineer. **ChatDev** (Qian et al., 2023) creates a virtual software company with communicating LLM agents. **AgentCoder** (Huang et al., 2023) separates code generation, test generation, and test execution into specialized agents.

These approaches improve the *process* of software development but typically use the same underlying model for all roles. The diversity comes from prompt engineering (role descriptions) rather than model diversity. Synthwave is orthogonal — Code MoA can be applied at the code-generation step within any agent framework, including the multi-agent ones above. In our deployment, the agent itself uses GSD-style staged execution (plan → build → verify) and dispatches every code-write through Synthwave's `write_synth_content.v1` profile.

### 2.4 Mixture of Agents for Text

**MoA** (Wang et al., 2024) demonstrates that querying multiple LLMs and aggregating their responses via a meta-model achieves state-of-the-art quality on AlpacaEval 2.0, outperforming a leading frontier model. The paper introduces the concept of "collaborativeness" — the property that LLMs can improve their responses when given other models' outputs as context.

Code MoA — and Synthwave's broader profile family — adapts this paradigm with two critical modifications:

1. **No cross-pollination during generation.** Unlike text MoA where models can see other agents' responses, generators in every Synthwave profile work independently. This prevents anchoring bias — when shown a "first draft," models tend to make minor edits rather than explore fundamentally different architectures.
2. **Structural synthesis, not textual blending.** The synthesizer must understand the underlying structure (code semantics for `write_synth_*`, tool-call argument schemas for `tool_chat`, message-channel discipline for harmony models) — not merely merge paragraphs.

### 2.5 LLM Ensembles

**LLM-Blender** (Jiang et al., 2023) proposes a two-stage ensemble: PairRanker for ranking candidates from diverse LLMs, followed by GenFuser for fusing top candidates. This is the closest prior work to Code MoA in spirit, but it was evaluated on text generation tasks, not code. Code MoA can be seen as a domain-specific adaptation of GenFuser's fusion concept, with a code-aware synthesis prompt replacing the generic fusion mechanism.

**"More Agents Is All You Need"** (Li et al., 2024) shows that simply scaling the number of LLM agents and using majority voting improves performance on reasoning and code tasks, with a scaling law analogous to compute scaling. Synthwave differs in using synthesis rather than voting — we want the best *output*, not the most common answer — and in routing to *different* models rather than scaling samples from one.

### 2.6 Program Synthesis

Classical program synthesis (Gulwani et al., 2017; Gulwani, 2011) constructs programs from formal specifications using constraint solving, enumerative search, or deductive methods. Genetic programming (Koza, 1992) evolves programs through mutation and crossover. Neural program synthesis (Abolafia et al., 2018; Chen et al., 2019) trains neural networks to generate programs from examples.

Code MoA occupies a unique position: it leverages pretrained LLMs (no task-specific training) in a multi-model ensemble (like GP populations) with an intelligent selection/recombination step (like stacking). The synthesizer performs a semantic analog of crossover — combining structural elements from multiple parents — but guided by natural language understanding rather than syntactic operators.

### 2.7 Mixture of Experts

Sparse Mixture of Experts models (Fedus et al., 2022; Jiang et al., 2024; Lepikhin et al., 2021) route tokens to specialized sub-networks within a single model. While superficially similar in name, MoE operates at the token level within one forward pass, while Code MoA operates at the program level across independent models. The key analogy is that both exploit specialization: MoE trains experts to handle different token distributions, Code MoA leverages models that naturally handle different code patterns.

---

## 3. Method

Synthwave is a single FastAPI service that exposes the OpenAI Chat Completions API and dispatches each request through one of three paths: a **single-upstream passthrough** (for non-MoA-eligible models), a **vision cascade** (for multimodal requests, routed to a designated upstream), or a **MoA fan-out + synthesis** (for profiles that declare two or more generators). The MoA path is the subject of this paper. Code MoA is a particular instantiation of that path — the `write_synth_content.v1` profile — but the same dispatch mechanism powers five other profiles in production.

### 3.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Request (OpenAI shape) → server.dispatch()                      │
│   1. Identity-prompt prepend (operator-defined system prompt)   │
│   2. Tool-call normalization (legacy `functions` → `tools`)     │
│   3. Multimodal short-circuit (image/video → vision cascade)    │
│   4. Resolve `model` → (alias map) → profile or upstream        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  passthrough     vision cascade    _dispatch_moa
  (1 upstream)    (1 upstream,      ┌────────────────────────────┐
                  policy-bound)     │ a) fanout: N parallel       │
                                    │    upstream POSTs           │
                                    │ b) shared-tail compaction   │
                                    │    if budget exceeded       │
                                    │ c) synthesizer.synthesize   │
                                    │    (mode=merge|best-of)     │
                                    │ d) reasoning-content rescue │
                                    │ e) tool-call deterministic  │
                                    │    merge (when applicable)  │
                                    │ f) record_moa_call → ring   │
                                    └────────────────────────────┘
                                                │
                                                ▼
                                  OpenAI-shaped response
                                  + X-MetaModel-* headers
```

The MoA path is implemented in `meta_model/moa/dispatch.py::_dispatch_moa()`, with concerns separated into modules: `fanout.py` for parallel HTTP, `synthesizer.py` for merge/best-of synthesis, `compaction.py` for shared-tail context compaction, `tools.py` for tool-call normalization and merging, `multimodal.py` for vision cascade policy, and `metrics.py` for the per-profile ringbuffer.

### 3.2 Profiles

A profile is the operator-facing unit of policy. Each profile names its generators (an ordered list of upstream names), its synthesizer (one upstream name), its synthesis mode (`merge` or `best-of`), and optional flags (`strip_tools`, `synthesizer_temperature`). Profiles are listed in `/v1/models` and selected either by setting `model: "<profile>"` in the request body or via the `x_meta_model.profile` extension header.

```toml
[profiles."write_synth_content.v1"]
type = "moa"
generators = ["primary", "reasoning", "fast"]
synthesizer = "primary"
synthesis_mode = "merge"
synthesizer_temperature = 0.2

[profiles."recovery_synthesis.v1"]
type = "moa"
generators = ["primary", "reasoning", "fast"]
synthesizer = "primary"
synthesis_mode = "merge"
strip_tools = true     # output must be terminal — no further tool calls
```

In production, six profiles share the MoA path:

| Profile | Purpose | Strip tools | Notes |
|---------|---------|-------------|-------|
| `write_synth_content.v1` | Code/file body synthesis (Code MoA proper) | no | Caller emits `{path}` (B-schema); synth fills body |
| `write_synth_planning.v1` | Planning-doc synthesis (PROJECT.md / ROADMAP.md / PLAN.md) | no | Same MoA, different prompt template |
| `tool_chat.v1` | Conversational + tool-calling agent turn | no | Tool-call deterministic merge layered on text merge |
| `text_synth.v1` | Plain-text conversational synthesis | no | Mostly for benchmarking and smoke tests |
| `cap_wrap.v1` | Tool-call-cap wrap-up message | no | Single closing turn after agent budget exhausted |
| `recovery_synthesis.v1` | Stuck-loop recovery — extract a final answer from history | yes | Strips tools so output is text-only |

This is what we mean by *generalization*: the same MoA primitive is reused across surfaces; only the prompt template, generator/synthesizer pairing, and tool policy change. Profiles can be added without touching dispatch code.

### 3.3 Anti-Anchoring (and the B-Schema)

A critical design property: generator models receive **only the task specification and any directly relevant context**, not any other generator's draft and never a "primary" first draft. This is essential because:

**Anchoring bias in LLMs is severe.** When shown existing code with instructions to "write your own version" or "improve this," models overwhelmingly make incremental edits rather than exploring fundamentally different approaches. In the original prototype, showing 24B code-completion model the primary model's output produced a near-copy (31 lines vs 72 lines, with only whitespace differences). Without anchoring, the same model produced genuinely different — if inferior — code.

In the original prototype, anti-anchoring was a *prompt discipline*: the agent's MoA module took care to remove the primary's draft before fan-out. This is fragile — any future change to the prompt assembly could re-introduce an anchor.

In Synthwave, anti-anchoring is *structural*. The autonomous agent's `write` tool emits only the **path** of the file to write — never a draft body. We call this the **B-schema**: the tool's argument is a single `{path}` field. The body is synthesized server-side by Synthwave's `write_synth_content.v1` profile from generators that all receive the same task specification (the surrounding agent context, the file path, the project plan), with no draft to anchor on. There is no place in the request shape where an "original" can leak into the generator prompt, because no original exists yet.

The B-schema also simplifies the agent: it no longer has to choose between writing code itself or delegating to MoA — the choice is structural. Every write is delegated, automatically.

### 3.4 Generator Prompt

Each generator model receives a minimal, task-focused prompt assembled by the profile's prompt template. For `write_synth_content.v1`:

```
System: You are writing the contents of a file. Output ONLY the file body —
        no explanations, no markdown fences, no commentary. Just the raw
        file content ready to be written to disk.

User:   Task: {task_context}
        File: {file_path}
        Write the complete file. Output ONLY the body.
```

Other profiles use shapes appropriate to their surface — `tool_chat.v1` preserves the original conversation messages and tool definitions, `cap_wrap.v1` injects a one-shot wrap-up prompt, etc. In all cases, generators see the task and only the task.

The prompts are deliberately terse to maximize the proportion of context available for the actual generation. No personality, no irrelevant conversation history, no tool definitions when not needed — just the task.

### 3.5 Synthesis Prompt

The synthesizer model receives all successful candidate implementations labeled by source. There is no "original" or "primary" candidate slot — the synthesizer evaluates candidates as peers:

```
System: You are synthesizing the best output from multiple implementations.
        Review each version and write ONE final version that combines:
        - The best architecture and structure
        - The most robust error handling
        - The cleanest, most readable code
        - The most efficient approach
        If one version is clearly superior, use it with minor improvements.
        Do NOT average or merge blindly — pick the best approach and refine it.
        Output ONLY the final code — no explanations, no markdown fences.

User:   File: {file_path}
        Task: {task_context}

        Implementations to review:

        --- candidate_a ---
        {generator_a_output}

        --- candidate_b ---
        {generator_b_output}

        --- candidate_c ---
        {generator_c_output}

        Write the final, best version. ONLY code, nothing else.
```

Key synthesis instructions:
- **"Pick the best approach and refine it"** — prevents Frankenstein code from averaging incompatible architectures.
- **"If one version is clearly superior, use it with minor improvements"** — allows the synthesizer to recognize when one model clearly won, rather than forcing artificial merging.
- **Low temperature (0.2 default, configurable per profile)** — reduces creative variance in the synthesis step; we want careful evaluation, not novel generation.

For `tool_chat.v1`, the synthesizer is given the candidate tool calls in normalized form and a deterministic merging procedure runs first (Section 3.8); only when candidates differ on text does the LLM-side synthesis step engage.

### 3.6 Configuration

The system is configured entirely through `meta-model.toml` (renamed to `synthwave.toml` post-rebrand). The shape:

```toml
[server]
host = "0.0.0.0"
port = 8400
api_key_env = "SYNTHWAVE_API_KEY" # optional but recommended for non-loopback
system_prompt = """You are Synthwave-1, ..."""   # operator-defined identity (Section 3.10)

[upstreams.primary]
base_url = "http://local-primary:8000/v1"
model_id = "primary-code-synthesizer"
api_key_env = "PRIMARY_API_KEY"
context = 262144
max_output = 16384

[upstreams.reasoning]
base_url = "http://127.0.0.1:8000/v1"
model_id = "reasoning-generator-20b"
context = 131072
max_output = 8192
request_overrides = { reasoning_effort = "low" }

[upstreams.fast]
base_url = "http://local-fast:8000/v1"
model_id = "fast-code-generator"
context = 32768
max_output = 8192

[profiles."write_synth_content.v1"]
type = "moa"
generators = ["primary", "reasoning", "fast"]
synthesizer = "primary"
synthesis_mode = "merge"

# … additional profiles …

[aliases]
"Synthwave-1" = "tool_chat.v1"
```

Key properties:

- **Decoupled categories**: profile-level names (`primary`, `reasoning`, `fast`) are abstract handles mapped to concrete upstreams. A user can swap models, providers, even hardware without touching profile definitions.
- **Per-upstream overrides**: `request_overrides` (e.g., `reasoning_effort`, `max_tokens`) are merged into the request body before forwarding. Used here to set reasoning generator to low reasoning effort.
- **Per-upstream context budgets**: every upstream declares its context window and max output; synthesizer caps and shared-tail compaction respect these as hard limits.
- **Aliases**: friendly model names (e.g., `Synthwave-1`) map to profiles, so OpenAI clients can call the service by a single identity.

### 3.7 Failure Handling

Synthwave's MoA path is designed to degrade gracefully:

| Failure mode | Behavior |
|-------------|----------|
| All generators time out | `503 no_quorum` returned to caller |
| Some generators fail | Synthesis proceeds with available candidates; degraded headers emitted |
| Single generator survives | Fast-path: that candidate is returned without a synth call |
| Synthesis fails (5xx, timeout, non-JSON, etc.) | Fallback to first candidate (`SYNTH_DECISION_FALLBACK_PRIMARY`) |
| Synthesizer max_output exceeded | Capped to upstream's `max_output` declaration |
| Caller-supplied `max_tokens > upstream.max_output` | Clamped so synth cannot exceed advertised reserve |
| Inbound request larger than budget | Shared-tail compaction; if still over, `413` with structured headers |
| Multimodal request | Routed to vision cascade, never enters MoA fan-out |

The invariant: **the MoA path never produces *worse* output than its best individual candidate.** If synthesis fails, the highest-quality candidate is returned unchanged with the degradation reason in `X-MetaModel-Fallback-Reason`.

### 3.8 Tool-Call Deterministic Merge

`tool_chat.v1` carries a constraint not present in pure-text profiles: when the agent expects a tool call, the response's `tool_calls` field must be a single coherent list, not arbitrary natural-language paraphrase of multiple models' attempts. We solve this in two stages:

1. **Deterministic merge**: For each tool name across candidates, normalize the argument JSON, hash the canonical form, and bucket-by-hash. If all candidates agree on the same canonical args, we **fast-path** — return one candidate's tool call as the final response without a synth LLM call. If candidates disagree, the synthesizer is invoked with all candidate tool calls in its prompt.
2. **Tool-aware synthesis**: When the LLM-side synth step runs on a tool-call disagreement, the synth body keeps `tools` / `tool_choice` / `parallel_tool_calls` so the synth model can re-emit a constraint-satisfying tool_call rather than text.

The fast-path captures a substantial fraction of `tool_chat.v1` traffic: in our chess production session (Section 5.6), every one of 35 `tool_chat.v1` calls had quorum 3/3, of which a meaningful subset converged on identical tool calls and went through the deterministic merge with no synthesizer involvement.

### 3.9 Reasoning-Content Rescue (F12)

Some open-weight models — notably the 20B reasoning generator family — emit responses in OpenAI's **harmony format**, which structures output into `analysis` (chain-of-thought), `commentary` (intermediate channel), and `final` (visible) channels. Under specific conditions (token budget exhausted while in the analysis channel; reasoning-effort `low` paired with short `max_tokens`; certain prompt shapes), a response can land with the visible content empty (`""`), or 8 bytes of harmony scaffolding only, while the `reasoning_content` field carries real text. We label this an **8-byte tombstone**.

Without intervention, a tombstone propagates as `(empty)` to the synthesizer's merge prompt, dropping that candidate from the comparison pool and inflating the perceived contribution of the other generators. Worse, draft-length metrics show `min=8 max=8` for the affected generator slot, masquerading as silent failure.

**The rescue path** (`synthesizer.py::_render_candidate` widened to use `_is_visible_content_missing`):

```python
def _render_candidate(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if _is_visible_content_missing(msg):
        fallback = _reasoning_fallback_text(msg)
        if fallback:
            return fallback
    elif isinstance(content, str):
        return content
    # … list-of-parts handling …
```

`_is_visible_content_missing` returns `True` when content is `None`, an empty string, a whitespace-only string, or a list of parts containing no text. The rescue lifts `reasoning_content` (the harmony `analysis` channel as exposed by vLLM) into the synthesizer's view of that candidate. The signature/hash paths use the same predicate, ensuring consistent treatment everywhere a candidate is inspected.

The rescue paired with a configuration fix — removing `include_reasoning = false` from the reasoning upstream's request_overrides, which had been silently stripping the field — closed the production tombstone class. In the chess session reported in Section 5.6, the reasoning generator's draft length distribution was min 54, max 2345, avg 724 chars across 35 `tool_chat.v1` calls — substantively contributing on every call. Pre-fix, the same generator produced `min=8 max=8` exclusively.

### 3.10 Operator-Defined Identity

A small but operationally important addition: a `[server] system_prompt` field that, when set, prepends a single `{role: "system", content: <prompt>}` message to every request before any dispatch branch runs. The caller's own system messages follow the operator's, so task framing is preserved.

This serves two purposes:

1. **Brand identity**: We deploy Synthwave with `system_prompt` set so that `Synthwave-1` (the friendly alias) consistently identifies itself as "an ensemble of multiple models" without naming the underlying upstreams. Without this, the surfaced model name is whichever upstream happens to be the synthesizer, which leaks implementation detail.
2. **Operator policies**: An operator can encode policies (refusal style, output language, safety constraints) in one place, enforced uniformly across every profile.

There is no heuristic gating on message content; every chat request gets the prepend. Empty / whitespace-only `system_prompt` → no behavior change.

### 3.11 Observability

Per-call records are stored in a fixed-size ringbuffer (default 100 per profile) and exposed at `/v1/metrics/moa`:

```json
{
  "ring_size": 100,
  "profiles": {
    "write_synth_content.v1": {
      "calls": 18, "quorum_avg": 2.944,
      "degraded_rate": 0.056,
      "synth_decisions": {"merged": 17, "fallback_primary": 1},
      "tool_call_rate": 0.0,
      "elapsed_ms_avg": 718.0,
      "draft_length": {
        "gen0": {"min": 191, "max": 3036, "avg": 2191, "samples": 18},
        "gen1": {"min": 140, "max": 3191, "avg": 1236, "samples": 18},
        "gen2": {"min": 2058, "max": 4077, "avg": 2726, "samples": 17}
      }
    }
  }
}
```

Per-position draft-length statistics are particularly load-bearing: they make it immediately visible if one generator slot is silently producing tombstones (min == max == small constant), drifting in length distribution (suggests prompt or model regression), or contributing materially less than the others. The endpoint is the primary diagnostic surface; production incidents in this paper were detected by reading these metrics, not by monitoring alerts.

### 3.12 Decoupled Readiness Probe

`/v1/health` originally probed each upstream with `POST /v1/chat/completions max_tokens=1` under a 5-second deadline. Under load, this probe queues behind real inference traffic and false-flags loaded upstreams as `down`. The fix — and the only fix that does not amount to a patch — is to change the probe path entirely:

- **Probe**: `GET /v1/models`, structurally inference-decoupled.
- **Verify**: `upstream.model_id` is in `data[*].id` of the catalog. Closes the separate failure mode where the upstream is running but serving a different model than configured.
- **Auth**: Same headers as `/chat/completions`; resolution failures classified as `misconfigured` rather than `down`.
- **Observability**: WARN on non-up, INFO on recovery transition.

After the fix, `/v1/health` correctly distinguishes `up`, `down`, and `misconfigured` even under heavy chat load — verified live during the chess session reported in Section 5.6.

---

## 4. Experimental Setup

### 4.1 Deployment Environment

Synthwave runs as a managed service on a dedicated accelerator host co-located with one of the inference upstreams. The other upstreams run on neighboring hosts in the same private cluster. The service is exposed through a protected HTTPS endpoint and consumed by:

- **An autonomous Rust agent**, which was the original platform for which Code MoA was built; agent-core's in-process MoA module (`agent-core::moa`) was hard-deleted once all five surfaces (write_synth_*, tool_chat, cap_wrap, recovery_synthesis) were migrated to Synthwave, replaced by a thin OpenAI-compatible client (`MetaModelClient` in `agent-llm`).
- **A public web product.** The platform's per-user agent talks to Synthwave via the same OpenAI-shaped API, behind an application proxy. The MoA-backed agent latency (25-35 s on heavy prompts, 60-90 s for multi-call sessions) is the dominant contributor to user-facing wall-clock time.

The deployment context is *naturalistic*: there is no synthetic load generator. Production data in this paper comes from real agent and user traffic that happened to run while we were measuring.

### 4.2 Models

The evaluated production lineup, anonymized by role:

| Slot | Model | Parameters | Architecture | Context | Served via |
|------|-------|-----------|-------------|---------|------------|
| `primary` (and synthesizer) | Primary code synthesizer | ~32B (est.) | code-specialized transformer | 262K | local OpenAI-compatible server, low-bit quant |
| `reasoning` | 20B reasoning generator | 20B | harmony-format transformer | 131K | local OpenAI-compatible server |
| `fast` | 32B fast generator | 32B | instruction-tuned code model | 32K | local OpenAI-compatible server, quantized |

All models are open-weight and served locally on the same cluster. No third-party API costs.

A vision profile (intentionally omitted from the production profile set in this paper's evaluation) routes multimodal requests through a designated upstream. We treat that as a single-upstream policy rather than a MoA path; multi-model vision synthesis is future work.

Earlier prototypes evaluated a 24B code-completion model as a generator and a 9B instruction model as a third generator. Both were excluded from the final lineup for reasons reported in Sections 5.3 and 5.4.

### 4.3 Tasks

We evaluate at two scopes.

**Original prototype CLI tasks** (qualitative grading, baseline):

| Task | Description |
|------|------------|
| fibonacci_server | HTTP server serving Fibonacci numbers, health endpoint |
| markdown_toc | Markdown table of contents generator with anchor links |
| json_differ | Recursive JSON comparison with colored terminal output |
| csv_pivot | CSV pivot table with group-by statistics |
| port_scanner | Concurrent TCP port scanner with service name resolution |

**Production-traffic snapshots** (live telemetry):

- A user-driven chess-game build through the agent's auto-mode — the agent plans the project, builds it, verifies it, all via Synthwave-backed `tool_chat.v1` and `write_synth_content.v1` calls.
- Public conversational-agent traffic against Synthwave's `tool_chat.v1` profile.

For both scopes we record per-call telemetry (`/v1/metrics/moa`) alongside the qualitative task outcome.

### 4.4 Evaluation

For each prototype task we capture:
1. The **original** implementation (primary model alone, single-upstream passthrough)
2. Each **generator's** independent implementation
3. The **synthesized** final version

All versions are logged for comparison. Evaluation is qualitative, assessed on:
- **Correctness**: Does it handle edge cases? Does it crash on bad input?
- **Robustness**: Error handling, input validation, resource cleanup
- **Features**: Does it go beyond the minimum specification?
- **Code quality**: Readability, structure, DRY principles, idiomatic patterns

For production-traffic snapshots we capture per-profile call counts, quorum-average, degraded-rate, synth-decision distribution, and per-slot draft-length statistics — computing the contribution of each generator to each call without instrumenting the consumer.

---

## 5. Results

### 5.1 Port Scanner (Controlled Comparison)

The port scanner task was run twice with different generator lineups on the same specification. (Historical baseline from the autonomous agent in-process prototype.)

**Configuration A: 24B code-completion model + 35B fast coder (generators), primary coder (synthesizer)**

| Version | Lines | Grade | Notes |
|---------|-------|-------|-------|
| Original (primary coder) | 72 | B+ | Configurable timeout/workers, port validation, `finally: sock.close()` |
| 24B code-completion model | 31 | D | Bare minimum. `getservbyport` without try/except (crashes on unknown ports). No validation, no configurability. |
| 35B fast coder | 62 | B | Real-time print on discovery, service name padding. No timeout/workers args, no port validation. |
| **Synthesized** | **71** | **A-** | Picked original (already best). Minimal improvement. |

**Configuration B: 20B reasoning generator + 35B fast coder (generators), primary coder (synthesizer)**

| Version | Lines | Grade | Notes |
|---------|-------|-------|-------|
| Original (primary coder) | 65 | B+ | Clean, separated `scan_ports()` helper. Manual `sock.close()`. |
| 20B reasoning generator | 94 | A- | Flexible port parsing (`1-1024,8080,443`), DNS validation, `create_connection` context manager, type hints. |
| 35B fast coder | 141 | B | Most features (file input, output file, default ports, examples). BUT: unused imports, loses `as_completed` benefit. |
| **Synthesized** | **110** | **A** | Reasoning generator's port parser + DNS validation + context manager. Fast coder's epilog. Added `OverflowError` catch. Dropped reasoning generator's over-engineering. |

**Finding 1:** The synthesized version in Configuration B is strictly superior to all three inputs. It is not a copy of any single model's output — it combines structural elements from the reasoning generator (port parser, context manager), UX elements from the fast coder (help examples), and adds its own improvement (OverflowError handling) that none of the generators included.

**Finding 2:** Generator model diversity matters enormously. The 24B code-completion model contributed nothing (grade D); the reasoning generator contributed the strongest architectural elements (grade A-). Replacing one generator changed the synthesized output from A- (minor improvement over original) to A (genuine multi-source synthesis).

(Note: this experiment used the original prototype's prompt-discipline anti-anchoring, before the B-schema. The findings stand because the generators in both configurations received only the task spec, not each other's output. The B-schema makes that property structural, not disciplinary.)

### 5.2 Cross-Task Summary

| Task | Original grade | Best generator | Synthesized grade | Synthesis type |
|------|---------------|---------------|-------------------|---------------|
| fibonacci_server | B+ | 35B fast coder (A-) | A | Feature addition (helper extraction, content-length, 404 handler) |
| markdown_toc | B | reasoning generator* (A) | A | Wholesale pick (reasoning generator clearly superior: unique anchors, stdin, --out) |
| json_differ | B+ | 35B fast coder (A-) | A- | Structural improvement (consolidated 3 color functions -> 1, extracted helper) |
| csv_pivot | B | 35B fast coder (A-) | A- | Feature addition (--output flag, auto-detect numerics, error handling) |
| port_scanner | B+ | reasoning generator (A-) | A | Multi-source synthesis (parser + validation + UX + novel error handling) |

*Note: markdown_toc was tested with an incorrect model configuration (the reasoning generator being treated as a code model). Included for completeness.

**Finding 3:** The synthesizer consistently grades ≥ the best individual generator. In no case did synthesis produce output worse than the best input. This confirms the graceful-degradation property: the floor is "pick the best one."

**Finding 4:** The synthesizer adds its own improvements in 3 of 5 tasks — OverflowError handling, function consolidation, and parameter validation that no generator included. The synthesizer is not merely selecting; it is reasoning about the code and improving it.

### 5.3 A 24B Code-Completion Model as a Generator (Negative Result)

A 24B code-completion model was evaluated as a generator but excluded from the production lineup due to:

1. **Severe anchoring**: When shown the primary's output as a "first draft," the model produced near-copies (whitespace-only differences). This motivated the original prompt-discipline anti-anchoring, and ultimately the B-schema.
2. **Minimal independent output**: Without anchoring, 24B code-completion model produced the shortest, least featured implementations (31 lines for port scanner vs 72-141 for others).
3. **Missing error handling**: `socket.getservbyport()` called without try/except, causing crashes on unknown port numbers.

This is a useful negative result: **not all models are good generators.** A model that excels at code completion (the model's strength) may underperform at independent code generation from specification. Synthwave benefits from generators with strong instruction-following and defensive-programming habits.

### 5.4 Generator Parity Bias (Negative Result)

A separate negative result emerged when a 9B instruction model was added as a third generator alongside the 20B reasoning generator and the 35B fast coder. The 9B consistently produced the most *structurally* organized output — clear section headers, numbered lists, machine-parseable formatting — even when its *content* was no better than the other generators.

This created a **pick bias**: the synthesizer, faced with one highly structured input and two less structured but equally substantive ones, defaulted to selecting the structured version wholesale rather than combining elements from all three. Synthesis quality degraded from genuine multi-source combination to single-source selection with minor edits.

When the 9B was removed, the remaining two generators (20B and 35B roles) were closer in structural quality, forcing the synthesizer to actually combine their perspectives. The result was surprising: **2 balanced generators produced better synthesis than 3 imbalanced ones.**

| Metric | 3 generators (with 9B) | 2 generators (without 9B) |
|--------|----------------------|--------------------------|
| ROADMAP.md synthesized | 119 lines | 218 lines |
| PROJECT.md synthesized | 119 lines | 152 lines |
| Synthesis type | Pick (9B selected) | Genuine combination |

This finding has practical implications for Synthwave configuration: **generator parity matters more than generator count.** Adding a generator that dominates on a superficial dimension (formatting, structure, length) can reduce synthesis quality by short-circuiting the comparative evaluation that makes synthesis valuable. The optimal generator set is one where each model contributes genuinely different *perspectives* at comparable *quality levels*, so the synthesizer is forced to evaluate and combine rather than pick.

### 5.5 Anchoring Effect (Historical → Structural)

We discovered the anchoring effect empirically when initial implementations showed generators producing near-copies of the primary's output. The root cause was the generator prompt including the primary's code:

```
A first draft exists:
{original_code}
Write your own version.
```

Removing the original code from the generator prompt immediately produced diverse, independent implementations. This is consistent with research on anchoring bias in human judgment (Tversky & Kahneman, 1974) and suggests that LLMs exhibit analogous anchoring effects in code generation.

Under Synthwave's B-schema, this is no longer a discipline question. The agent's `write` tool emits only `{path}` — there is no draft body in the request that could leak into a generator prompt. The structural shape of the system prevents the anchoring class entirely.

### 5.6 Production Telemetry — Live Chess Session

A user-driven chess game build through the agent's auto-mode produced the following per-profile telemetry (from `/v1/metrics/moa`, captured at session end on 2026-05-04):

```
tool_chat.v1: 35 calls, decisions={merged: 35}, degraded_rate 0.0
  gen0 (primary)         210–1046, avg 745
  gen1 (reasoning)        54–2345, avg 724
  gen2 (fast)            226– 929, avg 770

write_synth_content.v1: 18 calls, decisions={merged: 17, fallback_primary: 1}, degraded_rate 0.056
  gen0 (primary)         191–3036, avg 2191
  gen1 (reasoning)       140–3191, avg 1236
  gen2 (fast)           2058–4077, avg 2726
```

**Finding 5 (production):** Across 35 `tool_chat.v1` calls, every call had quorum 3/3 — all three generators contributed substantively to every chat turn the agent made. Across 18 `write_synth_content.v1` calls, 17 had quorum 3/3 and 1 fell back to the first candidate (a single timeout in an earlier session, retained in the ringbuffer).

**Finding 6 (production):** The reasoning generator (gen1, reasoning generator) consistently contributed substantive draft text in production after the F12 reasoning-content rescue landed. The minimum draft length of 54 chars on `tool_chat.v1` is well above the 8-byte tombstone class; pre-fix the same slot exclusively produced `min=8 max=8`. The rescue is the difference between a silently-broken third-party model and a productive ensemble member.

**Finding 7 (production):** Wall-clock cost for a multi-call session is acceptable for autonomous agents and tolerable for interactive users. The agent's chess-game build completed end-to-end in ≈60–90 seconds wall-clock, including 22 new `tool_chat.v1` and 12 new `write_synth_content.v1` MoA fan-outs (against a pre-session ringbuffer that already held earlier traffic, hence the 35 / 18 totals above). Per-call elapsed (after the elapsed_ms threading fix landed) typically falls in the 500–1500 ms range for `tool_chat.v1` and 1–3 s for `write_synth_content.v1`.

### 5.7 Tool-Call Determinism (Production)

Of the 35 `tool_chat.v1` calls in the chess session, all 35 produced a coherent single tool call in the response, with `synth_decisions` = `{merged: 35}`. The deterministic-merge fast-path captures cases where all three generators converge on identical canonical-arg tool calls; the LLM-side synth path handles cases where they diverge. The combination of the two — none of which existed in the prototype — is what makes a `tool_chat.v1` profile viable in production: pure-text MoA cannot guarantee a tool-shaped response, and pure-deterministic merging cannot resolve disagreement.

---

## 6. Discussion

### 6.1 Why Not Just Use the Best Model?

The synthesized output is consistently better than any individual model, including the synthesizer model acting alone (as the "primary" generating the original code). This seems paradoxical — how can a model improve upon its own output by reviewing alternatives?

The answer is that **generation and evaluation are different tasks with different difficulty profiles.** Generating a flexible port parser from scratch requires the model to *invent* the feature. But *recognizing* that reasoning generator's port parser is superior to a simpler positional-args approach, and *extracting* it into the synthesized version, is an easier evaluation task. The synthesizer sees multiple solutions and can make comparative judgments that would be unavailable during generation.

This is analogous to why human code review improves quality: the reviewer doesn't need to be a better programmer than the author — they just need to evaluate presented alternatives.

### 6.2 Scaling Laws

Our experiments use 2–3 generators + 1 synthesizer (3–4 models total). Based on ensemble learning theory (Breiman, 1996), we expect:

- **Diminishing returns**: Each additional generator adds less diversity than the previous one, as the synthesis already captures the most important patterns.
- **Quality floor from diversity**: Adding a weak generator does not degrade the output because the synthesizer simply ignores inferior contributions — provided the generator is parity-matched in structural traits (Section 5.4).
- **Synthesis bottleneck**: The synthesizer's ability to evaluate and compose is bounded by its own capabilities. A stronger synthesizer could extract more value from the same generators.

We cap generators at 5 to prevent context overflow in the synthesis prompt (5 implementations × ~4K tokens each = ~20K tokens, within most models' context limits, with shared-tail compaction protecting against caller-side history overrun).

### 6.3 Relationship to Genetic Programming

Synthwave's MoA path shares structural similarities with genetic programming (Koza, 1992):

| GP Concept | Synthwave Analog |
|-----------|------------------|
| Population | N generator outputs |
| Fitness evaluation | Synthesizer's qualitative assessment |
| Crossover | Structural element combination in synthesis |
| Mutation | Synthesizer's novel additions (e.g., OverflowError catch) |
| Selection | "Pick the best approach and refine it" |

The key difference is that GP operates syntactically (swapping subtrees) while Synthwave operates semantically (the synthesizer understands what the code does, not just its structure). This enables meaningful cross-architecture combination that would be invalid under syntactic crossover.

### 6.4 Limitations

1. **No automated correctness verification.** Unlike AlphaCode, we do not filter by test execution. The synthesizer relies on code understanding, not empirical testing. Adding execution-based filtering (CodeT-style) would be a natural extension.

2. **Qualitative evaluation.** Our grading is expert assessment, not automated metrics. HumanEval/MBPP-style evaluation would provide more rigorous comparison, but our tasks are more representative of real-world coding (full CLI tools, not isolated functions).

3. **Latency overhead.** Per-call latency for `write_synth_content.v1` typically falls in the 1–3 s range when all generators succeed; agent sessions accumulate that cost across many calls. This is acceptable for autonomous agents and tolerable for interactive users on heavy prompts; it is *not* yet competitive with single-model latency for simple chat. Faster generators or shorter prompts could close that gap.

4. **Synthesizer ceiling.** The synthesized output cannot exceed the synthesizer's own capabilities. If none of the generators produce a crucial feature, the synthesizer cannot invent it from nothing.

5. **Limited evaluation scale.** The qualitative comparisons are demonstrations, not benchmarks. The production-traffic telemetry is real but not externally-replicable. Systematic evaluation across hundreds of tasks with diverse specifications would strengthen the claims.

6. **No multimodal MoA.** The vision cascade is single-upstream. Multi-model vision synthesis is structurally compatible with the same dispatch path but unevaluated; we treat it as future work.

### 6.5 Future Work

- **Execution-based filtering**: Run generated code against automatically generated test cases (CodeT-style) before synthesis, providing the synthesizer with empirical correctness signals.
- **Iterative refinement**: Run the MoA path in multiple rounds — synthesize, generate tests, run tests, re-synthesize with failure information.
- **Process-level reward models**: Train a code-specific verifier (Lightman et al., 2023) to score implementations before synthesis, allowing the synthesizer to focus on the most promising candidates.
- **Cross-language synthesis**: Apply Code MoA across language boundaries — generate Python, Rust, and Go implementations, synthesize insights about architecture and algorithms back into the target language.
- **Automated benchmarking**: Systematic evaluation on HumanEval+, MBPP, and SWE-bench with pass@1 metrics for rigorous comparison.
- **Adaptive generator selection**: Dynamically select generators based on task characteristics — use models strong in systems programming for infrastructure tasks, models strong in data processing for analytics tasks.
- **Multi-tenancy at scale**: Per-tenant profile sets, per-tenant rate limiting, per-tenant identity prompts — Synthwave is structured to support this and we have the early ingredients (bearer auth, profile isolation), but production hardening is unstarted.

### 6.6 Operating Realities

A few notes about running an MoA service in production that the original prototype did not surface:

**Observability is the diagnostic surface.** Every production incident in the migration to Synthwave was first detected by reading `/v1/metrics/moa`, not by alerts. The 8-byte tombstone (Section 3.9) was diagnosed from `min=8 max=8` in `draft_length`. Synth-timeout regressions showed up as `degraded_rate` > 0 paired with `synth_decisions.fallback_primary` > 0. The decision to thread `elapsed_ms` through was driven by production confusion: a synth-fallback log line emitted "ReadTimeout" with no duration, and a wall-clock measurement of the user session contradicted the timeout-derived estimate. We could not reconcile the two without `elapsed_ms`.

**Identity is operator-defined.** A multi-model service answers questions about its identity from whichever upstream happens to synthesize. Users interpret the resulting answer as the service's identity — not the synthesizer's. The `system_prompt` prepend (Section 3.10) is the cheapest possible fix and worked immediately when deployed. It also discourages tying the service's brand to a specific upstream choice, which makes upstream rotation easier later.

**Health probes must be inference-decoupled.** Probing inference endpoints with inference requests creates a load-coupled feedback loop: the more loaded an upstream is, the more likely the probe is to time out, the more aggressively the system routes around the upstream, the worse the service degrades. Probing `/v1/models` with a tight deadline cleanly separates "is the upstream alive and serving the configured model" from "is the upstream currently busy" — the second question is answered by metrics, not the health endpoint.

**independent reviewer collab as design discipline.** Major architectural changes — the GenerationGateway split, the deletion of the in-process audit pipeline, the reasoning-content rescue, the readiness-probe rewrite — went through cold review by independent reviewer (a strong external coding model at high reasoning effort) before being shipped. We treat the cold review as a non-rubber-stamp constraint: a SHIP verdict is a precondition for landing changes that would otherwise be hard to reverse. Several findings in this paper (the `auth-resolution-can-raise` HIGH on the readiness-probe rewrite, the per-call `elapsed_ms` instrumentation, the `last_error` visibility on health) trace to independent reviewer review comments that the authors initially missed. We document this not as a particular tooling endorsement but because the practice of *cold review of the architecture spec by an independent strong model* is something any small team building production AI infrastructure can adopt.

---

## 7. Implementation

Synthwave is implemented in Python (FastAPI + httpx + tomllib), approximately 10K lines of source plus tests. The implementation consists of:

1. **HTTP server** (`server.py`): FastAPI app, routes (`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models`, `/v1/health`, `/v1/metrics/moa`, `/tokenize`), middleware (CORS, bearer auth).
2. **Config loader** (`config.py`): TOML schema with required/optional blocks (`upstreams`, `profiles`, `server`, `aliases`); strict validation, no silent defaults for load-bearing fields.
3. **Dispatch** (`moa/dispatch.py`): Central request router. Identity-prompt prepend, tool-call normalization, multimodal short-circuit, profile resolution, MoA fan-out, single-upstream passthrough, vision cascade. Per-call observability hooks.
4. **Fan-out** (`moa/fanout.py`): Parallel HTTP to *N* upstreams via `asyncio.gather` + `httpx.AsyncClient`. Per-upstream timeout, structured success/failure records, degraded-headers construction.
5. **Synthesis** (`moa/synthesizer.py`): Merge and best-of synthesis modes. Anchored on the lowest-index successful candidate when synth fails. Reasoning-content rescue. Tool-aware merge. Synth timeout margin to ensure fallback fits within caller's deadline.
6. **Compaction** (`moa/compaction.py`): Shared-tail compaction. When fan-out budget exceeds upstream context, the same compacted-tail prompt is sent to every generator; the synthesizer sees the same shape. 413 path with structured error headers when even compacted prompts exceed budget.
7. **Tools** (`moa/tools.py`): `tools` / `tool_choice` / `parallel_tool_calls` normalization, deterministic tool-call merge, canonical-args hashing.
8. **Multimodal** (`moa/multimodal.py`): Image/video/audio detection, vision cascade routing, modality-aware multimodal_max_active enforcement.
9. **Streaming** (`moa/streaming.py`): SSE streaming for the synth output token stream after MoA convergence; server resolves the merged candidate, then streams the synth's tokens to the caller.
10. **Metrics** (`metrics.py`): Per-profile ringbuffer (deque), aggregator producing `/v1/metrics/moa` payload, per-call elapsed_ms instrumentation.
11. **Health** (`health.py`): Per-upstream readiness probe via `GET /v1/models` + catalog membership check, auth-aware error classification.

The service is fronted by a single bearer token; production deploys provision the token via managed secret. Tests: 641 passing at the time of writing, run on every commit (`pytest`). Major changes go through cold review by an independent strong model (Section 6.6).

The autonomous agent's MoA module (`agent-core::moa`) was hard-deleted once all five surfaces were migrated to remote mode. The agent now uses `MetaModelClient` (a thin OpenAI-compatible HTTP client in `agent-llm`) to talk to Synthwave. The migration removed roughly 6K lines of Rust orchestration code from the agent and centralized all MoA policy in Synthwave.

Source code: github.com/[redacted]/synthwave (Python service); github.com/[redacted]/agent (Rust agent client).

---

## 8. Conclusion

Code MoA demonstrates that the ensemble principle — combining diverse models improves over any individual — applies to LLM-based code generation with minimal engineering effort. By routing code generation tasks to multiple diverse models in parallel and using an intelligent synthesizer to compose the best elements, we consistently achieve code quality equal to or better than any individual model, with zero additional training.

Synthwave generalizes that primitive into a productionized OpenAI-compatible service whose six profiles cover code synthesis, planning-doc synthesis, conversational tool-calling, plain text, cap-wrap closures, and stuck-loop recovery. The structural diversity that makes Code MoA work also makes those broader policies work: independent generators reveal different perspectives, the synthesizer picks the best and refines, and per-profile policies (tool stripping, deterministic tool-call merge, vision-modality cascade) encode surface-specific constraints without forcing surface-specific orchestration code.

The path from prototype to production exposed several design decisions that the in-agent prototype did not: the B-schema as a structural anti-anchoring guarantee; the reasoning-content rescue for harmony-format models; operator-defined identity injection; and an inference-decoupled readiness probe. Each was prompted by an operational incident, none by a research goal. We document them so that the next team productionizing an MoA service does not have to find them by failing at the same points.

The key insight remains unchanged from the original Code MoA paper: **architectural diversity, not technical complexity.** Different models make different mistakes; a smart synthesizer can exploit this. As the LLM ecosystem continues to diversify with models optimized for different niches — code, reasoning, instruction following, safety — the value of multi-model synthesis will only increase. Productionizing it requires modest infrastructure (a single FastAPI service, a TOML config, a ringbuffer endpoint) and disciplined operation. The infrastructure cost was repaid within weeks by the elimination of the in-process MoA code in every consumer.

---

## References

Abolafia, D.A., Norouzi, M., Shen, J., Zhao, R., Le, Q.V. (2018). Neural Program Synthesis with Priority Queue Training. *arXiv:1801.03526*.

Austin, J., Odena, A., Nye, M., Bosma, M., et al. (2021). Program Synthesis with Large Language Models. *arXiv:2108.07732*.

Bai, Y., Kadavath, S., Kundu, S., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*.

Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123–140.

Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.

Chen, B., Zhang, F., Nguyen, A., et al. (2022). CodeT: Code Generation with Generated Tests. *arXiv:2207.10397*.

Chen, M., Tworek, J., Jun, H., et al. (2021). Evaluating Large Language Models Trained on Code. *arXiv:2107.03374*.

Chen, X., Liu, C., Song, D. (2019). Execution-Guided Neural Program Synthesis. *ICLR 2019*.

Cobbe, K., Kosaraju, V., Bavarian, M., et al. (2021). Training Verifiers to Solve Math Word Problems. *arXiv:2110.14168*.

Fedus, W., Zoph, B., Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. *JMLR 2022*.

Freund, Y., Schapire, R.E. (1997). A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. *Journal of Computer and System Sciences*, 55(1).

Gulwani, S. (2011). Automating String Processing in Spreadsheets Using Input-Output Examples. *POPL 2011*.

Gulwani, S., Polozov, O., Singh, R. (2017). Program Synthesis. *Foundations and Trends in Programming Languages*, 4(1-2).

Guo, D., Zhu, Q., Yang, D., et al. (2024). DeepSeek-Coder: When the Large Language Model Meets Programming. *arXiv:2401.14196*.

Hong, S., Zhuge, M., Chen, J., et al. (2023). Meta Programming framework: Meta Programming for a Multi-Agent Collaborative Framework. *ICLR 2024*.

Huang, D., Bu, Q., Zhang, J., et al. (2023). AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and Optimisation. *arXiv:2312.13010*.

Jiang, A.Q., Sablayrolles, A., Roux, A., et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.

Jiang, D., Ren, X., Lin, B.Y. (2023). LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion. *ACL 2023*.

Jimenez, C.E., Yang, J., Wettig, A., et al. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? *ICLR 2024*.

Koza, J.R. (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press.

Le, H., Wang, Y., Gotmare, A.D., Savarese, S., Hoi, S.C.H. (2022). CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning. *NeurIPS 2022*.

Lepikhin, D., Lee, H., Xu, Y., et al. (2021). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. *ICLR 2021*.

Leviathan, Y., Kalman, M., Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML 2023*.

Li, J., Zhang, Q., Yu, Y., Fu, Q., Ye, D. (2024). More Agents Is All You Need. *arXiv:2402.05120*.

Li, R., Allal, L.B., Zi, Y., et al. (2023). StarCoder: May the Source Be with You! *TMLR 2023*.

Li, Y., Choi, D., Chung, J., et al. (2022). Competition-Level Code Generation with AlphaCode. *Science*.

Lightman, H., Kosaraju, V., Burda, Y., et al. (2023). Let's Verify Step by Step. *ICLR 2024*.

Liu, J., Xia, C.S., Wang, Y., Zhang, L. (2023). Is Your Code Generated by chat model Really Correct? *NeurIPS 2023*.

Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS 2022*.

Qian, C., Cong, X., Yang, C., et al. (2023). Communicative Agents for Software Development. *ACL 2024*.

Snell, C., Lee, J., Xu, K., Kumar, A. (2024). Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters. *arXiv:2408.03314*.

Tversky, A., Kahneman, D. (1974). Judgment under Uncertainty: Heuristics and Biases. *Science*, 185(4157), 1124–1131.

Wang, J., Wang, X., Jiang, D., et al. (2024). Mixture-of-Agents Enhances Large Language Model Capabilities. *arXiv:2406.04692*.

Wang, X., Wei, J., Schuurmans, D., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. *ICLR 2023*.

Wolpert, D.H. (1992). Stacked Generalization. *Neural Networks*, 5(2), 241–259.

Zhang, S., Chen, Z., Shen, Y., et al. (2023). Planning with Large Language Models for Code Generation. *ICLR 2023*.

Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*.

Zhuo, T.Y. (2023). ICE-Score: Instructing Large Language Models to Evaluate Code. *EACL 2024 Findings*.

Zhou, S., Alon, U., Agarwal, S., Neubig, G. (2023). CodeBERTScore: Evaluating Code Generation with Pretrained Models of Code. *EMNLP 2023*.
