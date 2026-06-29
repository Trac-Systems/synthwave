# Synthwave Meta-Model

**Synthwave makes several AI models smarter by making them work together.** It is a Mixture-of-Agents (MoA) service by **Trac Systems**: instead of routing your prompt to one model, Synthwave asks several independent models to each draft an answer, then has a final *synthesizer* model read every draft and fuse them into one best response. The combined system reasons better, makes fewer mistakes, and covers more ground than any of its individual members — all behind a single OpenAI-compatible endpoint.

You call it like any normal model API (`POST /v1/chat/completions` with `model: "synthwave"`). Operators define the model fleet and how the models combine — parallel fan-out with synthesis, fallback cascades, or voting — in a TOML config.

## Benchmark: AIME 2026

> [!IMPORTANT]
> **Synthwave scores 90.0% (27/30) on AIME 2026** — landing in the frontier band, above DeepSeek V3.2 and Claude Opus 4.6, on a freshly released, uncontaminated competition set. This is achieved with internal reasoning **mostly off**.

Measured on the full AIME 2026 set (American Invitational Mathematics Examination I + II, 30 problems, single run; final answers graded as exact integers). The three misses are the three hardest items — the two final AIME II problems and one diagram-dependent problem.

### Ranking — AIME 2026

| Model | AIME 2026 | Internal thinking |
| --- | --- | --- |
| GPT-5 | ~100% | full |
| Gemini 3.1 Pro | ~95% | full |
| Grok 4.2 | ~93% | full |
| **Synthwave (this project)** | **90.0%** | **mostly off** |
| DeepSeek V3.2 | ~88% | full |
| Claude Opus 4.6 | ~85% | full |
| Llama 4 Scout | ~80% | full |
| Qwen 3.5 | ~76% | full |

Competitor figures come from public AIME 2026 leaderboards and can be checked directly: [MathArena — AIME 2026](https://matharena.ai/?comp=aime--aime_2026) and the [LLM leaderboard](https://www.clickrank.ai/llm-leaderboard/). Only the Synthwave row is measured by us; treat the other rows as indicative (sources and exact evaluation conditions vary).

> [!NOTE]
> **Thinking was mostly off — and turning it on _may_ increase intelligence.** Of the four models in the ensemble, only one generator (Ornith) runs with internal step-by-step reasoning enabled; the other two generators and the synthesizer run without it. So 90% reflects an ensemble that is largely *not* using extended thinking. Enabling reasoning across more of the fleet is untested headroom that may push the score higher.

## The Benchmarked Setup

The configuration above runs across **two NVIDIA DGX Spark (GB10) nodes**, each model served with [vLLM](https://github.com/vllm-project/vllm) behind an OpenAI-compatible endpoint and **NVFP4-quantized** for the GB10 hardware. The Mixture-of-Agents profile fans out to three **generators** (each writes an independent draft, in parallel) and fuses their drafts with one **synthesizer**:

| Role | Model | Notes |
| --- | --- | --- |
| Generator | **Ornith-1.0-35B** | reasoning generator — the only member with internal thinking **on** |
| Generator | **Qwen3.6-35B-A3B** | generalist (also the image/video input path) |
| Generator | **Qwen3-Coder-Next** | code-focused generator |
| Synthesizer | **Gemma-4-26B-A4B** | reads every draft, writes the final fused answer (thinking off) |

Synthesis settings: **merge** mode, generator temperature **0.6**, synthesizer temperature **0.2**. The generators are queried concurrently; the synthesizer then reads all drafts and produces a single answer. Only Ornith uses extended thinking — the other generators and the synthesizer do not.

### What this demonstrates

This runs on **two DGX Spark (GB10) desktop-class nodes** — not a datacenter — yet the ensemble reaches **frontier territory for intelligence**. None of the individual models is a frontier model: three mid-size (26–35B) generators, each on its own well below the top of the table, are fused into a result that ranks alongside the best. The intelligence is a property of the **composition**, not of any single model or of raw scale.

That principle scales *down*, too. The Mixture-of-Agents mechanism is model-agnostic — it lifts the effective intelligence of whatever fleet it is given, including **single-machine setups running smaller models**. You do not need this exact lineup or this hardware; you need a *diverse, balanced set of models* and a synthesizer to combine them. Because intelligence here depends on the **composition of models**, improving the mix — stronger generators, more diversity, more reasoning enabled — raises the result further than scaling any one model could.

---

Synthwave exposes a normal `/v1` model API while internally routing each request through one of several operator-defined profiles: single-upstream passthrough, parallel fan-out with synthesis, fallback cascades, or voting. Clients can use it as a drop-in model endpoint; operators control the model fleet and profile behavior in TOML.

## What It Provides

- OpenAI-compatible `POST /v1/chat/completions`.
- Compatibility adapters for `POST /v1/completions` and `POST /v1/responses`.
- `GET /v1/models` with profile/model capabilities.
- `GET /v1/health` readiness checks against configured upstreams.
- `GET /v1/metrics/moa` for recent MoA dispatch telemetry.
- Optional bearer auth on the public API.
- Per-upstream bearer or basic auth.
- Server-owned profiles for MoA, cascade, and voting behavior.
- Tool/function-call normalization and sanitization.
- Optional image, video, and audio routing through server-level cascades.
- Optional single-model facade via `server.model_name`.

## Repository Layout

```text
src/meta_model/              FastAPI service and dispatch code
src/meta_model/moa/          Fan-out, synthesis, multimodal, and tool policy
tests/                       Contract and behavior tests
meta-model.toml.example      Annotated operator configuration
pyproject.toml               Python package metadata
PAPER.md                     System paper and design notes
```

## Requirements

- Python `3.11+`.
- One or more upstream model endpoints that speak an OpenAI-compatible `/v1/chat/completions` API.
- Optional upstream `/v1/models` support for readiness probes.

The upstreams can be local inference servers, private network services, cloud gateways, or a mixture of those. Synthwave only needs the base URL, model id, context budget, output budget, modalities, and auth settings.

## Install

```bash
cd synthwave
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For a runtime-only install, omit the dev extra:

```bash
pip install -e .
```

## Configure

Start from the example config:

```bash
cp meta-model.toml.example meta-model.toml
$EDITOR meta-model.toml
```

The config has three layers:

1. `server`: public API binding, auth, CORS, facade name, and optional system identity.
2. `upstreams`: available models and how to call them.
3. `profiles`: callable model/profile names exposed through `/v1/models`.

### Minimal Single-Upstream Setup

Use this when you want Synthwave to expose one existing model as an OpenAI-compatible endpoint with health checks, auth, and a stable model name.

```toml
[server]
host = "127.0.0.1"
port = 8400
bearer_token = "change-me"
model_name = "synthwave-local"

[upstreams.main]
model_id = "provider/model-id"
base_url = "http://127.0.0.1:8000/v1"
context = 32768
max_output = 4096
modalities = ["text"]
api_key_env = "MAIN_MODEL_KEY"

[profiles."chat.v1"]
type = "cascade"
upstreams = ["main"]
aliases = ["synthwave-local"]
```

Run with:

```bash
export MAIN_MODEL_KEY="upstream-token-if-needed"
export META_MODEL_CONFIG=./meta-model.toml
meta-model --config "$META_MODEL_CONFIG"
```

Or directly with Uvicorn:

```bash
META_MODEL_CONFIG=./meta-model.toml \
uvicorn meta_model.server:app --host 127.0.0.1 --port 8400
```

### MoA Setup With Your Available Models

Use this when you have multiple models and want independent drafts synthesized into one final answer.

```toml
[server]
host = "127.0.0.1"
port = 8400
bearer_token = "change-me"
model_name = "synthwave"
request_timeout_secs = 600

[upstreams.primary]
model_id = "your-best-synthesizer"
base_url = "http://127.0.0.1:8001/v1"
context = 131072
max_output = 8192
modalities = ["text"]
api_key_env = "PRIMARY_MODEL_KEY"

[upstreams.fast]
model_id = "your-fast-coder"
base_url = "http://127.0.0.1:8002/v1"
context = 32768
max_output = 4096
modalities = ["text"]
api_key_env = "FAST_MODEL_KEY"

[upstreams.reasoning]
model_id = "your-reasoning-model"
base_url = "http://127.0.0.1:8003/v1"
context = 32768
max_output = 4096
modalities = ["text"]
api_key_env = "REASONING_MODEL_KEY"
request_overrides = { include_reasoning = false }

[profiles."write_synth.v1"]
type = "moa"
generators = ["fast", "reasoning"]
synthesizer = "primary"
synthesis_mode = "merge"
generator_temperature = 0.4
synthesizer_temperature = 0.2
non_client_synth_reserve_tokens = 8192
aliases = ["synthwave"]
```

How to choose upstream roles:

- `synthesizer`: use the model with the best instruction following, longest context, and strongest final-answer quality.
- `generators`: use diverse models. Speed matters because the MoA wall clock is bounded by the slowest required generator.
- `context`: set the real usable input context, not the marketing maximum.
- `max_output`: set the largest generation budget you are willing to allocate to that upstream.
- `modalities`: use `['text']` for text-only models and add `image`, `video`, or `audio` only when the upstream really supports that input shape.
- `request_overrides`: use for server-owned upstream quirks such as disabling hidden reasoning output, forcing a parser mode, or setting model-specific request fields.

### Vision, Video, And Audio

Multimodal routing is server-level. Profiles do not opt in individually. Configure ranked cascades by upstream name:

```toml
[upstreams.vision_primary]
model_id = "your-vision-model"
base_url = "http://127.0.0.1:8010/v1"
context = 32768
max_output = 4096
modalities = ["text", "image"]

[vision]
endpoints = ["vision_primary"]

[video]
endpoints = []

[audio]
endpoints = []
```

If a modality list is empty, requests using that input type return a typed unsupported-modality error instead of being sent to a text-only model.

## Run

With the console script:

```bash
META_MODEL_CONFIG=./meta-model.toml \
meta-model --config ./meta-model.toml
```

With Uvicorn:

```bash
META_MODEL_CONFIG=./meta-model.toml \
uvicorn meta_model.server:app --host 127.0.0.1 --port 8400
```

If your config sets `[server] host` and `port`, prefer the console script so the service reads those values directly.

## Test The Service

Health:

```bash
curl -s http://127.0.0.1:8400/v1/health | jq
```

Model catalog:

```bash
curl -s http://127.0.0.1:8400/v1/models \
  -H 'Authorization: Bearer change-me' | jq
```

Chat completion:

```bash
curl -s http://127.0.0.1:8400/v1/chat/completions \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "synthwave",
    "messages": [{"role": "user", "content": "Write a compact hello-world in Python."}],
    "max_tokens": 200,
    "temperature": 0.2
  }' | jq
```

Select a specific profile:

```json
{
  "model": "write_synth.v1",
  "x_meta_model": {"profile": "write_synth.v1"},
  "messages": [{"role": "user", "content": "Implement the requested file."}],
  "max_tokens": 1200
}
```

## Endpoint Summary

```text
GET  /v1/health              Readiness; 503 when any required upstream is unhealthy
GET  /health                 Alias for /v1/health
GET  /v1/models              OpenAI-style model/profile catalog
POST /v1/chat/completions    Main OpenAI-compatible chat endpoint
POST /v1/completions         Legacy text-completion adapter
POST /v1/responses           Responses-style adapter for supported request shapes
POST /tokenize               Token counting via configured tokenizer upstream
GET  /v1/metrics/moa         Recent MoA dispatch telemetry
GET  /metrics                Alias for /v1/metrics/moa
```

## Operational Notes

- Treat `/v1/health` as readiness, not liveness. A model outage should stop traffic, not necessarily restart the service.
- MoA latency is determined by the slowest required generator plus synthesis time.
- Keep `request_timeout_secs` above the expected worst-case fan-out and synthesis wall clock.
- Use aliases or `server.model_name` when external clients need one stable model id.
- Keep secrets in environment variables with `api_key_env` or `basic_auth_pass_env`.
- Use `GET /v1/metrics/moa` to inspect generator success, fallback behavior, draft sizes, and elapsed time.

## Test Suite

```bash
cd synthwave
source .venv/bin/activate
pytest
ruff check .
```

## Authorship

Synthwave Meta-Model is authored by **Markus Bopp from Trac Systems**.
