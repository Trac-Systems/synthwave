"""Chat Completions request/response schema (OpenAI-compatible).

Pydantic v2 models. `extra="allow"` on the top-level request — OpenAI
clients send fields the meta-model doesn't know about, and we pass
unknown keys through to upstreams rather than 400ing the call.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Messages ────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """One chat message. `content` may be a str OR a list of parts
    (multimodal). We don't validate part shape here — that lives in
    D.3.1 vision policy."""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool", "developer", "function"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


# ── Tool schemas ────────────────────────────────────────────────────


class ToolDef(BaseModel):
    """OpenAI's `tools[*]` shape. Schema accepts any `type` string so
    `normalize_tool_request` can return a typed
    `feature_not_supported_in_v1` error for non-function shapes (custom
    tools, novel types). D.3.1 is function-tools-only at the dispatch
    layer; the schema itself stays permissive."""

    model_config = ConfigDict(extra="allow")

    type: str = "function"
    function: dict[str, Any] | None = None


# ── Meta-model extension ───────────────────────────────────────────


class MetaModelExt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str | None = None
    trace: bool = False


# ── Request ─────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """OpenAI Chat Completions request.

    `extra="allow"` so clients sending fields we don't model (e.g.
    `seed`, `frequency_penalty`, `logit_bias`, etc.) pass through
    unmolested. That's the OpenAI compat contract.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)

    # Tool calling — accept legacy + modern. `normalize_tool_request`
    # in moa/tools.py owns shape validation so error codes are typed
    # (`feature_not_supported_in_v1` vs `invalid_request_error`)
    # rather than pydantic's generic 400.
    tools: list[ToolDef] | None = None
    functions: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    function_call: str | dict[str, Any] | None = None
    parallel_tool_calls: bool = True

    # Output budget (one or the other, not both).
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)

    # Streaming.
    stream: bool = False
    stream_options: dict[str, Any] | None = None

    # Standard sampling knobs.
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: str | list[str] | None = None
    response_format: dict[str, Any] | None = None
    n: int | None = Field(default=None, ge=1)

    # Meta-model extension (server-owned profiles).
    x_meta_model: MetaModelExt | None = None
