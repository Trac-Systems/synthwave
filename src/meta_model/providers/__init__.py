"""Pluggable upstream transport providers.

Each provider translates synthwave's internal OpenAI Chat-Completions
request/response shape to and from a specific backend's native wire
protocol, so a single MoA ensemble can fan out across heterogeneous
backends transparently (OpenAI-compatible vLLM, the OpenAI API's
reasoning models, and the Anthropic Messages API).

A provider's public entry point takes the same arguments as
`meta_model.upstream.forward_chat_completion` and returns an
`httpx.Response` whose JSON body is OpenAI-shaped, so every existing
caller (fanout, synthesizer, cascade) consumes it unchanged. Providers
are selected per-upstream via `UpstreamConfig.protocol`; the default
("openai") needs no provider and stays on the original fast path.
"""

from __future__ import annotations

from .anthropic import forward_anthropic_messages

__all__ = ["forward_anthropic_messages"]
