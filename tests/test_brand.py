"""F10 — server-wide model_name brand override.

Coverage:
- Config validation
  - empty / whitespace-only model_name normalizes to None (no-op)
  - collision with profile / upstream / alias rejected at load time
- Response.model rewrite across all endpoints
  - /v1/chat/completions non-streaming
  - /v1/chat/completions streaming SSE chunks
  - /v1/completions (legacy) — propagates via chat-side rewrite
  - /v1/responses
- Resolver resolves brand to first callable non-voting profile
- /v1/models lists brand as featured first entry
- Brand unset → no behavior change (back-compat)
"""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from meta_model.config import parse_config_str
from meta_model.server import app, set_config, set_upstream_transport


_BRAND_FIXTURE = """
[server]
model_name = "Synthwave-1"

[upstreams.a]
model_id = "ma"
base_url = "http://a.local/v1"
context = 8192
max_output = 512

[upstreams.b]
model_id = "mb"
base_url = "http://b.local/v1"
context = 8192
max_output = 512

[profiles."alpha.v1"]
type = "moa"
generators = ["a", "b"]
synthesizer = "a"

[profiles."beta.v1"]
type = "moa"
generators = ["b"]
synthesizer = "b"
"""

_NO_BRAND_FIXTURE = """
[upstreams.a]
model_id = "ma"
base_url = "http://a.local/v1"
context = 8192
max_output = 512

[profiles."alpha.v1"]
type = "moa"
generators = ["a"]
synthesizer = "a"
"""


@pytest.fixture
def brand_config():
    cfg = parse_config_str(_BRAND_FIXTURE)
    set_config(cfg)
    yield cfg
    set_config(None)


@pytest.fixture
def no_brand_config():
    cfg = parse_config_str(_NO_BRAND_FIXTURE)
    set_config(cfg)
    yield cfg
    set_config(None)


def _client() -> TestClient:
    return TestClient(app)


def _ok_chat_handler(req: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "synth-1",
            "object": "chat.completion",
            "created": 0,
            "model": "ma",  # the upstream's model_id
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


# ── Config validation ───────────────────────────────────────────────


def test_brand_unset_default_is_none() -> None:
    cfg = parse_config_str(_NO_BRAND_FIXTURE)
    assert cfg.server.model_name is None


def test_brand_whitespace_only_normalizes_to_none() -> None:
    """Defends operator-typo configs that leave whitespace.
    `model_name = "   "` strips to None (no-op)."""
    cfg = parse_config_str("""
[server]
model_name = "   "

[upstreams.a]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."alpha.v1"]
type = "moa"
generators = ["a"]
synthesizer = "a"
""")
    assert cfg.server.model_name is None


def test_brand_strips_surrounding_whitespace() -> None:
    """``"  Synthwave-1  "`` stores as ``"Synthwave-1"``."""
    cfg = parse_config_str("""
[server]
model_name = "  Synthwave-1  "

[upstreams.a]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."alpha.v1"]
type = "moa"
generators = ["a"]
synthesizer = "a"
""")
    assert cfg.server.model_name == "Synthwave-1"


def test_brand_collision_with_profile_rejected() -> None:
    """Brand must not collide with any profile name (case-insensitive)."""
    with pytest.raises(Exception, match=r"(?i)collide"):
        parse_config_str("""
[server]
model_name = "alpha.v1"

[upstreams.a]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."alpha.v1"]
type = "moa"
generators = ["a"]
synthesizer = "a"
""")


def test_brand_collision_with_alias_rejected() -> None:
    """Brand must not collide with any profile alias (case-insensitive)."""
    with pytest.raises(Exception, match=r"(?i)collide"):
        parse_config_str("""
[server]
model_name = "Synthwave-1"

[upstreams.a]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."alpha.v1"]
type = "moa"
generators = ["a"]
synthesizer = "a"
aliases = ["synthwave-1"]
""")


def test_brand_collision_with_upstream_rejected() -> None:
    """Brand must not collide with any raw upstream key."""
    with pytest.raises(Exception, match=r"(?i)collide"):
        parse_config_str("""
[server]
model_name = "primary"

[upstreams.primary]
model_id = "ma"
base_url = "http://a/v1"
context = 8192
max_output = 512

[profiles."alpha.v1"]
type = "moa"
generators = ["primary"]
synthesizer = "primary"
""")


# ── Response.model rewrite — non-streaming chat ─────────────────────


def test_chat_completions_response_model_uses_brand(brand_config) -> None:
    """`response.model` is the brand, not the upstream's model_id or
    the profile name. Independent of which profile the client called."""
    set_upstream_transport(httpx.MockTransport(_ok_chat_handler))
    try:
        r = _client().post(
            "/v1/chat/completions",
            json={"model": "alpha.v1", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "Synthwave-1"
        # Profile metadata still in headers for introspection-aware clients.
        assert r.headers.get("x-metamodel-profile") == "alpha.v1"
    finally:
        set_upstream_transport(None)


def test_chat_completions_response_model_unchanged_when_brand_unset(
    no_brand_config,
) -> None:
    """Back-compat: brand unset → response.model is profile name (today's
    behavior)."""
    set_upstream_transport(httpx.MockTransport(_ok_chat_handler))
    try:
        r = _client().post(
            "/v1/chat/completions",
            json={"model": "alpha.v1", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Pre-F10 behavior: model echoes whatever dispatch set (synth
        # passthrough returns upstream's model_id; cascade rewrites to
        # profile_name). Either way, NOT "Synthwave-1".
        assert body["model"] != "Synthwave-1"
    finally:
        set_upstream_transport(None)


# ── Brand resolves as a callable model parameter ────────────────────


def test_brand_resolves_to_first_callable_profile(brand_config) -> None:
    """Calling with model='Synthwave-1' routes to alpha.v1 (first
    callable in config order). Headers + response confirm."""
    set_upstream_transport(httpx.MockTransport(_ok_chat_handler))
    try:
        r = _client().post(
            "/v1/chat/completions",
            json={
                "model": "Synthwave-1",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 200, r.text
        # Routed internally to alpha.v1.
        assert r.headers.get("x-metamodel-profile") == "alpha.v1"
        # Response.model still reports the brand (not alpha.v1).
        assert r.json()["model"] == "Synthwave-1"
    finally:
        set_upstream_transport(None)


def test_brand_resolves_case_insensitive(brand_config) -> None:
    """`synthwave-1`, `SYNTHWAVE-1`, `SyNtHwAvE-1` all resolve."""
    set_upstream_transport(httpx.MockTransport(_ok_chat_handler))
    try:
        for variant in ["synthwave-1", "SYNTHWAVE-1", "SyNtHwAvE-1"]:
            r = _client().post(
                "/v1/chat/completions",
                json={
                    "model": variant,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert r.status_code == 200, f"{variant} failed: {r.text}"
            assert r.headers.get("x-metamodel-profile") == "alpha.v1"
    finally:
        set_upstream_transport(None)


# ── /v1/models lists the brand ──────────────────────────────────────


def test_models_endpoint_features_brand_first(brand_config) -> None:
    r = _client().get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    ids = [m["id"] for m in data]
    # Brand is the FIRST entry.
    assert ids[0] == "Synthwave-1"
    # Profiles still listed afterward.
    assert "alpha.v1" in ids
    assert "beta.v1" in ids
    # Brand entry has no `alias_of` (it's not an alias of a specific
    # profile in the F4-A sense; resolution goes through F10's
    # dedicated path).
    brand_entry = data[0]
    assert "alias_of" not in brand_entry
    # Capabilities mirror the brand-target profile (alpha.v1 is the
    # first callable; both alpha and the brand should report the same
    # max_model_len + capability shape).
    alpha_entry = next(m for m in data if m["id"] == "alpha.v1")
    assert brand_entry["max_model_len"] == alpha_entry["max_model_len"]
    assert brand_entry["capabilities"] == alpha_entry["capabilities"]


def test_models_endpoint_unchanged_when_brand_unset(no_brand_config) -> None:
    r = _client().get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    ids = [m["id"] for m in data]
    assert "Synthwave-1" not in ids
    assert "alpha.v1" in ids


# ── Streaming SSE: model field on every chunk ───────────────────────


def test_streaming_chat_chunks_use_brand(brand_config) -> None:
    """Streaming SSE: every `data: {...}` chunk's `model` field is
    the brand. Walk the SSE bytes and assert."""
    set_upstream_transport(httpx.MockTransport(_ok_chat_handler))
    try:
        with _client().stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "alpha.v1",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        ) as r:
            assert r.status_code == 200
            body_bytes = b"".join(r.iter_bytes())
        body_text = body_bytes.decode()
        # Parse each `data: {...}` line as JSON; assert model == brand.
        chunks_seen = 0
        for line in body_text.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):].strip()
            if payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if "model" in chunk:
                assert chunk["model"] == "Synthwave-1", (
                    f"chunk has model={chunk['model']!r}, expected Synthwave-1"
                )
                chunks_seen += 1
        assert chunks_seen >= 1, f"no SSE chunks parsed; raw body: {body_text[:300]!r}"
    finally:
        set_upstream_transport(None)


# ── Legacy /v1/completions inherits via chat-side rewrite ────────────


def test_legacy_completions_uses_brand(brand_config) -> None:
    """F8 reshape preserves the model field from chat_resp; chat
    already wrote the brand, so legacy inherits it."""
    set_upstream_transport(httpx.MockTransport(_ok_chat_handler))
    try:
        r = _client().post(
            "/v1/completions",
            json={"model": "alpha.v1", "prompt": "hi", "max_tokens": 5},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "Synthwave-1"
        assert body["object"] == "text_completion"  # F8 still working
    finally:
        set_upstream_transport(None)


# ── /v1/responses uses brand ────────────────────────────────────────


def test_responses_endpoint_uses_brand(brand_config) -> None:
    set_upstream_transport(httpx.MockTransport(_ok_chat_handler))
    try:
        r = _client().post(
            "/v1/responses",
            json={"model": "alpha.v1", "input": "hi"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "Synthwave-1"
    finally:
        set_upstream_transport(None)


# ── Brand-only fleet (every profile is voting → brand 404s) ─────────


def test_brand_with_no_callable_non_voting_returns_404() -> None:
    """If [features].voting=true and the only callable profile is
    voting, the brand has nothing to route to and returns the typed
    `model_not_found` envelope."""
    cfg = parse_config_str("""
[features]
voting = true

[server]
model_name = "Synthwave-1"

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

[profiles."voter.v1"]
type = "voting"
upstreams = ["a", "b"]
""")
    set_config(cfg)
    try:
        r = _client().post(
            "/v1/chat/completions",
            json={
                "model": "Synthwave-1",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 404, r.text
        body = r.json()
        assert body["error"]["code"] == "model_not_found"
    finally:
        set_config(None)
