"""F4-core — endpoint integration tests for /openapi.json, /health
alias, /metrics alias, /v1/completions, /tokenize."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from meta_model.config import parse_config_str
from meta_model.server import app, set_config, set_upstream_transport


_FIXTURE = """
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

[upstreams.same]
model_id = "shared"
base_url = "http://same.local/v1"
context = 8192
max_output = 512

[upstreams.same2]
model_id = "shared"
base_url = "http://same2.local/v1"
context = 8192
max_output = 512

[profiles."plain.v1"]
type = "moa"
generators = ["a", "b"]
synthesizer = "a"

[profiles."single.v1"]
type = "moa"
generators = ["a"]
synthesizer = "a"

[profiles."pinned.v1"]
type = "moa"
generators = ["a", "b"]
synthesizer = "a"
tokenizer_upstream = "b"

[profiles."homogeneous.v1"]
type = "moa"
generators = ["same", "same2"]
synthesizer = "same"

# F9: cascade profile for first-upstream tokenizer fallback test.
[profiles."cascade.v1"]
type = "cascade"
upstreams = ["a", "b"]
"""


@pytest.fixture
def loaded_config():
    cfg = parse_config_str(_FIXTURE)
    set_config(cfg)
    yield cfg
    set_config(None)


def _client() -> TestClient:
    return TestClient(app)


# ── /openapi.json ────────────────────────────────────────────────────


def test_openapi_json_is_served() -> None:
    """F4-core: previously suppressed via openapi_url=None. Now exposed
    so tooling can introspect the API surface."""
    r = _client().get("/openapi.json")
    assert r.status_code == 200
    body = r.json()
    assert body["openapi"].startswith("3.")
    paths = body.get("paths", {})
    assert "/v1/chat/completions" in paths
    assert "/v1/health" in paths
    assert "/health" in paths
    assert "/v1/completions" in paths
    assert "/tokenize" in paths


# ── /health alias ───────────────────────────────────────────────────


def test_health_root_alias_same_as_v1_health() -> None:
    """F4-core: root-level /health alias must return the same body
    shape as /v1/health (review r1 F4 may flag drift if they diverge)."""
    if hasattr(app.state, "upstream_health"):
        app.state.upstream_health = None
    set_config(None)
    a = _client().get("/v1/health")
    b = _client().get("/health")
    assert a.status_code == b.status_code
    assert a.json()["status"] == b.json()["status"]


# ── /metrics alias ──────────────────────────────────────────────────


def test_metrics_root_alias_same_as_v1() -> None:
    a = _client().get("/v1/metrics/moa")
    b = _client().get("/metrics")
    assert a.status_code == 200 and b.status_code == 200
    # Same JSON shape — both call aggregate_metrics().
    assert a.json().keys() == b.json().keys()


# ── /v1/completions ─────────────────────────────────────────────────


def _ok_chat_handler(req: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "ma",
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


def test_completions_translates_prompt_to_chat(loaded_config) -> None:
    """Legacy `prompt` field → user message → chat completions.

    Review r1 F4-core LOW: assert the upstream actually saw the
    rewritten messages (not just that the response passes through).

    F8: also asserts the response wire shape is the legacy
    text_completion envelope (object/id/choices[i].text/logprobs),
    not the chat-completion envelope. Drop-in clients targeting
    OpenAI's classic /v1/completions parse `.text` and would crash
    on a chat-shape body."""
    seen_bodies: list[dict] = []

    def capturing_handler(req: httpx.Request) -> httpx.Response:
        import json

        seen_bodies.append(json.loads(req.content))
        return _ok_chat_handler(req)

    set_upstream_transport(httpx.MockTransport(capturing_handler))
    try:
        r = _client().post(
            "/v1/completions",
            json={"model": "single.v1", "prompt": "hello", "max_tokens": 5},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # F8: legacy wire shape, not chat shape.
        assert body["object"] == "text_completion"
        assert body["id"].startswith("cmpl-"), body["id"]
        assert "message" not in body["choices"][0]
        assert body["choices"][0]["text"] == "ok"
        assert "logprobs" in body["choices"][0]
        assert body["choices"][0]["finish_reason"] == "stop"
        # Upstream saw rewritten chat shape — prompt removed, messages added.
        assert len(seen_bodies) >= 1
        upstream_body = seen_bodies[-1]
        assert "prompt" not in upstream_body
        assert upstream_body["messages"] == [
            {"role": "user", "content": "hello"}
        ]
    finally:
        set_upstream_transport(None)


def test_completions_content_length_matches_body(loaded_config) -> None:
    """Review r1 F8 HIGH regression. The legacy reshape changes the body
    length; Starlette only auto-computes Content-Length when the header
    is absent. Forwarding `content-length` from the chat response would
    mis-frame the wire (broken keep-alive on HTTP/1.1, truncated reads
    on strict clients). Strip the header before reconstructing."""
    set_upstream_transport(httpx.MockTransport(_ok_chat_handler))
    try:
        r = _client().post(
            "/v1/completions",
            json={"model": "single.v1", "prompt": "hello", "max_tokens": 5},
        )
        assert r.status_code == 200, r.text
        # TestClient surfaces the rendered body bytes via .content.
        assert "content-length" in r.headers
        assert int(r.headers["content-length"]) == len(r.content), (
            f"declared content-length={r.headers['content-length']} "
            f"vs actual body length={len(r.content)}"
        )
    finally:
        set_upstream_transport(None)


def test_completions_invalid_chat_body_returns_400(loaded_config) -> None:
    """Review r1 F4-core HIGH regression: in-process forward must NOT
    let pydantic ValidationError become a 500. Field validation runs
    before dispatch."""
    r = _client().post(
        "/v1/completions",
        json={"model": "single.v1", "prompt": "hello", "max_tokens": 0},
    )
    # max_tokens must be >0 per ChatRequest schema → 400 typed envelope.
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"


def test_completions_x_meta_model_profile_uses_typed_resolver(
    loaded_config,
) -> None:
    """Review r1 F4-core MED: `x_meta_model.profile` override must
    flow through `routing.resolve_profile` so unknown values surface
    the typed F4 envelope (not the chat-path 404 shape)."""
    r = _client().post(
        "/v1/completions",
        json={
            "model": "single.v1",
            "prompt": ".",
            "x_meta_model": {"profile": "ghost.v1"},
        },
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["code"] == "model_not_found"
    assert body["error"]["param"] == "model"


def test_completions_unknown_model_returns_404(loaded_config) -> None:
    r = _client().post(
        "/v1/completions",
        json={"model": "nope", "prompt": "."},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "model"


def test_completions_array_prompt_returns_501(loaded_config) -> None:
    r = _client().post(
        "/v1/completions",
        json={"model": "single.v1", "prompt": ["one", "two"]},
    )
    assert r.status_code == 501
    body = r.json()
    assert body["error"]["code"] == "unsupported_legacy_param"
    assert body["error"]["param"] == "prompt"


@pytest.mark.parametrize(
    "field",
    ["echo", "suffix", "best_of", "logprobs", "response_format",
     "max_completion_tokens", "stream_options"],
)
def test_completions_unsupported_param_returns_501(
    loaded_config, field: str
) -> None:
    r = _client().post(
        "/v1/completions",
        json={"model": "single.v1", "prompt": ".", field: "anything"},
    )
    assert r.status_code == 501
    body = r.json()
    assert body["error"]["code"] == "unsupported_legacy_param"
    assert body["error"]["param"] == field


def test_completions_n_gt_1_returns_501(loaded_config) -> None:
    r = _client().post(
        "/v1/completions",
        json={"model": "single.v1", "prompt": ".", "n": 2},
    )
    assert r.status_code == 501
    body = r.json()
    assert body["error"]["code"] == "unsupported_legacy_param"
    assert body["error"]["param"] == "n"


def test_completions_missing_prompt_returns_400(loaded_config) -> None:
    r = _client().post(
        "/v1/completions",
        json={"model": "single.v1"},
    )
    assert r.status_code == 400


# ── /tokenize ───────────────────────────────────────────────────────


def _ok_tokenize_handler(req: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"tokens": [1, 2, 3], "count": 3, "max_model_len": 8192},
    )


def test_tokenize_routes_to_explicit_tokenizer_upstream(
    loaded_config,
) -> None:
    """`pinned.v1` has tokenizer_upstream=b. Probe must hit b.local."""
    seen_urls: list[str] = []

    def capturing_handler(req: httpx.Request) -> httpx.Response:
        seen_urls.append(str(req.url))
        return _ok_tokenize_handler(req)

    set_upstream_transport(httpx.MockTransport(capturing_handler))
    try:
        r = _client().post(
            "/tokenize",
            json={"model": "pinned.v1", "prompt": "hello"},
        )
        assert r.status_code == 200
        # Path: /v1 stripped, /tokenize appended at root.
        assert any("b.local/tokenize" in u for u in seen_urls)
    finally:
        set_upstream_transport(None)


def test_tokenize_homogeneous_fleet_picks_first_silently(
    loaded_config,
) -> None:
    """Both `same` and `same2` have model_id='shared'. /tokenize must
    succeed without an explicit override."""
    set_upstream_transport(httpx.MockTransport(_ok_tokenize_handler))
    try:
        r = _client().post(
            "/tokenize",
            json={"model": "homogeneous.v1", "prompt": "hello"},
        )
        assert r.status_code == 200
        assert r.json()["count"] == 3
    finally:
        set_upstream_transport(None)


def test_tokenize_moa_falls_back_to_synthesizer(loaded_config) -> None:
    """F9: MoA profile with heterogeneous generators no longer 409s —
    falls back to the synthesizer's tokenizer (the model whose context
    governs F1's advertised max_model_len and produces the wire response).

    plain.v1 has generators a (ma) + b (mb) and synthesizer=a. Probe must
    hit a.local with a's model_id rewritten in the body, not 409."""
    seen_urls: list[str] = []
    seen_bodies: list[dict] = []

    def capturing_handler(req: httpx.Request) -> httpx.Response:
        import json as _json

        seen_urls.append(str(req.url))
        seen_bodies.append(_json.loads(req.content))
        return _ok_tokenize_handler(req)

    set_upstream_transport(httpx.MockTransport(capturing_handler))
    try:
        r = _client().post(
            "/tokenize",
            json={"model": "plain.v1", "prompt": "hello"},
        )
        assert r.status_code == 200, r.text
        # Synth (a) was picked. Probe hit a.local; body carries a's model_id.
        assert any("a.local/tokenize" in u for u in seen_urls)
        assert seen_bodies[-1]["model"] == "ma"
        # Header surfaces the resolved upstream so clients can verify.
        assert r.headers.get("x-metamodel-tokenizer-upstream") == "a"
    finally:
        set_upstream_transport(None)


def test_tokenize_cascade_falls_back_to_first_upstream(loaded_config) -> None:
    """F9: Cascade profile falls back to the first upstream's tokenizer
    (the one always tried first; if it serves, that's what the client got).

    cascade.v1 has upstreams=[a, b]; probe must hit a.local."""
    seen_urls: list[str] = []

    def capturing_handler(req: httpx.Request) -> httpx.Response:
        seen_urls.append(str(req.url))
        return _ok_tokenize_handler(req)

    set_upstream_transport(httpx.MockTransport(capturing_handler))
    try:
        r = _client().post(
            "/tokenize",
            json={"model": "cascade.v1", "prompt": "hello"},
        )
        assert r.status_code == 200, r.text
        assert any("a.local/tokenize" in u for u in seen_urls)
        assert r.headers.get("x-metamodel-tokenizer-upstream") == "a"
    finally:
        set_upstream_transport(None)


_VOTING_FIXTURE = """
[features]
voting = true

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

[profiles."voter.v1"]
type = "voting"
upstreams = ["a", "b"]
"""


def test_tokenize_voting_profile_still_409s() -> None:
    """F9: voting profile has no principled tokenizer default — every
    voter contributes to a YES/NO consensus, no single one is "the"
    tokenizer. Confirms the 409 fallback survives for voting and
    operators must explicitly pin `tokenizer_upstream`."""
    cfg = parse_config_str(_VOTING_FIXTURE)
    set_config(cfg)
    try:
        r = _client().post(
            "/tokenize",
            json={"model": "voter.v1", "prompt": "hello"},
        )
        assert r.status_code == 409, r.text
        body = r.json()
        assert body["error"]["code"] == "heterogeneous_tokenizer"
        assert "tokenizer_upstream" in body["error"]["message"]
    finally:
        set_config(None)


def test_tokenize_response_carries_upstream_header(loaded_config) -> None:
    """F9: every successful /tokenize response includes
    `X-MetaModel-Tokenizer-Upstream` so clients counting tokens across
    calls can verify the picked upstream is stable and matches their
    budget assumptions."""
    set_upstream_transport(httpx.MockTransport(_ok_tokenize_handler))
    try:
        # Explicit pin: pinned.v1 has tokenizer_upstream="b".
        r = _client().post(
            "/tokenize",
            json={"model": "pinned.v1", "prompt": "hi"},
        )
        assert r.status_code == 200
        assert r.headers.get("x-metamodel-tokenizer-upstream") == "b"

        # Implicit single-upstream: single.v1 → "a".
        r = _client().post(
            "/tokenize",
            json={"model": "single.v1", "prompt": "hi"},
        )
        assert r.status_code == 200
        assert r.headers.get("x-metamodel-tokenizer-upstream") == "a"
    finally:
        set_upstream_transport(None)


def test_tokenize_single_upstream_works_without_override(
    loaded_config,
) -> None:
    set_upstream_transport(httpx.MockTransport(_ok_tokenize_handler))
    try:
        r = _client().post(
            "/tokenize",
            json={"model": "single.v1", "prompt": "hello"},
        )
        assert r.status_code == 200
    finally:
        set_upstream_transport(None)


def test_tokenize_unknown_model_returns_404(loaded_config) -> None:
    r = _client().post(
        "/tokenize",
        json={"model": "nope", "prompt": "."},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"


def test_tokenize_raw_upstream_routes_directly(loaded_config) -> None:
    """Raw upstream addressing: model='a' bypasses profile lookup
    (synthetic single-element MoA from resolve_profile)."""
    seen_urls: list[str] = []

    def capturing_handler(req: httpx.Request) -> httpx.Response:
        seen_urls.append(str(req.url))
        return _ok_tokenize_handler(req)

    set_upstream_transport(httpx.MockTransport(capturing_handler))
    try:
        r = _client().post(
            "/tokenize",
            json={"model": "a", "prompt": "."},
        )
        assert r.status_code == 200
        assert any("a.local/tokenize" in u for u in seen_urls)
    finally:
        set_upstream_transport(None)


def test_tokenize_missing_model_returns_400(loaded_config) -> None:
    r = _client().post(
        "/tokenize",
        json={"prompt": "."},
    )
    assert r.status_code == 400


def test_tokenize_invalid_tokenizer_upstream_rejected_at_config_load() -> None:
    """tokenizer_upstream typo → config load fails."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc:
        parse_config_str(
            _FIXTURE
            + """
[profiles."typo.v1"]
type = "moa"
generators = ["a", "b"]
synthesizer = "a"
tokenizer_upstream = "ghost"
"""
        )
    assert "tokenizer_upstream" in str(exc.value).lower()


# ── F4-A: alias resolution across all 3 endpoints ───────────────────


_ALIAS_FIXTURE = _FIXTURE + """
[profiles."aliased.v1"]
type = "moa"
generators = ["a"]
synthesizer = "a"
aliases = ["fast", "MODEL-4"]
"""


@pytest.fixture
def aliased_config():
    cfg = parse_config_str(_ALIAS_FIXTURE)
    set_config(cfg)
    yield cfg
    set_config(None)


def test_chat_completions_resolves_alias(aliased_config) -> None:
    """Verdict (f-2): /v1/chat/completions accepts an alias as `model`."""
    set_upstream_transport(httpx.MockTransport(_ok_chat_handler))
    try:
        r = _client().post(
            "/v1/chat/completions",
            json={"model": "MODEL-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 200, r.text
        # Profile id reported in headers reflects the canonical name.
        assert r.headers.get("x-metamodel-profile") == "aliased.v1"
    finally:
        set_upstream_transport(None)


def test_completions_resolves_alias(aliased_config) -> None:
    """Verdict (f-2): /v1/completions accepts an alias (case-insensitive)."""
    set_upstream_transport(httpx.MockTransport(_ok_chat_handler))
    try:
        r = _client().post(
            "/v1/completions",
            json={"model": "fast", "prompt": "hi", "max_tokens": 5},
        )
        assert r.status_code == 200, r.text
    finally:
        set_upstream_transport(None)


def test_responses_resolves_alias(aliased_config) -> None:
    """Verdict (f-2): /v1/responses accepts an alias (case-insensitive).

    The Responses envelope echoes the model field the client supplied —
    standard OpenAI convention. Resolution is observable via
    successful dispatch (200) rather than by rewriting the response
    `model`.
    """
    set_upstream_transport(httpx.MockTransport(_ok_chat_handler))
    try:
        r = _client().post(
            "/v1/responses",
            json={"model": "FAST", "input": "hi", "max_output_tokens": 5},
        )
        assert r.status_code == 200, r.text
    finally:
        set_upstream_transport(None)


def test_unknown_alias_returns_typed_404_on_completions(aliased_config) -> None:
    r = _client().post(
        "/v1/completions",
        json={"model": "ghost-alias", "prompt": "hi", "max_tokens": 5},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"
    assert body["error"]["type"] == "invalid_request_error"


def test_models_lists_aliases_with_alias_of(aliased_config) -> None:
    """F4-A: each alias appears as its own /v1/models entry, marked
    with `alias_of` pointing at the canonical profile."""
    r = _client().get("/v1/models")
    assert r.status_code == 200
    data = r.json()["data"]
    by_id = {m["id"]: m for m in data}
    # Canonical entry has no alias_of marker.
    assert "aliased.v1" in by_id
    assert "alias_of" not in by_id["aliased.v1"]
    # Alias entries marked, casing preserved.
    assert "fast" in by_id
    assert by_id["fast"]["alias_of"] == "aliased.v1"
    assert "MODEL-4" in by_id
    assert by_id["MODEL-4"]["alias_of"] == "aliased.v1"
    # Capability surface mirrors the canonical entry.
    assert by_id["fast"]["capabilities"] == by_id["aliased.v1"]["capabilities"]
    assert by_id["fast"]["max_model_len"] == by_id["aliased.v1"]["max_model_len"]


def test_models_omits_aliases_of_hidden_voting_profiles() -> None:
    """Alias entries respect callable_profiles — aliases of voting
    profiles are hidden when [features].voting=false."""
    cfg = parse_config_str("""
[features]
voting = false

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
aliases = ["fast"]

[profiles."vote.v1"]
type = "voting"
upstreams = ["a", "b"]
aliases = ["consensus"]
""")
    set_config(cfg)
    try:
        ids = [m["id"] for m in _client().get("/v1/models").json()["data"]]
        # vote.v1 + its alias both hidden; plain.v1 + its alias both shown.
        assert "vote.v1" not in ids
        assert "consensus" not in ids
        assert "plain.v1" in ids
        assert "fast" in ids
    finally:
        set_config(None)
