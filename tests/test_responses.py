"""F4-Responses — tests for /v1/responses adapter (request/response
mappers, tool_choice, streaming event taxonomy, typed-501 advisories)."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from meta_model.config import parse_config_str
from meta_model.responses import (
    ResponsesAdapterError,
    chat_to_responses,
    new_response_id,
    responses_to_chat,
    stream_responses_events,
)
from meta_model.server import app, set_config, set_upstream_transport


_FIXTURE = """
[upstreams.a]
model_id = "ma"
base_url = "http://a.local/v1"
context = 8192
max_output = 512

[profiles."single.v1"]
type = "moa"
generators = ["a"]
synthesizer = "a"
"""


@pytest.fixture
def loaded_config():
    cfg = parse_config_str(_FIXTURE)
    set_config(cfg)
    yield cfg
    set_config(None)


def _client() -> TestClient:
    return TestClient(app)


def _ok_chat(message_content: str = "hello world", *, tool_calls=None):
    msg: dict = {"role": "assistant"}
    if tool_calls is not None:
        msg["content"] = None
        msg["tool_calls"] = tool_calls
    else:
        msg["content"] = message_content
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "ma",
        "choices": [
            {"index": 0, "message": msg, "finish_reason": "stop"},
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        },
    }


# ── Pure request mapper ─────────────────────────────────────────────


def test_responses_to_chat_string_input() -> None:
    chat = responses_to_chat({"model": "single.v1", "input": "hello"})
    assert chat["model"] == "single.v1"
    assert chat["messages"] == [{"role": "user", "content": "hello"}]


def test_responses_to_chat_instructions_prepended() -> None:
    chat = responses_to_chat(
        {"model": "single.v1", "instructions": "be terse", "input": "hi"}
    )
    assert chat["messages"][0] == {"role": "system", "content": "be terse"}
    assert chat["messages"][1] == {"role": "user", "content": "hi"}


def test_responses_to_chat_message_items() -> None:
    chat = responses_to_chat(
        {
            "model": "single.v1",
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "hello"}]},
            ],
        }
    )
    assert chat["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_responses_to_chat_input_image_maps_to_image_url() -> None:
    chat = responses_to_chat(
        {
            "model": "single.v1",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "what is this?"},
                    {"type": "input_image",
                     "image_url": "http://img.example/cat.png"},
                ]},
            ],
        }
    )
    msg = chat["messages"][0]
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    assert msg["content"][0] == {"type": "text", "text": "what is this?"}
    assert msg["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "http://img.example/cat.png"},
    }


def test_responses_to_chat_function_call_input() -> None:
    chat = responses_to_chat(
        {
            "model": "single.v1",
            "input": [
                {"type": "function_call", "call_id": "call_1",
                 "name": "lookup", "arguments": '{"q":"x"}'},
            ],
        }
    )
    msg = chat["messages"][0]
    assert msg["role"] == "assistant"
    assert msg["content"] is None
    assert msg["tool_calls"] == [
        {"id": "call_1", "type": "function",
         "function": {"name": "lookup", "arguments": '{"q":"x"}'}},
    ]


def test_responses_to_chat_function_call_output_str() -> None:
    chat = responses_to_chat(
        {
            "model": "single.v1",
            "input": [
                {"type": "function_call_output", "call_id": "call_1",
                 "output": "result"},
            ],
        }
    )
    assert chat["messages"][0] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "result",
    }


def test_responses_to_chat_function_call_output_input_text_array() -> None:
    """Agents-SDK convention: tool output as array of input_text parts."""
    chat = responses_to_chat(
        {
            "model": "single.v1",
            "input": [
                {"type": "function_call_output", "call_id": "call_1",
                 "output": [
                    {"type": "input_text", "text": "result "},
                    {"type": "input_text", "text": "concat"},
                 ]},
            ],
        }
    )
    assert chat["messages"][0]["content"] == "result concat"


def test_responses_to_chat_function_call_output_output_text_compat() -> None:
    """`output_text` array (compat kindness, not canonical)."""
    chat = responses_to_chat(
        {
            "model": "single.v1",
            "input": [
                {"type": "function_call_output", "call_id": "call_1",
                 "output": [{"type": "output_text", "text": "compat"}]},
            ],
        }
    )
    assert chat["messages"][0]["content"] == "compat"


def test_responses_to_chat_tool_choice_string() -> None:
    chat = responses_to_chat(
        {"model": "single.v1", "input": "x", "tool_choice": "required"}
    )
    assert chat["tool_choice"] == "required"


def test_responses_to_chat_tool_choice_function() -> None:
    chat = responses_to_chat(
        {
            "model": "single.v1",
            "input": "x",
            "tool_choice": {"type": "function", "name": "lookup"},
        }
    )
    assert chat["tool_choice"] == {
        "type": "function",
        "function": {"name": "lookup"},
    }


def test_responses_to_chat_tools_function_nested_passthrough() -> None:
    """Already-nested chat-shape tool (compat client) passes through."""
    tool = {"type": "function", "function": {"name": "lookup", "parameters": {}}}
    chat = responses_to_chat(
        {"model": "single.v1", "input": "x", "tools": [tool]}
    )
    assert chat["tools"] == [tool]


def test_responses_to_chat_tools_function_flat_shape_rewritten() -> None:
    """Review r1 F4-Responses HIGH: Responses uses FLAT function tool
    shape `{type, name, parameters}`; chat needs nested
    `{type:"function", function:{...}}`. Adapter must rewrite."""
    flat = {
        "type": "function",
        "name": "lookup",
        "description": "look something up",
        "parameters": {"type": "object", "properties": {}},
    }
    chat = responses_to_chat(
        {"model": "single.v1", "input": "x", "tools": [flat]}
    )
    assert chat["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "look something up",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_responses_to_chat_tools_function_flat_missing_name_400() -> None:
    """Flat tool without `name` — typed 400, not silent dispatch failure."""
    with pytest.raises(ResponsesAdapterError) as exc:
        responses_to_chat(
            {
                "model": "single.v1",
                "input": "x",
                "tools": [{"type": "function", "parameters": {}}],
            }
        )
    assert exc.value.status_code == 400
    assert exc.value.code == "invalid_tools"


def test_responses_to_chat_max_output_tokens_maps() -> None:
    chat = responses_to_chat(
        {"model": "single.v1", "input": "x", "max_output_tokens": 64}
    )
    assert chat["max_tokens"] == 64
    assert "max_output_tokens" not in chat


def test_responses_to_chat_strips_stream_flag() -> None:
    chat = responses_to_chat(
        {"model": "single.v1", "input": "x", "stream": True}
    )
    # Stream=False inside the chat body so dispatch runs non-streaming.
    assert chat["stream"] is False


# ── Typed 501 advisories ────────────────────────────────────────────


@pytest.mark.parametrize(
    "field,expected_param,truthy_value",
    [
        ("previous_response_id", "previous_response_id", "resp_old"),
        ("background", "background", True),
        # Review r1 F4-Responses MED: official field name is
        # `conversation` (singular), not `conversations`.
        ("conversation", "conversation", "conv_x"),
    ],
)
def test_responses_to_chat_unsupported_top_level(
    field: str, expected_param: str, truthy_value
) -> None:
    with pytest.raises(ResponsesAdapterError) as exc:
        responses_to_chat({"model": "single.v1", "input": "x", field: truthy_value})
    assert exc.value.status_code == 501
    assert exc.value.code == "unsupported_responses_param"
    assert exc.value.param == expected_param


@pytest.mark.parametrize(
    "field,falsy_value",
    [
        ("previous_response_id", None),
        ("background", False),
        ("conversation", None),
    ],
)
def test_responses_to_chat_falsy_top_level_passes_through(
    field: str, falsy_value
) -> None:
    """Review r1 F4-Responses MED: clients send fields with default
    null/false values as part of the request shape; presence alone
    must not 501."""
    chat = responses_to_chat(
        {"model": "single.v1", "input": "x", field: falsy_value}
    )
    assert chat["messages"] == [{"role": "user", "content": "x"}]


def test_responses_to_chat_text_format_json_schema_returns_501() -> None:
    """Review r1 F4-Responses MED: structured outputs via `text.format`
    are not implemented — clients silently getting plain-text back is
    worse than a typed 501 advisory."""
    with pytest.raises(ResponsesAdapterError) as exc:
        responses_to_chat(
            {
                "model": "single.v1",
                "input": "x",
                "text": {"format": {"type": "json_schema", "schema": {}}},
            }
        )
    assert exc.value.status_code == 501
    assert exc.value.code == "unsupported_text_format"
    assert exc.value.param == "text.format"


def test_responses_to_chat_text_format_text_passes_through() -> None:
    """Plain `text.format.type=text` is a no-op."""
    chat = responses_to_chat(
        {
            "model": "single.v1",
            "input": "x",
            "text": {"format": {"type": "text"}},
        }
    )
    assert chat["messages"] == [{"role": "user", "content": "x"}]


@pytest.mark.parametrize(
    "ptype",
    ["input_file", "input_audio", "input_video",
     "computer_call", "web_search_call", "file_search_call"],
)
def test_responses_to_chat_unsupported_input_part_type(ptype: str) -> None:
    with pytest.raises(ResponsesAdapterError) as exc:
        responses_to_chat(
            {
                "model": "single.v1",
                "input": [
                    {"type": "message", "role": "user", "content": [
                        {"type": ptype, "data": "x"},
                    ]},
                ],
            }
        )
    assert exc.value.status_code == 501


def test_responses_to_chat_unsupported_function_call_output_part() -> None:
    with pytest.raises(ResponsesAdapterError) as exc:
        responses_to_chat(
            {
                "model": "single.v1",
                "input": [
                    {"type": "function_call_output", "call_id": "c",
                     "output": [{"type": "input_image", "image_url": "u"}]},
                ],
            }
        )
    assert exc.value.status_code == 501
    assert exc.value.code == "unsupported_function_call_output_part"


def test_responses_to_chat_unsupported_tool_choice_allowed_tools() -> None:
    with pytest.raises(ResponsesAdapterError) as exc:
        responses_to_chat(
            {
                "model": "single.v1",
                "input": "x",
                "tool_choice": {"type": "allowed_tools", "tools": []},
            }
        )
    assert exc.value.status_code == 501
    assert exc.value.code == "unsupported_tool_choice"


def test_responses_to_chat_unsupported_builtin_tool() -> None:
    with pytest.raises(ResponsesAdapterError) as exc:
        responses_to_chat(
            {
                "model": "single.v1",
                "input": "x",
                "tools": [{"type": "web_search"}],
            }
        )
    assert exc.value.status_code == 501
    assert exc.value.code == "unsupported_tool"


# ── Pure response mapper ────────────────────────────────────────────


def test_chat_to_responses_text_message() -> None:
    chat = _ok_chat("hello world")
    body = chat_to_responses(chat, response_id="resp_x", model_name="single.v1")
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["model"] == "single.v1"
    assert body["output_text"] == "hello world"
    assert len(body["output"]) == 1
    item = body["output"][0]
    assert item["type"] == "message"
    assert item["role"] == "assistant"
    assert item["content"][0]["type"] == "output_text"
    assert item["content"][0]["text"] == "hello world"


def test_chat_to_responses_function_call() -> None:
    chat = _ok_chat(tool_calls=[
        {"id": "call_42", "type": "function",
         "function": {"name": "lookup", "arguments": '{"q":"x"}'}},
    ])
    body = chat_to_responses(chat, response_id="resp_x", model_name="single.v1")
    assert body["output_text"] == ""
    assert len(body["output"]) == 1
    item = body["output"][0]
    assert item["type"] == "function_call"
    assert item["call_id"] == "call_42"
    assert item["name"] == "lookup"
    assert item["arguments"] == '{"q":"x"}'


def test_chat_to_responses_text_plus_function_call() -> None:
    chat = _ok_chat()
    chat["choices"][0]["message"] = {
        "role": "assistant",
        "content": "let me check",
        "tool_calls": [
            {"id": "c", "type": "function",
             "function": {"name": "x", "arguments": "{}"}},
        ],
    }
    body = chat_to_responses(chat, response_id="resp_x", model_name="single.v1")
    assert len(body["output"]) == 2
    assert body["output"][0]["type"] == "message"
    assert body["output"][1]["type"] == "function_call"
    assert body["output_text"] == "let me check"


def test_chat_to_responses_finish_reason_length_marks_incomplete() -> None:
    """Review r1 F4-Responses MED: chat finish_reason `length` maps to
    Responses-spec `max_output_tokens`, not raw `length`."""
    chat = _ok_chat()
    chat["choices"][0]["finish_reason"] = "length"
    body = chat_to_responses(chat, response_id="resp_x", model_name="single.v1")
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}


def test_chat_to_responses_finish_reason_content_filter_preserved() -> None:
    chat = _ok_chat()
    chat["choices"][0]["finish_reason"] = "content_filter"
    body = chat_to_responses(chat, response_id="resp_x", model_name="single.v1")
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "content_filter"}


def test_chat_to_responses_usage_translated() -> None:
    chat = _ok_chat()
    body = chat_to_responses(chat, response_id="resp_x", model_name="single.v1")
    assert body["usage"] == {
        "input_tokens": 5,
        "output_tokens": 7,
        "total_tokens": 12,
    }


# ── Streaming event taxonomy ────────────────────────────────────────


def _events(response_body, **kwargs):
    return list(stream_responses_events(response_body, **kwargs))


def test_stream_events_text_message_sequence() -> None:
    body = chat_to_responses(_ok_chat("hello world"),
                             response_id="resp_x", model_name="single.v1")
    events = _events(body, chunk_size=4)
    types = [e["event"] for e in events]
    # Required structure for a single text message:
    assert types[0] == "response.created"
    assert types[1] == "response.in_progress"
    assert types[2] == "response.output_item.added"
    assert types[3] == "response.content_part.added"
    # Multiple deltas (chunk_size=4 over "hello world" = 11 chars → 3 chunks)
    delta_count = sum(1 for t in types if t == "response.output_text.delta")
    assert delta_count == 3
    assert "response.output_text.done" in types
    assert "response.content_part.done" in types
    assert "response.output_item.done" in types
    assert types[-1] == "response.completed"
    # Sequence numbers are monotonically increasing.
    seqs = [e["data"]["sequence_number"] for e in events]
    assert seqs == sorted(seqs)


def test_stream_events_function_call_uses_function_call_arguments_delta() -> None:
    chat = _ok_chat(tool_calls=[
        {"id": "c", "type": "function",
         "function": {"name": "lookup",
                      "arguments": '{"long":"argument string"}'}},
    ])
    body = chat_to_responses(chat, response_id="resp_x", model_name="single.v1")
    events = _events(body, chunk_size=8)
    types = [e["event"] for e in events]
    # Critical: must use function_call_arguments.delta (NOT
    # function_call.arguments.delta — review r1 anticipated trap).
    assert "response.function_call_arguments.delta" in types
    assert "response.function_call_arguments.done" in types
    # Output_text events MUST NOT appear for function_call items.
    assert "response.output_text.delta" not in types


def test_stream_events_completed_has_usage() -> None:
    body = chat_to_responses(_ok_chat("hi"),
                             response_id="resp_x", model_name="single.v1")
    events = _events(body)
    completed = [e for e in events if e["event"] == "response.completed"][0]
    assert completed["data"]["response"]["usage"]["output_tokens"] == 7


def test_stream_events_created_envelope_is_in_progress_shape() -> None:
    """Review r1 F4-Responses HIGH: created/in_progress must NOT
    expose final output / usage. Those land only in the terminal
    completed event and the per-item .done events."""
    body = chat_to_responses(_ok_chat("hello"),
                             response_id="resp_x", model_name="single.v1")
    events = _events(body)
    created = events[0]
    assert created["event"] == "response.created"
    env = created["data"]["response"]
    assert env["status"] == "in_progress"
    assert env["output"] == []
    assert env["output_text"] == ""
    assert env["usage"] is None


def test_stream_events_output_item_added_carries_in_progress_shape() -> None:
    """Review r1 F4-Responses HIGH: output_item.added must show the
    item with empty content / arguments — the deltas haven't shipped
    yet. Final shape lands in output_item.done."""
    body = chat_to_responses(_ok_chat("hello world"),
                             response_id="resp_x", model_name="single.v1")
    events = _events(body, chunk_size=4)
    added = [e for e in events if e["event"] == "response.output_item.added"][0]
    item = added["data"]["item"]
    assert item["status"] == "in_progress"
    assert item["content"] == []
    done = [e for e in events if e["event"] == "response.output_item.done"][0]
    assert done["data"]["item"]["status"] == "completed"
    assert done["data"]["item"]["content"][0]["text"] == "hello world"


def test_stream_events_content_part_added_has_empty_text() -> None:
    """The added content part is empty; deltas populate it; final
    text appears only in `.done`."""
    body = chat_to_responses(_ok_chat("xyz"),
                             response_id="resp_x", model_name="single.v1")
    events = _events(body)
    added = [e for e in events if e["event"] == "response.content_part.added"][0]
    assert added["data"]["part"]["text"] == ""
    text_done = [e for e in events if e["event"] == "response.output_text.done"][0]
    assert text_done["data"]["text"] == "xyz"


def test_stream_events_function_call_added_has_empty_arguments() -> None:
    chat = _ok_chat(tool_calls=[
        {"id": "c", "type": "function",
         "function": {"name": "lookup", "arguments": '{"q":"x"}'}},
    ])
    body = chat_to_responses(chat, response_id="resp_x", model_name="single.v1")
    events = _events(body, chunk_size=8)
    added = [e for e in events if e["event"] == "response.output_item.added"][0]
    assert added["data"]["item"]["arguments"] == ""
    done = [e for e in events
            if e["event"] == "response.function_call_arguments.done"][0]
    assert done["data"]["arguments"] == '{"q":"x"}'


def test_stream_events_incomplete_emits_response_incomplete() -> None:
    """Review r1 F4-Responses MED: incomplete responses end with
    `response.incomplete`, not `response.completed`. Review r2 MED
    follow-up: in-progress envelopes must NOT leak the terminal
    `incomplete_details` reason before `response.incomplete` lands."""
    chat = _ok_chat()
    chat["choices"][0]["finish_reason"] = "length"
    body = chat_to_responses(chat, response_id="resp_x", model_name="single.v1")
    events = _events(body)
    final = events[-1]
    assert final["event"] == "response.incomplete"
    assert final["data"]["response"]["status"] == "incomplete"
    assert final["data"]["response"]["incomplete_details"]["reason"] == "max_output_tokens"
    # In-progress envelopes must NOT carry the terminal reason —
    # clients tracking event order would see "incomplete" prematurely.
    for e in events[:-1]:
        if e["event"] in ("response.created", "response.in_progress"):
            assert e["data"]["response"]["incomplete_details"] is None


def test_stream_events_sequence_numbers_contiguous() -> None:
    """Sequence numbers must be 1..N with no gaps."""
    body = chat_to_responses(_ok_chat("hi there"),
                             response_id="resp_x", model_name="single.v1")
    events = _events(body)
    seqs = [e["data"]["sequence_number"] for e in events]
    assert seqs == list(range(1, len(events) + 1))


# ── Endpoint integration ────────────────────────────────────────────


def test_endpoint_simple_text_round_trip(loaded_config) -> None:
    set_upstream_transport(httpx.MockTransport(lambda req: httpx.Response(200, json=_ok_chat("hi back"))))
    try:
        r = _client().post(
            "/v1/responses",
            json={"model": "single.v1", "input": "hi"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "response"
        assert body["output_text"] == "hi back"
    finally:
        set_upstream_transport(None)


def test_endpoint_unknown_model_returns_typed_404(loaded_config) -> None:
    r = _client().post(
        "/v1/responses",
        json={"model": "nope", "input": "."},
    )
    assert r.status_code == 404
    body = r.json()
    assert body["error"]["code"] == "model_not_found"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["param"] == "model"


def test_endpoint_previous_response_id_returns_501(loaded_config) -> None:
    r = _client().post(
        "/v1/responses",
        json={"model": "single.v1", "input": ".", "previous_response_id": "resp_old"},
    )
    assert r.status_code == 501
    body = r.json()
    assert body["error"]["code"] == "unsupported_responses_param"
    assert body["error"]["param"] == "previous_response_id"


def test_endpoint_streaming_emits_responses_event_taxonomy(loaded_config) -> None:
    set_upstream_transport(httpx.MockTransport(lambda req: httpx.Response(200, json=_ok_chat("hello"))))
    try:
        with _client().stream(
            "POST",
            "/v1/responses",
            json={"model": "single.v1", "input": "hi", "stream": True},
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            raw = b"".join(resp.iter_bytes()).decode("utf-8")
        # Parse SSE event lines.
        events = [
            line[len("event: ") :]
            for line in raw.splitlines()
            if line.startswith("event: ")
        ]
        assert events[0] == "response.created"
        assert events[1] == "response.in_progress"
        assert events[-1] == "response.completed"
        assert "response.output_text.delta" in events
    finally:
        set_upstream_transport(None)


def test_endpoint_unsupported_tool_choice_returns_501(loaded_config) -> None:
    r = _client().post(
        "/v1/responses",
        json={
            "model": "single.v1",
            "input": ".",
            "tool_choice": {"type": "allowed_tools", "tools": []},
        },
    )
    assert r.status_code == 501
    body = r.json()
    assert body["error"]["code"] == "unsupported_tool_choice"


def test_new_response_id_format() -> None:
    rid = new_response_id()
    assert rid.startswith("resp_")
    assert len(rid) == len("resp_") + 24
