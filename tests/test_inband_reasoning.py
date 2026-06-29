from meta_model.reasoning import (
    strip_inband_reasoning as S,
    strip_message_inband_reasoning as SM,
)


def test_paired_think():
    assert S("<think>plan the answer</think>The answer is 42.") == "The answer is 42."


def test_thinking_tag_variant():
    assert S("<thinking>x</thinking>real answer") == "real answer"


def test_prefilled_open_trailing_close():
    assert S("reasoning text here\n</think>\n\nFinal answer here.") == "Final answer here."


def test_cutoff_open_no_close():
    assert S("<think>still reasoning and ran out of tokens") == ""


def test_unicode_deepseek():
    assert S("◁think▷secret plan◁/think▷done") == "done"


def test_harmony_final_channel():
    assert S("<|channel|>analysis<|message|>reason<|end|><|channel|>final<|message|>Yes, it works.") == "Yes, it works."


def test_harmony_analysis_only_drops():
    assert S("<|channel|>analysis<|message|>only thinking, no final<|end|>") == ""


def test_plain_answer_untouched():
    t = "Use a < b comparison and 3 > 2 logic here."
    assert S(t) == t


def test_no_markup_untouched():
    t = "A normal complete answer with no thinking at all."
    assert S(t) == t


def test_leading_preamble_with_answer_marker():
    assert S("Here's a thinking process:\nstep one\nstep two\n\nAnswer:\nThe result is X.") == "The result is X."


def test_message_helper_strips_content():
    m = {"role": "assistant", "content": "<think>z</think>hi there"}
    SM(m)
    assert m["content"] == "hi there"


def test_message_helper_preserves_tool_calls():
    m = {"role": "assistant", "content": "<think>z</think>", "tool_calls": [{"id": "1", "type": "function"}]}
    SM(m)
    assert m["tool_calls"] == [{"id": "1", "type": "function"}]


def test_message_helper_list_content():
    m = {"role": "assistant", "content": [{"type": "text", "text": "<think>q</think>final"}]}
    SM(m)
    assert m["content"][0]["text"] == "final"
