"""`meta_model.moa.multimodal` — modality detection.

F6 made multimodal serving server-level (top-level `[vision]` /
`[video]` / `[audio]` blocks routed through the cascade in
`dispatch._multimodal_cascade`). The per-profile filter logic this
module previously owned is gone; only the modality scanner remains.
"""

from __future__ import annotations

from typing import Any

from meta_model.moa.multimodal import MessageModality, detect_message_modality


def _msg(role: str, content: Any) -> dict[str, Any]:
    return {"role": role, "content": content}


def _image_part(url: str = "https://example.com/img.png") -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def _video_part(url: str = "https://example.com/clip.mp4") -> dict[str, Any]:
    return {"type": "video_url", "video_url": {"url": url}}


def _audio_part_data() -> dict[str, Any]:
    return {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}}


def _audio_part_url() -> dict[str, Any]:
    return {"type": "input_audio", "input_audio": {"url": "https://example.com/clip.wav"}}


def _text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


# ── detect_message_modality — text & default ───────────────────────


def test_detect_text_only_messages_no_modality() -> None:
    messages = [_msg("user", "hello"), _msg("assistant", "hi")]
    out = detect_message_modality(messages)
    assert out.image_count == 0
    assert out.video_count == 0
    assert out.audio_count == 0
    assert out.has_images is False
    assert out.has_videos is False
    assert out.has_audios is False
    assert out.is_multimodal is False
    assert out.unsupported_parts == ()


def test_detect_text_part_only_no_modality() -> None:
    messages = [_msg("user", [_text_part("hello")])]
    out = detect_message_modality(messages)
    assert out.image_count == 0


def test_detect_handles_null_content() -> None:
    """Tool-result messages can have content=None. Skip cleanly."""
    messages = [{"role": "tool", "content": None, "tool_call_id": "x"}]
    out = detect_message_modality(messages)
    assert out.image_count == 0
    assert out.unsupported_parts == ()


# ── image parts ────────────────────────────────────────────────────


def test_detect_image_part_counted() -> None:
    messages = [_msg("user", [_text_part("describe"), _image_part()])]
    out = detect_message_modality(messages)
    assert out.image_count == 1
    assert out.has_images is True
    assert out.is_multimodal is True


def test_detect_multiple_images_counted() -> None:
    messages = [
        _msg("user", [_image_part("a"), _image_part("b")]),
        _msg("user", [_image_part("c")]),
    ]
    out = detect_message_modality(messages)
    assert out.image_count == 3


def test_detect_malformed_image_url_object() -> None:
    messages = [_msg("user", [{"type": "image_url", "image_url": "not-an-object"}])]
    out = detect_message_modality(messages)
    assert out.image_count == 0
    assert "image_url_malformed" in out.unsupported_parts


def test_detect_image_url_missing_url() -> None:
    messages = [_msg("user", [{"type": "image_url", "image_url": {}}])]
    out = detect_message_modality(messages)
    assert out.image_count == 0
    assert "image_url_missing_url" in out.unsupported_parts


# ── F6: video parts ────────────────────────────────────────────────


def test_detect_video_part_counted() -> None:
    messages = [_msg("user", [_text_part("watch"), _video_part()])]
    out = detect_message_modality(messages)
    assert out.video_count == 1
    assert out.image_count == 0
    assert out.has_videos is True
    assert out.is_multimodal is True


def test_detect_video_url_malformed() -> None:
    messages = [_msg("user", [{"type": "video_url", "video_url": "not-an-object"}])]
    out = detect_message_modality(messages)
    assert out.video_count == 0
    assert "video_url_malformed" in out.unsupported_parts


def test_detect_video_url_missing_url() -> None:
    messages = [_msg("user", [{"type": "video_url", "video_url": {}}])]
    out = detect_message_modality(messages)
    assert out.video_count == 0
    assert "video_url_missing_url" in out.unsupported_parts


# ── F6: audio parts ────────────────────────────────────────────────


def test_detect_audio_part_inline_data_counted() -> None:
    messages = [_msg("user", [_text_part("listen"), _audio_part_data()])]
    out = detect_message_modality(messages)
    assert out.audio_count == 1
    assert out.has_audios is True
    assert out.is_multimodal is True


def test_detect_audio_part_url_counted() -> None:
    messages = [_msg("user", [_audio_part_url()])]
    out = detect_message_modality(messages)
    assert out.audio_count == 1


def test_detect_audio_part_missing_payload() -> None:
    messages = [_msg("user", [{"type": "input_audio", "input_audio": {}}])]
    out = detect_message_modality(messages)
    assert out.audio_count == 0
    assert "input_audio_missing_payload" in out.unsupported_parts


def test_detect_audio_part_malformed() -> None:
    messages = [_msg("user", [{"type": "input_audio", "input_audio": "not-an-object"}])]
    out = detect_message_modality(messages)
    assert out.audio_count == 0
    assert "input_audio_malformed" in out.unsupported_parts


# ── mixed modality ─────────────────────────────────────────────────


def test_detect_image_plus_video_plus_audio() -> None:
    messages = [_msg("user", [_image_part(), _video_part(), _audio_part_data()])]
    out = detect_message_modality(messages)
    assert out.image_count == 1
    assert out.video_count == 1
    assert out.audio_count == 1
    assert out.is_multimodal is True


# ── unsupported parts / malformed ──────────────────────────────────


def test_detect_unsupported_part_type_recorded() -> None:
    messages = [_msg("user", [{"type": "input_file", "file_id": "f-1"}])]
    out = detect_message_modality(messages)
    assert out.unsupported_parts == ("input_file",)


def test_detect_unsupported_dedupes() -> None:
    messages = [
        _msg("user", [{"type": "input_file", "file_id": "f-1"}]),
        _msg("user", [{"type": "input_file", "file_id": "f-2"}]),
    ]
    out = detect_message_modality(messages)
    # Each unique unsupported type appears once.
    assert out.unsupported_parts == ("input_file",)


def test_detect_part_missing_type_is_unsupported() -> None:
    messages = [_msg("user", [{"text": "no type"}])]
    out = detect_message_modality(messages)
    assert "missing_type" in out.unsupported_parts


def test_detect_non_dict_part_recorded_as_malformed() -> None:
    messages = [_msg("user", ["bare-string-part"])]
    out = detect_message_modality(messages)
    assert "malformed_part" in out.unsupported_parts


def test_detect_non_list_non_string_content_recorded() -> None:
    messages = [_msg("user", {"some": "dict"})]
    out = detect_message_modality(messages)
    assert "malformed_content" in out.unsupported_parts


def test_detect_malformed_text_part_recorded() -> None:
    """Text part with non-string `text` is malformed."""
    out = detect_message_modality([_msg("user", [{"type": "text", "text": 42}])])
    assert "text_malformed" in out.unsupported_parts


def test_detect_text_part_missing_text_field_recorded() -> None:
    out = detect_message_modality([_msg("user", [{"type": "text"}])])
    assert "text_malformed" in out.unsupported_parts


# ── dataclass invariant ────────────────────────────────────────────


def test_message_modality_dataclass_is_frozen() -> None:
    out = MessageModality(image_count=2)
    try:
        out.image_count = 5  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("MessageModality should be frozen")
