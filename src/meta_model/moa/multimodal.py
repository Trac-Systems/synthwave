"""Per-request modality detection.

F6 made multimodal serving server-level (top-level `[vision]` /
`[video]` / `[audio]` blocks routed through `_multimodal_cascade` in
`dispatch.py`). The per-profile `MultimodalPolicy` filter logic this
module previously owned is gone. What remains here:

- ``MessageModality`` — counts of image / video / audio parts seen
  in a request, plus the deduped list of malformed / unrecognized
  part types.
- ``detect_message_modality(messages)`` — single-pass scan that
  populates the above. Dispatch consults `has_images / has_videos /
  has_audios` to short-circuit into the matching cascade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Recognized content-part types. F6: video and audio now recognized
# at detection time (counted), but dispatch decides whether the
# server actually supports them via `[video].endpoints` /
# `[audio].endpoints`. Anything else still surfaces as
# `unsupported_content_part`.
#
# Part-type aliases:
#   - image: OpenAI Chat shape uses "image_url"; Responses API uses
#     "input_image" (already adapted by responses.py before reaching
#     here, so chat-side only sees image_url).
#   - video: "video_url" — modeled on image_url's shape for symmetry.
#     Responses API uses "input_video"; not yet supported here.
#   - audio: "input_audio" — Chat API shape, holds {data, format}
#     OR a {url}-shaped object for inline-vs-fetched audio.
_SUPPORTED_PART_TYPES = frozenset(
    {"text", "image_url", "video_url", "input_audio"}
)


@dataclass(frozen=True)
class MessageModality:
    """Outcome of scanning a request's messages for content parts.

    Counts each non-text modality independently so the cascade router
    knows which `[vision]/[video]/[audio]` block to consult.
    `unsupported_parts` lists the unrecognized `type` values seen
    (deduped, in first-seen order).
    """

    image_count: int = 0
    video_count: int = 0
    audio_count: int = 0
    unsupported_parts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_images(self) -> bool:
        return self.image_count > 0

    @property
    def has_videos(self) -> bool:
        return self.video_count > 0

    @property
    def has_audios(self) -> bool:
        return self.audio_count > 0

    @property
    def is_multimodal(self) -> bool:
        return self.has_images or self.has_videos or self.has_audios


def detect_message_modality(messages: list[dict[str, Any]]) -> MessageModality:
    """Scan messages for content parts; categorize.

    String-content messages are text-only. List-content messages are
    walked; each entry's `type` is inspected. Image parts are
    structurally validated (must have `image_url` object with string
    `url`). Anything else with an unrecognized `type` is recorded.

    Structural validation failures (image_url shape) surface as
    `unsupported_content_part` — strict mode, since malformed image
    parts will confuse upstreams.
    """
    image_count = 0
    video_count = 0
    audio_count = 0
    unsupported: list[str] = []
    seen_unsupported: set[str] = set()

    def _record_unsupported(label: str) -> None:
        if label in seen_unsupported:
            return
        seen_unsupported.add(label)
        unsupported.append(label)

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) or content is None:
            continue  # text-only message
        if not isinstance(content, list):
            # Unexpected shape — treat as malformed.
            _record_unsupported("malformed_content")
            continue
        for part in content:
            if not isinstance(part, dict):
                _record_unsupported("malformed_part")
                continue
            ptype = part.get("type")
            if not isinstance(ptype, str):
                _record_unsupported("missing_type")
                continue
            if ptype not in _SUPPORTED_PART_TYPES:
                _record_unsupported(ptype)
                continue
            if ptype == "text":
                # Review r32 L1: validate text-part shape too. With global
                # entry-time rejection now the typed contract, malformed
                # text parts shouldn't quietly pass to the upstream.
                tval = part.get("text")
                if not isinstance(tval, str):
                    _record_unsupported("text_malformed")
                continue
            if ptype == "image_url":
                # Structural shape only — review r31 #8: skip deep URL
                # validation but require object + string url.
                iu = part.get("image_url")
                if not isinstance(iu, dict):
                    _record_unsupported("image_url_malformed")
                    continue
                url = iu.get("url")
                if not isinstance(url, str) or not url:
                    _record_unsupported("image_url_missing_url")
                    continue
                image_count += 1
                continue
            if ptype == "video_url":
                # Mirrors image_url's structural validation.
                vu = part.get("video_url")
                if not isinstance(vu, dict):
                    _record_unsupported("video_url_malformed")
                    continue
                url = vu.get("url")
                if not isinstance(url, str) or not url:
                    _record_unsupported("video_url_missing_url")
                    continue
                video_count += 1
                continue
            if ptype == "input_audio":
                # OpenAI Chat audio shape carries either {data, format}
                # for inline base64 OR a {url} for fetched audio.
                # Accept both; reject empty / wrong-shaped.
                ia = part.get("input_audio")
                if not isinstance(ia, dict):
                    _record_unsupported("input_audio_malformed")
                    continue
                has_data = isinstance(ia.get("data"), str) and ia.get("data")
                has_url = isinstance(ia.get("url"), str) and ia.get("url")
                if not (has_data or has_url):
                    _record_unsupported("input_audio_missing_payload")
                    continue
                audio_count += 1
                continue

    return MessageModality(
        image_count=image_count,
        video_count=video_count,
        audio_count=audio_count,
        unsupported_parts=tuple(unsupported),
    )


__all__ = [
    "MessageModality",
    "detect_message_modality",
]
