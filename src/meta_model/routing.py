"""F4-core — shared profile resolver for the new OpenAI-compatible
endpoints (`/v1/completions`, `/tokenize`, `/v1/responses`).

Wraps the resolution logic that already lives in
`moa/dispatch.py::resolve_profile` (which the chat-completions path
uses) with an exception-based API so the new endpoints get uniform
typed-404 behavior:

    {"error": {
        "type": "invalid_request_error",
        "code": "model_not_found",
        "message": "model 'foo' is not configured",
        "param": "model"
    }}

Why a separate module: dispatch's tuple return is convenient inline
but every new endpoint would have to repeat the (None, None,
"model_not_found") → 404 envelope construction. Centralising that
here also gives F4-A a single site to wire the alias map at — once
F4-A lands, alias lookups happen inside this module so every new
endpoint inherits aliasing without further changes.

The chat-completions path (`/v1/chat/completions`) keeps using
`dispatch.resolve_profile` directly because dispatch needs the
typed error code separately from the message (e.g.
`feature_disabled` flips to a different status / wording than
`model_not_found`). Both code paths stay in sync because this
module is a thin wrapper over the dispatch resolver.
"""

from __future__ import annotations

from typing import Any

from fastapi.responses import JSONResponse

from .config import MetaModelConfig, Profile
from .errors import error_envelope
from .moa.dispatch import resolve_profile as _resolve_profile_tuple


class ProfileResolutionError(Exception):
    """Typed error raised by `resolve_profile` for unknown / disabled
    targets. Endpoints catch it and convert via `error_response`."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        param: str | None = "model",
        type_: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.param = param
        # Override the default OpenAI taxonomy mapping. 404 normally maps
        # to "not_found_error" via `error_type_for_status`, but the
        # OpenAI SDK expects model-not-found to surface as
        # "invalid_request_error" so the client raises BadRequestError —
        # match that convention.
        self.type_ = type_ or "invalid_request_error"


def resolve_profile(
    cfg: MetaModelConfig,
    model: str | None,
    ext_profile: str | None = None,
) -> tuple[Profile, str]:
    """Resolve a request's target profile or raise.

    Order: extension override > exact profile name > alias map (F4-A,
    no-op until populated) > raw upstream key > raise
    ProfileResolutionError.

    Returns `(profile, canonical_name)`. The synthetic single-upstream
    MoA profile produced for raw-upstream addressing is the same
    object dispatch builds, so downstream code paths stay uniform.
    """
    if not model and not ext_profile:
        raise ProfileResolutionError(
            400, "missing_model", "request body missing 'model'"
        )
    profile, name, err = _resolve_profile_tuple(cfg, model or "", ext_profile)
    if err == "model_not_found":
        raise ProfileResolutionError(
            404,
            "model_not_found",
            f"model {name!r} is not configured",
        )
    if err == "feature_disabled":
        raise ProfileResolutionError(
            400,
            "feature_disabled",
            f"profile {name!r} requires features.voting=true",
        )
    assert profile is not None and name is not None
    return profile, name


def error_response(err: ProfileResolutionError) -> JSONResponse:
    """Convert a `ProfileResolutionError` into the typed JSON envelope."""
    body: dict[str, Any] = error_envelope(
        err.message,
        status=err.status_code,
        param=err.param,
        code=err.code,
        type_=err.type_,
    )
    return JSONResponse(status_code=err.status_code, content=body)


__all__ = [
    "ProfileResolutionError",
    "error_response",
    "resolve_profile",
]
