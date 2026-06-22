"""Shared OpenAI-compatible error envelope helpers.

Extracted from `server.py` so non-server modules (e.g. `streaming.py`)
can build the same envelope without circular imports. The wire shape
is identical to OpenAI's REST error responses:

    {"error": {"message": "...", "type": "...", "param": ..., "code": ...}}

`type` follows OpenAI's small enumerated set (invalid_request_error,
authentication_error, permission_error, not_found_error,
rate_limit_error, service_unavailable_error, api_error). `param` and
`code` are explicitly null when absent — that keeps the wire shape
stable for clients that index into the envelope by key.
"""

from __future__ import annotations

from typing import Any


def error_type_for_status(status: int) -> str:
    """Map an HTTP status code to OpenAI's `error.type` taxonomy."""
    if status == 401:
        return "authentication_error"
    if status == 403:
        return "permission_error"
    if status == 404:
        return "not_found_error"
    if status == 429:
        return "rate_limit_error"
    if status == 503:
        return "service_unavailable_error"
    if status >= 500:
        return "api_error"
    return "invalid_request_error"


def error_envelope(
    message: str,
    *,
    status: int,
    param: str | None = None,
    code: str | None = None,
    type_: str | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-compatible error body: ``{"error": {...}}``."""
    err: dict[str, Any] = {
        "message": message,
        "type": type_ or error_type_for_status(status),
    }
    err["param"] = param
    err["code"] = code
    return {"error": err}


__all__ = ["error_envelope", "error_type_for_status"]
