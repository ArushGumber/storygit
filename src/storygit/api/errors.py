"""Turning typed engine errors into one JSON shape.

Every failing route returns the same body — ``{kind, detail, retry_after}`` — with ``kind``
set to the exception's class name. That gives the interface exactly one error path to
handle, and lets it tell a rate limit (retry in 45 seconds) from a locked node (tell the
writer) without parsing English.

It also means no route leaks an internal traceback, which matters here because exception
text from the provider layer can contain a URL, and a URL can contain a key. Every message
goes through the settings redactor on the way out.
"""

from __future__ import annotations

from typing import Any

from storygit.domain.errors import (
    ApplyError,
    LockedNodeError,
    SnapshotNotFoundError,
    StoryGitError,
)
from storygit.providers.base import (
    BudgetExceeded,
    OpenRouterDisabledError,
    ProviderDown,
    ProviderError,
    RateLimited,
    SchemaParseError,
)

STATUS_FOR: list[tuple[type[Exception], int]] = [
    (LockedNodeError, 409),
    (SnapshotNotFoundError, 404),
    (ApplyError, 422),
    (RateLimited, 429),
    (BudgetExceeded, 402),
    (OpenRouterDisabledError, 403),
    (ProviderDown, 503),
    (SchemaParseError, 502),
    (ProviderError, 502),
    (StoryGitError, 400),
]
"""Ordered most specific first; the first match wins."""


def status_for(error: Exception) -> int:
    """The HTTP status an engine error maps to.

    Args:
        error: The exception.

    Returns:
        A status code; 500 for anything unrecognized.
    """
    for kind, status in STATUS_FOR:
        if isinstance(error, kind):
            return status
    return 500


def problem(error: Exception, *, redact: Any = None) -> dict[str, Any]:
    """The JSON body for a failure.

    Args:
        error: The exception.
        redact: A scrubbing function, so no secret can reach the client.

    Returns:
        ``{kind, detail, retry_after}``.
    """
    detail = str(error)
    if redact is not None:
        detail = redact(detail)
    body: dict[str, Any] = {"kind": type(error).__name__, "detail": detail}
    retry_after = getattr(error, "retry_after", None)
    if retry_after is not None:
        body["retry_after"] = float(retry_after)
    return body
