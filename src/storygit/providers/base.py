"""The provider interface every backend implements, and the errors they raise.

One request type, one response type, one method. Rotation, caching, cost accounting and
logging are layered around this rather than baked into each backend, so adding a
provider means writing one ``complete`` and nothing else, and so the router can swap
providers per purpose without any caller noticing.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from storygit.domain.errors import StoryGitError


class Role(StrEnum):
    """Who is speaking in a message."""

    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    """One turn of a prompt."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class LLMRequest(BaseModel):
    """Everything a backend needs to make one call.

    Attributes:
        messages: The prompt.
        purpose: A dotted tag such as ``propose.beat`` or ``extract.facts``. The router
            uses it to choose a provider, the call log groups by it, and the eval
            harness reports cost per purpose. Every call must carry one.
        model: Overrides the provider's default model.
        temperature: Sampling temperature.
        max_tokens: Output cap.
        json_schema: A JSON Schema the response must satisfy. Backends that support
            structured output enforce it; the rest have it appended to the prompt.
        sample_index: Distinguishes parallel samples of the *same* prompt. It is part of
            the cache key, so six axis-conditioned candidates do not collapse onto one
            cached response.
        bypass_cache: Skip the read-through cache for this call.
        stop: Optional stop sequences.
    """

    model_config = ConfigDict(frozen=True)

    messages: tuple[Message, ...]
    purpose: str
    model: str | None = None
    temperature: float = Field(default=0.9, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    json_schema: dict[str, Any] | None = None
    sample_index: int = 0
    bypass_cache: bool = False
    stop: tuple[str, ...] = ()

    def system_text(self) -> str:
        """Concatenated system messages, for backends with a separate system field."""
        return "\n\n".join(m.content for m in self.messages if m.role is Role.system)

    def conversation(self) -> tuple[Message, ...]:
        """Messages excluding system turns."""
        return tuple(m for m in self.messages if m.role is not Role.system)


class LLMResponse(BaseModel):
    """What came back, plus everything needed to account for it.

    Attributes:
        text: The completion.
        model: The model that actually served it, which may be a fallback.
        provider: Backend name.
        prompt_tokens: Input tokens, estimated when the API does not report them.
        completion_tokens: Output tokens.
        latency_s: Wall-clock seconds; zero for a cache hit.
        cached: Whether this came from the disk cache.
        key_index: Which rotation slot served it — an index, never a key.
        cost_usd: Estimated cost from the static price table.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    key_index: int | None = None
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class Provider(Protocol):
    """A model backend.

    Attributes:
        name: Stable identifier used in the call log and the routing table.
    """

    name: str

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one completion.

        Args:
            request: The call to make.

        Returns:
            The response.

        Raises:
            ProviderError: Any subclass, on failure.
        """
        ...


# --- errors -------------------------------------------------------------------


class ProviderError(StoryGitError):
    """Base class for every provider failure."""


class RateLimited(ProviderError):
    """Every key or the whole provider is rate limited.

    Attributes:
        retry_after: Seconds until the soonest key is expected to be usable again.
    """

    def __init__(self, message: str, retry_after: float = 60.0) -> None:
        """Create the error.

        Args:
            message: What happened.
            retry_after: Seconds until a retry could succeed.
        """
        super().__init__(message)
        self.retry_after = retry_after


class ProviderDown(ProviderError):
    """The provider returned a server error or could not be reached."""


class SchemaParseError(ProviderError):
    """The model's output did not parse against the requested schema.

    Attributes:
        raw: The offending text, so the caller can attempt a repair prompt.
    """

    def __init__(self, message: str, raw: str = "") -> None:
        """Create the error.

        Args:
            message: What failed to parse.
            raw: The raw model output.
        """
        super().__init__(message)
        self.raw = raw


class BudgetExceeded(ProviderError):
    """A call was refused because it would cross the spending cap."""


class OpenRouterDisabledError(ProviderError):
    """OpenRouter is locked.

    Raised before any request is constructed, so no key is read, no URL is built, and
    nothing reaches the network. The lock opens only when ``OPENROUTER_ENABLED`` is
    exactly ``"true"`` *and* the budget guard permits the call.
    """


def estimate_tokens(text: str) -> int:
    """Rough token count for logging when an API does not report usage.

    Four characters per token is close enough for cost accounting and for the eval
    harness's tokens-per-action metric; nothing branches on this number.

    Args:
        text: The text to measure.

    Returns:
        An estimated token count, at least 1 for non-empty text.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)
