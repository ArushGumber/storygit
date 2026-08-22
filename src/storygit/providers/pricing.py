"""Static price table, in US dollars per million tokens.

Development runs on free tiers, so every real cost here is zero. The table still
matters: it is what turns the call log into the "what would this have cost" number the
evaluation reports, and it is what the budget guard checks against before an OpenRouter
call in chunk 7.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Price(BaseModel):
    """Dollars per million tokens for one model."""

    model_config = ConfigDict(frozen=True)

    prompt: float
    completion: float


FREE = Price(prompt=0.0, completion=0.0)

PRICES: dict[str, Price] = {
    # Gemini free tier: real quota, no dollars.
    "gemini:*": FREE,
    # Groq free tier.
    "groq:*": FREE,
    # Local models cost electricity, which we do not bill.
    "local:*": FREE,
    # OpenRouter, chunk 7 only. Published rates for the models we might use.
    "openrouter:anthropic/claude-3.5-sonnet": Price(prompt=3.0, completion=15.0),
    "openrouter:anthropic/claude-sonnet-4": Price(prompt=3.0, completion=15.0),
    "openrouter:openai/gpt-4o": Price(prompt=2.5, completion=10.0),
    "openrouter:google/gemini-2.5-pro": Price(prompt=1.25, completion=10.0),
    "openrouter:*": Price(prompt=3.0, completion=15.0),
}
"""Keys are ``provider:model``; ``provider:*`` is the fallback for that provider."""


def price_for(provider: str, model: str) -> Price:
    """Look up a model's price, falling back to the provider's default.

    Args:
        provider: Backend name.
        model: Model id.

    Returns:
        The price entry; a conservative provider default when the model is unlisted.
    """
    exact = PRICES.get(f"{provider}:{model}")
    if exact is not None:
        return exact
    return PRICES.get(f"{provider}:*", FREE)


def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Dollar cost of one call.

    Args:
        provider: Backend name.
        model: Model id.
        prompt_tokens: Input tokens.
        completion_tokens: Output tokens.

    Returns:
        Estimated cost in dollars.
    """
    price = price_for(provider, model)
    return (prompt_tokens * price.prompt + completion_tokens * price.completion) / 1_000_000
