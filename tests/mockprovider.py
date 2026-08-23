"""A provider that returns canned JSON, so generation is testable without a network.

Two modes. Give it a queue of responses and it serves them in order; give it a callable
and it decides per request. It records every request it saw, which is how the tests
check that prompts carry slices rather than whole states, that axis fragments differ,
and that a call never happened at all.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from storygit.providers.base import LLMRequest, LLMResponse, estimate_tokens


class MockProvider:
    """Canned-response provider.

    Attributes:
        name: Provider name to report; set it to impersonate ``gemini`` or ``groq``.
        model: Model name to report.
        requests: Every request served, in order.
    """

    def __init__(
        self,
        responses: list[str] | Callable[[LLMRequest], str] | None = None,
        *,
        name: str = "gemini",
        model: str = "mock-model",
    ) -> None:
        """Create the provider.

        Args:
            responses: A list served in order (the last one repeats once exhausted), or
                a callable taking the request and returning the response text.
            name: Provider name to report.
            model: Model name to report.
        """
        self.name = name
        self.model = model
        self._responses = responses if responses is not None else ["{}"]
        self.requests: list[LLMRequest] = []
        self._index = 0

    def _next(self, request: LLMRequest) -> str:
        if callable(self._responses):
            return self._responses(request)
        if not self._responses:
            return "{}"
        text = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return text

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Serve one canned response."""
        self.requests.append(request)
        text = self._next(request)
        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            prompt_tokens=estimate_tokens("".join(m.content for m in request.messages)),
            completion_tokens=estimate_tokens(text),
            latency_s=0.001,
        )

    async def aclose(self) -> None:
        """Nothing to close."""

    # --- helpers for tests ------------------------------------------------------

    def prompts_for(self, purpose_prefix: str) -> list[str]:
        """All prompt text sent for purposes starting with a prefix."""
        return [
            "\n".join(m.content for m in r.messages)
            for r in self.requests
            if r.purpose.startswith(purpose_prefix)
        ]


def canned(payload: dict[str, Any]) -> str:
    """Serialize a dict as the JSON a model would have returned."""
    return json.dumps(payload)
