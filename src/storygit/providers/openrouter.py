"""OpenRouter — written in full, and hard-locked.

This client is complete and would work. It cannot run. The constructor reads
``OPENROUTER_ENABLED`` and, unless it is exactly the string ``"true"``, every call
raises :class:`~storygit.providers.base.OpenRouterDisabledError` *before* a key is read,
a URL is built, or a transport is touched. When it is unlocked (chunk 7, with Arush
present), every call still passes through the budget guard first, which refuses anything
that could cross the in-code cap.

Two locks in series, both fail-closed, and neither can be tripped by a typo:
``"True"``, ``"1"``, and ``"yes"`` all leave it locked.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import httpx

from storygit.providers.base import (
    LLMRequest,
    LLMResponse,
    OpenRouterDisabledError,
    ProviderDown,
    RateLimited,
    estimate_tokens,
)
from storygit.providers.budget import BudgetGuard
from storygit.providers.pricing import estimate_cost

API_URL = "https://openrouter.ai/api/v1/chat/completions"
ENABLED_SENTINEL = "true"
"""The one string that unlocks this provider. Compared exactly, no normalization."""


class OpenRouterProvider:
    """Metered provider, disabled unless explicitly and exactly enabled.

    Attributes:
        name: ``"openrouter"``.
        model: Default model id.
        enabled: Whether the lock is open.
    """

    name = "openrouter"

    def __init__(
        self,
        api_key: str,
        *,
        model: str,
        enabled_flag: str,
        budget: BudgetGuard,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 120.0,
        redactor: Callable[[str], str] | None = None,
    ) -> None:
        """Create the provider in its locked or unlocked state.

        Args:
            api_key: OpenRouter key. Not read at all while locked.
            model: Default model id.
            enabled_flag: The raw ``OPENROUTER_ENABLED`` value, compared exactly against
                ``"true"``.
            budget: The guard every unlocked call must pass.
            transport: httpx transport.
            timeout_s: Per-request timeout.
            redactor: Scrubs secrets from error text.
        """
        self.enabled = enabled_flag == ENABLED_SENTINEL
        self.model = model
        self._budget = budget
        self._key = api_key
        self._timeout = timeout_s
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        if redactor is None:
            from storygit.config import get_settings

            redactor = get_settings().redact
        self._redact = redactor

    def _require_enabled(self) -> None:
        """Raise unless the lock is open.

        Raises:
            OpenRouterDisabledError: Whenever the flag is not exactly ``"true"``.
        """
        if not self.enabled:
            raise OpenRouterDisabledError(
                "OpenRouter is disabled. It stays disabled until OPENROUTER_ENABLED is "
                'exactly "true" and the budget guard permits the call. This is a '
                "deliberate lock, not a configuration error."
            )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one completion, if and only if both locks are open.

        Args:
            request: The call to make.

        Returns:
            The response.

        Raises:
            OpenRouterDisabledError: If the provider is locked. Raised before any key is
                read or any network activity occurs.
            BudgetExceeded: If the call could cross the in-code cap.
            RateLimited: On HTTP 429.
            ProviderDown: On any other failure.
        """
        self._require_enabled()

        model = request.model or self.model
        prompt_text = "".join(m.content for m in request.messages)
        prompt_tokens = estimate_tokens(prompt_text)
        self._budget.check(
            self.name, model, prompt_tokens=prompt_tokens, max_output_tokens=request.max_tokens
        )

        if self._client is None:
            self._client = httpx.AsyncClient(transport=self._transport, timeout=self._timeout)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.stop:
            payload["stop"] = list(request.stop)
        if request.json_schema is not None:
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            http_response = await self._client.post(
                API_URL,
                json=payload,
                headers={
                    "authorization": f"Bearer {self._key}",
                    "content-type": "application/json",
                    "x-title": "storygit",
                },
            )
        except httpx.HTTPError as exc:
            raise ProviderDown(f"openrouter transport error: {self._redact(str(exc))}") from exc
        latency = time.perf_counter() - started

        if http_response.status_code == 429:
            raise RateLimited("openrouter: rate limited", retry_after=30.0)
        if http_response.status_code >= 400:
            body = self._redact(http_response.text[:400])
            raise ProviderDown(f"openrouter: HTTP {http_response.status_code}: {body}")

        try:
            data = http_response.json()
        except json.JSONDecodeError as exc:
            raise ProviderDown("openrouter: response was not JSON") from exc

        choices = data.get("choices") or []
        text = choices[0]["message"]["content"] if choices else ""
        usage = data.get("usage") or {}
        prompt_used = int(usage.get("prompt_tokens") or prompt_tokens)
        completion_used = int(usage.get("completion_tokens") or estimate_tokens(text or ""))
        cost = estimate_cost(self.name, model, prompt_used, completion_used)
        self._budget.record(self.name, model, request.purpose, cost)
        return LLMResponse(
            text=text or "",
            model=data.get("model") or model,
            provider=self.name,
            prompt_tokens=prompt_used,
            completion_tokens=completion_used,
            latency_s=latency,
            cost_usd=cost,
        )

    async def aclose(self) -> None:
        """Close the HTTP client, if one was ever opened."""
        if self._client is not None:
            await self._client.aclose()
