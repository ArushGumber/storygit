"""Groq over REST — the fast, cheap lane for extraction and classification.

Groq serves a small Llama model at very low latency on a free tier. It is not the right
model for prose, and the router never sends prose to it. What it is right for is the
high-volume structured work: pulling facts out of an accepted paragraph, summarizing an
edit into a style rule, and the simulated writer's rewrites during evaluation. Those are
the calls that would otherwise burn the Gemini quota that prose actually needs.
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
    ProviderDown,
    RateLimited,
    estimate_tokens,
)

API_URL = "https://api.groq.com/openai/v1/chat/completions"


class _JsonModeRejected(Exception):
    """Groq's server-side JSON validation rejected its own model's output.

    Internal to this module: it triggers one retry with JSON mode off and never escapes.
    """


class GroqProvider:
    """OpenAI-shaped chat client pointed at Groq.

    Attributes:
        name: ``"groq"``.
        model: Default model id.
    """

    name = "groq"

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "openai/gpt-oss-20b",
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 60.0,
        redactor: Callable[[str], str] | None = None,
    ) -> None:
        """Create the provider.

        Args:
            api_key: Groq API key, held in memory only.
            model: Default model id.
            transport: httpx transport; tests inject a mock.
            timeout_s: Per-request timeout.
            redactor: Scrubs secrets from error text.

        Raises:
            ValueError: If no key was supplied.
        """
        if not api_key:
            raise ValueError("GroqProvider needs an API key")
        self._key = api_key
        self.model = model
        self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
        if redactor is None:
            from storygit.config import get_settings

            redactor = get_settings().redact
        self._redact = redactor

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one completion.

        Groq validates JSON-mode output server side and rejects the whole request with a
        400 when its own model produces malformed JSON. That is a generation failure, not
        an outage, so it is retried once with JSON mode off — the prompt already carries
        the schema and this model returns clean JSON without the mode. Only if that also
        fails does the router get to fall back to a different provider.

        Args:
            request: The call to make.

        Returns:
            The response.

        Raises:
            RateLimited: On HTTP 429.
            ProviderDown: On any other failure.
        """
        try:
            return await self._attempt(request, json_mode=request.json_schema is not None)
        except _JsonModeRejected:
            return await self._attempt(request, json_mode=False)

    async def _attempt(self, request: LLMRequest, *, json_mode: bool) -> LLMResponse:
        """One HTTP call, with JSON mode on or off."""
        model = request.model or self.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.stop:
            payload["stop"] = list(request.stop)
        if json_mode:
            # Groq supports OpenAI-style JSON mode; the schema itself is carried in the
            # prompt by the agents layer, and validated on the way back by Pydantic.
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            http_response = await self._client.post(
                API_URL,
                json=payload,
                headers={
                    "authorization": f"Bearer {self._key}",
                    "content-type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise ProviderDown(f"groq transport error: {self._redact(str(exc))}") from exc
        latency = time.perf_counter() - started

        if http_response.status_code == 429:
            retry_after = http_response.headers.get("retry-after")
            raise RateLimited(
                "groq: rate limited",
                retry_after=float(retry_after) if retry_after else 30.0,
            )
        if http_response.status_code >= 400:
            body = self._redact(http_response.text[:400])
            if json_mode and "json_validate_failed" in body:
                raise _JsonModeRejected(body)
            raise ProviderDown(f"groq: HTTP {http_response.status_code}: {body}")

        try:
            data = http_response.json()
        except json.JSONDecodeError as exc:
            raise ProviderDown("groq: response was not JSON") from exc

        choices = data.get("choices") or []
        text = choices[0]["message"]["content"] if choices else ""
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text or "",
            model=data.get("model") or model,
            provider=self.name,
            prompt_tokens=int(usage.get("prompt_tokens") or estimate_tokens(str(payload))),
            completion_tokens=int(usage.get("completion_tokens") or estimate_tokens(text or "")),
            latency_s=latency,
        )

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
