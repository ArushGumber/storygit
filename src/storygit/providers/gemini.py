"""Gemini over REST, with six free-tier keys in rotation.

Six keys on the free tier is a real quota, but only if you use them properly. Each key
carries its own state: a cooldown timestamp set when that key returns a 429 or a quota
error, and rolling counters for reporting. A call takes the next healthy key; when every
key is cooling, the call retries on the fallback models in order (their quotas are
separate); when nothing is left, it raises :class:`RateLimited` carrying the soonest time
a retry could work, so the caller can wait exactly long enough rather than guessing.

Nothing here ever logs a key. The key is passed in a header, the call log stores a
rotation *index*, and any error text is scrubbed through the settings redactor before it
leaves the module.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from storygit.providers.base import (
    LLMRequest,
    LLMResponse,
    ProviderDown,
    RateLimited,
    Role,
    estimate_tokens,
)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_COOLDOWN_S = 60.0
"""Cooldown applied when a 429 arrives without a usable ``Retry-After`` header."""


@dataclass
class KeyState:
    """Health and usage of one rotation slot.

    Cooldowns are tracked per ``(key, model)`` because that is how the free tier's
    quotas actually work: exhausting a key on the primary model says nothing about that
    same key's allowance on a fallback model. Tracking them per key alone would throw
    away the fallback capacity the runbook depends on.

    Attributes:
        index: 1-based rotation slot, the only thing about a key that is ever logged.
        cooldowns: Model id to the monotonic timestamp before which it must not be used.
        requests: Successful requests served by this key.
        tokens: Total tokens served by this key.
        failures: Consecutive failures; reset on success.
    """

    index: int
    cooldowns: dict[str, float] = field(default_factory=dict)
    requests: int = 0
    tokens: int = 0
    failures: int = 0

    def healthy(self, now: float, model: str) -> bool:
        """Whether this key is out of cooldown for this model."""
        return now >= self.cooldowns.get(model, 0.0)

    def cool(self, now: float, model: str, seconds: float) -> None:
        """Put this key/model pair on cooldown for at least ``seconds``."""
        self.cooldowns[model] = max(self.cooldowns.get(model, 0.0), now + seconds)
        self.failures += 1

    def cooling_until(self, models: list[str]) -> float:
        """Soonest moment any of these models becomes usable on this key."""
        return min((self.cooldowns.get(m, 0.0) for m in models), default=0.0)


@dataclass
class AttemptFailure:
    """One key/model attempt that did not work.

    Surfaced so the router can log intra-rotation failures. A call that succeeded on
    key 3 because keys 1 and 2 were rate limited should leave three rows in the call
    log, not one, or the log understates what the quota is doing.

    Attributes:
        key_index: The rotation slot that failed.
        model: The model that was tried.
        error: The exception type and message, already scrubbed of secrets.
    """

    key_index: int
    model: str
    error: str


@dataclass
class RotationReport:
    """What the rotation did, for logs and tests.

    Attributes:
        key_index: The slot that eventually served the call, or -1 if none did.
        model: The model that served it, possibly a fallback.
        attempts: Every ``(key index, model)`` tried, in order.
        failures: The attempts that failed, with their (scrubbed) errors.
    """

    key_index: int
    model: str
    attempts: list[tuple[int, str]] = field(default_factory=list)
    failures: list[AttemptFailure] = field(default_factory=list)


class GeminiProvider:
    """Round-robin Gemini client with per-key rate-limit state.

    Attributes:
        name: ``"gemini"``.
        model: Primary model id.
        fallback_models: Models tried when every key is cooling on the primary.
    """

    name = "gemini"

    def __init__(
        self,
        api_keys: list[str],
        *,
        model: str,
        fallback_models: list[str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_s: float = 90.0,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], Any] | None = None,
        redactor: Callable[[str], str] | None = None,
    ) -> None:
        """Create the provider.

        Args:
            api_keys: Keys in rotation order. Held in memory only.
            model: Primary model id.
            fallback_models: Models to try when the primary is exhausted everywhere.
            transport: httpx transport; tests inject a mock so no socket is opened.
            timeout_s: Per-request timeout.
            clock: Monotonic clock, injectable so cooldown expiry is testable.
            sleeper: Awaitable sleep, injectable so tests do not actually wait.
            redactor: Scrubs secrets from error text.

        Raises:
            ValueError: If no keys were supplied.
        """
        if not api_keys:
            raise ValueError("GeminiProvider needs at least one API key")
        self._keys = list(api_keys)
        self.model = model
        self.fallback_models = list(fallback_models or [])
        self._states = [KeyState(index=i + 1) for i in range(len(self._keys))]
        self._cursor = 0
        self._clock = clock or time.monotonic
        self._sleep = sleeper or asyncio.sleep
        self._timeout = timeout_s
        self._client = httpx.AsyncClient(transport=transport, timeout=timeout_s)
        if redactor is None:
            from storygit.config import get_settings

            redactor = get_settings().redact
        self._redact = redactor
        self.last_rotation: RotationReport | None = None

    # --- rotation ---------------------------------------------------------------

    def key_states(self) -> list[KeyState]:
        """Current per-key health, for the call log and diagnostics."""
        return list(self._states)

    def _next_healthy(self, model: str) -> int | None:
        """Index into ``self._keys`` of the next key healthy for this model."""
        now = self._clock()
        for offset in range(len(self._keys)):
            candidate = (self._cursor + offset) % len(self._keys)
            if self._states[candidate].healthy(now, model):
                self._cursor = (candidate + 1) % len(self._keys)
                return candidate
        return None

    def _soonest_retry(self, models: list[str]) -> float:
        """Seconds until the first key/model pair comes out of cooldown."""
        now = self._clock()
        soonest = min((state.cooling_until(models) for state in self._states), default=now)
        return max(0.0, soonest - now)

    # --- request construction ---------------------------------------------------

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        """Translate an LLMRequest into Gemini's ``generateContent`` body."""
        contents: list[dict[str, Any]] = []
        for message in request.conversation():
            role = "model" if message.role is Role.assistant else "user"
            contents.append({"role": role, "parts": [{"text": message.content}]})
        if not contents:
            contents = [{"role": "user", "parts": [{"text": ""}]}]

        generation: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": request.max_tokens,
        }
        if request.stop:
            generation["stopSequences"] = list(request.stop)
        if request.json_schema is not None:
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = _to_gemini_schema(request.json_schema)

        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        system = request.system_text()
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    # --- the call ---------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one completion, rotating keys and then models as needed.

        Args:
            request: The call to make.

        Returns:
            The response, with ``key_index`` set to the rotation slot that served it.

        Raises:
            RateLimited: Every key is cooling on every model. ``retry_after`` says how
                long until the soonest one is usable.
            ProviderDown: A non-rate-limit failure that did not recover.
        """
        models = [request.model or self.model, *self.fallback_models]
        attempts: list[tuple[int, str]] = []
        failures: list[AttemptFailure] = []
        last_error: Exception | None = None

        for model in models:
            for _ in range(len(self._keys)):
                slot = self._next_healthy(model)
                if slot is None:
                    break
                state = self._states[slot]
                attempts.append((state.index, model))
                try:
                    response = await self._call_once(request, model, slot)
                except (RateLimited, ProviderDown) as exc:
                    cooldown = exc.retry_after if isinstance(exc, RateLimited) else 5.0
                    state.cool(self._clock(), model, cooldown)
                    failures.append(
                        AttemptFailure(
                            key_index=state.index,
                            model=model,
                            error=self._redact(f"{type(exc).__name__}: {exc}"),
                        )
                    )
                    last_error = exc
                    continue
                state.failures = 0
                state.requests += 1
                state.tokens += response.total_tokens
                self.last_rotation = RotationReport(
                    key_index=state.index, model=model, attempts=attempts, failures=failures
                )
                return response

        retry_after = self._soonest_retry(models)
        self.last_rotation = RotationReport(
            key_index=-1, model=models[-1], attempts=attempts, failures=failures
        )
        message = (
            f"gemini: all {len(self._keys)} keys are rate limited across "
            f"{len(models)} model(s); soonest retry in {retry_after:.0f}s"
        )
        if isinstance(last_error, ProviderDown):
            raise ProviderDown(f"{message} (last error: {last_error})") from last_error
        raise RateLimited(message, retry_after=retry_after or DEFAULT_COOLDOWN_S)

    async def _call_once(self, request: LLMRequest, model: str, slot: int) -> LLMResponse:
        """One HTTP call against one key.

        Raises:
            RateLimited: On 429 or a quota-flavoured 4xx.
            ProviderDown: On 5xx, transport failure, or an unparseable body.
        """
        url = f"{API_BASE}/models/{model}:generateContent"
        payload = self._build_payload(request)
        started = time.perf_counter()
        try:
            http_response = await self._client.post(
                url,
                json=payload,
                headers={
                    "x-goog-api-key": self._keys[slot],
                    "content-type": "application/json",
                },
            )
        except httpx.HTTPError as exc:
            raise ProviderDown(f"gemini transport error: {self._redact(str(exc))}") from exc
        latency = time.perf_counter() - started

        if http_response.status_code == 429:
            raise RateLimited(
                "gemini: rate limited", retry_after=_retry_after(http_response, DEFAULT_COOLDOWN_S)
            )
        if http_response.status_code >= 500:
            raise ProviderDown(f"gemini: HTTP {http_response.status_code}")
        if http_response.status_code >= 400:
            body = self._redact(http_response.text[:400])
            if "quota" in body.lower() or "resource_exhausted" in body.lower():
                raise RateLimited(
                    "gemini: quota exhausted", retry_after=_retry_after(http_response, 300.0)
                )
            raise ProviderDown(f"gemini: HTTP {http_response.status_code}: {body}")

        try:
            data = http_response.json()
        except json.JSONDecodeError as exc:
            raise ProviderDown("gemini: response was not JSON") from exc

        text = _extract_text(data)
        usage = data.get("usageMetadata") or {}
        return LLMResponse(
            text=text,
            model=data.get("modelVersion") or model,
            provider=self.name,
            prompt_tokens=int(usage.get("promptTokenCount") or estimate_tokens(str(payload))),
            completion_tokens=int(usage.get("candidatesTokenCount") or estimate_tokens(text)),
            latency_s=latency,
            key_index=self._states[slot].index,
        )

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


def _retry_after(response: httpx.Response, default: float) -> float:
    """Read ``Retry-After`` when the API supplies it, else use a default."""
    raw = response.headers.get("retry-after")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


def _extract_text(data: dict[str, Any]) -> str:
    """Pull the completion text out of a ``generateContent`` response."""
    candidates = data.get("candidates") or []
    for candidate in candidates:
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        if text:
            return text
    return ""


_SCHEMA_KEYS = {
    "type",
    "description",
    "enum",
    "items",
    "properties",
    "required",
    "nullable",
    "format",
    # Length bounds are forwarded deliberately: without them a small model can fall into
    # a repetition loop and fill a string field until it hits the token cap, which turns
    # a good candidate into a truncated-JSON parse failure.
    "maxLength",
    "minLength",
    "maxItems",
    "minItems",
}


def _to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Reduce a JSON Schema to the subset Gemini's ``responseSchema`` accepts.

    Pydantic emits ``$defs``/``$ref``, ``anyOf``, ``title`` and ``default``, none of
    which Gemini accepts. This inlines references and drops the rest, which is lossy but
    safe: the schema is a generation *hint*, and the authoritative validation is the
    Pydantic parse on the way back in.

    Args:
        schema: A JSON Schema, typically from ``Model.model_json_schema()``.

    Returns:
        A Gemini-compatible schema.
    """
    defs = schema.get("$defs", {})

    def convert(node: Any) -> Any:
        if isinstance(node, list):
            return [convert(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref = str(node["$ref"]).rsplit("/", 1)[-1]
            return convert(defs.get(ref, {"type": "string"}))
        if "anyOf" in node:
            options = [opt for opt in node["anyOf"] if opt.get("type") != "null"]
            converted = convert(options[0]) if options else {"type": "string"}
            if len(options) < len(node["anyOf"]) and isinstance(converted, dict):
                converted["nullable"] = True
            return converted
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key not in _SCHEMA_KEYS:
                continue
            out[key] = (
                {k: convert(v) for k, v in value.items()}
                if key == "properties" and isinstance(value, dict)
                else convert(value)
            )
        if out.get("type") == "object" and "properties" not in out:
            out["properties"] = {}
        return out

    return dict(convert({k: v for k, v in schema.items() if k != "$defs"}))
