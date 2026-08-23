"""One routing table, and the wrapper that makes every call cached, costed, and logged.

The routing table is the whole cost policy in one readable place: prose goes to Gemini,
extraction and classification go to Groq, embeddings and NLI stay local, and OpenRouter
is unreachable. Because routing is keyed on the *purpose tag* rather than chosen at the
call site, changing the policy is a one-line edit here and nothing else moves.

Around the chosen backend, the router adds four things in a fixed order:

1. **Cache** — a read-through lookup on the request digest. A repeated call never leaves
   the process, which is what makes a 15-episode evaluation run affordable.
2. **Budget** — checked before the call for metered providers.
3. **Fallback** — Gemini and Groq back each other up for non-prose purposes, per the
   runbook. Prose never falls back to the small model; a bad sentence is worse than a
   missing one.
4. **Logging** — one row per call, always, including failures.
"""

from __future__ import annotations

from typing import Any

from storygit.config import Settings, get_settings
from storygit.providers.base import (
    LLMRequest,
    LLMResponse,
    OpenRouterDisabledError,
    Provider,
    ProviderDown,
    ProviderError,
    RateLimited,
)
from storygit.providers.budget import BudgetGuard
from storygit.providers.cache import LLMCache, cache_key
from storygit.providers.calllog import CallLog
from storygit.providers.pricing import estimate_cost

PROSE_PURPOSES = ("propose.",)
"""Purpose prefixes that must never be served by the small fast model."""

ROUTING: dict[str, str] = {
    # Generation of anything a reader will see.
    "propose": "gemini",
    # Structured extraction and cheap classification.
    "extract": "groq",
    "classify": "groq",
    "simulate": "groq",
    # Scoring and soft critique.
    "judge": "gemini",
    # Chunk 7 only, and only with Arush present.
    "final": "openrouter",
}
"""Purpose *family* (the part before the first dot) to provider name."""


class Router:
    """Chooses a provider per purpose and wraps it in cache, budget, and logging.

    Attributes:
        providers: Available backends by name.
        cache: The read-through cache.
        calllog: The call log.
        budget: The spend guard.
    """

    def __init__(
        self,
        providers: dict[str, Provider],
        *,
        cache: LLMCache | None = None,
        calllog: CallLog | None = None,
        budget: BudgetGuard | None = None,
        routing: dict[str, str] | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Wire the router.

        Args:
            providers: Backends by name. Missing ones simply become unroutable.
            cache: Read-through cache; an in-memory one is created if omitted.
            calllog: Call log; an in-memory one is created if omitted.
            budget: Spend guard; one at the configured cap is created if omitted.
            routing: Purpose family to provider. Defaults to :data:`ROUTING`.
            settings: Settings; loaded from the environment if omitted.
        """
        self._settings = settings or get_settings()
        self.providers = dict(providers)
        self.cache = cache if cache is not None else LLMCache(":memory:")
        self.calllog = calllog if calllog is not None else CallLog(":memory:")
        self.budget = (
            budget
            if budget is not None
            else BudgetGuard(":memory:", cap_usd=self._settings.openrouter_budget_usd)
        )
        self.routing = dict(routing or ROUTING)

    # --- routing ----------------------------------------------------------------

    def provider_for(self, purpose: str) -> Provider:
        """The backend that serves a purpose tag.

        Args:
            purpose: A dotted tag such as ``propose.beat``.

        Returns:
            The provider.

        Raises:
            ProviderDown: If the purpose has no route, or the routed provider is not
                configured (typically a missing key).
        """
        family = purpose.split(".", 1)[0]
        name = self.routing.get(family)
        if name is None:
            raise ProviderDown(f"no route for purpose {purpose!r}")
        provider = self.providers.get(name)
        if provider is None:
            raise ProviderDown(f"provider {name!r} (for {purpose!r}) is not configured")
        return provider

    def _fallback_for(self, purpose: str, primary: str) -> Provider | None:
        """The backup backend for a purpose, if one is allowed.

        Prose purposes have no fallback: the runbook says fall back to Groq for
        *non-prose* tasks, and silently serving a beat from a 8B model would be a
        quality regression the writer could not see the cause of.
        """
        if any(purpose.startswith(prefix) for prefix in PROSE_PURPOSES):
            return None
        alternative = "groq" if primary == "gemini" else "gemini"
        return self.providers.get(alternative)

    # --- the call ---------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Serve one request: cache, budget, provider, fallback, log.

        Args:
            request: The call to make. Its ``purpose`` decides the route.

        Returns:
            The response, from cache or from a provider.

        Raises:
            ProviderError: Any subclass, when the primary and its fallback both fail.
                The failure is logged before it is re-raised.
        """
        provider = self.provider_for(request.purpose)
        model: str = request.model or str(getattr(provider, "model", "unknown"))

        key = cache_key(provider.name, model, request)
        if not request.bypass_cache:
            hit = self.cache.get(key)
            if hit is not None:
                self.calllog.record(
                    provider=hit.provider,
                    model=hit.model,
                    purpose=request.purpose,
                    key_index=hit.key_index,
                    prompt_tokens=hit.prompt_tokens,
                    completion_tokens=hit.completion_tokens,
                    latency_s=0.0,
                    cached=True,
                    cost_usd=0.0,
                )
                return hit

        try:
            response = await self._call(provider, request, model)
        except OpenRouterDisabledError:
            # The lock is not a transient failure and must never be routed around.
            raise
        except (RateLimited, ProviderDown) as primary_error:
            fallback = self._fallback_for(request.purpose, provider.name)
            if fallback is None:
                self._log_error(provider.name, model, request, primary_error)
                raise
            fallback_model = str(request.model or getattr(fallback, "model", "unknown"))
            fallback_key = cache_key(fallback.name, fallback_model, request)
            if not request.bypass_cache:
                # A flaky primary should not mean paying for the fallback twice.
                hit = self.cache.get(fallback_key)
                if hit is not None:
                    self._log_error(provider.name, model, request, primary_error)
                    self.calllog.record(
                        provider=hit.provider,
                        model=hit.model,
                        purpose=request.purpose,
                        key_index=hit.key_index,
                        prompt_tokens=hit.prompt_tokens,
                        completion_tokens=hit.completion_tokens,
                        cached=True,
                    )
                    return hit
            try:
                response = await self._call(fallback, request, request.model)
            except ProviderError as fallback_error:
                self._log_error(provider.name, model, request, primary_error)
                self._log_error(fallback.name, fallback_model, request, fallback_error)
                raise
            self._log_error(provider.name, model, request, primary_error)
            key = cache_key(response.provider, response.model, request)

        self._log_rotation_failures(request)
        if not request.bypass_cache:
            self.cache.put(key, request.purpose, response)
        self.calllog.record(
            provider=response.provider,
            model=response.model,
            purpose=request.purpose,
            key_index=response.key_index,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_s=response.latency_s,
            cached=False,
            cost_usd=response.cost_usd,
        )
        return response

    async def _call(
        self, provider: Provider, request: LLMRequest, model: str | None
    ) -> LLMResponse:
        """Invoke one provider, costing the response if the provider did not."""
        response = await provider.complete(request)
        if response.cost_usd == 0.0:
            cost = estimate_cost(
                response.provider,
                response.model,
                response.prompt_tokens,
                response.completion_tokens,
            )
            if cost:
                response = response.model_copy(update={"cost_usd": cost})
        return response

    def _log_rotation_failures(self, request: LLMRequest) -> None:
        """Record key/model attempts that failed inside a provider's own rotation.

        A call that succeeded on key 3 because keys 1 and 2 were rate limited should
        leave three rows, not one; otherwise the log understates quota pressure and the
        cost report looks healthier than the run actually was.
        """
        for provider in self.providers.values():
            report = getattr(provider, "last_rotation", None)
            if report is None:
                continue
            for failure in getattr(report, "failures", ()):
                self.calllog.record(
                    provider=provider.name,
                    model=failure.model,
                    purpose=request.purpose,
                    key_index=failure.key_index,
                    error=failure.error,
                )
            report.failures.clear()

    def _log_error(
        self, provider_name: str, model: str, request: LLMRequest, error: Exception
    ) -> None:
        """Record a failed attempt, with the message scrubbed of secrets."""
        self.calllog.record(
            provider=provider_name,
            model=model,
            purpose=request.purpose,
            error=f"{type(error).__name__}: {error}",
        )

    # --- lifecycle --------------------------------------------------------------

    def mark(self) -> int:
        """A call-log marker, so a later :meth:`summary` can be scoped to one run."""
        return self.calllog.mark()

    def summary(self, *, since: int = 0) -> dict[str, Any]:
        """Call-log summary plus cache statistics.

        Args:
            since: Only count calls after this :meth:`mark`; zero summarises everything.
        """
        summary = self.calllog.summary(since=since)
        summary["cache"] = self.cache.stats()
        summary["budget_remaining_usd"] = self.budget.remaining()
        return summary

    async def aclose(self) -> None:
        """Close every backend's HTTP client and the local databases."""
        for provider in self.providers.values():
            closer = getattr(provider, "aclose", None)
            if closer is not None:
                await closer()
        self.cache.close()
        self.calllog.close()
        self.budget.close()


STRONG_MODEL_ROUTING: dict[str, str] = dict(ROUTING) | {
    "propose": "openrouter",
    "judge": "openrouter",
}
"""Routing for the strong-model rerun: generation and judging move to the metered provider.

Extraction and classification stay on the free tier deliberately. The question the rerun
answers is whether the *system* works or the *model* is good, and that question is about
the prose and the ranking. Moving extraction too would change a second variable and cost
money to learn nothing.

This is the "one-line edit" the routing decision was made for: the policy lives in one
table, so redirecting the expensive half of the pipeline is a dictionary literal rather
than a sweep through forty call sites.
"""


def build_router(
    settings: Settings | None = None,
    *,
    cache_path: str | None = None,
    calllog_path: str | None = None,
    strong_model: str | None = None,
) -> Router:
    """Construct the production router from settings.

    Providers with no key configured are simply omitted, so the router degrades to
    whatever is actually available rather than failing at import time. OpenRouter is
    always constructed and always locked.

    Args:
        settings: Settings; loaded from the environment if omitted.
        cache_path: Override the cache database path.
        calllog_path: Override the call-log database path.
        strong_model: Route generation and judging to the metered provider using this
            model id. The lock and the budget guard are untouched: the provider still
            refuses every call unless ``OPENROUTER_ENABLED`` is exactly ``"true"``, and
            still refuses any call it cannot afford under the cap.

    Returns:
        A wired router.
    """
    settings = settings or get_settings()
    providers: dict[str, Provider] = {}

    if settings.gemini_keys:
        from storygit.providers.gemini import GeminiProvider

        providers["gemini"] = GeminiProvider(
            settings.gemini_keys,
            model=settings.gemini_model,
            fallback_models=settings.gemini_fallbacks,
            timeout_s=settings.request_timeout_s,
            redactor=settings.redact,
        )
    if settings.groq_api_key.get_secret_value():
        from storygit.providers.groq import GroqProvider

        providers["groq"] = GroqProvider(
            settings.groq_api_key.get_secret_value(),
            model=settings.groq_model,
            timeout_s=settings.request_timeout_s,
            redactor=settings.redact,
        )

    cache_db = cache_path or str(settings.resolved_cache_path)
    log_db = calllog_path or str(settings.resolved_cache_path.parent / "calllog.db")
    budget = BudgetGuard(
        settings.resolved_cache_path.parent / "spend.db",
        cap_usd=settings.openrouter_budget_usd,
    )

    from storygit.providers.openrouter import OpenRouterProvider

    providers["openrouter"] = OpenRouterProvider(
        settings.openrouter_api_key.get_secret_value(),
        model=strong_model or settings.openrouter_model,
        enabled_flag=settings.openrouter_enabled,
        budget=budget,
        timeout_s=settings.request_timeout_s,
        redactor=settings.redact,
    )

    return Router(
        providers,
        cache=LLMCache(cache_db),
        calllog=CallLog(log_db, redactor=settings.redact),
        budget=budget,
        routing=STRONG_MODEL_ROUTING if strong_model else None,
        settings=settings,
    )
