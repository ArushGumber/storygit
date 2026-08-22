"""Provider layer: rotation, cooldowns, caching, the OpenRouter lock, and key hygiene.

Every test here is offline. The transports are `httpx.MockTransport` instances that
assert on what they were handed; none of them opens a socket.
"""

from __future__ import annotations

import json

import httpx
import pytest

from storygit.providers.base import (
    BudgetExceeded,
    LLMRequest,
    Message,
    OpenRouterDisabledError,
    ProviderDown,
    ProviderError,
    RateLimited,
    Role,
)
from storygit.providers.budget import BudgetGuard
from storygit.providers.cache import LLMCache, cache_key
from storygit.providers.calllog import CallLog
from storygit.providers.gemini import GeminiProvider
from storygit.providers.groq import GroqProvider
from storygit.providers.openrouter import OpenRouterProvider
from storygit.providers.router import Router

SENTINEL_KEYS = [f"SENTINEL_KEY_VALUE_{i}" for i in range(1, 7)]
"""Fake keys with a value we can grep for anywhere a secret might leak."""


def request(purpose: str = "propose.beat", **kwargs: object) -> LLMRequest:
    """A minimal request."""
    return LLMRequest(
        messages=(Message(role=Role.user, content="write a beat"),),
        purpose=purpose,
        **kwargs,  # type: ignore[arg-type]
    )


def gemini_body(text: str = "ok") -> dict[str, object]:
    """A well-formed generateContent response."""
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7},
        "modelVersion": "gemini-test",
    }


class Clock:
    """A monotonic clock the test advances by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def _no_sleep(_seconds: float) -> None:
    """Stand-in for asyncio.sleep so tests never actually wait."""


# --- rotation -----------------------------------------------------------------


async def test_rotation_skips_rate_limited_keys() -> None:
    seen: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        key = req.headers["x-goog-api-key"]
        seen.append(key)
        if key in SENTINEL_KEYS[:2]:
            return httpx.Response(429, headers={"retry-after": "30"}, json={"error": "slow down"})
        return httpx.Response(200, json=gemini_body())

    provider = GeminiProvider(
        SENTINEL_KEYS,
        model="m1",
        transport=httpx.MockTransport(handler),
        clock=Clock(),
        sleeper=_no_sleep,
        redactor=lambda t: t,
    )
    response = await provider.complete(request())

    assert response.key_index == 3, "keys 1 and 2 returned 429, so the call lands on key 3"
    assert seen == SENTINEL_KEYS[:3]
    states = provider.key_states()
    assert not states[0].healthy(1000.0, "m1") and not states[1].healthy(1000.0, "m1")
    assert states[2].requests == 1


async def test_all_keys_cooling_falls_back_to_the_next_model() -> None:
    models: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        model = str(req.url).split("/models/")[1].split(":")[0]
        models.append(model)
        if model == "primary":
            return httpx.Response(429, json={"error": "quota"})
        return httpx.Response(200, json=gemini_body("fallback answer"))

    provider = GeminiProvider(
        SENTINEL_KEYS[:2],
        model="primary",
        fallback_models=["backup"],
        transport=httpx.MockTransport(handler),
        clock=Clock(),
        sleeper=_no_sleep,
        redactor=lambda t: t,
    )
    response = await provider.complete(request())

    assert response.text == "fallback answer"
    assert models[:2] == ["primary", "primary"]
    assert "backup" in models


async def test_total_exhaustion_raises_rate_limited_with_a_retry_hint() -> None:
    def handler(_req: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "45"}, json={"error": "no"})

    clock = Clock()
    provider = GeminiProvider(
        SENTINEL_KEYS,
        model="m1",
        transport=httpx.MockTransport(handler),
        clock=clock,
        sleeper=_no_sleep,
        redactor=lambda t: t,
    )
    with pytest.raises(RateLimited) as excinfo:
        await provider.complete(request())
    assert 0 < excinfo.value.retry_after <= 45


async def test_cooldowns_expire() -> None:
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "30"}, json={})
        return httpx.Response(200, json=gemini_body())

    clock = Clock()
    provider = GeminiProvider(
        [SENTINEL_KEYS[0]],
        model="m1",
        transport=httpx.MockTransport(handler),
        clock=clock,
        sleeper=_no_sleep,
        redactor=lambda t: t,
    )
    with pytest.raises(RateLimited):
        await provider.complete(request())

    clock.advance(31)
    response = await provider.complete(request())
    assert response.key_index == 1, "the key comes back once its cooldown has passed"


async def test_server_error_becomes_provider_down() -> None:
    provider = GeminiProvider(
        [SENTINEL_KEYS[0]],
        model="m1",
        transport=httpx.MockTransport(lambda _r: httpx.Response(503, text="upstream boom")),
        clock=Clock(),
        sleeper=_no_sleep,
        redactor=lambda t: t,
    )
    with pytest.raises(ProviderDown):
        await provider.complete(request())


# --- cache --------------------------------------------------------------------


async def test_cache_serves_the_second_identical_call() -> None:
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=gemini_body(f"answer {calls['n']}"))

    provider = GeminiProvider(
        [SENTINEL_KEYS[0]],
        model="m1",
        transport=httpx.MockTransport(handler),
        clock=Clock(),
        sleeper=_no_sleep,
        redactor=lambda t: t,
    )
    router = Router({"gemini": provider})

    first = await router.complete(request())
    second = await router.complete(request())

    assert calls["n"] == 1, "the second identical request must not reach the transport"
    assert second.cached is True and first.cached is False
    assert second.text == first.text
    assert router.cache.stats()["hits"] == 1


async def test_sample_index_and_bypass_defeat_the_cache() -> None:
    calls = {"n": 0}

    def handler(_req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=gemini_body(f"answer {calls['n']}"))

    provider = GeminiProvider(
        [SENTINEL_KEYS[0]],
        model="m1",
        transport=httpx.MockTransport(handler),
        clock=Clock(),
        sleeper=_no_sleep,
        redactor=lambda t: t,
    )
    router = Router({"gemini": provider})

    await router.complete(request())
    await router.complete(request(sample_index=1))
    assert calls["n"] == 2, "parallel samples of one prompt must not collapse onto one entry"

    await router.complete(request(bypass_cache=True))
    assert calls["n"] == 3


def test_cache_key_depends_on_everything_that_changes_the_answer() -> None:
    base = request()
    key = cache_key("gemini", "m1", base)
    assert key != cache_key("gemini", "m2", base)
    assert key != cache_key("groq", "m1", base)
    assert key != cache_key("gemini", "m1", base.model_copy(update={"temperature": 0.1}))
    assert key != cache_key("gemini", "m1", base.model_copy(update={"sample_index": 2}))
    assert key == cache_key("gemini", "m1", base.model_copy(update={"purpose": "judge.soft"}))


def test_cache_round_trips_a_response() -> None:
    from storygit.providers.base import LLMResponse

    cache = LLMCache(":memory:")
    response = LLMResponse(text="hello", model="m", provider="gemini", prompt_tokens=3)
    cache.put("k", "propose.beat", response)
    loaded = cache.get("k")
    assert loaded is not None
    assert loaded.text == "hello" and loaded.cached is True
    assert cache.stats()["entries"] == 1


# --- the OpenRouter lock ------------------------------------------------------


@pytest.mark.parametrize("flag", ["", "false", "False", "True", "TRUE", "1", "yes", "0", " true"])
async def test_openrouter_is_locked_for_every_flag_but_exactly_true(flag: str) -> None:
    touched = {"transport": False}

    def handler(_req: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        touched["transport"] = True
        return httpx.Response(200, json={})

    provider = OpenRouterProvider(
        "SENTINEL_OPENROUTER_KEY",
        model="anthropic/claude-3.5-sonnet",
        enabled_flag=flag,
        budget=BudgetGuard(":memory:", cap_usd=12.0),
        transport=httpx.MockTransport(handler),
        redactor=lambda t: t,
    )
    assert provider.enabled is False
    with pytest.raises(OpenRouterDisabledError):
        await provider.complete(request("final.eval"))
    assert touched["transport"] is False, "the lock must raise before any network activity"


async def test_router_never_routes_around_the_openrouter_lock() -> None:
    provider = OpenRouterProvider(
        "SENTINEL_OPENROUTER_KEY",
        model="m",
        enabled_flag="false",
        budget=BudgetGuard(":memory:"),
        redactor=lambda t: t,
    )
    gemini = GeminiProvider(
        [SENTINEL_KEYS[0]],
        model="m1",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=gemini_body())),
        clock=Clock(),
        sleeper=_no_sleep,
        redactor=lambda t: t,
    )
    router = Router({"openrouter": provider, "gemini": gemini})
    with pytest.raises(OpenRouterDisabledError):
        await router.complete(request("final.eval"))


async def test_budget_guard_refuses_a_call_that_would_cross_the_cap() -> None:
    budget = BudgetGuard(":memory:", cap_usd=12.0)
    budget.record("openrouter", "anthropic/claude-3.5-sonnet", "final.eval", 11.999)
    provider = OpenRouterProvider(
        "SENTINEL_OPENROUTER_KEY",
        model="anthropic/claude-3.5-sonnet",
        enabled_flag="true",  # unlocked only inside this unit test, against no transport
        budget=budget,
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})
        ),
        redactor=lambda t: t,
    )
    assert provider.enabled is True
    with pytest.raises(BudgetExceeded):
        await provider.complete(request("final.eval", max_tokens=4000))
    assert budget.remaining() < 0.01


def test_budget_ignores_free_providers() -> None:
    budget = BudgetGuard(":memory:", cap_usd=0.0)
    budget.check("gemini", "anything", prompt_tokens=10_000, max_output_tokens=10_000)
    budget.check("groq", "anything", prompt_tokens=10_000, max_output_tokens=10_000)


# --- key hygiene --------------------------------------------------------------


async def test_no_key_material_reaches_logs_cache_keys_or_errors() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.headers["x-goog-api-key"] == SENTINEL_KEYS[0]:
            return httpx.Response(
                403, text=f"Invalid key {SENTINEL_KEYS[0]} for project", json=None
            )
        return httpx.Response(200, json=gemini_body())

    def redact(text: str) -> str:
        for key in SENTINEL_KEYS:
            text = text.replace(key, "<redacted>")
        return text

    provider = GeminiProvider(
        SENTINEL_KEYS,
        model="m1",
        transport=httpx.MockTransport(handler),
        clock=Clock(),
        sleeper=_no_sleep,
        redactor=redact,
    )
    calllog = CallLog(":memory:", redactor=redact)
    router = Router({"gemini": provider}, calllog=calllog)

    response = await router.complete(request())
    assert response.key_index == 2

    haystack = json.dumps(calllog.rows()) + json.dumps(router.summary(), default=str)
    for key in SENTINEL_KEYS:
        assert key not in haystack, "a key leaked into the call log or summary"
    assert "<redacted>" in haystack, "the failure was recorded, with the key scrubbed"
    assert any(row["key_index"] == 2 for row in calllog.rows())

    for key in SENTINEL_KEYS:
        assert key not in cache_key("gemini", "m1", request())


async def test_error_messages_are_scrubbed() -> None:
    def redact(text: str) -> str:
        return text.replace(SENTINEL_KEYS[0], "<redacted>")

    provider = GeminiProvider(
        [SENTINEL_KEYS[0]],
        model="m1",
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(400, text=f"bad key {SENTINEL_KEYS[0]}")
        ),
        clock=Clock(),
        sleeper=_no_sleep,
        redactor=redact,
    )
    with pytest.raises(ProviderDown) as excinfo:
        await provider.complete(request())
    assert SENTINEL_KEYS[0] not in str(excinfo.value)
    assert "<redacted>" in str(excinfo.value)


# --- routing and logging ------------------------------------------------------


async def test_routing_sends_prose_to_gemini_and_extraction_to_groq() -> None:
    gemini = GeminiProvider(
        [SENTINEL_KEYS[0]],
        model="gm",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=gemini_body("prose"))),
        clock=Clock(),
        sleeper=_no_sleep,
        redactor=lambda t: t,
    )
    groq = GroqProvider(
        "SENTINEL_GROQ_KEY",
        model="gq",
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "facts"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                    "model": "gq",
                },
            )
        ),
        redactor=lambda t: t,
    )
    router = Router({"gemini": gemini, "groq": groq})

    assert (await router.complete(request("propose.beat"))).provider == "gemini"
    assert (await router.complete(request("extract.facts"))).provider == "groq"
    assert (await router.complete(request("judge.soft"))).provider == "gemini"

    purposes = {row["purpose"]: row for row in router.calllog.rows()}
    assert set(purposes) == {"propose.beat", "extract.facts", "judge.soft"}
    assert purposes["extract.facts"]["provider"] == "groq"

    summary = router.summary()
    assert summary["calls"] == 3
    assert summary["total_tokens"] > 0
    assert {row["purpose"] for row in summary["by_purpose"]} == set(purposes)


async def test_non_prose_falls_back_to_the_other_provider_but_prose_does_not() -> None:
    def dead_gemini() -> GeminiProvider:
        return GeminiProvider(
            [SENTINEL_KEYS[0]],
            model="gm",
            transport=httpx.MockTransport(lambda _r: httpx.Response(503, text="down")),
            clock=Clock(),
            sleeper=_no_sleep,
            redactor=lambda t: t,
        )

    groq_calls = {"n": 0}

    def groq_handler(_req: httpx.Request) -> httpx.Response:
        groq_calls["n"] += 1
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "rescued"}}], "model": "gq"}
        )

    def groq() -> GroqProvider:
        return GroqProvider(
            "SENTINEL_GROQ_KEY",
            model="gq",
            transport=httpx.MockTransport(groq_handler),
            redactor=lambda t: t,
        )

    # A judging call is non-prose, so the runbook's Groq fallback applies.
    rescued = await Router({"gemini": dead_gemini(), "groq": groq()}).complete(
        request("judge.soft")
    )
    assert rescued.provider == "groq" and rescued.text == "rescued"
    assert groq_calls["n"] == 1

    # Prose must never be quietly served by the small model.
    prose_router = Router({"gemini": dead_gemini(), "groq": groq()})
    with pytest.raises(ProviderError):
        await prose_router.complete(request("propose.beat"))
    assert groq_calls["n"] == 1, "a failed prose call must not silently downgrade to Groq"
    assert all(row["provider"] != "groq" for row in prose_router.calllog.rows())


async def test_unroutable_purpose_is_a_typed_error() -> None:
    router = Router({})
    with pytest.raises(ProviderDown, match="no route"):
        await router.complete(request("nonsense.thing"))
