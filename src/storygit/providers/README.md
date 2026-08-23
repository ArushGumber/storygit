# `storygit.providers`

Model access: one interface, four backends, and everything around them — rotation,
caching, cost, and logging — layered on top rather than baked into each backend.

## The interface

```python
class Provider(Protocol):
    name: str

    async def complete(self, request: LLMRequest) -> LLMResponse: ...
```

Every `LLMRequest` carries a **purpose tag** (`propose.beat`, `extract.facts`,
`judge.soft`). The router chooses the backend from it, the call log groups by it, and
the evaluation reports tokens and dollars per purpose. Nothing calls a provider directly;
everything goes through `Router.complete`.

| Backend | File | What it serves | Cost |
|---|---|---|---|
| Gemini | `gemini.py` | Prose, planning, judging | Free tier, six keys |
| Groq | `groq.py` | Extraction, classification, the simulated writer's edits | Free tier |
| Local | `local.py` | bge-small embeddings, DeBERTa-MNLI cross-encoder | CPU |
| OpenRouter | `openrouter.py` | Chunk 7 only | Metered, capped at $12 |

## Key rotation

Six free-tier keys are a real quota only if they are used properly. Each key carries
per-`(key, model)` cooldown state, because free-tier quota is per model per key —
exhausting a key on the primary model says nothing about its allowance on a fallback.

A call takes the next healthy key. On a 429 or a quota error that key/model pair goes on
cooldown (honouring `Retry-After` when present) and the call rotates. When every key is
cooling on the primary model, the fallback models are tried in order. When nothing is
left, `RateLimited` is raised carrying the soonest time a retry could work, so the caller
can wait exactly long enough instead of guessing.

Failures *inside* a rotation are surfaced and logged too: a call that succeeded on key 3
because keys 1 and 2 were rate limited leaves three rows in the log, not one. Otherwise
the log understates quota pressure and the cost report looks healthier than the run was.

## The OpenRouter lock

Two locks in series, both fail-closed:

1. `OPENROUTER_ENABLED` must be **exactly** the string `"true"`. `"True"`, `"1"`, and
   `"yes"` all leave it locked — a lock that can be tripped by a typo is not a lock.
   `OpenRouterDisabledError` is raised *before* a key is read, a URL is built, or a
   transport is touched.
2. Even unlocked, every call passes the budget guard, which refuses anything whose
   worst-case cost (assuming the model emits its full output allowance) would cross the
   in-code cap. A guard that only notices after the money is gone is not a guard.

Tests assert the lock holds for nine different flag values and that the transport is
never invoked.

## Keys are never logged

The call log schema has a `key_index` column and no column that could hold a key. Every
error message is scrubbed through `Settings.redact` before it is stored or raised. Cache
keys are digests of prompts; no secret is an input.

`tests/test_providers.py::test_no_key_material_reaches_logs_cache_keys_or_errors` runs a
rotation and a failure with sentinel keys and asserts the sentinel appears in no log row,
no summary, no cache key, and no exception message.

## Caching

Read-through, keyed on the SHA-256 of provider, model, messages, temperature, token cap,
schema, stop sequences, and **`sample_index`**. That last field is what lets six
axis-conditioned candidates for one beat be six cache entries instead of one. The cache
lives outside the repository (`<workspace>/.storygit_cache/`), because it fills with
model output and a cache checked into a deliverable ships somebody's API responses.

A second run of `scripts/smoke_live.py` makes zero network calls. That is what makes a
15-episode evaluation affordable on a free tier.

## Routing

```python
ROUTING = {
    "propose": "gemini",  # anything a reader will see
    "extract": "groq",  # structured reading
    "classify": "groq",
    "simulate": "groq",  # the evaluation's simulated writer
    "judge": "gemini",
    "final": "openrouter",  # chunk 7, with Arush present
}
```

Non-prose purposes fall back to the other provider when the primary is down. Prose does
**not**: silently serving a beat from an 8B model is a quality regression the writer
cannot see the cause of, and a missing sentence is better than a bad one.

## Worked example

```python
router = build_router()  # reads the workspace .env
response = await router.complete(
    LLMRequest(
        messages=(Message(role=Role.user, content=prompt),),
        purpose="propose.beat",
        sample_index=2,
    )
)
print(router.calllog.render_summary())
```
