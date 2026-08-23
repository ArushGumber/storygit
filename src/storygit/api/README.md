# `storygit.api`

A thin adapter over the engine. No business logic lives here.

That constraint is the reason chunks 1–5 are testable with no web machinery at all, and it
is why the whole HTTP layer is a few hundred lines. Routes translate JSON into engine calls
and engine results back into JSON; anything more interesting than that belongs a layer down.

## Files

| File | What it does |
|---|---|
| `app.py` | The app factory, the lifespan that opens and closes shared resources, CORS for the dev origin, the exception handlers, and static serving of the built frontend. |
| `deps.py` | `AppState`: repository, provider router, settings, and one engine per branch, built once and injected. |
| `schemas.py` | The wire format, and the mapping functions between it and the domain models. |
| `errors.py` | Typed engine errors → HTTP statuses and one problem shape. |
| `routers/story.py` | Reads: tree, node, world slice, threads, flags, ledger, authorship, history. |
| `routers/actions.py` | Everything that changes the story. |
| `routers/branches.py` | Branch, switch, diff, merge. |
| `routers/artifacts.py` | The Gallery sessions and the Eval figures. |

## The wire format is not the domain model

Serializing domain models directly would make every internal refactor a breaking API change
and would leak fields the frontend has no business seeing. The interface also needs
different shapes: a plan-tree node on screen wants its children inlined and its authorship
ratio computed, and neither belongs in the object that gets hashed into a snapshot.

The mapping functions in `schemas.py` are the seam, and they are the only place that has to
change when either side moves. Where a domain model happens to be exactly right — `Fact`,
`Thread`, `Flag` — it is reused rather than duplicated, because a wrapper that adds nothing
will drift.

## One error shape

Every failing route returns `{kind, detail, retry_after}`, with `kind` set to the
exception's class name. The interface gets one error path to handle, and it can tell a rate
limit (retry, and it knows for how long) from a locked node (tell the writer) without
parsing English.

```
LockedNodeError      → 409     RateLimited          → 429 (+ retry_after)
SnapshotNotFound     → 404     BudgetExceeded       → 402
ApplyError           → 422     OpenRouterDisabled   → 403
ProviderDown         → 503     SchemaParseError     → 502
```

No route leaks a traceback. That matters here specifically because provider exception text
can contain a URL, and a URL can contain a key — so every message goes through the settings
redactor on the way out.

## Single writer, one process

There is no session, no auth, and no locking between concurrent writers, because there are
none: one story database, one writer, one process. That is the right shape for a tool a
novelist runs on their own machine, and it is stated in `deps.py` because it is load-bearing
and easy to miss.

Two consequences are handled rather than assumed. SQLite connections are opened with
`check_same_thread=False`, because FastAPI serves synchronous routes from a thread pool; and
`Repository.commit_diff` takes a write lock, so the read-apply-write sequence is atomic even
if two requests arrive together. Without the lock, two concurrent commits could both branch
from the same parent and one would silently vanish.

`learning_systems.tex` describes what would change at platform scale.

## Everything goes through `commit_diff`

There is no other path by which a POST can change state. That is what makes the snapshot
history a complete record rather than a partial one — if it happened, there is a snapshot
for it, with the author and the intent line attached.

## Long calls

Proposing takes seconds: six samples plus six judge calls. They run concurrently on the
event loop, so one request costs roughly one round trip rather than twelve, and the
interface shows progress. There is no queue and no worker, because there is one writer.

## Development versus production

**Development** is two processes: Vite on 5173 proxying `/api` to uvicorn on 8000. The
proxy keeps the browser on one origin, which makes CORS a non-issue during development; the
CORS configuration exists only for anyone who bypasses the proxy.

**Production** is one process on one port: uvicorn serves the API and the built frontend
from the same origin, so CORS never applies. Unknown paths fall through to `index.html`,
because the frontend routes client-side and a hard refresh on `/gallery` must not 404.

```bash
uvicorn storygit.api.app:app --reload            # dev
STORYGIT_DB=story.db python -m storygit.api.app  # prod
scripts/e2e_smoke.sh                             # both, end to end, with a canned provider
```
