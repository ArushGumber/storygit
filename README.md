# storygit

**Version control for story state.** An AI-assisted storytelling system that helps a
human writer develop a one-line premise into a structured serial — Story → Episodes →
Scenes → Beats → Prose — while keeping the writer in charge of every change.

Built as a take-home for a serialized-fiction platform (Research Engineer). The reference point is Google
DeepMind's *Fabula*; the goal is not to reproduce it, but to fix the structural problems
its own user study surfaced.

---

## The problem

Naive long-form LLM generation fails in five reliable ways:

1. **State drift.** Facts established early are forgotten or contradicted later. The
   worst case is epistemic: a character acts on knowledge they have not been given.
2. **Broken edit semantics.** Editing one scene either propagates nowhere or rewrites
   everything, because nothing records what depends on what.
3. **Agency erosion.** The writer becomes an arbiter of good-or-bad instead of an author.
4. **Exploration overwhelm.** Unlabeled alternatives cost more to review than writing
   from scratch.
5. **No learning.** Iteration 30 repeats iteration 1's mistakes; the system never builds
   a model of *this* writer's taste.

And, for serialized audio fiction specifically: no serial mechanics — hooks,
cliffhangers, recaps, and the open threads that make a listener press *next episode*.

## The thesis

Treat the story as a **versioned, typed state graph**. Make every AI action a
**reviewable diff** against it. Make dependencies **explicit**, so edits propagate by
*marking*, not rewriting. Let the writer **own the objective**. Learn the writer's taste
from accept / reject / edit signals.

The division of labour: **the AI handles coherence and mechanics; the human handles
taste, voice, and retention.**

Fabula is a chain of prompts with a UI. This is a state machine with a learning loop.

---

## Architecture

```
                 ┌──────────────────────────────────────────────┐
   writer  ─────▶│  UI: plan tree │ labeled diffs │ world state  │
                 └───────┬──────────────────────────────┬───────┘
                         │ intent                       │ accept / reject / edit
                         ▼                              ▼
                 ┌───────────────┐              ┌───────────────┐
                 │ proposal      │  candidates  │ preference    │
                 │ engine        │─────────────▶│ layer         │
                 └───────┬───────┘              └───────┬───────┘
                         │ Diff                         │ ranking
                         ▼                              │
                 ┌───────────────┐   flags      ┌───────────────┐
                 │ continuity    │◀─────────────│ selection     │
                 │ checker (3L)  │              │ (MMR / DPP)   │
                 └───────┬───────┘              └───────────────┘
                         │
                         ▼
    ┌────────────────────────────────────────────────────────┐
    │ state store — content-addressed snapshots, branches      │
    │   plan tree · world graph (facts, epistemic edges)       │
    │   open threads · writer ledger · provenance              │
    └───────────────────────┬────────────────────────────────┘
                            │ changed facts
                            ▼
                    ┌───────────────┐
                    │ propagation   │  marks stale / review — never rewrites
                    └───────────────┘
```

| Package | What it does |
|---|---|
| `src/storygit/domain/` | The typed state: plan tree, world graph, threads, writer ledger, provenance, and the 31 diff operations that are the only way to change any of it. |
| `src/storygit/store/` | Git-shaped persistence: content-addressed objects, snapshot manifests, branches, three-way merge. |
| `src/storygit/graph/` | Deterministic dependency edges, staleness propagation, and the entity-scoped slices that generation prompts consume. |
| `src/storygit/providers/` | One interface over Gemini (six keys, rotated), Groq, local CPU models, and a hard-locked metered provider. Read-through cache, budget guard, call log. |
| `src/storygit/agents/` | Schema-constrained generation converted into typed diffs, and fact extraction from accepted prose. |
| `src/storygit/selection/` | Named conditioning axes, MMR / DPP / top-k behind one signature, and the coherent-to-surprising dial. |
| `src/storygit/continuity/` | Three checker layers, the bible diff, and the periodic audit. |
| `src/storygit/preference/` | Exemplars, edit mining, a contrastive voice model, the ranking head, and a Thompson bandit. |
| `src/storygit/api/` | A thin HTTP adapter. No logic lives here. |
| `eval/` | Simulated writers, injected ground truth, metrics, ablations, and the gallery recorder. |
| `frontend/` | Vite + React + TypeScript. Five tabs; the tool is three panes. |

Each package has its own `README.md` covering what it does, its key types, a worked
example, and the invariants it maintains.

---

## Quickstart

```bash
uv venv --python 3.11 && uv pip install -e '.[dev,ml,api]'
cd frontend && npm install && cd ..

make check        # tests, lint, types
make all          # + frontend build, end-to-end smoke, and the paper
make help         # every regeneration command, named
```

`make` exists because every artifact here has a command that recreates it. A figure or a
number nobody can regenerate is a figure or a number nobody can check.

### Run the tool

```bash
# development: two processes, Vite proxies /api
uvicorn storygit.api.app:app --reload
cd frontend && npm run dev          # http://localhost:5173

# production: one process, one port
cd frontend && npm run build && cd ..
STORYGIT_DB=story.db .venv/bin/python -m storygit.api.app   # http://127.0.0.1:8000
```

Model keys live in a workspace-level `.env` **outside this repository** and are read by the
provider layer; `.env.example` lists every variable, with the reason for each. Without keys
the tool still runs — everything deterministic works — and generation reports that no
provider is configured. Google retires model ids without notice, so if a long run starts
failing, `scripts/smoke_live.py` is the cheapest way to find out which one went.

### Run the evaluation

```bash
make numbers                                  # deterministic metrics + the paper's macros
.venv/bin/python -m eval.run --config smoke   # one tiny live run
make eval                                     # four personas, budgeted and checkpointed
make gallery                                  # the eight replayable sessions
```

Every measured number in `docs/presentable.tex` is a macro generated from the evaluation's
own output by `eval/texnumbers.py`. Nothing in the paper is typed by hand, so no figure can
go stale without the regeneration step noticing.

### Check it end to end

```bash
scripts/e2e_smoke.sh              # boots the real server, walks the levels over HTTP
scripts/smoke_live.py             # two live provider calls; run twice, the second is cached
scripts/screenshots.py            # shoots every tab for the visual audit
```

## A minimal session against the state layer

```python
from storygit.domain.diff import AddNode, Diff, DiffAuthor
from storygit.domain.ids import IdGenerator
from storygit.domain.nodes import Episode
from storygit.seed import seed_story
from storygit.store.repository import Repository

ids = IdGenerator(seed=0)
repo = Repository.open("story.db")
seed_story(repo, ids)                       # the orphan premise and its opening cast

repo.commit_diff(
    Diff(
        ops=(AddNode(node=Episode(id=ids.node(), parent_id=repo.state().root_id,
                                  title="Ashfall")),),
        author=DiffAuthor.human,
        intent="open on the orphan's city",
    )
)
```

## What is measured

Every claim is either measured or explicitly marked as unmeasured. The deterministic tier
needs no model calls, so it is exact and cannot be improved by rerunning:

| | |
|---|---|
| Continuity checker | layer 1 alone 80% recall, + layer 2 100%, at **0.00 false positives per beat** on a clean story |
| Staleness prediction | declared edges P=1.00 R=0.67; + embedding edges at threshold 0.68 P=1.00 R=1.00 |
| Selector diversity | top-k 0.254 / MMR 0.483 / DPP 0.555 mean pairwise distance, at 0.90 / 0.83 / 0.78 quality |
| Bandit | Thompson pseudo-regret 4.8 and flat after ~50 rounds; ε-greedy 6.3 and still climbing |
| Preference prior | +0.08 held-out pairwise accuracy over uniform after a fresh writer's first 10 comparisons (and −0.01 before any comparison — the prior earns its place in the small-data middle, not at zero) |

The live tier: four simulated writers drove the real engine through 139 decisions against
the free-tier provider, with no run cut short. Learning is measured on a **held-out probe** —
12 frozen decisions sampled from *other* writers' runs, replayed after every episode, so
nothing in the curve can move because the task got harder. Three of four runs end at or
above the population prior on that set and above an uninformed head, at 69% top-1 agreement
with the writer's own first choice. Weight recovery averages 0.46 against an **oracle
ceiling of 0.64** — the same estimator on the same data fitted to weights it is told — so
72% of what was achievable at this sample size, with one run at its ceiling. Acceptance held
or rose in three runs and fell in one; that trend is confounded with rising task difficulty
and `docs/presentable.tex` says so rather than reading it either way.

Two predictions written in advance turned out **wrong**, and both changed the system rather
than the write-up: MMR at the conventional λ = 0.7 was identical to the top-k baseline, and
embedding dependency edges did better than expected. Both are in
`docs/presentable.tex` §"What the measurements changed".

What this cannot measure: real taste, fatigue, or trust. The evaluation uses simulated
writers, so every number is a property of the machinery. Only a human study would settle
the rest, and `docs/presentable.tex` §Limitations says so.

## Cost

**Everything here ran on free-tier keys, at $0.00 across roughly ten million tokens** —
six rotated Gemini keys for generation and judging, Groq for extraction, CPU models on one
laptop for embeddings and NLI.

That is a decision about where risk sits, not a constraint worked around. A metered key
from day one buys better prose immediately and hides every cost bug until the bill arrives.
A free tier forces the cost controls to exist before they are needed — purpose-tag routing,
a read-through cache keyed on the full request, key rotation with per-(key, model)
cooldowns, a budget guard that refuses a call it cannot afford — and those are the same
controls a production system needs. Each was exercised in anger rather than written and
hoped for.

The metered budget is consequently unspent, and reserved for the strong-model rerun that
separates *the system works* from *the model is good*. Nothing needs re-engineering for it:
the model id is a configuration value.

## Design decisions

The full decision log — every choice, the alternative rejected, and why — is in
`docs/presentable.tex`, which also carries the evaluation and the limitations. Package-level
notes live in each package's `README.md`.

## Licence

Proprietary; written as an interview deliverable.
