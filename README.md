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
| `src/storygit/domain/` | The typed state: plan tree, world graph, threads, writer ledger, provenance, and the diff operations that are the only way to change any of it. |
| `src/storygit/store/` | Git-shaped persistence: content-addressed objects, snapshot manifests, branches, three-way merge. |
| `src/storygit/graph/` | Deterministic dependency edges, staleness propagation, and the entity-scoped slices that generation prompts consume. |

Later chunks add `providers/` (model routing), `agents/` (proposals as diffs),
`selection/` (diverse labeled candidates), `continuity/` (the three-layer checker),
`preference/` (learning the writer's taste), `eval/`, `api/`, and `frontend/`.

---

## Quickstart

```bash
uv venv --python 3.11 && uv pip install -e '.[dev]'
.venv/bin/python -m pytest          # offline unit tests
.venv/bin/ruff check src tests      # lint
.venv/bin/mypy src                  # types
```

A minimal session against the state layer:

```python
from storygit.domain.diff import AddNode, Diff, DiffAuthor
from storygit.domain.ids import IdGenerator
from storygit.domain.nodes import Story
from storygit.domain.state import StoryState
from storygit.store.repository import Repository

ids = IdGenerator(seed=0)
repo = Repository.open("story.db")
root = Story(id=ids.node(), title="Untitled", seed="A powerless orphan discovers an ability...")
repo.initialize(StoryState.build(nodes={root.id: root}))

episode_id = ids.node()
repo.commit_diff(
    Diff(
        ops=(AddNode(node=Episode(id=episode_id, parent_id=root.id, title="Ashfall")),),
        author=DiffAuthor.human,
        intent="open on the orphan's city",
    )
)
```

## Design decisions

The full decision log — every choice, the alternatives rejected, and why — is in
`docs/presentable.tex`. Package-level notes live in each package's `README.md`.

## Licence

Proprietary; written as an interview deliverable.
