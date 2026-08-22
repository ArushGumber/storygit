# `storygit.store`

Git, for story state. Content-addressed objects, snapshot manifests, branch pointers,
structural diff, and three-way merge — all in one SQLite file.

## How it works

Every object (node, entity, fact, epistemic edge, thread, ledger) is serialized to
canonical JSON — sorted keys, no whitespace — and stored under `sha256("kind:payload")`.
A **snapshot** is a manifest: `{kind: {logical id: content hash}}`, plus a parent
pointer, author, and intent. A **branch** is a name pointing at a snapshot.

Three things fall out of this for free:

1. **Cheap versions.** Committing a change that touches three nodes writes three new
   object rows and one manifest. A 500-snapshot history of a 200-node story costs
   roughly the changed nodes, not 500 full copies.
2. **Structural diff without heuristics.** What changed between two versions is a set
   comparison over two manifests. No model is asked to guess.
3. **Reproducibility.** Identical content produces an identical manifest, so
   "apply the same diff twice and get the same hashes" is a test, not a hope.

## Key types

| Type | File | What it is |
|---|---|---|
| `connect()` | `db.py` | Opens/creates the database, applies the schema, turns on WAL. |
| `SnapshotStore` | `snapshots.py` | `put_object` / `get_object`, `write_snapshot(state)`, `load_state(id)`, `history(id)`. |
| `BranchStore` | `branches.py` | Named pointers into the snapshot chain. |
| `structural_diff(before, after)` | `branches.py` | The diff that turns one state into another, ordered removals-before-additions and parents-before-children, with a fixup pass that guarantees the round trip. |
| `merge(...)` → `MergeResult` | `branches.py` | Three-way merge at object granularity. Non-overlapping changes combine; same-object changes come back as `MergeConflict`s for the writer. Nothing is auto-resolved. |
| `Repository` | `repository.py` | The facade. `state()`, `commit_diff()`, `preview_apply()`, `create_branch()`, `diff()`, `merge_branches()`. |

## Worked example

```python
repo = Repository.open("story.db")
repo.initialize(StoryState.build(nodes={root.id: root}))

repo.commit_diff(diff)  # apply + snapshot + move the branch
repo.create_branch("what-if")
repo.commit_diff(other_diff, branch="what-if")

result = repo.merge_branches("main", "what-if")
if result.clean:
    repo.commit_diff(result.diff)  # take their work
else:
    show(result.conflicts)  # ask the writer
```

## Invariants

- `Repository.commit_diff` is the only path that writes state. If it happened, there is
  a snapshot for it.
- Snapshots are immutable. Reloading an old snapshot after ten commits returns exactly
  what it held.
- Merge never resolves a conflict on its own.

## Why SQLite

One writer, one story, a few megabytes, and a file the writer can copy. Postgres buys
concurrency this system does not have. See `learning_systems.tex` for what would change
at a serialized-fiction platform scale.
