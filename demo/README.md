# demo/

The finished demo story, committed so `make demo` works from a fresh clone.

`story.db` is a normal storygit repository: content-addressed objects, snapshot manifests,
branch pointers. It was produced by a writer working through the HTTP API — the same
actions the interface emits — not by a script that wrote rows.

```bash
make demo        # http://127.0.0.1:8000
```

**Browsing needs no keys.** The plan tree, the world state, the bible, every recorded
snapshot and branch, the audit, and the diff between any two versions are all computed from
this file. *Proposing* needs provider keys, because that is the part that calls a model —
so a reader with no `.env` can see everything the system knows and everything it did, and
cannot ask it for a new candidate.

`scripts/seed_demo.py` recreates the starting point (premise and opening cast); the story
on top of it is the writer's.

The walkthrough keyed to this state is `arush/demo_script.md`, which is private and not
part of the deliverable.
