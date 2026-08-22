# `storygit.graph`

Dependencies, propagation, and slicing. Deterministic — nothing in this package calls a
model, by construction.

## Why it exists

Fabula's writers reported two opposite failures: an edit that propagated nowhere, and an
edit that silently rewrote the plan. Both come from the same absence — nothing records
what depends on what. Here a beat declares the facts it establishes (`produces`) and the
facts it relies on (`consumes`), so "what did this edit affect" is a graph walk with an
exact answer, and the answer is a *mark*, never a rewrite.

## Key functions

| Function | File | What it does |
|---|---|---|
| `dependents_of_facts(state, facts)` | `dependency.py` | Transitive closure: fact → consuming beats → the facts those beats produce → onward. Stops at locked nodes. |
| `hard_constraints(state)` | `dependency.py` | Everything generation must not contradict — the writer's own rules plus every fact established by a locked node. This is what a lock *means* operationally. |
| `propagate_change(before, after)` | `propagation.py` | The normal entry point. Works out which facts changed and returns `StaleMark`s. |
| `marks_to_diff(state, marks)` | `propagation.py` | Turns marks into ordinary status ops, so staleness flows through the same snapshot machinery as everything else. |
| `preview(state, diff)` | `propagation.py` | Dry run: what a candidate *would* mark. Feeds the "would mark 2 beats stale" line under every proposal. |
| `entity_slice(state, entities, at_beat)` | `slices.py` | The only thing a generation prompt ever sees: those entities, their facts valid at that beat, who knows what, open threads, the plan path, locks, style notes, and the writer's criteria. |

## The three propagation rules

1. **Locked nodes are never marked**, and the walk does not pass through them — a locked
   beat is not going to change, so nothing it produces can change either.
2. **Human prose is flagged for review, never staled.** The system does not tell an
   author their own sentences are out of date.
3. **Nothing is ever rewritten.** The writer chooses regenerate, edit by hand, or dismiss
   ("still works").

Marks come in three strengths: `stale` (declared dependency), `review` (human prose),
and `maybe_affected` (the optional embedding-similarity edge provider — chunk 3's
ablation, wired here as a `Protocol` so the deterministic core never grows a model
dependency).

## Worked example

```python
after = apply(state, edit)
marks = propagate_change(state, after)
# [StaleMark(node_id=beat_b, kind=stale,
#            reason='Depends on a fact that changed: "Kael is at Ashfall."
#                    (established in Beat A) changed.', origin_beat=beat_a)]
final = apply(after, marks_to_diff(after, marks))
```

## Why slices and not the whole story

Sending the whole state to a model does not scale past a few episodes and does not help.
A slice is a few hundred tokens instead of tens of thousands, and because it is retrieval
over a *typed graph* rather than over text chunks, what comes back is exactly true rather
than approximately relevant.
