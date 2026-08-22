# `storygit.domain`

The typed story state, and the operations that are the only way to change it. Pure
Python: no database, no network, no models.

## Key types

| Type | File | What it is |
|---|---|---|
| `Story`, `Episode`, `Scene`, `Beat`, `Prose` | `nodes.py` | The plan tree. Every node carries the four planning questions (what happens / what the audience learns / how they feel / where and when) plus a `status` and a `stale_reason`. `Episode` adds the serial mechanics: hook, cliffhanger, recap, threads in/out, writer-set goals. `Beat` adds `produces` / `consumes` — the declared fact dependencies propagation walks. |
| `Entity`, `Fact`, `Knows` | `world.py` | The world graph. A `Fact` is an edge with a validity interval `[valid_from_beat, valid_until_beat)` over the global beat order, so a fact that changes does not overwrite its own history. `Knows` is separate on purpose: characters acting on information they were never given is the failure mode readers notice first. |
| `Predicate` | `world.py` | Closed vocabulary (`location`, `alive`, `injury`, `possesses`, `relationship`, `relationship_status`, `goal`, `secret`, `trait`, `faction`) plus `note`. Closed is what makes checker layer 1 a dict lookup instead of a model call. |
| `Thread` | `threads.py` | One open narrative question, with when it was opened and last advanced. |
| `WriterLedger` | `ledger.py` | Locks, hard constraints, style notes (writer-written or mined), writer-defined `Criterion` scoring axes, rejected directions, and the coherent↔surprising dial. |
| `ProvenanceSpan` | `provenance.py` | Who wrote which sentences, per span, so a paragraph the writer half-rewrote reads as exactly that. |
| `Diff`, `Op` | `diff.py` | 29 frozen operation types plus `delta_summary()`, which renders a diff as the English a novelist reads under a candidate. |
| `StoryState` | `state.py` | The immutable aggregate plus its derived indices — beat sequence, facts by `(subject, predicate)`, producers and consumers per fact, alias index. |
| `apply()` | `apply.py` | `(state, diff) -> new state`. Pure. Validates every precondition; raises typed errors from `errors.py`. |

## Worked example

```python
from storygit.domain.apply import apply
from storygit.domain.diff import AddFact, Diff, DiffAuthor
from storygit.domain.state import StoryState
from storygit.domain.world import Fact, Predicate

diff = Diff(
    ops=(AddFact(fact=Fact(
        id=fact_id, subject=kael, predicate=Predicate.location,
        object_entity=ashfall, valid_from_beat=beat_a, established_by_beat=beat_a,
    )),),
    author=DiffAuthor.ai,
    intent="put Kael in the market",
)
new_state = apply(state, diff)               # state is untouched
assert new_state.producer_of[fact_id] == beat_a
assert new_state.facts_valid_at(beat_b, subject=kael, predicate=Predicate.location)
```

## Invariants

- Every model is frozen. Changing anything means producing a new object, which is what
  lets the store hash it and share unchanged pieces between snapshots.
- `apply` never mutates its input. Applying the same diff to the same state twice gives
  byte-identical results, which is the basis of the store's content addressing.
- A node's `locked` status and the ledger's `locks` set are written together by
  `SetLock` / `ClearLock` and never separately, so they cannot drift apart.
- The tree's shape (`ALLOWED_CHILDREN`) is enforced on every structural op, and a beat
  has at most one prose child.
