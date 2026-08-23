# `storygit.agents`

Where a language model's answer stops being text and becomes a proposed change to the
story.

## The pipeline

```
StateSlice + intent  ->  prompt  ->  schema-constrained call  ->  validated object
                                                                      |
                          alias resolution, fact/knows/thread mapping  v
                                                                    Diff
```

Nothing is interpreted twice, and nothing reaches the state without passing through
`apply`'s validation.

| File | What it does |
|---|---|
| `schemas.py` | One response model per level (`PremiseProposal`, `EpisodeProposal`, `SceneProposal`, `BeatProposal`, `ProseProposal`) plus `FactDraft`, `CharacterDraft`, and `ExtractionResult`. Every field is bounded. |
| `prompts.py` | Every prompt in the system, named and documented, in one file. |
| `parse.py` | `complete_structured`: call, validate, and repair once on failure. |
| `aliases.py` | Name → entity resolution. Exact match after normalization, otherwise create *and tell the writer*. |
| `propose.py` | `Proposer.propose(level, ...)` → `list[Proposal]`, and `Proposer.extract(...)` → `Diff`. |

## Why the schema carries the requirements

A `BeatProposal` cannot come back without `produces` and `consumes`, because they are
fields of the schema. The dependency graph therefore cannot quietly go unmaintained: the
model is structurally unable to hand back a beat that does not declare what it
establishes and what it relies on.

`FactDraft.known_by` works the same way. It lists the characters who *learn* the fact at
this beat — not everyone present, and not everyone it is true of. A fact can be true
without anyone knowing it, and a later beat where a character acts on something they were
never told is the most noticeable continuity error in serialized fiction.

## Why every field is bounded

The `max_length` bounds in `schemas.py` do two jobs. They keep proposals readable — a
writer scanning three candidates will not read an essay per field. And they are a hard
stop on a real failure mode observed during the chunk 2 live smoke test: a small model
fell into a repetition loop and filled a string field until it hit the token cap, turning
a perfectly good candidate into truncated JSON. The bounds are forwarded into the
provider's response schema, so the model is stopped rather than corrected.

## Two rules for prompts

**Prompts are fed slices, never state.** The input is always an entity-scoped
`StateSlice`, rendered compactly, plus the writer's intent and their standing
instructions. Serializing the whole story would not fit, would cost a fortune, and
measurably makes the output vaguer.

**The writer's constraints come last and are phrased as prohibitions.** Hard constraints
— their own rules plus every fact a locked node established — sit at the end of the
system message, where instruction-following is strongest.

## Alias resolution is deliberately conservative

Exact match after normalization (lowercase, collapse whitespace, strip a leading
article), otherwise mint a new entity *and* attach a note: `introduces new entity "the
Warden" — same as "Warden of Kell"? merge in the world-state pane`. No embedding
similarity is used. A fuzzy match that is occasionally wrong would weld two characters
together silently, which is precisely the failure the rest of the system exists to
prevent. The suggestion is a prompt for the writer to look; the merge is theirs to make.

## Worked example

```python
proposer = Proposer(router, IdGenerator(seed=7))
proposals = await proposer.propose(
    state, Level.beat, target_node_id=scene_id, intent="Kael discovers what he can do", k=3
)

p = proposals[0]
p.rationale  # "Puts the two of them in a room and gives Kael something to refuse."
p.delta_summary  # ['adds beat "The Warden's offer"', 'establishes: Kael keeps a secret: ...']
p.stale_preview  # 0  -- nothing downstream would be affected
apply(state, p.diff)  # validated before anything is committed
```
