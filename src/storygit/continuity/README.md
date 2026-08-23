# `storygit.continuity`

Three layers, cheapest and most certain first. Nothing is ever auto-fixed.

## Why three layers and not one model call

The obvious design is "ask a good model whether this contradicts anything". It costs money
per check, it is slow, and it is wrong some fraction of the time in a way you cannot
characterize. Worse, you cannot report a recall number for it, which means the system's
central claim becomes an assertion rather than a measurement.

So each check is pushed to the cheapest layer that can decide it, and each layer's recall
is reported separately.

| Layer | File | Mechanism | Cost | Catches |
|---|---|---|---|---|
| 1 | `layer1.py` | Dict lookup + equality over the closed predicate vocabulary | Free, instant | Conflicting single-valued facts, dead actors, possession conflicts, epistemic violations |
| 2 | `layer2_nli.py` | DeBERTa-MNLI cross-encoder, local CPU | Free, ~ms/pair | Contradictions between free-text `note` facts and borderline multi-valued pairs |
| 3 | `layer3_judge.py` | LLM soft judge, argue-before-rate | One call per candidate | Motivation implausibility, tone breaks; also supplies the ranking score |

**Layer 1 cannot call a model, by construction.** It imports nothing from `providers/`, and
`tests/test_continuity.py::test_layer1_cannot_call_a_model` parses its imports and asserts
so. Determinism here is a structural property, not a convention.

## What layer 1 does and does not flag

Catches:

- **Conflicting single-valued facts.** A character is in one place, is alive or not,
  belongs to one faction. Two simultaneously valid values is a contradiction.
- **Dead actors.** A character established as dead who then does something.
- **Possession conflicts.** Two holders of the same object at the same beat.
- **Epistemic violations.** A beat relies on a fact (`consumes`), performed by a character
  who has not been told it. The flag says whether they learn it later — *"learns it in Beat
  D"* — or never. This is the failure readers notice most and it is invisible to any check
  that only looks at what is *true*.

Deliberately does **not** flag:

- A properly ended fact being replaced. That is a change, not a contradiction.
- A fact re-established with the same value. That is repetition.
- Two goals, traits, or possessions at once. That is a character.
- A character acting on their own secret. You know your own secrets.

## Every flag cites its establishing beat

`"Kael cannot be in Ashfall here"` is useless. `"Kael was established in Kell in The
Warden's Offer"` is actionable, because the writer can go and look. `Flag.established_by`
is a required field for layers 1 and 2, and the interface links to it.

Hard flags (deterministic) and soft flags (model opinions) are structurally distinct in the
data model, sorted apart, and rendered differently. A soft flag never blocks anything.

## The NLI threshold is calibrated, not guessed

Measured against the local checkpoint:

| pair | P(contradiction) |
|---|---|
| "can command the ash" / "never able to affect it at all" | 1.00 |
| "can command the ash" / "the ash never obeys anyone" | 0.43 |
| "can command the ash" / "does what Kael tells it" | 0.00 |

The middle row is the interesting one — a real contradiction, but universal-versus-specific,
and the model is only moderately confident. A 0.5 threshold misses it. The default is 0.35,
which catches it with an enormous margin above the paraphrase. The evaluation sweeps it and
reports the precision/recall curve.

## Bible diff and strike

On accept the writer sees what changed in the world:

```
+ Kael keeps a secret: he can hear the ash.
~ Kael is at Ashfall. (no longer true from here)
- Kael has the Warden's token.
```

Striking a fact is not an undo. It produces a normal diff — `RemoveFact` if it was never
true, `InvalidateFact` if it stopped being true — which propagates like any other change,
and layer 1 re-runs on whatever depended on it. Even correcting the machine goes through the
same reviewable path as everything else.

## The audit

Per-accept checking is incremental and can miss slow drift: a contradiction assembled over
ten episodes where no single accept was wrong. `audit.run_audit(state)` walks the whole
graph through layers 1 and 2 and reports per-layer counts.

It also answers a question no per-fact check can: **which threads are being dropped.** A
thread opened in episode 2 and untouched since episode 3 is not a contradiction, so nothing
else in the system would ever mention it — and for a serial it is the most expensive kind
of mistake.
