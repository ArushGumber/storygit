# `eval`

Turning the system's claims into numbers, and being honest about which claims cannot be
turned into numbers at all.

## Two tiers, on purpose

**Offline metrics** (`offline.py`) need no model calls. Checker recall against injected
contradictions, staleness precision and recall against a hand-written dependency chain, the
soft-edge threshold sweep, selector diversity on real embeddings, the MMR λ sweep, bandit
regret, and whether pretraining earns its place. These are *exact* — no sampling noise, so
a regression is a real regression — they are free, and they cannot be improved by rerunning
until the model has a good day.

**Live runs** (`run.py`) drive the real engine with simulated writers. These produce the
acceptance and edit-distance curves, the held-out probe curve, and the weight-recovery
numbers, and they cost quota.

```bash
python -m eval.offline                          # everything deterministic, seconds
python -m eval.run --config smoke --max-calls 200
python -m eval.run --config full --max-calls 3000
python -m eval.run --offline-only               # regenerate figures without spending quota
python -m eval.run --config probesample         # sample the probe fixture (see below)
python -m eval.probe --build                    # rebuild it from those runs
python -m eval.costing                          # project a metered rerun, arithmetic only
```

## The held-out probe, and why it exists

Acceptance rate over a run measures two things at once. The writer's bar does not move, but
the task gets harder every episode — each new candidate has to stay consistent with more
established facts, more open threads, and more accumulated rules. So a falling acceptance
curve is not evidence the preference layer failed, and a rising one would not be evidence
that it worked. **A metric that moves for two reasons measures neither**, and no statistic
computed on that curve separates them.

The fix is a different experiment. `probe.py` freezes a set of decision points and replays
them after every episode, ranking with the head as it stands at that moment and scoring
that ranking against what the persona would privately have chosen (Kendall tau, plus top-1
agreement). The probe set never changes, so difficulty is held fixed by construction and
anything that moves is the head.

Two properties keep it honest:

- **No leakage.** Points come from a dedicated `probesample` run that is never itself
  probed, and `ProbeSet.for_persona` additionally refuses any point sourced from the
  persona being measured. A persona is never probed on a decision it trained on.
- **No provider calls.** Each point stores the candidate-intrinsic features computed at
  decision time. Replay recomputes only the two features that depend on the *learner's*
  state — voice cosine and edit-direction projection — and re-ranks. Asserted by a test,
  because a probe that quietly regenerated a candidate would stop being frozen.

## The ceiling on weight recovery

A correlation of 0.4 between a fitted weight vector and a hidden one is meaningless without
a scale, and the scale is not 1.0 — that is the answer for infinite data. Twenty-odd noisy
comparisons over thirteen features do not identify thirteen weights.

`ceiling.py` measures the estimator's best case directly: take the feature matrices a
persona actually saw and its actual decision count, generate choices from a *known* random
weight vector at that persona's noise level, fit the same head the same way, correlate, and
repeat over 200 seeds. Every reported recovery number carries this ceiling beside it,
because 0.44 against 0.50 and 0.44 against 0.95 are different claims — and only one of them
is about the system rather than about the sample size.

The offline tier carries the same computation as a curve against decision count, drawn from
the ranges real candidates occupy rather than from a uniform cube. That choice does most of
the work: uniform draws report 0.88 at 25 decisions where realistic ones report 0.72,
because real features are clustered and correlated and the design matrix is far worse
conditioned than random data.

## The simulated writers

Four personas, each **a hidden weight vector over exactly the feature space the preference
head is fitted on**. That is the design decision the whole evaluation rests on, because it
turns the central question into a measurable one: after N decisions, does the fitted weight
vector correlate with the hidden one? Not *"did acceptance go up"* — which a degenerate
system achieves by proposing the same thing repeatedly — but *"did the machinery recover the
taste it was shown"*.

| Persona | Weighted towards | Forbidden | Character |
|---|---|---|---|
| the Serialist | momentum, consequence | flashbacks | Writes for retention |
| the Minimalist | specificity, **negative** on length | prophecy | Short, concrete, exposition-averse |
| the Maximalist | voice, interiority, length | romance | Dial at 0.70, dialogue low |
| the Controller | continuity, own criteria | killing the mentor, prophecy | Accepts rarely, edits heavily, locks constantly |

Three constraints keep this from being circular, and the first is asserted in the tests:

1. The engine is **never** told the hidden weights.
2. The persona sees only what the engine chose to show — never the feature vectors.
3. The edit model is told the persona's **style**, never its objective. Otherwise the
   engine could learn the answer from text the persona produced.

### What this cannot measure

Real taste, fatigue, trust, whether the prose is any good, or whether a listener would press
next episode. A persona is a linear functional plus noise: it cannot be surprised, cannot
change its mind, and cannot tell you the tool is exhausting to use. Everything measured here
is a property of the *machinery*, and that is the only claim made.

There is a deeper circularity worth naming: the preference layer is validated against
writers made of the same feature vector it is fitted on. That is exactly why the headline
metric is **recoverability** rather than "learns taste". The `pretrain.py` proto-personas
(which the prior is fitted on) and these personas (which the system is measured on) are
deliberately separate modules, so the prior cannot be fitted on the test set.

## Ground truth by construction

`inject.py` builds a small clean story and then injects one contradiction of each class,
each into its own copy so they cannot interact — a location conflict must not accidentally
also be a possession conflict, or the per-class recall numbers stop meaning anything.

It also supplies the two other ground truths: hand-written passages with known fact sets
(extraction recall) and a scripted edit whose true blast radius is known because the
dependency edges were written by hand.

That last one includes a beat that **genuinely depends on the changed fact but never
declared it**. Without such a beat, the soft-edge ablation could only add false positives
and never recall, which would be an unfair test of an idea worth testing properly.

## What the offline numbers say

```
continuity checker      layer 1 alone 80%, + layer 2 100%
                        0.00 false positives per beat on a clean story
staleness (declared)    P=1.00  R=0.67   -- exact, and blind to undeclared dependencies
staleness (+ soft@0.68) P=1.00  R=1.00   -- picks up the undeclared one cleanly here
selector diversity      top-k 0.254 / MMR(0.5) 0.483 / DPP 0.555, at 0.90 / 0.83 / 0.78 quality
bandit pseudo-regret    Thompson 4.8 (flat after ~50), epsilon-greedy 6.3 and climbing
pretraining lift        +0.13 at 10 comparisons, -0.02 at 40
```

Two of these contradicted what the design predicted in advance, and both are reported as
found rather than quietly re-explained:

- **Soft edges did better than expected.** The prediction was "high recall, poor precision,
  crossover too weak to use". On this case, thresholds at or above 0.68 add the undeclared
  dependency at no precision cost. The case is six beats with lexically distinctive
  locations, so this is suggestive rather than settled — but it is the opposite of the
  stated prediction and the write-up says so.
- **MMR at the conventional λ = 0.7 was identical to the top-k baseline.** Bi-encoder cosine
  has a high floor (~0.33–0.82 across six candidates here), so a redundancy penalty that is
  nearly constant changes no ordering. Fixed two ways: the similarity matrix is now rescaled
  within the candidate set, and the default λ moved to 0.5 — a value chosen from the sweep
  in `mmr_lambda_sweep.svg` rather than from convention.

The second one also produced the clearest argument for DPPs the project has: the DPP reaches
the diverse answer with **no parameter at all**, because its kernel is multiplicative — a
near-duplicate's contribution to the determinant collapses however good it is — whereas
MMR's additive penalty can always be outvoted by a large enough quality gap.

## Ablations

Each removes exactly one thing (asserted in `test_each_ablation_removes_exactly_one_thing`)
and each carries a **stated expectation** written before the run. A prediction made after
the numbers arrive is not a prediction.

| Ablation | What should get worse |
|---|---|
| `no_preference` | Acceptance stays flat; fitted weights do not correlate with hidden ones |
| `no_propagation` | Contradictions accumulate — nothing says what you built on has moved |
| `no_checker` | Contradictions survive into accepted state; the Controller suffers most |
| `temperature_only` | Diversity collapses at similar quality |
| `dpp` | Should match or beat MMR on diversity at similar quality |
| `no_dial` | The Maximalist (dial 0.70) notices; low-dial personas barely differ |
| `soft_edges` | Stale recall up, precision down |

## Running long

A 15-episode run on a free tier will be interrupted. So:

- **Checkpoints after every episode.** A run that cannot resume never finishes.
- **Budgets calls, not time.** The binding constraint is the daily request quota, so the
  driver counts calls and stops cleanly at the cap.
- **Reports what did not run.** `summary.md` names every skipped configuration and why. A
  results table that silently omits the runs that failed is a lie.

Rough cost, before caching: a `full`-shaped run is ~33 decisions and ~430 calls per
(persona, configuration). The dominant term is the judge — one call per candidate — so if
quota binds, the honest lever is `n`, not the judge.

## The Gallery recorder

`gallery_record.py` records a session as ordered steps, each naming the snapshot it happened
at. Replay reads the story out of the store rather than out of a script, so the Gallery tab
cannot show something the system did not actually do — and replay makes **zero provider
calls**, which is asserted in the tests.

Each step captures what a writer would have seen: candidates with their labels, flags, and
scores; what they did; and what changed — the bible diff and the stale marks. That is enough
to reconstruct all three panes of the interface at any point in the session.
