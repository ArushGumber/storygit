# Evaluation summary

Generated 2026-08-23T13:36:19.736796+00:00.

## Offline metrics (deterministic, no model calls)

### Continuity checker

| configuration | recall | precision | F1 |
|---|---|---|---|
| layer 1 only | 80% | 100% | 0.89 |
| layer 1 + 2 | 100% | 100% | 1.00 |

False positives on a clean story: 0.00 per beat.

### Staleness prediction

| configuration | precision | recall | F1 |
|---|---|---|---|
| declared only | 1.00 | 0.67 | 0.80 |
| + soft edges @ 0.55 | 0.60 | 1.00 | 0.75 |
| + soft edges @ 0.62 | 0.75 | 1.00 | 0.86 |
| + soft edges @ 0.68 | 1.00 | 1.00 | 1.00 |
| + soft edges @ 0.72 | 1.00 | 1.00 | 1.00 |
| + soft edges @ 0.78 | 1.00 | 1.00 | 1.00 |
| + soft edges @ 0.85 | 1.00 | 1.00 | 1.00 |

### Selector diversity

| selector | diversity | mean quality |
|---|---|---|
| MMR (lambda=0.5) | 0.483 | 0.833 |
| DPP | 0.555 | 0.780 |
| top-k (temperature only) | 0.254 | 0.900 |

### Bandit

Pseudo-regret after 400 rounds: Thompson 4.8, epsilon-greedy 6.3. Pulls: {'safe': 16, 'explore': 384}.

### Pretraining

| comparisons made | from prior | from uniform | lift |
|---|---|---|---|
| 0 | 0.785 | 0.795 | -0.010 |
| 5 | 0.795 | 0.685 | +0.110 |
| 10 | 0.780 | 0.700 | +0.080 |
| 20 | 0.805 | 0.765 | +0.040 |
| 40 | 0.825 | 0.815 | +0.010 |

## Live runs

| run | decisions | acceptance | first third | last third | mean edit dist | weight recovery | same-n ceiling | tokens/action |
|---|---|---|---|---|---|---|---|---|
| full/the Controller | 33 | 100% | 100% | 100% | 0.00 | 0.69 | 0.67 ± 0.12 | 56580 |
| full/the Maximalist | 33 | 100% | 100% | 100% | 0.93 | 0.44 | 0.61 ± 0.14 | 54855 |
| full/the Minimalist | 38 | 82% | 100% | 77% | 0.32 | 0.29 | 0.66 ± 0.12 | 49582 |
| full/the Serialist | 35 | 94% | 91% | 100% | 0.00 | 0.43 | 0.63 ± 0.13 | 55195 |

### Held-out probe

The same frozen decisions, drawn from other personas' runs, re-ranked by the head after every episode. Nothing here can move because the task got harder: the probe set does not change.

| run | probe points | tau, first | tau, last | uniform head | prior head | top-1, last |
|---|---|---|---|---|---|---|
| full/the Controller | 12 | +0.667 | +0.778 | +0.667 | +0.778 | 92% |
| full/the Maximalist | 12 | +0.667 | +0.611 | +0.500 | +0.611 | 75% |
| full/the Minimalist | 12 | +0.444 | +0.111 | +0.000 | +0.056 | 33% |
| full/the Serialist | 12 | +0.611 | +0.611 | +0.889 | +0.833 | 75% |

The informed retry was offered on 7 rejected candidate set(s) and rescued 5 of them; the rest were rejected twice, which is a writer meaning it.

## Provider cost

- calls: 1993
- tokens: 7493302
- cache hit rate: 8%
- estimated cost: $0.0000 (free tier; the price table would apply on a metered provider)
