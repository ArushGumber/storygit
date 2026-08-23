# Evaluation summary

Generated 2026-08-23T17:13:21.843412+00:00.

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
| full/the Controller | 3 | 100% | 100% | 100% | 0.20 | 0.20 | 0.43 ± 0.16 | 33534 |
| full/the Maximalist | 3 | 100% | 100% | 100% | 0.00 | 0.51 | 0.37 ± 0.19 | 68916 |
| full/the Minimalist | 26 | 54% | 68% | 48% | 0.46 | -0.06 | 0.61 ± 0.12 | 32045 |
| full/the Serialist | 37 | 70% | 89% | 61% | 0.48 | 0.42 | 0.64 ± 0.13 | 44080 |

### Held-out probe

The same frozen decisions, drawn from other personas' runs, re-ranked by the head after every episode. Nothing here can move because the task got harder: the probe set does not change.

| run | probe points | tau, first | tau, last | uniform head | prior head | top-1, last |
|---|---|---|---|---|---|---|
| full/the Minimalist | 18 | +0.148 | +0.259 | +0.370 | +0.444 | 44% |
| full/the Serialist | 18 | +0.741 | +0.667 | +0.222 | +0.296 | 67% |

The informed retry was offered on 9 rejected candidate set(s) and rescued 4 of them; the rest were rejected twice, which is a writer meaning it.

## Errors during runs

- `full/the Controller`: RateLimited: gemini: all 6 keys are rate limited across 4 model(s); soonest retry in 57s
- `full/the Maximalist`: RateLimited: groq: rate limited
- `full/the Minimalist`: RateLimited: gemini: all 6 keys are rate limited across 4 model(s); soonest retry in 0s

## Provider cost

- calls: 1655
- tokens: 2771486
- cache hit rate: 20%
- estimated cost: $0.0000 (free tier; the price table would apply on a metered provider)
