# Evaluation summary

Generated 2026-08-23T15:55:21.147833+00:00.

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
| probesample/the Controller | 23 | 83% | 100% | 86% | 0.50 | 0.25 | 0.57 ± 0.13 | 40713 |
| probesample/the Maximalist | 25 | 84% | 81% | 92% | 0.79 | 0.53 | 0.61 ± 0.12 | 40682 |
| probesample/the Minimalist | 27 | 63% | 84% | 46% | 0.27 | 0.46 | 0.56 ± 0.13 | 35231 |
| probesample/the Serialist | 23 | 87% | 96% | 77% | 0.44 | 0.60 | 0.62 ± 0.13 | 39545 |

The informed retry was offered on 10 rejected candidate set(s) and rescued 7 of them; the rest were rejected twice, which is a writer meaning it.

## Provider cost

- calls: 2128
- tokens: 3814214
- cache hit rate: 12%
- estimated cost: $0.0000 (free tier; the price table would apply on a metered provider)
