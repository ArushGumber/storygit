# Evaluation summary

Generated 2026-08-23T12:40:21.885442+00:00.

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

| run | decisions | acceptance | first third | last third | mean edit dist | weight recovery | tokens/action |
|---|---|---|---|---|---|---|---|
| probesample/the Controller | 11 | 100% | 100% | 100% | 0.80 | 0.46 | 34440 |
| probesample/the Maximalist | 12 | 83% | 100% | 75% | 0.92 | 0.27 | 36574 |
| probesample/the Minimalist | 12 | 92% | 100% | 91% | 0.00 | 0.58 | 31053 |
| probesample/the Serialist | 11 | 100% | 100% | 100% | 0.00 | 0.34 | 36696 |

## Provider cost

- calls: 658
- tokens: 1594018
- cache hit rate: 0%
- estimated cost: $0.0000 (free tier; the price table would apply on a metered provider)
