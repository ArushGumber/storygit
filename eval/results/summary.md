# Evaluation summary

Generated 2026-08-24T16:05:55.559986+00:00.

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
| full/the Controller | 14 | 71% | 100% | 66% | 0.33 | 0.32 | 0.54 ± 0.13 | 62226 |
| full/the Maximalist | 36 | 81% | 64% | 89% | 0.86 | 0.28 | 0.65 ± 0.12 | 65936 |
| full/the Minimalist | 40 | 75% | 89% | 62% | 0.01 | 0.32 | 0.64 ± 0.13 | 57060 |
| full/the Serialist | 35 | 83% | 93% | 86% | 0.27 | 0.81 | 0.54 ± 0.14 | 75457 |
| half/the Controller | 2 | 0% | 0% | 0% | 0.00 | 0.38 | 0.00 ± 0.00 | 45823 |
| half/the Maximalist | 2 | 0% | 0% | 0% | 0.00 | 0.06 | 0.00 ± 0.00 | 45415 |
| half/the Minimalist | 16 | 25% | 71% | 22% | 0.37 | -0.60 | 0.17 ± 0.21 | 55840 |
| half/the Serialist | 2 | 0% | 0% | 0% | 0.00 | 0.23 | 0.00 ± 0.00 | 40698 |

### Held-out probe

The same frozen decisions, drawn from other personas' runs, re-ranked by the head after every episode. Nothing here can move because the task got harder: the probe set does not change.

| run | probe points | tau, first | tau, last | uniform head | prior head | top-1, last |
|---|---|---|---|---|---|---|
| full/the Controller | 18 | +0.000 | +0.000 | +0.000 | -0.111 | 22% |
| full/the Maximalist | 18 | +0.333 | +0.407 | +0.222 | +0.333 | 56% |
| full/the Minimalist | 18 | +0.333 | +0.444 | +0.222 | +0.296 | 56% |
| full/the Serialist | 18 | +0.593 | +0.741 | +0.222 | +0.333 | 67% |
| half/the Minimalist | 18 | +0.185 | +0.185 | +0.481 | +0.519 | 50% |

The informed retry was offered on 25 rejected candidate set(s) and rescued 15 of them; the rest were rejected twice, which is a writer meaning it.

## Errors during runs

- `full/the Controller`: episode 2 was not accepted; stopping
- `half/the Controller`: episode 1 was not accepted; stopping
- `half/the Maximalist`: episode 1 was not accepted; stopping
- `half/the Serialist`: episode 1 was not accepted; stopping

## Provider cost

- calls: 2164
- tokens: 9325578
- cache hit rate: 3%
- estimated cost: $6.7940 (free tier; the price table would apply on a metered provider)
