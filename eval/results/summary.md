# Evaluation summary

Generated 2026-08-23T02:32:37.376527+00:00.

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
| 0 | 0.790 | 0.735 | +0.055 |
| 5 | 0.770 | 0.660 | +0.110 |
| 10 | 0.775 | 0.645 | +0.130 |
| 20 | 0.795 | 0.755 | +0.040 |
| 40 | 0.815 | 0.835 | -0.020 |

## Live runs

| run | decisions | acceptance | first third | last third | mean edit dist | weight recovery | tokens/action |
|---|---|---|---|---|---|---|---|
| full/the Controller | 0 | 0% | 0% | 0% | 0.00 | 0.33 | 0 |
| full/the Maximalist | 0 | 0% | 0% | 0% | 0.00 | 0.06 | 0 |
| full/the Minimalist | 1 | 100% | 100% | 100% | 0.00 | 0.47 | 1393682 |
| full/the Serialist | 33 | 94% | 91% | 98% | 0.44 | 0.65 | 41102 |

## Errors during runs

- `full/the Controller`: RateLimited: gemini: all 6 keys are rate limited across 4 model(s); soonest retry in 4s
- `full/the Maximalist`: RateLimited: gemini: all 6 keys are rate limited across 4 model(s); soonest retry in 4s
- `full/the Minimalist`: RateLimited: groq: rate limited

## Provider cost

- calls: 498
- tokens: 1393682
- cache hit rate: 9%
- estimated cost: $0.0000 (free tier; the price table would apply on a metered provider)
