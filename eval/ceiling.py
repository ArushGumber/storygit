"""What weight recovery can be at our sample sizes, before asking what it is.

A correlation of 0.44 between a fitted weight vector and a hidden one is meaningless
without a scale. The scale that matters is not 1.0 — that is the answer for infinite data —
but *what the estimator achieves on this many decisions, at this noise level, when the
model is correctly specified by construction*. Twenty-odd noisy pairwise comparisons over
thirteen features do not identify thirteen weights, and no amount of engineering makes
them.

So this measures the ceiling directly. For each persona, take the feature matrices it
actually saw and the number of decisions it actually made, generate choices from a *known*
random weight vector at that persona's noise level, fit the same head the same way, and
correlate. Repeat over many seeds. The mean is what a perfect run would score; the spread
says how much of any single run's number is luck.

Reported beside the measured value everywhere it appears. A measured 0.44 against a ceiling
of 0.50 is a different claim from a measured 0.44 against a ceiling of 0.95, and only one of
those two claims is about the system rather than about the sample size.

Deterministic and free: no provider calls, seeded, part of the offline metrics tier.
"""

from __future__ import annotations

import random
import statistics
from typing import Any

from eval.metrics import weight_recovery
from storygit.preference.bt_head import BTWeights, fit
from storygit.preference.features import BASE_FEATURES, FeatureVector

SEEDS = 200
"""How many synthetic runs per persona. Enough for a stable mean and a meaningful sd."""


def _pairs_from(
    matrices: list[list[dict[str, float]]],
    hidden: dict[str, float],
    noise: float,
    rng: random.Random,
) -> list[tuple[FeatureVector, FeatureVector]]:
    """Turn recorded candidate sets into comparisons a known weight vector would make.

    Args:
        matrices: One list of candidate feature dicts per decision, as recorded.
        hidden: The weight vector generating the choices.
        noise: Decision noise, matching the persona's.
        rng: Draw source.

    Returns:
        ``(winner, loser)`` pairs — the chosen candidate against each one it beat, which is
        exactly the pairing the real head is fitted on.
    """
    pairs: list[tuple[FeatureVector, FeatureVector]] = []
    for shown in matrices:
        if len(shown) < 2:
            continue
        vectors = [FeatureVector(values=dict(v)) for v in shown]
        scored = [
            sum(hidden.get(k, 0.0) * v for k, v in vec.values.items()) + rng.gauss(0.0, noise)
            for vec in vectors
        ]
        best = scored.index(max(scored))
        for index, vector in enumerate(vectors):
            if index != best:
                pairs.append((vectors[best], vector))
    return pairs


def ceiling_for(
    matrices: list[list[dict[str, float]]],
    *,
    noise: float,
    seeds: int = SEEDS,
    l2: float = 1.0,
) -> dict[str, float]:
    """The estimator's best case on this much data.

    Args:
        matrices: The candidate sets the persona actually saw, as feature dicts.
        noise: The persona's decision noise.
        seeds: How many synthetic runs to average.
        l2: Regularization, matching the live head.

    Returns:
        ``{"mean": ..., "sd": ..., "n_decisions": ..., "n_pairs": ...}``. A mean of 0.0
        with no decisions means there was nothing to measure.
    """
    if not matrices:
        return {"mean": 0.0, "sd": 0.0, "n_decisions": 0.0, "n_pairs": 0.0}

    scores: list[float] = []
    pair_counts: list[int] = []
    for seed in range(seeds):
        rng = random.Random(90210 + seed)
        hidden = {name: rng.uniform(-1.0, 1.0) for name in BASE_FEATURES}
        pairs = _pairs_from(matrices, hidden, noise, rng)
        pair_counts.append(len(pairs))
        if not pairs:
            continue
        fitted = fit(pairs, prior=BTWeights.uniform(), l2=l2)
        learned = dict(zip(fitted.names, fitted.weights, strict=False))
        scores.append(weight_recovery(learned, hidden))

    if not scores:
        return {"mean": 0.0, "sd": 0.0, "n_decisions": float(len(matrices)), "n_pairs": 0.0}
    return {
        "mean": statistics.fmean(scores),
        "sd": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "n_decisions": float(len(matrices)),
        "n_pairs": float(statistics.fmean(pair_counts)),
    }


def for_runs(runs: list[dict[str, Any]], noise_by_persona: dict[str, float]) -> dict[str, Any]:
    """Ceilings for every recorded run, keyed by persona.

    Args:
        runs: Loaded RunLog dictionaries.
        noise_by_persona: Each persona's decision noise.

    Returns:
        Persona name to ceiling statistics.
    """
    out: dict[str, Any] = {}
    for run in runs:
        persona = run.get("persona", "")
        matrices = [
            list(a["features"]) for a in run.get("actions", ()) if len(a.get("features", ())) >= 2
        ]
        out[persona] = ceiling_for(matrices, noise=noise_by_persona.get(persona, 0.08))
    return out
