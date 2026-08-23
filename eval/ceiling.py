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
from collections.abc import Sequence
from typing import Any

from eval.metrics import weight_recovery, weight_recovery_on_axes_the_writer_has
from storygit.preference.bt_head import fit
from storygit.preference.features import BASE_FEATURES, FeatureVector
from storygit.preference.layer import (
    EDITED_WIN_WEIGHT,
    MIN_PAIRS_TO_FIT,
    SHRINK_UNIDENTIFIED,
)
from storygit.preference.pretrain import fit_prior

LIVE_L2 = 2.0
"""Matches ``PreferenceLayer._fit_head``. The ceiling used 1.0, and did not say so."""

SEEDS = 200
"""How many synthetic runs per persona. Enough for a stable mean and a meaningful sd."""


def _pairs_from(
    matrices: list[list[dict[str, float]]],
    hidden: dict[str, float],
    noise: float,
    rng: random.Random,
    *,
    edited: Sequence[bool] | None = None,
) -> tuple[list[tuple[FeatureVector, FeatureVector]], list[float]]:
    """Turn recorded candidate sets into comparisons a known weight vector would make.

    Args:
        matrices: One list of candidate feature dicts per decision, as recorded. Decisions
            the writer rejected are the caller's to drop -- see :func:`for_runs`.
        hidden: The weight vector generating the choices.
        noise: Decision noise, matching the persona's.
        rng: Draw source.
        edited: Per decision, whether the writer edited the winner before accepting. The
            live head halves those pairs' weight, so the ceiling must too.

    Returns:
        ``(winner, loser)`` pairs — the chosen candidate against each one it beat, which is
        exactly the pairing the real head is fitted on — and one sample weight per pair.
    """
    # The persona divides by the L1 norm of its weights before adding noise
    # (``personas._raw_score``), so a fixed sigma means a fixed signal-to-noise ratio for
    # every persona. Scoring an un-normalised dot product here and adding the same sigma
    # made the oracle's labels several times cleaner than the writer's, which inflates the
    # ceiling -- conservatively, but by an unstated amount.
    scale = sum(abs(w) for w in hidden.values()) or 1.0

    pairs: list[tuple[FeatureVector, FeatureVector]] = []
    weights: list[float] = []
    for index, shown in enumerate(matrices):
        if len(shown) < 2:
            continue
        vectors = [FeatureVector(values=dict(v)) for v in shown]
        scored = [
            sum(hidden.get(k, 0.0) * v for k, v in vec.values.items()) / scale
            + rng.gauss(0.0, noise)
            for vec in vectors
        ]
        best = scored.index(max(scored))
        weight = EDITED_WIN_WEIGHT if edited and index < len(edited) and edited[index] else 1.0
        for other, vector in enumerate(vectors):
            if other != best:
                pairs.append((vectors[best], vector))
                weights.append(weight)
    return pairs, weights


def _hidden_like(reference: dict[str, float], rng: random.Random) -> dict[str, float]:
    """A random weight vector with the same shape as a real persona's.

    The first version drew dense ``uniform(-1, 1)`` over all thirteen features. The four
    personas are sparse -- each defines two criteria, so two criterion slots are exactly
    zero, and five or six coordinates in total are -- and their signed features are a known
    subset. Recovering a dense random vector and recovering a sparse structured one are
    different estimation problems, and only the second is the one the runs face, so a
    ceiling built from the first is not a ceiling for this problem.

    Args:
        reference: The persona's hidden weights, used for its sparsity and sign pattern
            only. The magnitudes are redrawn, so nothing about the persona leaks into the
            fit.
        rng: Draw source.

    Returns:
        A weight vector zero exactly where the reference is zero.
    """
    signed = {"length", "dialogue_ratio", "voice_cosine", "edit_direction"}
    out: dict[str, float] = {}
    for name in BASE_FEATURES:
        if reference and reference.get(name, 0.0) == 0.0:
            out[name] = 0.0
        elif name in signed:
            out[name] = rng.uniform(-1.0, 1.0)
        else:
            out[name] = rng.uniform(0.1, 1.2)
    return out


def ceiling_for(
    matrices: list[list[dict[str, float]]],
    *,
    noise: float,
    seeds: int = SEEDS,
    l2: float = LIVE_L2,
    edited: Sequence[bool] | None = None,
    like: dict[str, float] | None = None,
) -> dict[str, float]:
    """The estimator's best case on this much data.

    "The same estimator, on the same candidate sets, at the same decision count" is the
    sentence this number's whole meaning rests on, so every clause of it is honoured here
    rather than approximately: the same regularization, the same population prior as the
    anchor, the same edited-win down-weighting, the same shrinkage flag, the same
    normalised noise scale, and the same sparsity in the vector being recovered. The
    caller supplies the same *decisions* by dropping the ones that produced no pairs.

    Args:
        matrices: The candidate sets the persona actually saw, as feature dicts, with
            rejected decisions already removed.
        noise: The persona's decision noise.
        seeds: How many synthetic runs to average.
        l2: Regularization, matching the live head.
        edited: Per decision, whether the winner was edited before being accepted.
        like: A persona's hidden weights, for sparsity and sign structure only.

    Returns:
        ``{"mean": ..., "sd": ..., "n_decisions": ..., "n_pairs": ...}``. A mean of 0.0
        with no decisions means there was nothing to measure.
    """
    if not matrices:
        return {"mean": 0.0, "sd": 0.0, "n_decisions": 0.0, "n_pairs": 0.0}

    prior = fit_prior()
    scores: list[float] = []
    identified: list[float] = []
    pair_counts: list[int] = []
    for seed in range(seeds):
        rng = random.Random(90210 + seed)
        hidden = _hidden_like(like or {}, rng)
        pairs, sample_weights = _pairs_from(matrices, hidden, noise, rng, edited=edited)
        pair_counts.append(len(pairs))
        if len(pairs) < MIN_PAIRS_TO_FIT:
            continue
        fitted = fit(
            pairs,
            prior=prior,
            l2=l2,
            sample_weights=sample_weights,
            shrink_unidentified=SHRINK_UNIDENTIFIED,
        )
        learned = dict(zip(fitted.names, fitted.weights, strict=False))
        scores.append(weight_recovery(learned, hidden))
        identified.append(weight_recovery_on_axes_the_writer_has(learned, hidden))

    if not scores:
        return {"mean": 0.0, "sd": 0.0, "n_decisions": float(len(matrices)), "n_pairs": 0.0}
    return {
        "mean": statistics.fmean(scores),
        "sd": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "identified_mean": statistics.fmean(identified) if identified else 0.0,
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
        # A rejected set produces no comparison in reality -- the writer said no to all of
        # them, so there is no winner to pair against anything. The oracle always picks a
        # best, so including those decisions fitted the ceiling on materially more
        # comparisons than the run it is meant to scale.
        usable = [
            a
            for a in run.get("actions", ())
            if len(a.get("features", ())) >= 2 and a.get("kind") != "reject"
        ]
        out[persona] = ceiling_for(
            [list(a["features"]) for a in usable],
            noise=noise_by_persona.get(persona, 0.08),
            edited=[a.get("kind") == "edit" for a in usable],
            like=run.get("hidden_weights") or None,
        )
    return out
