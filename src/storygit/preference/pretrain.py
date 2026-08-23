"""Pretraining the preference head across synthetic proto-personas.

The honest problem with a per-writer preference model is that a real writer produces about
twenty comparisons in a session, and twenty comparisons is not enough to fit anything from
scratch. Every per-user preference system faces this, and the standard answer is a
population prior: fit across many users, then adapt per user from that starting point.

We have no population of writers. So we simulate one — not the full personas of the
evaluation, just procedural weight vectors over the same feature space with noise — fit a
prior on their pooled comparisons, and start each writer from it.

**The limitation, stated plainly.** A prior fitted on synthetic personas cannot contain
real human taste; nothing in this data ever met a reader. What it *can* contain is the
structure of the problem: that features have consistent signs, that the comparison signal
is noisy, that some axes matter more than others. The design bet is that the **adaptation
machinery** transfers even though the prior's specific values do not — and that bet is
testable, which is what :func:`storygit.preference.pretrain.evaluate_prior` measures. If a
prior-initialized head does not beat a uniform one over a fresh writer's first ten
decisions, the prior is worthless and should be dropped.

This module is deliberately separate from the evaluation's simulated writers: those are
richer (style quirks, forbidden moves, LLM edits) and are the *measurement*. These are
crude and are the *training data*, and keeping them apart is what stops the evaluation
marking its own homework.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from storygit.preference.bt_head import BTWeights, fit, pairwise_accuracy
from storygit.preference.features import BASE_FEATURES, FeatureVector


class ProtoPersona(BaseModel):
    """A synthetic writer: weights over the feature space, plus decision noise.

    Attributes:
        name: Label.
        weights: Hidden preference weights by feature name.
        noise: Standard deviation of the noise added to each comparison, so the generated
            preferences are not perfectly separable. Real choices are not.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    weights: dict[str, float]
    noise: float = 0.25

    def prefers(self, a: FeatureVector, b: FeatureVector, rng: random.Random) -> bool:
        """Whether this persona would choose ``a`` over ``b``."""
        score_a = sum(self.weights.get(k, 0.0) * v for k, v in a.values.items())
        score_b = sum(self.weights.get(k, 0.0) * v for k, v in b.values.items())
        return (score_a - score_b) + rng.gauss(0.0, self.noise) > 0.0


def make_personas(n: int = 8, *, seed: int = 0) -> list[ProtoPersona]:
    """Generate proto-personas with random but structured weights.

    Weights are drawn from a distribution that keeps continuity and the judge axes mostly
    positive — nobody prefers contradictions — while letting the stylistic axes (length,
    dialogue ratio, voice) take either sign, because those genuinely differ between
    writers.

    Args:
        n: How many to make.
        seed: RNG seed.

    Returns:
        The personas.
    """
    rng = random.Random(seed)
    signed = {"length", "dialogue_ratio", "voice_cosine", "edit_direction"}
    out: list[ProtoPersona] = []
    for index in range(n):
        weights = {
            name: (rng.uniform(-1.0, 1.0) if name in signed else rng.uniform(0.1, 1.2))
            for name in BASE_FEATURES
        }
        out.append(ProtoPersona(name=f"proto-{index}", weights=weights))
    return out


def random_features(rng: random.Random) -> FeatureVector:
    """A random point in feature space, for generating synthetic comparisons."""
    return FeatureVector(values={name: rng.random() for name in BASE_FEATURES})


def generate_pairs(
    personas: Sequence[ProtoPersona],
    *,
    per_persona: int = 200,
    seed: int = 0,
) -> list[tuple[FeatureVector, FeatureVector]]:
    """Generate ``(winner, loser)`` pairs from a set of proto-personas.

    Args:
        personas: The synthetic writers.
        per_persona: Comparisons each one makes.
        seed: RNG seed.

    Returns:
        Pooled comparisons, winner first.
    """
    rng = random.Random(seed)
    pairs: list[tuple[FeatureVector, FeatureVector]] = []
    for persona in personas:
        for _ in range(per_persona):
            a, b = random_features(rng), random_features(rng)
            pairs.append((a, b) if persona.prefers(a, b, rng) else (b, a))
    return pairs


def fit_prior(
    *,
    n_personas: int = 8,
    per_persona: int = 200,
    seed: int = 0,
    l2: float = 1.0,
) -> BTWeights:
    """Fit the population prior.

    Args:
        n_personas: How many proto-personas to pool.
        per_persona: Comparisons each makes.
        seed: RNG seed.
        l2: Regularization.

    Returns:
        The prior weights, to be used as the starting point for every new writer.
    """
    personas = make_personas(n_personas, seed=seed)
    pairs = generate_pairs(personas, per_persona=per_persona, seed=seed + 1)
    return fit(pairs, prior=BTWeights.uniform(), l2=l2)


def evaluate_prior(
    prior: BTWeights,
    *,
    n_decisions: int = 10,
    n_test: int = 200,
    seed: int = 99,
    shrink_unidentified: bool = False,
) -> dict[str, float]:
    """Does starting from the prior beat starting from uniform, early on?

    Simulates a *fresh* writer, gives both a prior-initialized and a uniform-initialized
    head the same first ``n_decisions`` comparisons, and measures held-out pairwise
    accuracy.

    This is the test that decides whether pretraining is worth keeping. It is reported
    rather than assumed.

    What it is not: a transfer test. The fresh writer is held out of the *sample* the
    prior was fitted on -- the caller must pass a seed no training persona used, or the
    number is measured on training data -- but it is drawn from the same generator, so it
    is in-distribution by construction. ``make_personas`` forces every judge axis,
    criterion and continuity weight positive, which is the sign structure the four
    hand-written evaluation personas also have, so an all-equal-positive vector is already
    a decent ranker on this population and the prior is *guaranteed* correlated with the
    writer before any data arrives. The four evaluated personas, by contrast, come from a
    distribution this generator cannot produce -- ``_weights`` zero-fills, and
    ``uniform(0.1, 1.2)`` never emits 0.0 -- so there is no leakage into them and also no
    evidence here that the prior transfers to a writer unlike the ones it was fitted on.

    Args:
        prior: The fitted prior.
        n_decisions: How many comparisons the fresh writer has made.
        n_test: Held-out comparisons to score on.
        seed: RNG seed for the fresh writer.
        shrink_unidentified: Pass through to the fit, so the shrinkage A/B can check that
            the change does not cost the prior the thing the prior is for.

    Returns:
        ``{"prior": accuracy, "uniform": accuracy, "lift": difference}``.
    """
    fresh = make_personas(1, seed=seed)[0]
    train = generate_pairs([fresh], per_persona=n_decisions, seed=seed + 1)
    test = generate_pairs([fresh], per_persona=n_test, seed=seed + 2)

    from_prior = fit(train, prior=prior, l2=2.0, shrink_unidentified=shrink_unidentified)
    from_uniform = fit(
        train, prior=BTWeights.uniform(), l2=2.0, shrink_unidentified=shrink_unidentified
    )

    prior_accuracy = pairwise_accuracy(from_prior, test)
    uniform_accuracy = pairwise_accuracy(from_uniform, test)
    return {
        "prior": prior_accuracy,
        "uniform": uniform_accuracy,
        "lift": prior_accuracy - uniform_accuracy,
    }
