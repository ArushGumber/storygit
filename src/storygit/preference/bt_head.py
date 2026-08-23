r"""The Bradley-Terry preference head.

The writer's choices are comparisons, not ratings: they accepted A while B and C were on
screen. Bradley-Terry is the standard model for exactly that data. Each candidate has a
latent worth; the probability of preferring $a$ over $b$ is

.. math:: P(a \succ b) = \sigma(w^\top (f_a - f_b))

where $f$ is the feature vector and $\sigma$ the logistic function. That is *logistic
regression on feature differences* — the intercept vanishes because it cancels in the
difference — so fitting it is a solved, convex problem with no tuning drama, and the
learned $w$ is directly readable: a positive weight on ``dialogue_ratio`` means this writer
likes dialogue.

This is the same construction as the reward model in RLHF, minus the policy-gradient half:
the same pairwise-comparison likelihood over the same kind of human feedback. Here the
learned score orders three candidates rather than training a policy, which is the right
scale for one writer's few dozen decisions.

Fitted with L2-regularized gradient descent in pure NumPy. At thirteen features and a few
hundred pairs the whole fit is milliseconds, needs no ML framework, and — usefully — is
deterministic under a seed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from storygit.preference.features import BASE_FEATURES, FeatureVector

if TYPE_CHECKING:  # pragma: no cover
    pass


class BTWeights(BaseModel):
    """A fitted (or prior) weight vector.

    Attributes:
        names: Feature names, in order. Persisted so a model outlives a code change that
            adds a feature.
        weights: One weight per name.
        pairs_seen: How many comparisons it was fitted on. Zero means the prior.
        l2: The regularization strength used.
    """

    model_config = ConfigDict(frozen=True)

    names: tuple[str, ...] = BASE_FEATURES
    weights: tuple[float, ...] = ()
    pairs_seen: int = 0
    l2: float = 1.0

    @staticmethod
    def uniform(names: tuple[str, ...] = BASE_FEATURES) -> BTWeights:
        """The cold-start model: every feature counts equally and positively.

        A new writer has produced no comparisons, and the head must still rank. Uniform
        positive weights say "good on every axis is good", which is a defensible default
        and is exactly what the pretraining prior improves on.
        """
        return BTWeights(names=names, weights=tuple(1.0 / len(names) for _ in names))

    def score(self, features: FeatureVector) -> float:
        """The latent worth of one candidate."""
        vector = features.as_list(self.names)
        weights = self.weights or BTWeights.uniform(self.names).weights
        return float(sum(w * v for w, v in zip(weights, vector, strict=True)))

    def probability(self, a: FeatureVector, b: FeatureVector) -> float:
        """P(the writer prefers ``a`` to ``b``)."""
        import math

        return 1.0 / (1.0 + math.exp(-(self.score(a) - self.score(b))))

    def score_of(self, name: str) -> float:
        """The weight on one named feature, or 0.0 if the model does not carry it."""
        if name not in self.names or not self.weights:
            return 0.0
        return float(self.weights[self.names.index(name)])

    def top_weights(self, n: int = 5) -> list[tuple[str, float]]:
        """The features driving the ranking, largest magnitude first.

        This is what makes the head explainable to the writer: "you have been choosing for
        specificity and against length" is a sentence the interface can show.
        """
        pairs = list(zip(self.names, self.weights or (), strict=False))
        return sorted(pairs, key=lambda kv: -abs(kv[1]))[:n]

    def save(self, path: Path | str) -> None:
        """Persist to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2))

    @staticmethod
    def load(path: Path | str) -> BTWeights:
        """Load from JSON, returning the uniform prior if the file is absent."""
        file = Path(path)
        if not file.exists():
            return BTWeights.uniform()
        return BTWeights.model_validate(json.loads(file.read_text()))


def fit(
    pairs: Sequence[tuple[FeatureVector, FeatureVector]],
    *,
    names: tuple[str, ...] = BASE_FEATURES,
    prior: BTWeights | None = None,
    l2: float = 1.0,
    epochs: int = 400,
    learning_rate: float = 0.5,
    sample_weights: Sequence[float] | None = None,
) -> BTWeights:
    """Fit weights to a set of (winner, loser) feature pairs.

    Args:
        pairs: ``(winner_features, loser_features)``, one per comparison.
        names: Feature order.
        prior: Weights to start from and regularize towards. This is how per-writer
            adaptation works: the pretrained prior is the starting point, and the writer's
            own comparisons move it. Regularizing *towards the prior* rather than towards
            zero is what makes ten comparisons useful instead of noise.
        l2: Regularization strength. High enough to matter at a few dozen pairs.
        epochs: Gradient steps.
        learning_rate: Step size.
        sample_weights: Per-pair weights. Edited-then-accepted wins are weaker evidence
            than clean accepts and are down-weighted by the caller.

    Returns:
        The fitted weights, carrying the number of pairs they saw.
    """
    import numpy as np

    start = prior if prior is not None else BTWeights.uniform(names)
    w = np.array(start.weights or BTWeights.uniform(names).weights, dtype=np.float64)
    prior_w = w.copy()
    if not pairs:
        return BTWeights(names=names, weights=tuple(w.tolist()), pairs_seen=0, l2=l2)

    x = np.array(
        [
            [a - b for a, b in zip(win.as_list(names), lose.as_list(names), strict=True)]
            for win, lose in pairs
        ],
        dtype=np.float64,
    )
    weights = (
        np.array(sample_weights, dtype=np.float64)
        if sample_weights is not None
        else np.ones(len(pairs), dtype=np.float64)
    )
    weights = weights / weights.sum() * len(pairs)

    for _ in range(epochs):
        # The winner is always the first element, so the label is always 1 and the
        # gradient of the log-likelihood is (1 - sigma(w.x)) * x.
        logits = x @ w
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        gradient = (x * ((1.0 - probabilities) * weights)[:, None]).sum(axis=0) / len(pairs)
        gradient -= l2 * (w - prior_w) / max(1, len(pairs))
        w += learning_rate * gradient

    return BTWeights(names=names, weights=tuple(w.tolist()), pairs_seen=len(pairs), l2=l2)


def pairwise_accuracy(
    weights: BTWeights, pairs: Sequence[tuple[FeatureVector, FeatureVector]]
) -> float:
    """Fraction of held-out comparisons the model gets right.

    Args:
        weights: The fitted model.
        pairs: ``(winner, loser)`` pairs it has not seen.

    Returns:
        Accuracy in ``[0, 1]``; 0.5 is chance.
    """
    if not pairs:
        return 0.0
    correct = sum(1 for win, lose in pairs if weights.score(win) > weights.score(lose))
    return correct / len(pairs)


def weight_correlation(weights: BTWeights, hidden: dict[str, float]) -> float:
    """Pearson correlation between fitted weights and a known hidden vector.

    Only meaningful in evaluation, where the simulated writer's preferences are literally a
    weight vector over these same features. It is the sharpest question the evaluation can
    ask: not "did acceptance go up" but "did the machinery recover the taste".

    Args:
        weights: Fitted weights.
        hidden: The true weights, by feature name.

    Returns:
        Pearson r, or 0.0 when either vector is constant.
    """
    import numpy as np

    a = np.array([weights.score_of(name) for name in weights.names], dtype=np.float64)
    b = np.array([hidden.get(name, 0.0) for name in weights.names], dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
