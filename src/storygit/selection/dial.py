"""The coherent-to-surprising dial.

Fabula's writers asked for an "absurdity dial" and the request is more interesting than it
sounds: it is a request to control the objective, not just the output. A writer at 3am on
episode 40 wants safe continuations; the same writer starting a new arc wants to be
surprised. Those are different objectives, and no single ranking serves both.

The implementation is deliberately simple and measurable. Take the greedy, temperature-0
continuation as the definition of "what this model would obviously do next". Every
candidate's *surprise* is its embedding distance from that. Effective quality is then

    (1 - d) * base_quality + d * surprise

where ``d`` is the dial. At ``d = 0`` the ranking is pure quality; at ``d = 1`` it is pure
distance from the obvious. The greedy continuation costs one extra model call per node and
is cached, so moving the dial costs nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from storygit.selection.embed import cosine, min_max

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


def surprise_scores(candidates: np.ndarray, greedy: np.ndarray) -> list[float]:
    """How far each candidate is from the obvious continuation.

    Args:
        candidates: ``(n, dim)`` candidate embeddings.
        greedy: ``(dim,)`` embedding of the greedy temperature-0 continuation.

    Returns:
        One distance per candidate, in ``[0, 2]`` before normalization.
    """
    return [1.0 - cosine(candidates[i], greedy) for i in range(candidates.shape[0])]


def effective_quality(
    base_quality: Sequence[float],
    surprise: Sequence[float],
    dial: float,
) -> list[float]:
    """Blend quality and surprise according to the dial.

    Both inputs are rescaled to ``[0, 1]`` first. They arrive on unrelated scales — a
    judge score against a cosine distance — and blending them raw would let whichever has
    the wider range dominate, so the dial would stop meaning what it says.

    Args:
        base_quality: Quality score per candidate.
        surprise: Distance-from-greedy per candidate.
        dial: 0.0 coherent, 1.0 surprising.

    Returns:
        Effective quality per candidate, in ``[0, 1]``.
    """
    if not base_quality:
        return []
    dial = min(1.0, max(0.0, dial))
    q = min_max(list(base_quality))
    s = min_max(list(surprise)) if surprise else [0.0] * len(q)
    return [(1.0 - dial) * q[i] + dial * s[i] for i in range(len(q))]
