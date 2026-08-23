r"""Maximal Marginal Relevance: pick k that are good *and* different from each other.

The greedy top-k by quality is the obvious thing and it is wrong here, because the three
highest-scoring candidates for a beat are usually three phrasings of the same idea. The
writer then has no choice to make. MMR fixes this by scoring each remaining candidate
against both its own quality and its similarity to what has already been picked, so the
second pick is the best candidate *that is not the first one again*.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


def mmr_select(
    quality: Sequence[float],
    similarity: np.ndarray,
    k: int,
    *,
    lambda_: float = 0.7,
) -> list[int]:
    r"""Select ``k`` indices by maximal marginal relevance.

    At each step the score of a remaining candidate ``i`` is

    .. math:: \lambda \cdot q_i - (1 - \lambda) \max_{j \in S} \mathrm{sim}(i, j)

    where ``S`` is what has been selected so far. The first pick is pure quality, since
    nothing has been selected to be similar to.

    Args:
        quality: One score per candidate. Any scale; only the ordering and the relative
            gaps matter, so callers normalize first.
        similarity: ``(n, n)`` similarity matrix.
        k: How many to select.
        lambda_: 1.0 is pure quality (equivalent to top-k), 0.0 is pure spread.

    Returns:
        Selected indices, in selection order.
    """
    n = len(quality)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    selected: list[int] = []
    remaining = set(range(n))

    for _ in range(k):
        best_index = -1
        best_score = float("-inf")
        for i in sorted(remaining):
            redundancy = max((float(similarity[i][j]) for j in selected), default=0.0)
            score = lambda_ * float(quality[i]) - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_index = i
        selected.append(best_index)
        remaining.discard(best_index)
    return selected
