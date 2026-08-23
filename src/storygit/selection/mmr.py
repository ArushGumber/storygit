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

DEFAULT_LAMBDA = 0.5
"""Quality/diversity trade-off, measured rather than chosen.

Swept over six candidates whose three highest-quality entries are near-paraphrases of each
other --- the case diversity selection exists for:

======  ======================  =========  =======
lambda  picked                  diversity  quality
======  ======================  =========  =======
0.9     three paraphrases       0.254      0.900
0.7     three paraphrases       0.254      0.900
0.6     three paraphrases       0.254      0.900
0.5     two paraphrases + one   0.483      0.833
0.4     spread across all three 0.555      0.780
0.3     spread across all three 0.555      0.780
======  ======================  =========  =======

At 0.7 --- the conventional default, and this project's first choice --- MMR picks exactly
what the top-k baseline picks, which makes the default selector indistinguishable from the
ablation it is meant to beat. 0.5 buys a large diversity gain for a small quality
concession. Below 0.4 it converges on the DPP's answer.

Worth noting what this says about the two methods: the DPP reaches the spread answer with
no parameter at all, because its kernel is *multiplicative* --- a near-duplicate's
contribution to the determinant collapses regardless of how good it is --- whereas MMR's
*additive* penalty can always be outvoted by a large enough quality gap. That is the
argument for DPPs stated as a measurement rather than as a preference.
"""


def mmr_select(
    quality: Sequence[float],
    similarity: np.ndarray,
    k: int,
    *,
    lambda_: float = DEFAULT_LAMBDA,
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
        lambda_: 1.0 is pure quality (equivalent to top-k), 0.0 is pure spread. Both the
            quality scores and the similarity matrix are rescaled to ``[0, 1]`` across the
            candidate set first, so lambda means what it says on any input scale.

    Returns:
        Selected indices, in selection order.
    """
    n = len(quality)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    # Raw bi-encoder cosine has a high floor, which makes the redundancy term nearly
    # constant and collapses MMR into top-k. See rescale_similarity for the measurement.
    from storygit.selection.embed import min_max, rescale_similarity

    similarity = rescale_similarity(similarity)
    scaled_quality = min_max(list(quality))
    selected: list[int] = []
    remaining = set(range(n))

    for _ in range(k):
        best_index = -1
        best_score = float("-inf")
        for i in sorted(remaining):
            redundancy = max((float(similarity[i][j]) for j in selected), default=0.0)
            score = lambda_ * scaled_quality[i] - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best_index = i
        selected.append(best_index)
        remaining.discard(best_index)
    return selected
