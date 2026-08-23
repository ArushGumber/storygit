r"""Determinantal point processes: diversity as volume, not as a penalty.

MMR is a heuristic — subtract a similarity term and hope the weighting is right. A DPP is
the principled version of the same instinct. Build a kernel
:math:`L = \mathrm{diag}(q)\, S\, \mathrm{diag}(q)` where ``q`` is quality and ``S`` is
cosine similarity; then
:math:`\det(L_S)` for a subset ``S`` is the squared volume of the parallelepiped spanned
by those candidates' (quality-scaled) vectors. Volume is large when the vectors are both
*long* (high quality) and *spread out* (mutually dissimilar), and collapses to zero when
two of them are parallel. So "pick the highest-determinant subset" is exactly "pick a good,
non-redundant set" — with no hand-tuned trade-off parameter.

Exact MAP inference is NP-hard in general. At n=6, k=3 it is trivial, and this module does
the standard greedy maximization, which has a known approximation guarantee and here is
usually exact.

The interface is deliberately identical to :func:`storygit.selection.mmr.mmr_select`, so
switching selectors is one config value and the evaluation can ablate it honestly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

JITTER = 1e-9
"""Added to the diagonal so a singular kernel stays invertible in float32."""


def build_kernel(quality: Sequence[float], similarity: np.ndarray) -> np.ndarray:
    """Build the quality-weighted DPP kernel.

    Args:
        quality: One non-negative score per candidate.
        similarity: ``(n, n)`` cosine similarity.

    Returns:
        ``L = diag(q) S diag(q)``, symmetric positive semi-definite.
    """
    import numpy as np

    q = np.clip(np.asarray(quality, dtype=np.float64), 1e-6, None)
    s = np.clip(np.asarray(similarity, dtype=np.float64), 0.0, None)
    kernel = np.outer(q, q) * s
    return np.asarray(kernel + np.eye(len(q)) * JITTER, dtype=np.float64)


def dpp_select(
    quality: Sequence[float],
    similarity: np.ndarray,
    k: int,
    *,
    lambda_: float = 0.7,
) -> list[int]:
    """Select ``k`` indices by greedy MAP inference over a quality-weighted DPP.

    Args:
        quality: One score per candidate, expected in ``[0, 1]``.
        similarity: ``(n, n)`` similarity matrix.
        k: How many to select.
        lambda_: Accepted for interface compatibility with
            :func:`~storygit.selection.mmr.mmr_select` and unused — a DPP gets its
            quality/diversity trade-off from the kernel itself, which is the point of
            using one.

    Returns:
        Selected indices, in selection order.
    """
    import numpy as np

    del lambda_  # the kernel encodes the trade-off; see the module docstring
    n = len(quality)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    kernel = build_kernel(quality, similarity)

    selected: list[int] = []
    remaining = list(range(n))
    for _ in range(k):
        best_index = -1
        best_gain = float("-inf")
        for i in remaining:
            subset = [*selected, i]
            gain = float(np.linalg.det(kernel[np.ix_(subset, subset)]))
            if gain > best_gain:
                best_gain = gain
                best_index = i
        if best_index < 0:  # pragma: no cover - defensive
            break
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


def topk_select(
    quality: Sequence[float],
    similarity: np.ndarray,
    k: int,
    *,
    lambda_: float = 0.7,
) -> list[int]:
    """Plain top-k by quality — the ablation baseline.

    This is what "diversity from temperature alone" looks like: sample at a high
    temperature and keep the best scoring candidates, with nothing preventing the top
    three from being the same idea three times. The evaluation compares MMR and DPP
    against it on mean pairwise distance at matched quality.

    Args:
        quality: One score per candidate.
        similarity: Ignored; present so the three selectors are interchangeable.
        k: How many to select.
        lambda_: Ignored.

    Returns:
        The ``k`` highest-quality indices, best first.
    """
    del similarity, lambda_
    order = sorted(range(len(quality)), key=lambda i: (-quality[i], i))
    return order[: min(k, len(order))]
