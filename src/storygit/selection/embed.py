"""Embeddings and similarity kernels shared by MMR, DPP, and the dial.

Everything here is a thin, testable layer over ``providers/local.py``. It exists so that
the selection algorithms take plain arrays and can be tested on synthetic geometry with no
model loaded at all — which is what makes the MMR and DPP tests fast, deterministic, and
actually about the algorithm rather than about the encoder.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Embed candidate texts with the local encoder, L2-normalized.

    Args:
        texts: One string per candidate.

    Returns:
        Shape ``(n, dim)``, float32, unit rows.
    """
    from storygit.providers.local import embed

    return embed(texts)


def similarity(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of unit-norm rows.

    Args:
        vectors: Shape ``(n, dim)``, rows already normalized.

    Returns:
        Shape ``(n, n)``, values in ``[-1, 1]``.
    """
    import numpy as np

    if vectors.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(vectors @ vectors.T, dtype=np.float32)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length, leaving zero rows alone.

    Args:
        vectors: Shape ``(n, dim)``.

    Returns:
        The same shape, with unit rows.
    """
    import numpy as np

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(vectors / norms, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity, or ``0.0`` if either vector is zero.
    """
    import numpy as np

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def min_max(values: Sequence[float]) -> list[float]:
    """Scale values into ``[0, 1]``.

    Selection blends a quality score with a distance, and the two arrive on unrelated
    scales — a judge score of 3.8 out of 5 against a cosine distance of 0.21. Blending
    them without rescaling would let whichever happens to have the larger range dominate,
    and the dial would stop meaning what it says.

    Args:
        values: The values to scale.

    Returns:
        Scaled values; all ``0.5`` when every input is equal, so a tie stays a tie.
    """
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return [0.5] * len(values)
    return [(v - low) / (high - low) for v in values]
