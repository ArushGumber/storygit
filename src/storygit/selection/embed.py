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


def rescale_similarity(matrix: np.ndarray) -> np.ndarray:
    """Rescale a similarity matrix so its off-diagonal entries span ``[0, 1]``.

    Bi-encoder cosine has a high floor: two unrelated English sentences embedded by
    bge-small still score around 0.6-0.7, because they share a language, a register, and a
    topic. Feeding those raw into MMR mis-scales the trade-off badly --- a redundancy
    penalty that is 0.7 for *everything* is a constant, and a constant subtracted from
    every candidate changes no ordering at all, so MMR silently degenerates into top-k.

    That is not hypothetical: it is what the first offline run showed. On six candidates
    where three were near-paraphrases, MMR at lambda=0.7 picked exactly the three
    paraphrases, identically to the top-k baseline.

    Rescaling within the candidate set restores the intended meaning: 1.0 is "as similar as
    anything here", 0.0 is "as different as anything here", and lambda is once again a
    trade-off rather than a decoration.

    Args:
        matrix: An ``(n, n)`` similarity matrix.

    Returns:
        The same shape, off-diagonal entries rescaled, diagonal forced to 1.0.
    """
    import numpy as np

    n = matrix.shape[0]
    if n < 2:
        return matrix
    off = ~np.eye(n, dtype=bool)
    values = matrix[off]
    low, high = float(values.min()), float(values.max())
    out = np.zeros_like(matrix) if high - low < 1e-9 else (matrix - low) / (high - low)
    out = np.clip(out, 0.0, 1.0)
    np.fill_diagonal(out, 1.0)
    return np.asarray(out, dtype=np.float32)


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
