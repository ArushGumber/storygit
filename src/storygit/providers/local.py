"""Local models: sentence embeddings and the NLI cross-encoder.

Two jobs, both free and both on CPU. Embeddings (bge-small, 384-dimensional) power
candidate diversity selection, exemplar retrieval, the voice model, and the soft
dependency-edge ablation. The DeBERTa-MNLI cross-encoder is layer 2 of the continuity
checker: it decides whether two free-text facts contradict each other.

Everything is loaded lazily and once. ``HF_HUB_OFFLINE=1`` is set before any model is
touched, so a cache miss fails loudly instead of silently downloading half a gigabyte
during an evaluation run. The models are pre-cached; if this raises, the cache is wrong
and that is worth knowing immediately.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    import numpy as np

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
"""Primary encoder. 384 dimensions, strong on short-text similarity, fast on CPU."""

ALT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
"""Alternate encoder, kept for the ablation comparison in the evaluation."""

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
"""Cross-encoder producing (contradiction, entailment, neutral) logits."""

_lock = threading.Lock()
_encoders: dict[str, Any] = {}
_nli: dict[str, Any] = {}


def _force_offline() -> None:
    """Make Hugging Face loads fail rather than download.

    The models are pre-downloaded into the cache. A download during an evaluation run
    would be slow, would need network, and would hide a broken environment.
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def get_encoder(model_name: str = EMBEDDING_MODEL) -> Any:
    """Load (once) and return a sentence-transformers encoder.

    Args:
        model_name: Hugging Face model id.

    Returns:
        The loaded ``SentenceTransformer``.
    """
    with _lock:
        if model_name not in _encoders:
            _force_offline()
            from sentence_transformers import SentenceTransformer

            _encoders[model_name] = SentenceTransformer(model_name, device="cpu")
        return _encoders[model_name]


def embed(texts: Sequence[str], *, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Embed a batch of texts, L2-normalized.

    Normalizing means the dot product *is* cosine similarity, which every consumer here
    wants — MMR, the DPP kernel, exemplar retrieval, and the voice centroid.

    Args:
        texts: Texts to embed.
        model_name: Which encoder to use.

    Returns:
        An array of shape ``(len(texts), dim)``, float32.
    """
    import numpy as np

    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    encoder = get_encoder(model_name)
    vectors = encoder.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return np.asarray(vectors, dtype=np.float32)


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of L2-normalized rows.

    Args:
        vectors: Normalized embeddings, shape ``(n, dim)``.

    Returns:
        An ``(n, n)`` similarity matrix.
    """
    import numpy as np

    if vectors.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(vectors @ vectors.T, dtype=np.float32)


def get_nli() -> tuple[Any, Any]:
    """Load (once) and return the NLI tokenizer and model.

    Returns:
        ``(tokenizer, model)``, the model in eval mode on CPU.
    """
    with _lock:
        if "pair" not in _nli:
            _force_offline()
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
            model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
            model.eval()
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
            _nli["pair"] = (tokenizer, model)
        tokenizer, model = _nli["pair"]
        return tokenizer, model


def nli_scores(pairs: Sequence[tuple[str, str]], *, batch_size: int = 16) -> list[dict[str, float]]:
    """Score premise/hypothesis pairs for contradiction, entailment, and neutrality.

    Args:
        pairs: ``(premise, hypothesis)`` sentence pairs.
        batch_size: How many pairs to run at once.

    Returns:
        One probability dict per pair, with keys ``contradiction``, ``entailment``,
        and ``neutral``.
    """
    if not pairs:
        return []
    import torch

    tokenizer, model = get_nli()
    labels = _label_order(model)
    out: list[dict[str, float]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            [p for p, _ in batch],
            [h for _, h in batch],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).tolist()
        for row in probabilities:
            out.append({labels[i]: float(row[i]) for i in range(len(row))})
    return out


def _label_order(model: Any) -> list[str]:
    """Map the model's label indices to our three canonical names.

    Different MNLI checkpoints order their labels differently, so this reads the
    model's own ``id2label`` rather than assuming.

    Args:
        model: The loaded classification model.

    Returns:
        Canonical label names in the model's index order.
    """
    id2label = getattr(model.config, "id2label", None) or {}
    canonical = {
        "contradiction": "contradiction",
        "entailment": "entailment",
        "neutral": "neutral",
    }
    ordered: list[str] = []
    for index in range(len(id2label) or 3):
        raw = str(id2label.get(index, index)).lower()
        ordered.append(canonical.get(raw, raw))
    return ordered or ["contradiction", "entailment", "neutral"]


def reset_cache() -> None:
    """Drop the loaded models. Used by tests that need a clean process state."""
    with _lock:
        _encoders.clear()
        _nli.clear()
