r"""The voice model: learning what this writer's prose *sounds* like.

A judge can score whether a passage is well constructed. It cannot tell you whether it
sounds like *this writer*, because that is not a quality — it is an identity, and it is
different for every writer. So it has to be learned from their own text.

**The setup.** Freeze a sentence encoder (bge-small, 384-d) and train a small linear
projection on top of it with InfoNCE. Anchors are prose the writer wrote or edited.
Positives are proposals they accepted. Negatives are proposals they rejected and the
pre-edit versions of things they changed. The head learns a subspace where "sounds like
this writer" is a direction.

**Why a frozen encoder and a small head.** A writer produces a few thousand words during a
session, not a corpus. Fine-tuning the encoder on that would destroy it. The frozen encoder
supplies general semantic and stylistic structure; the projection only has to find which
part of that structure this writer cares about — 384x128 parameters, seconds on CPU,
learnable from tens of examples.

**Edit direction vectors.** A second, even cheaper signal from the same data. Take the mean
of ``embed(after) - embed(before)`` over every edit pair. That vector points from what the
model writes towards what the writer wants. Scoring a candidate by its projection onto it
answers "is this already in the direction the writer keeps pulling me?" — and it needs no
training at all, which means it works from the very first edit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

PROJECTION_DIM = 128
TEMPERATURE = 0.07
"""InfoNCE temperature. The standard value; see the docstring on why it matters."""


class VoiceModel(BaseModel):
    """A trained projection head plus the writer's voice centroid.

    Attributes:
        projection: Flattened ``(input_dim, PROJECTION_DIM)`` matrix.
        input_dim: Encoder dimension.
        output_dim: Projection dimension.
        centroid: The writer's voice direction in projected space, unit length.
        direction: Mean ``embed(after) - embed(before)`` over edit pairs, in *raw*
            encoder space. Separate from the projection on purpose: it is meaningful with
            one edit pair, long before the head has enough data to train.
        anchors_seen: How many anchor texts the head was trained on.
        edits_seen: How many edit pairs the direction was computed from.
    """

    model_config = ConfigDict(frozen=True)

    projection: tuple[float, ...] = ()
    input_dim: int = 384
    output_dim: int = PROJECTION_DIM
    centroid: tuple[float, ...] = ()
    direction: tuple[float, ...] = ()
    anchors_seen: int = 0
    edits_seen: int = 0

    @property
    def is_trained(self) -> bool:
        """Whether a projection has actually been fitted."""
        return bool(self.projection) and bool(self.centroid)

    def project(self, vectors: np.ndarray) -> np.ndarray:
        """Map encoder embeddings into the voice space, L2-normalized.

        Falls back to the identity when untrained, so an untrained model degrades to plain
        cosine against the centroid rather than crashing.

        Args:
            vectors: ``(n, input_dim)`` encoder embeddings.

        Returns:
            ``(n, output_dim)`` unit-norm projections.
        """
        import numpy as np

        if not self.projection:
            return vectors
        matrix = np.array(self.projection, dtype=np.float32).reshape(
            self.input_dim, self.output_dim
        )
        projected = vectors @ matrix
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.asarray(projected / norms, dtype=np.float32)

    def score(self, texts: Sequence[str]) -> list[float]:
        """How much each text sounds like this writer, in ``[0, 1]``.

        Args:
            texts: Candidate prose.

        Returns:
            One score per text; 0.5 for everything when the model is untrained, so a new
            writer's ranking is simply unaffected by voice rather than skewed by it.
        """
        import numpy as np

        if not texts:
            return []
        if not self.is_trained:
            return [0.5] * len(texts)
        from storygit.selection.embed import embed_texts

        projected = self.project(embed_texts(list(texts)))
        centroid = np.array(self.centroid, dtype=np.float32)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        cosines = projected @ centroid
        return [float((c + 1.0) / 2.0) for c in cosines]

    def direction_scores(self, texts: Sequence[str]) -> list[float]:
        """Projection of each text onto the mean edit direction, in ``[0, 1]``.

        Args:
            texts: Candidate prose.

        Returns:
            One score per text; 0.5 when no edits have been seen.
        """
        import numpy as np

        if not texts:
            return []
        if not self.direction:
            return [0.5] * len(texts)
        from storygit.selection.embed import embed_texts

        vectors = embed_texts(list(texts))
        direction = np.array(self.direction, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            return [0.5] * len(texts)
        projections = vectors @ (direction / norm)
        low, high = float(projections.min()), float(projections.max())
        if high - low < 1e-9:
            return [0.5] * len(texts)
        return [float((p - low) / (high - low)) for p in projections]

    def save(self, path: Path | str) -> None:
        """Persist to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json())

    @staticmethod
    def load(path: Path | str) -> VoiceModel:
        """Load from JSON, returning an untrained model if the file is absent."""
        file = Path(path)
        if not file.exists():
            return VoiceModel()
        return VoiceModel.model_validate(json.loads(file.read_text()))


def edit_direction(before: Sequence[str], after: Sequence[str]) -> tuple[float, ...]:
    r"""Mean of :math:`\mathrm{embed}(\text{after}) - \mathrm{embed}(\text{before})`.

    Args:
        before: The model's versions.
        after: The writer's versions, index-aligned with ``before``.

    Returns:
        The direction vector, empty when there are no pairs.
    """
    if not before or len(before) != len(after):
        return ()
    from storygit.selection.embed import embed_texts

    deltas = embed_texts(list(after)) - embed_texts(list(before))
    return tuple(float(v) for v in deltas.mean(axis=0))


def train(
    anchors: Sequence[str],
    positives: Sequence[str],
    negatives: Sequence[str],
    *,
    edit_before: Sequence[str] = (),
    edit_after: Sequence[str] = (),
    dim: int = PROJECTION_DIM,
    epochs: int = 200,
    learning_rate: float = 0.05,
    temperature: float = TEMPERATURE,
    seed: int = 0,
) -> VoiceModel:
    r"""Train the projection head with InfoNCE and compute the voice centroid.

    The loss, for anchor :math:`a` with positive :math:`p` against negatives
    :math:`n_1..n_k`:

    .. math::
        \mathcal{L} = -\log
        \frac{\exp(\mathrm{sim}(a, p) / \tau)}
             {\exp(\mathrm{sim}(a, p)/\tau) + \sum_j \exp(\mathrm{sim}(a, n_j)/\tau)}

    which is exactly cross-entropy over "which of these is the positive". Minimizing it
    pulls anchors and positives together and pushes negatives away, in the projected space
    only --- the encoder never moves.

    Args:
        anchors: Human-written and human-edited prose.
        positives: Accepted proposals.
        negatives: Rejected proposals and pre-edit versions.
        edit_before: Pre-edit texts, for the direction vector.
        edit_after: Post-edit texts, index-aligned with ``edit_before``.
        dim: Projection dimension.
        epochs: Gradient steps.
        learning_rate: Step size.
        temperature: InfoNCE temperature.
        seed: Initialization seed, so training is reproducible.

    Returns:
        The trained model. With too little data it returns an untrained model carrying
        only the edit direction, which is the honest degradation: no voice model is better
        than one fitted to three examples.
    """
    import numpy as np

    direction = edit_direction(edit_before, edit_after)
    if len(anchors) < 2 or not positives or not negatives:
        return VoiceModel(direction=direction, edits_seen=len(edit_before))

    from storygit.selection.embed import embed_texts

    anchor_vectors = embed_texts(list(anchors))
    positive_vectors = embed_texts(list(positives))
    negative_vectors = embed_texts(list(negatives))
    input_dim = int(anchor_vectors.shape[1])

    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 1.0 / np.sqrt(input_dim), size=(input_dim, dim))

    def normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.asarray(x / norms, dtype=np.float64)

    for _ in range(epochs):
        a = normalize(anchor_vectors @ matrix)
        p = normalize(positive_vectors @ matrix)
        n = normalize(negative_vectors @ matrix)

        # Each anchor is paired with the positive at the same index (cycling), and every
        # negative acts as a negative for every anchor.
        pos_index = np.arange(len(a)) % len(p)
        logits_pos = np.einsum("ij,ij->i", a, p[pos_index]) / temperature
        logits_neg = (a @ n.T) / temperature

        all_logits = np.concatenate([logits_pos[:, None], logits_neg], axis=1)
        shifted = all_logits - all_logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)

        # d loss / d a  =  (sum_j q_j * z_j - z_pos) / temperature, with z the partners.
        partners = np.concatenate([p[pos_index][:, None, :], np.tile(n, (len(a), 1, 1))], axis=1)
        weighted = np.einsum("ik,ikj->ij", probabilities, partners)
        grad_a = (weighted - p[pos_index]) / temperature

        # Backprop through the normalization is dropped: with unit-norm inputs and a small
        # step size the direction term dominates, and keeping it makes the update far more
        # expensive for no measurable gain at this scale.
        grad_matrix = anchor_vectors.T @ grad_a / len(a)
        matrix -= learning_rate * grad_matrix

    projected_anchors = normalize(anchor_vectors @ matrix)
    projected_positives = normalize(positive_vectors @ matrix)
    centroid = np.concatenate([projected_anchors, projected_positives], axis=0).mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) or 1.0)

    return VoiceModel(
        projection=tuple(float(v) for v in matrix.flatten()),
        input_dim=input_dim,
        output_dim=dim,
        centroid=tuple(float(v) for v in centroid),
        direction=direction,
        anchors_seen=len(anchors),
        edits_seen=len(edit_before),
    )
