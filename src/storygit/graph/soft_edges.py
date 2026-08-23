"""Soft dependency edges — the ablation the declared graph is measured against.

The declared `produces`/`consumes` edges are exact but incomplete: a beat can depend on an
earlier one in ways nobody wrote down, through a callback, an image, or an implication.
The obvious alternative is to infer dependencies from embedding similarity — if two beats
read alike, changing one probably affects the other.

This module implements that alternative, deliberately, as a *weaker* signal and as
something to measure rather than something to trust. Its marks are `maybe_affected`, never
`stale`; they are visually distinct in the interface; and the evaluation sweeps the
similarity threshold and reports precision and recall against ground truth so the
trade-off is a number rather than an opinion.

The expectation, worth stating in advance so the result is falsifiable: high recall at low
thresholds with poor precision, and the crossover point too poor to use as a primary
mechanism. If that turns out to be wrong the design should change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from storygit.domain.ids import NodeId
from storygit.domain.nodes import Beat
from storygit.domain.state import StoryState

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

DEFAULT_THRESHOLD = 0.72
"""Cosine similarity above which two beats are called "maybe related"."""


def beat_text(state: StoryState, beat_id: NodeId) -> str:
    """The text used to represent a beat for similarity purposes.

    Planning fields plus the prose, if any: a beat's dependence on another is as likely to
    live in what was written as in what was planned.

    Args:
        state: The state to read.
        beat_id: The beat.

    Returns:
        A single string, empty if the node is missing.
    """
    node = state.nodes.get(beat_id)
    if node is None:
        return ""
    parts = [node.title, node.what_happens, node.audience_learns, node.location]
    prose = state.prose_for(beat_id)
    if prose is not None and prose.text:
        parts.append(prose.text)
    return " ".join(p for p in parts if p)


class EmbeddingEdgeProvider:
    """Suggests non-declared dependencies from beat-to-beat embedding similarity.

    Satisfies :class:`storygit.graph.dependency.EdgeProvider`, so it slots into
    :func:`storygit.graph.propagation.propagate` without the deterministic core gaining
    any dependency on an ML package.

    Attributes:
        threshold: Cosine similarity above which an edge is suggested.
        forward_only: When true, only later beats can be affected by an earlier one.
    """

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        forward_only: bool = True,
        embed_fn: object | None = None,
    ) -> None:
        """Create the provider.

        Args:
            threshold: Similarity cut-off.
            forward_only: Restrict suggestions to beats later in reading order. A story
                does not usually reach backwards, and allowing it doubles the false
                positives.
            embed_fn: Override the embedding function; tests inject synthetic vectors so
                the threshold behaviour is testable without loading a model.
        """
        self.threshold = threshold
        self.forward_only = forward_only
        self._embed_fn = embed_fn
        self._cache: dict[str, np.ndarray] = {}

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self._embed_fn is not None:
            return self._embed_fn(texts)  # type: ignore[operator, no-any-return]
        from storygit.selection.embed import embed_texts

        return embed_texts(texts)

    def extra_dependents(self, state: StoryState, node_id: NodeId) -> tuple[NodeId, ...]:
        """Beats that read closely enough on ``node_id`` to be worth checking.

        Args:
            state: Current state.
            node_id: The beat that changed.

        Returns:
            Beat ids above the similarity threshold, in reading order. Never includes
            ``node_id`` itself or beats that already declare a dependency on it.
        """
        import numpy as np

        beats = [b for b in state.beats_in_order() if isinstance(b, Beat)]
        if len(beats) < 2 or node_id not in {b.id for b in beats}:
            return ()

        texts = [beat_text(state, b.id) for b in beats]
        if not any(texts):
            return ()
        vectors = self._embed(texts)
        index = {b.id: i for i, b in enumerate(beats)}
        origin = vectors[index[node_id]]
        similarities = np.asarray(vectors @ origin, dtype=np.float64)

        declared: set[NodeId] = set()
        source = state.nodes.get(node_id)
        for fact_id in getattr(source, "produces", ()):
            declared.update(state.consumers_of.get(fact_id, ()))

        origin_seq = state.seq.get(node_id, 0)
        out: list[NodeId] = []
        for beat in beats:
            if beat.id == node_id or beat.id in declared:
                continue
            if self.forward_only and state.seq.get(beat.id, 0) <= origin_seq:
                continue
            if float(similarities[index[beat.id]]) >= self.threshold:
                out.append(beat.id)
        return tuple(out)
