"""Retrieval of the writer's own work into the prompt.

The cheapest way to make a model sound like a particular writer is to show it that
writer's sentences. No training, no weights, and it works from the first accepted
paragraph — which matters because every other learner in this package needs data the writer
has not produced yet.

Two pools. **Positives** are accepted proposals and prose the writer typed themselves.
**Negatives** are rejected proposals: showing a model what was turned down is a stronger
instruction than describing it, and it is the only use for a rejection other than deleting
it.

Retrieval is nearest-neighbour by embedding against the current intent, with a recency
tilt. Recency matters because taste drifts within a session and the most recent evidence is
the most relevant, and because early accepts happened when the writer had fewer
alternatives to compare against.

The whole block is behind a flag, so the no-preference ablation can switch it off and get
byte-identical chunk 3 behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

RECENCY_HALF_LIFE = 12.0
"""How many exemplars back an item's weight halves. Tilts, does not dominate."""


class Exemplar(BaseModel):
    """One piece of text the writer liked or rejected.

    Attributes:
        text: The prose.
        positive: True for accepted or human-written, False for rejected.
        age: How many exemplars have been added since this one; 0 is newest.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    positive: bool
    age: int = 0


class ExemplarPool:
    """The writer's accepted and rejected work, retrievable by similarity.

    Attributes:
        positives: Accepted and human-written texts, newest last.
        negatives: Rejected texts, newest last.
    """

    def __init__(
        self,
        positives: Sequence[str] = (),
        negatives: Sequence[str] = (),
        *,
        half_life: float = RECENCY_HALF_LIFE,
    ) -> None:
        """Create a pool.

        Args:
            positives: Accepted and human-written prose, oldest first.
            negatives: Rejected proposals, oldest first.
            half_life: Recency half-life in items.
        """
        self.positives = [t for t in positives if t.strip()]
        self.negatives = [t for t in negatives if t.strip()]
        self.half_life = half_life

    def __len__(self) -> int:
        """Total number of exemplars held."""
        return len(self.positives) + len(self.negatives)

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to retrieve. True for a brand new writer."""
        return not self.positives and not self.negatives

    def retrieve(self, query: str, *, m: int = 2) -> tuple[list[str], list[str]]:
        """The ``m`` most relevant positives and negatives for a query.

        Args:
            query: The current intent or slice text.
            m: How many of each to return. Small on purpose: exemplars compete with the
               state slice for prompt space, and two good ones beat six mediocre ones.

        Returns:
            ``(positives, negatives)``.
        """
        if self.is_empty:
            return [], []
        if not query.strip():
            return self.positives[-m:], self.negatives[-m:]
        return (
            self._nearest(self.positives, query, m),
            self._nearest(self.negatives, query, m),
        )

    def _nearest(self, pool: list[str], query: str, m: int) -> list[str]:
        """Recency-weighted nearest neighbours from one pool."""
        if not pool:
            return []
        if len(pool) <= m:
            return list(pool)
        from storygit.selection.embed import embed_texts

        vectors = embed_texts([query, *pool])
        similarities = vectors[1:] @ vectors[0]
        scored = []
        for index, similarity in enumerate(similarities):
            age = len(pool) - 1 - index
            recency = 0.5 ** (age / self.half_life)
            scored.append((float(similarity) * (0.5 + 0.5 * recency), pool[index]))
        scored.sort(key=lambda pair: -pair[0])
        return [text for _, text in scored[:m]]

    def render(self, query: str, *, m: int = 2, enabled: bool = True) -> str:
        """The prompt block, or an empty string.

        Args:
            query: The current intent, for retrieval.
            m: How many of each polarity to include.
            enabled: When False, returns "" — this is the ablation switch.

        Returns:
            A block naming what the writer liked and turned down, or "".
        """
        if not enabled or self.is_empty:
            return ""
        positives, negatives = self.retrieve(query, m=m)
        lines: list[str] = []
        if positives:
            lines.append("THE WRITER KEPT THESE. Match this voice.")
            lines.extend(f"  ---\n  {text.strip()}" for text in positives)
        if negatives:
            lines.append("THE WRITER TURNED THESE DOWN. Do not write like this.")
            lines.extend(f"  ---\n  {text.strip()}" for text in negatives)
        return "\n".join(lines)
