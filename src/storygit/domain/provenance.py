"""Who wrote which sentence, and the authorship ratios that fall out of it.

Provenance is per sentence span rather than per node because the interesting case
is mixed: the AI proposed a paragraph, the writer rewrote two sentences of it. A
node-level flag would lose exactly the information the writer cares about.
"""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from storygit.domain.ids import ProposalId


class Authorship(StrEnum):
    """Origin of a span of prose."""

    human = "human"
    ai = "ai"
    ai_edited_by_human = "ai_edited_by_human"


class ProvenanceSpan(BaseModel):
    """A half-open range of sentence indices and who is responsible for it.

    Attributes:
        start: First sentence index covered (inclusive).
        end: One past the last sentence index covered (exclusive).
        source: Who wrote it.
        proposal_id: The proposal the span came from, when it came from one.
    """

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    source: Authorship
    proposal_id: ProposalId | None = None

    @model_validator(mode="after")
    def _check_range(self) -> ProvenanceSpan:
        if self.end <= self.start:
            raise ValueError(f"empty or inverted provenance span [{self.start}, {self.end})")
        return self

    @property
    def length(self) -> int:
        """Number of sentences the span covers."""
        return self.end - self.start


def authorship_ratio(spans: tuple[ProvenanceSpan, ...]) -> dict[Authorship, float]:
    """Fraction of sentences attributable to each source.

    Overlapping spans are resolved last-writer-wins, so an edit recorded on top of
    an AI span correctly reads as ``ai_edited_by_human``.

    Args:
        spans: Provenance spans of one prose node, in the order they were recorded.

    Returns:
        A mapping from source to its share of sentences. Empty when there are no
        spans.
    """
    if not spans:
        return {}
    owner: dict[int, Authorship] = {}
    for span in spans:
        for index in range(span.start, span.end):
            owner[index] = span.source
    counts = Counter(owner.values())
    total = sum(counts.values())
    return {source: counts[source] / total for source in sorted(counts, key=lambda s: s.value)}


def merge_ratios(ratios: list[tuple[dict[Authorship, float], int]]) -> dict[Authorship, float]:
    """Combine per-node authorship ratios into a subtree ratio.

    Args:
        ratios: ``(ratio, sentence_count)`` pairs, one per prose node.

    Returns:
        The sentence-weighted mean of the ratios.
    """
    total = sum(count for _, count in ratios)
    if total == 0:
        return {}
    merged: dict[Authorship, float] = {}
    for ratio, count in ratios:
        for source, share in ratio.items():
            merged[source] = merged.get(source, 0.0) + share * count
    return {source: value / total for source, value in sorted(merged.items(), key=lambda kv: kv[0])}
