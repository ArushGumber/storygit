"""The writer ledger: everything the writer has told the system about their taste.

This is the object that keeps the human in charge. Locks, hard constraints, style
notes, and writer-defined criteria all end up in prompts or in the ranking, and all
of them are visible and deletable — including the rules the system mined from the
writer's own edits, which must never influence generation invisibly.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from storygit.domain.ids import NodeId, ProposalId


class StyleNoteSource(StrEnum):
    """Where a style note came from."""

    writer = "writer"
    mined = "mined"


class StyleNote(BaseModel):
    """A soft constraint on how prose should read.

    Attributes:
        text: The rule, in plain language ("shorter sentences").
        source: Written by the writer, or mined from their edits.
        count: How many times the rule was observed; mined rules only enter
            prompts once this reaches the mining threshold.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    source: StyleNoteSource = StyleNoteSource.writer
    count: int = Field(default=1, ge=1)


class Criterion(BaseModel):
    """A writer-defined scoring axis, in the writer's own words.

    The judge scores every candidate on these alongside the fixed narratology
    criteria, and the preference head gets them as features. This is how the writer
    owns the objective rather than being an arbiter of the system's objective.

    Attributes:
        name: Short label shown in the UI.
        description: Plain-language description handed to the judge verbatim.
        weight: Writer-set importance in ``[0, 1]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class RejectedDirection(BaseModel):
    """A direction the writer has explicitly turned down.

    Attributes:
        text: What was rejected, in one line.
        proposal_id: The proposal it came from, when known.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    proposal_id: ProposalId | None = None


class WriterLedger(BaseModel):
    """The writer's standing instructions to the system.

    Attributes:
        locks: Nodes the writer has frozen. Propagation never marks them and their
            facts are exported as hard constraints into every prompt.
        hard_constraints: Free-text rules that must not be violated.
        style_notes: Soft prose rules, writer-written or mined from edits.
        criteria: Writer-defined scoring axes.
        rejected_directions: Directions already turned down, fed to generation as
            negative exemplars.
        dial: Coherent (0.0) to surprising (1.0).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    locks: frozenset[NodeId] = frozenset()
    hard_constraints: tuple[str, ...] = ()
    style_notes: tuple[StyleNote, ...] = ()
    criteria: tuple[Criterion, ...] = ()
    rejected_directions: tuple[RejectedDirection, ...] = ()
    dial: float = Field(default=0.35, ge=0.0, le=1.0)

    def active_style_notes(self, mined_threshold: int = 2) -> tuple[StyleNote, ...]:
        """Style notes that are allowed into prompts.

        Writer-written notes always count. Mined rules have to have been observed
        ``mined_threshold`` times, so one idiosyncratic edit does not become a
        standing instruction.

        Args:
            mined_threshold: Observation count a mined rule needs.

        Returns:
            The notes to render into the prompt.
        """
        return tuple(
            note
            for note in self.style_notes
            if note.source is StyleNoteSource.writer or note.count >= mined_threshold
        )
