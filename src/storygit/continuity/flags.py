"""What a continuity flag is, and what it must always carry.

Two properties are non-negotiable, and both come from the same principle: a flag is a
message to a novelist, not a log line.

**Every flag cites its establishing beat.** "Kael cannot be in Ashfall here" is useless;
"Kael was established in Kell in *The Warden's Offer* (episode 2, beat 4)" is actionable,
because the writer can go and look. The citation is a required field, not an optional
extra, and the interface links to it.

**Nothing is ever auto-fixed.** A flag is information. The writer decides whether the
contradiction is an error or the point — characters lie, narrators are unreliable, and a
system that "corrects" a deliberate inconsistency is worse than one that says nothing.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from storygit.domain.ids import EntityId, FactId, NodeId


class Severity(StrEnum):
    """How confident the system is that this is a real problem.

    ``hard`` flags come from deterministic graph checks and are almost always genuine
    contradictions. ``soft`` flags come from an NLI model or a language-model judge and
    are opinions — they are shown differently and never block anything.
    """

    hard = "hard"
    soft = "soft"


class FlagKind(StrEnum):
    """What sort of problem was found."""

    contradiction = "contradiction"
    epistemic = "epistemic"
    dead_actor = "dead_actor"
    possession = "possession"
    tone = "tone"
    motivation = "motivation"
    dropped_thread = "dropped_thread"
    duplicate_entity = "duplicate_entity"
    hard_constraint = "hard_constraint"


class Flag(BaseModel):
    """One continuity problem, with everything the writer needs to judge it.

    Attributes:
        kind: What sort of problem.
        severity: Hard (deterministic) or soft (model opinion).
        layer: Which checker layer produced it — 1 deterministic, 2 NLI, 3 judge.
        message: The writer-facing sentence.
        node_id: Where the problem shows up.
        fact_ids: The facts involved.
        established_by: The beat that established the fact being contradicted. Required
            for layers 1 and 2; the interface links to it.
        entity_ids: Entities involved, for highlighting the world-state subgraph.
        subgraph: Nodes and facts the interface should show around the flag.
        score: For soft flags, the model's confidence in ``[0, 1]``.
    """

    model_config = ConfigDict(frozen=True)

    kind: FlagKind
    severity: Severity
    layer: int
    message: str
    node_id: NodeId | None = None
    fact_ids: tuple[FactId, ...] = ()
    established_by: NodeId | None = None
    entity_ids: tuple[EntityId, ...] = ()
    subgraph: tuple[str, ...] = ()
    score: float | None = None

    @property
    def is_hard(self) -> bool:
        """Whether this is a deterministic contradiction rather than an opinion."""
        return self.severity is Severity.hard


def sort_flags(flags: list[Flag]) -> list[Flag]:
    """Order flags the way the interface shows them: hard first, then by layer.

    Args:
        flags: Flags in any order.

    Returns:
        Hard flags before soft ones, earlier layers before later, then by message so the
        order is stable across runs.
    """
    return sorted(
        flags,
        key=lambda f: (0 if f.is_hard else 1, f.layer, f.kind.value, f.message),
    )
