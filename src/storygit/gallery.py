"""Recorded sessions: the shape they take, and how to replay one.

A recorded session is a list of steps, each naming the snapshot it happened at. Replay reads
the story back out of those snapshots rather than out of a serialized copy, so a Gallery
cannot show something the system did not do — and replaying makes **zero provider calls**,
which is asserted in the tests.

The *reader* lives here rather than in the evaluation harness because the API serves it, and
a deliverable package that imports its own test harness breaks the moment it runs from
anywhere but the repository root. The *recorder* stays in ``eval/`` — writing a session is
authoring, and only the harness authors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from storygit.domain.ids import NodeId, ProposalId, SnapshotId


class ShownCandidate(BaseModel):
    """One candidate as the writer saw it.

    Attributes:
        proposal_id: Its id.
        axis_label: The named direction it was generated along.
        delta_summary: What it would change, in English.
        rationale: Why the model proposed it.
        text: The candidate's prose or plan text.
        flags: Flags on it, flattened for the interface.
        base_quality: Judge or head score.
        surprise: Distance from the greedy continuation.
        effective_quality: What it was actually ranked on.
        selected: Whether it made the shortlist.
    """

    model_config = ConfigDict(frozen=True)

    proposal_id: ProposalId
    axis_label: str = ""
    delta_summary: tuple[str, ...] = ()
    rationale: str = ""
    text: str = ""
    flags: tuple[dict[str, Any], ...] = ()
    base_quality: float = 0.0
    surprise: float = 0.0
    effective_quality: float = 0.0
    selected: bool = False


class Step(BaseModel):
    """One thing that happened, and everything it changed.

    Attributes:
        index: Position in the session.
        title: What this step demonstrates.
        note: One sentence explaining why it is in the session.
        snapshot_id: The snapshot after this step.
        node_id: The node it concerned.
        level: Plan level.
        intent: The writer's instruction, if any.
        shown: The candidates that were on screen.
        action: What the writer did.
        chosen: The proposal acted on.
        bible_diff: Facts added, ended, and struck.
        marks: Stale and review marks the action produced.
        flags: Continuity flags on the resulting state.
    """

    model_config = ConfigDict(frozen=True)

    index: int
    title: str = ""
    note: str = ""
    snapshot_id: SnapshotId | None = None
    node_id: NodeId | None = None
    level: str = ""
    intent: str = ""
    shown: tuple[ShownCandidate, ...] = ()
    action: str = ""
    chosen: ProposalId | None = None
    bible_diff: tuple[str, ...] = ()
    marks: tuple[dict[str, Any], ...] = ()
    flags: tuple[dict[str, Any], ...] = ()


class Session(BaseModel):
    """A recorded, replayable session.

    Attributes:
        name: Filename-safe identifier.
        title: What it demonstrates.
        summary: One paragraph for the Gallery's index.
        branch: Branch it was recorded on.
        steps: The steps, in order.
        db_path: The story database, so replay can load the snapshots.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    title: str = ""
    summary: str = ""
    branch: str = "main"
    steps: tuple[Step, ...] = ()
    db_path: str = ""

    def save(self, directory: Path | str) -> Path:
        """Write the session as JSON."""
        target = Path(directory) / f"{self.name}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2))
        return target

    @staticmethod
    def load(path: Path | str) -> Session:
        """Read a session back."""
        return Session.model_validate(json.loads(Path(path).read_text()))


class Replay(BaseModel):
    """One replayed step, resolved against the story database.

    Attributes:
        step: The recorded step.
        tree: The plan tree as it stood.
        facts: Facts true at that point, as sentences.
    """

    model_config = ConfigDict(frozen=True)

    step: Step
    tree: tuple[dict[str, Any], ...] = ()
    facts: tuple[str, ...] = Field(default_factory=tuple)


def replay(session: Session, repo: Any) -> list[Replay]:
    """Reconstruct each step's state from the snapshots it names.

    Makes no provider calls — everything comes out of the store. That is what makes the
    Gallery a record rather than a recording.

    Args:
        session: The recorded session.
        repo: The repository holding the snapshots.

    Returns:
        One entry per step, with the plan tree and world state as they stood.
    """
    out: list[Replay] = []
    for step in session.steps:
        if step.snapshot_id is None:
            out.append(Replay(step=step))
            continue
        state = repo.state_at(step.snapshot_id)
        names = state.entity_names()
        tree = tuple(
            {
                "id": node.id,
                "type": node.node_type.value,
                "title": node.title,
                "status": node.status.value,
                "stale_reason": node.stale_reason,
            }
            for node in sorted(state.nodes.values(), key=lambda n: state.seq.get(n.id, 0))
        )
        facts = tuple(
            fact.sentence(names) for fact in sorted(state.facts.values(), key=lambda f: f.id)
        )
        out.append(Replay(step=step, tree=tree, facts=facts))
    return out
