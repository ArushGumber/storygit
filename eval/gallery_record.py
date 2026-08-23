"""Recording replayable sessions from real snapshots.

A recorded session is the honest form of a demo. Every step names the snapshot it happened
at, so replay reads the story out of the store rather than out of a script — the Gallery
tab cannot show something the system did not actually do, and replay makes **zero provider
calls**, which is asserted in the tests.

Each step captures what a writer would have seen: the candidates with their labels, flags,
and scores; what they did; and what changed as a result — the bible diff and the stale
marks. That is enough to reconstruct all three panes of the interface at any point.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from storygit.domain.ids import NodeId, ProposalId, SnapshotId
from storygit.engine import AcceptResult, Engine
from storygit.selection.select import Candidate, candidate_text


class ShownCandidate(BaseModel):
    """One candidate as the writer saw it.

    Attributes:
        proposal_id: Its id.
        axis_label: The named direction it was generated along.
        delta_summary: What it would change, in English.
        rationale: Why the model proposed it.
        text: The candidate's prose or plan text.
        flags: Flags on it, as ``(severity, layer, message, established_by)``.
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
        title: What this step demonstrates, for the Gallery's step list.
        note: One sentence explaining why this step is in the session.
        snapshot_id: The snapshot after this step.
        node_id: The node it concerned.
        level: Plan level.
        intent: The writer's instruction, if any.
        shown: The candidates that were on screen.
        action: ``accept``, ``edit``, ``reject``, ``lock``, ``strike``, or ``branch``.
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


def _flag_dict(flag: Any) -> dict[str, Any]:
    return {
        "kind": flag.kind.value,
        "severity": flag.severity.value,
        "layer": flag.layer,
        "message": flag.message,
        "established_by": flag.established_by,
        "node_id": flag.node_id,
    }


def _mark_dict(mark: Any) -> dict[str, Any]:
    return {
        "node_id": mark.node_id,
        "kind": mark.kind.value,
        "reason": mark.reason,
        "origin_beat": mark.origin_beat,
    }


def shown_from(candidates: Sequence[Candidate]) -> tuple[ShownCandidate, ...]:
    """Convert engine candidates into the recorded form."""
    return tuple(
        ShownCandidate(
            proposal_id=c.proposal.id,
            axis_label=c.axis_label,
            delta_summary=c.proposal.delta_summary,
            rationale=c.proposal.rationale,
            text=candidate_text(c.proposal),
            flags=tuple(_flag_dict(f) for f in c.flags),
            base_quality=round(c.base_quality, 4),
            surprise=round(c.surprise, 4),
            effective_quality=round(c.effective_quality, 4),
            selected=c.selected,
        )
        for c in candidates
    )


class Recorder:
    """Wraps an engine and records what the writer would have seen.

    Attributes:
        engine: The engine being recorded.
        steps: Steps recorded so far.
    """

    def __init__(self, engine: Engine, name: str, *, title: str = "", summary: str = "") -> None:
        """Create a recorder.

        Args:
            engine: The engine to record.
            name: Filename-safe session name.
            title: What the session demonstrates.
            summary: One paragraph for the Gallery index.
        """
        self.engine = engine
        self.name = name
        self.title = title or name
        self.summary = summary
        self.steps: list[Step] = []

    def record(
        self,
        *,
        title: str,
        note: str = "",
        candidates: Sequence[Candidate] = (),
        action: str = "",
        chosen: ProposalId | None = None,
        result: AcceptResult | None = None,
        node_id: NodeId | None = None,
        level: str = "",
        intent: str = "",
        marks: Sequence[Any] = (),
        flags: Sequence[Any] = (),
        snapshot_id: SnapshotId | None = None,
    ) -> Step:
        """Append one step.

        Args:
            title: What this step demonstrates.
            note: Why it is in the session.
            candidates: What was on screen.
            action: What the writer did.
            chosen: What they acted on.
            result: The accept result, when there was one.
            node_id: The node concerned.
            level: Plan level.
            intent: The writer's instruction.
            marks: Marks, when not carried by ``result``.
            flags: Flags, when not carried by ``result``.
            snapshot_id: The resulting snapshot, when not carried by ``result``.

        Returns:
            The recorded step.
        """
        step = Step(
            index=len(self.steps),
            title=title,
            note=note,
            snapshot_id=(result.snapshot_id if result else snapshot_id)
            or self.engine.repo.head(self.engine.branch),
            node_id=node_id,
            level=level,
            intent=intent,
            shown=shown_from(candidates),
            action=action,
            chosen=chosen,
            bible_diff=result.bible_diff.lines if result else (),
            marks=tuple(_mark_dict(m) for m in (result.marks if result else marks)),
            flags=tuple(_flag_dict(f) for f in (result.flags if result else flags)),
        )
        self.steps.append(step)
        return step

    def session(self, db_path: str = "") -> Session:
        """Assemble the recorded steps into a session."""
        return Session(
            name=self.name,
            title=self.title,
            summary=self.summary,
            branch=self.engine.branch,
            steps=tuple(self.steps),
            db_path=db_path,
        )


class Replay(BaseModel):
    """One replayed step, resolved against the story database.

    Attributes:
        step: The recorded step.
        tree: The plan tree as it stood, as ``(id, type, title, status)``.
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
