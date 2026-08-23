"""Recording replayable sessions from real snapshots.

The session *shape* and the replay function live in ``storygit.gallery`` because the API
serves them, and a deliverable package must not import its own test harness. This module is
the authoring half: it wraps an engine and writes down what a writer would have seen.


A recorded session is the honest form of a demo. Every step names the snapshot it happened
at, so replay reads the story out of the store rather than out of a script — the Gallery
tab cannot show something the system did not actually do, and replay makes **zero provider
calls**, which is asserted in the tests.

Each step captures what a writer would have seen: the candidates with their labels, flags,
and scores; what they did; and what changed as a result — the bible diff and the stale
marks. That is enough to reconstruct all three panes of the interface at any point.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from storygit.domain.ids import NodeId, ProposalId, SnapshotId
from storygit.engine import AcceptResult, Engine
from storygit.gallery import Replay, Session, ShownCandidate, Step, replay
from storygit.selection.select import Candidate, candidate_text

__all__ = ["Recorder", "Replay", "Session", "ShownCandidate", "Step", "replay", "shown_from"]


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
