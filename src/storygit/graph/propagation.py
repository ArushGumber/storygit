"""Propagation: marking what an edit affected, and never rewriting it.

This is the direct answer to the complaint Fabula's writers made loudest — that
editing one scene either changed nothing downstream or silently rewrote the plan.
Here an accepted change walks the declared dependency edges and *marks* what it
touched. Three rules make it safe to leave running:

1. Locked nodes are never marked and are not traversed through.
2. Prose the human wrote or edited is flagged for **review**, never staled — the
   system does not get to tell an author their own sentences are out of date.
3. Nothing is ever rewritten. The writer chooses regenerate, edit, or dismiss.

The marks themselves are emitted as a normal ``Diff`` of status ops, so staleness
flows through the same snapshot machinery as every other change.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from storygit.domain.apply import apply
from storygit.domain.diff import Diff, DiffAuthor, Op, SetNodeStatus
from storygit.domain.ids import FactId, NodeId
from storygit.domain.nodes import NodeStatus
from storygit.domain.provenance import Authorship
from storygit.domain.state import StoryState
from storygit.graph.dependency import EdgeProvider, dependents_of_facts


class MarkKind(StrEnum):
    """How strongly a node is implicated by an upstream change."""

    stale = "stale"
    review = "review"
    maybe_affected = "maybe_affected"


class StaleMark(BaseModel):
    """One node implicated by a change, with the citation that explains why.

    Attributes:
        node_id: The affected node.
        kind: Stale (regenerable), review (human prose), or maybe-affected (soft
            edge from the embedding ablation).
        reason: Writer-facing explanation.
        origin_fact: The fact whose change implicated this node.
        origin_beat: The beat that established that fact.
    """

    model_config = ConfigDict(frozen=True)

    node_id: NodeId
    kind: MarkKind
    reason: str
    origin_fact: FactId | None = None
    origin_beat: NodeId | None = None


def changed_facts(before: StoryState, after: StoryState) -> set[FactId]:
    """Facts that a change removed or altered.

    Newly added facts are deliberately excluded: adding information cannot
    invalidate a beat that never referred to it.

    Args:
        before: State before the change.
        after: State after the change.

    Returns:
        Ids of facts that were removed or whose content changed.
    """
    removed = before.facts.keys() - after.facts.keys()
    altered = {
        fid
        for fid in before.facts.keys() & after.facts.keys()
        if before.facts[fid] != after.facts[fid]
    }
    return set(removed) | altered


def _human_touched(state: StoryState, beat_id: NodeId) -> bool:
    prose = state.prose_for(beat_id)
    if prose is None:
        return False
    return any(
        span.source in (Authorship.human, Authorship.ai_edited_by_human) for span in prose.spans
    )


def propagate(
    state: StoryState,
    fact_ids: set[FactId],
    *,
    edge_state: StoryState | None = None,
    edge_provider: EdgeProvider | None = None,
) -> list[StaleMark]:
    """Mark everything downstream of a set of changed facts.

    Args:
        state: The state *after* the change; marks are reported against it.
        fact_ids: Facts that changed or were removed.
        edge_state: Where to read dependency edges from. Defaults to ``state``, but
            a removal deletes the very ``consumes`` entries that make it
            interesting, so callers pass the pre-change state. Use
            :func:`propagate_change`, which handles this.
        edge_provider: Optional source of soft, non-declared edges; their results
            are marked ``maybe_affected`` and are always weaker than declared ones.

    Returns:
        Marks in reading order, at most one per node.
    """
    edges = edge_state if edge_state is not None else state
    locks = frozenset(state.ledger.locks)
    names = state.entity_names()
    implicated = dependents_of_facts(edges, fact_ids, stop_at=locks)

    marks: dict[NodeId, StaleMark] = {}
    for node_id, fact_id in implicated.items():
        if node_id in locks or node_id not in state.nodes:
            continue
        fact = state.facts.get(fact_id)
        origin_beat = state.producer_of.get(fact_id) or edges.producer_of.get(fact_id)
        origin_label = _label(state if origin_beat in state.nodes else edges, origin_beat)
        sentence = fact.sentence(names) if fact is not None else "a fact it relies on"
        if fact is None:
            detail = f"a fact established in {origin_label} was struck"
        else:
            detail = f"“{sentence}” (established in {origin_label}) changed"
        if _human_touched(state, node_id):
            marks[node_id] = StaleMark(
                node_id=node_id,
                kind=MarkKind.review,
                reason=f"Your prose here may be affected: {detail}.",
                origin_fact=fact_id,
                origin_beat=origin_beat,
            )
        else:
            marks[node_id] = StaleMark(
                node_id=node_id,
                kind=MarkKind.stale,
                reason=f"Depends on a fact that changed: {detail}.",
                origin_fact=fact_id,
                origin_beat=origin_beat,
            )

    if edge_provider is not None:
        origins = {edges.producer_of[f] for f in fact_ids if f in edges.producer_of}
        for origin in sorted(origins):
            for node_id in edge_provider.extra_dependents(state, origin):
                if node_id in locks or node_id in marks or node_id not in state.nodes:
                    continue
                marks[node_id] = StaleMark(
                    node_id=node_id,
                    kind=MarkKind.maybe_affected,
                    reason=(
                        f"Reads closely on {_label(state, origin)}, which changed. "
                        "Not a declared dependency — check if it still works."
                    ),
                    origin_beat=origin,
                )

    return sorted(marks.values(), key=lambda m: (state.seq.get(m.node_id, 0), m.node_id))


def propagate_change(
    before: StoryState,
    after: StoryState,
    *,
    edge_provider: EdgeProvider | None = None,
) -> list[StaleMark]:
    """Marks produced by moving from one state to another.

    The normal entry point: it works out which facts changed and reads dependency
    edges from the pre-change state, which is the only place a struck fact's
    consumers still exist.

    Args:
        before: State before the change.
        after: State after the change.
        edge_provider: Optional soft-edge provider.

    Returns:
        The marks.
    """
    return propagate(
        after,
        changed_facts(before, after),
        edge_state=before,
        edge_provider=edge_provider,
    )


def marks_to_diff(state: StoryState, marks: list[StaleMark]) -> Diff:
    """Turn marks into the status ops that record them.

    ``stale`` marks change status. ``review`` and ``maybe_affected`` marks leave the
    status alone and only set the reason, so a node the writer accepted never
    silently loses that standing because of a soft signal.

    Args:
        state: State the marks were computed against.
        marks: Marks to record.

    Returns:
        A ``system``-authored diff.
    """
    ops: list[Op] = []
    for mark in marks:
        node = state.nodes.get(mark.node_id)
        if node is None:
            continue
        status = NodeStatus.stale if mark.kind is MarkKind.stale else node.status
        if node.status is status and node.stale_reason == mark.reason:
            continue
        ops.append(SetNodeStatus(node_id=mark.node_id, status=status, stale_reason=mark.reason))
    return Diff(ops=tuple(ops), author=DiffAuthor.system, intent="propagate staleness")


def preview(
    state: StoryState,
    diff: Diff,
    *,
    edge_provider: EdgeProvider | None = None,
) -> list[StaleMark]:
    """What a diff *would* mark, without committing anything.

    Used for the "would mark 2 beats stale" line under every candidate, which is
    the whole point: the writer sees the blast radius before deciding.

    Args:
        state: Current state.
        diff: The candidate diff.
        edge_provider: Optional soft-edge provider.

    Returns:
        The marks the diff would produce.
    """
    return propagate_change(state, apply(state, diff), edge_provider=edge_provider)


def apply_marks(state: StoryState, marks: list[StaleMark]) -> StoryState:
    """Apply marks to a state (used by tests and the engine's accept path)."""
    return apply(state, marks_to_diff(state, marks))


def _label(state: StoryState, node_id: NodeId | None) -> str:
    if node_id is None:
        return "an earlier beat"
    node = state.nodes.get(node_id)
    if node is None:
        return "a removed beat"
    return node.title or f"{node.node_type.value} {node_id}"
