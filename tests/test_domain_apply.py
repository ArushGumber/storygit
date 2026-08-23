"""Apply-level validation: preconditions, purity, and structural rules."""

from __future__ import annotations

import pytest
from tests.conftest import Fixture

from storygit.domain.apply import apply
from storygit.domain.diff import (
    AddFact,
    AddNode,
    Diff,
    DiffAuthor,
    MoveNode,
    RemoveNode,
    SetLock,
    SetNodeStatus,
    UpdateNode,
)
from storygit.domain.errors import (
    CycleError,
    DuplicateIdError,
    InvalidStructureError,
    LockedNodeError,
    UnknownEntityError,
    UnknownNodeError,
)
from storygit.domain.ids import EntityId, FactId, IdGenerator, NodeId
from storygit.domain.nodes import Beat, Episode, NodeStatus, Scene
from storygit.domain.world import Fact, Predicate


def test_unknown_node_raises_and_leaves_state_untouched(fixture: Fixture) -> None:
    state = fixture.repo.state()
    before = dict(state.nodes)
    with pytest.raises(UnknownNodeError):
        apply(state, Diff(ops=(UpdateNode(node_id=NodeId("nope"), fields={"title": "x"}),)))
    assert state.nodes == before


def test_unknown_entity_in_fact_raises(fixture: Fixture, ids: IdGenerator) -> None:
    state = fixture.repo.state()
    bad = Fact(
        id=ids.fact(),
        subject=EntityId("ghost"),
        predicate=Predicate.trait,
        object_text="brave",
        valid_from_beat=fixture.beat_a,
        established_by_beat=fixture.beat_a,
    )
    with pytest.raises(UnknownEntityError):
        apply(state, Diff(ops=(AddFact(fact=bad),)))


def test_duplicate_node_id_raises(fixture: Fixture) -> None:
    state = fixture.repo.state()
    clash = Scene(id=fixture.scene, parent_id=fixture.episode, title="clash")
    with pytest.raises(DuplicateIdError):
        apply(state, Diff(ops=(AddNode(node=clash),)))


def test_tree_shape_is_enforced(fixture: Fixture, ids: IdGenerator) -> None:
    state = fixture.repo.state()
    illegal = Episode(id=ids.node(), parent_id=fixture.scene, title="episode under a scene")
    with pytest.raises(InvalidStructureError):
        apply(state, Diff(ops=(AddNode(node=illegal),)))


def test_move_into_own_subtree_is_a_cycle(fixture: Fixture) -> None:
    state = fixture.repo.state()
    with pytest.raises(CycleError):
        apply(
            state,
            Diff(
                ops=(
                    MoveNode(node_id=fixture.episode, new_parent_id=fixture.beat_a, new_position=0),
                )
            ),
        )


def test_locked_nodes_reject_edits_and_removal(fixture: Fixture) -> None:
    state = apply(fixture.repo.state(), Diff(ops=(SetLock(node_id=fixture.beat_b),)))
    assert state.nodes[fixture.beat_b].status is NodeStatus.locked
    assert fixture.beat_b in state.ledger.locks
    with pytest.raises(LockedNodeError):
        apply(state, Diff(ops=(UpdateNode(node_id=fixture.beat_b, fields={"title": "no"}),)))
    with pytest.raises(LockedNodeError):
        apply(state, Diff(ops=(RemoveNode(node_id=fixture.beat_b),)))
    with pytest.raises(LockedNodeError):
        apply(
            state,
            Diff(ops=(SetNodeStatus(node_id=fixture.beat_b, status=NodeStatus.stale),)),
        )


def test_apply_is_pure_and_repeatable(fixture: Fixture, ids: IdGenerator) -> None:
    state = fixture.repo.state()
    diff = Diff(
        ops=(
            AddNode(
                node=Beat(id=NodeId("n_fixed"), parent_id=fixture.scene, title="E", position=4)
            ),
        ),
        author=DiffAuthor.ai,
    )
    first = apply(state, diff)
    second = apply(state, diff)
    assert first.nodes == second.nodes
    assert len(state.nodes) == len(first.nodes) - 1


def test_removing_a_beat_cascades_its_facts(fixture: Fixture) -> None:
    state = fixture.repo.state()
    assert fixture.fact_f in state.facts
    after = apply(state, Diff(ops=(RemoveNode(node_id=fixture.beat_a),)))
    assert fixture.fact_f not in after.facts
    assert fixture.fact_f not in after.nodes[fixture.beat_b].consumes  # type: ignore[union-attr]


def test_add_fact_registers_the_producing_beat(fixture: Fixture) -> None:
    state = fixture.repo.state()
    beat_a = state.nodes[fixture.beat_a]
    assert isinstance(beat_a, Beat)
    assert fixture.fact_f in beat_a.produces
    assert state.producer_of[fixture.fact_f] == fixture.beat_a
    assert state.consumers_of[fixture.fact_f] == (fixture.beat_b,)


def test_unknown_field_update_is_rejected(fixture: Fixture) -> None:
    state = fixture.repo.state()
    with pytest.raises(InvalidStructureError):
        apply(state, Diff(ops=(UpdateNode(node_id=fixture.beat_a, fields={"nonsense": 1}),)))


def test_fact_needs_exactly_one_object(fixture: Fixture, ids: IdGenerator) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Fact(
            id=FactId("f_x"),
            subject=fixture.kael,
            predicate=Predicate.location,
            object_entity=fixture.ashfall,
            object_text="also text",
            valid_from_beat=fixture.beat_a,
            established_by_beat=fixture.beat_a,
        )


def test_a_locked_node_and_the_lock_set_can_never_disagree(fixture: Fixture) -> None:
    """`node.status is locked` and `node_id in ledger.locks` are one fact, stored twice.

    Only SetLock and ClearLock write either, and every other status change on a locked node
    is refused -- otherwise a dismiss-stale could set a locked node to `accepted` while it
    stayed in the lock set, and the interface would show a lock the engine no longer
    honoured.
    """
    from storygit.domain.diff import ClearLock

    state = apply(fixture.repo.state(), Diff(ops=(SetLock(node_id=fixture.beat_b),)))
    assert state.nodes[fixture.beat_b].status is NodeStatus.locked
    assert fixture.beat_b in state.ledger.locks

    for status in (NodeStatus.accepted, NodeStatus.draft, NodeStatus.stale):
        with pytest.raises(LockedNodeError):
            apply(state, Diff(ops=(SetNodeStatus(node_id=fixture.beat_b, status=status),)))

    cleared = apply(state, Diff(ops=(ClearLock(node_id=fixture.beat_b),)))
    assert cleared.nodes[fixture.beat_b].status is NodeStatus.accepted
    assert fixture.beat_b not in cleared.ledger.locks
