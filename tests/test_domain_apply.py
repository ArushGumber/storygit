"""Apply-level validation: preconditions, purity, and structural rules."""

from __future__ import annotations

import pytest
from tests.conftest import Fixture

from storygit.domain.apply import apply
from storygit.domain.diff import (
    AddEntity,
    AddFact,
    AddKnows,
    AddNode,
    Diff,
    DiffAuthor,
    InvalidateFact,
    MergeEntities,
    MoveNode,
    OpenThread,
    RemoveEntity,
    RemoveFact,
    RemoveNode,
    SetLock,
    SetNodeStatus,
    SetProse,
    TouchThread,
    UpdateNode,
    UpdateThread,
)
from storygit.domain.errors import (
    CycleError,
    DuplicateIdError,
    InvalidStructureError,
    LockedNodeError,
    UnknownEntityError,
    UnknownFactError,
    UnknownNodeError,
    UnknownThreadError,
)
from storygit.domain.ids import EntityId, FactId, IdGenerator, NodeId, ThreadId
from storygit.domain.nodes import Beat, Episode, NodeStatus, Prose, Scene
from storygit.domain.threads import Thread
from storygit.domain.world import Entity, EntityKind, Fact, Knows, Predicate


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


def _guards(fixture: Fixture, ids: IdGenerator) -> list[tuple[str, Diff, type[Exception]]]:
    """Every rejection `apply` claims to make, as (name, offending diff, error)."""
    beat_a, beat_d, scene = fixture.beat_a, fixture.beat_d, fixture.scene
    prose_id, fact_id, thread_id = ids.node(), ids.fact(), ids.thread()
    unknown_node, unknown_entity = NodeId("n_nope"), EntityId("e_nope")
    unknown_fact, unknown_thread = FactId("f_nope"), ThreadId("t_nope")
    a_thread = Thread(
        id=thread_id, description="a thread", opened_at_beat=beat_a, last_touched_beat=beat_a
    )
    return [
        (
            "prose text on a node that is not prose",
            Diff(ops=(SetProse(node_id=beat_a, text="x"),)),
            InvalidStructureError,
        ),
        (
            "a second entity with an id already in use",
            Diff(
                ops=(
                    AddEntity(entity=Entity(id=fixture.kael, kind=EntityKind.character, name="x")),
                )
            ),
            DuplicateIdError,
        ),
        (
            "an entity learning a fact that does not exist",
            Diff(
                ops=(
                    AddKnows(
                        knows=Knows(character=fixture.kael, fact=unknown_fact, since_beat=beat_a)
                    ),
                )
            ),
            UnknownFactError,
        ),
        (
            "a character who does not exist learning a fact",
            Diff(
                ops=(
                    AddKnows(
                        knows=Knows(
                            character=unknown_entity, fact=fixture.fact_f, since_beat=beat_a
                        )
                    ),
                )
            ),
            UnknownEntityError,
        ),
        (
            "a thread opened at a beat that does not exist",
            Diff(
                ops=(
                    OpenThread(thread=a_thread.model_copy(update={"opened_at_beat": unknown_node})),
                )
            ),
            UnknownNodeError,
        ),
        (
            "advancing a thread nobody opened",
            Diff(ops=(TouchThread(thread_id=unknown_thread, beat_id=beat_a),)),
            UnknownThreadError,
        ),
        (
            "editing fields of a thread nobody opened",
            Diff(ops=(UpdateThread(thread_id=unknown_thread, fields={"description": "x"}),)),
            UnknownThreadError,
        ),
        (
            "invalidating a fact that does not exist",
            Diff(ops=(InvalidateFact(fact_id=unknown_fact, valid_until_beat=beat_d),)),
            UnknownFactError,
        ),
        (
            "a root that is not a story node",
            Diff(ops=(AddNode(node=Episode(id=ids.node(), parent_id=None, title="orphan")),)),
            InvalidStructureError,
        ),
        (
            "a second block of prose under one beat",
            Diff(
                ops=(
                    AddNode(node=Prose(id=prose_id, parent_id=beat_a, text="first")),
                    AddNode(node=Prose(id=ids.node(), parent_id=beat_a, text="second")),
                )
            ),
            InvalidStructureError,
        ),
        (
            "changing a node's immutable identity",
            Diff(ops=(UpdateNode(node_id=beat_a, fields={"node_type": "scene"}),)),
            InvalidStructureError,
        ),
        (
            "removing a node with children without asking for the subtree",
            Diff(ops=(RemoveNode(node_id=scene, recursive=False),)),
            InvalidStructureError,
        ),
        (
            "moving a locked node",
            Diff(
                ops=(
                    SetLock(node_id=beat_a),
                    MoveNode(node_id=beat_a, new_parent_id=scene, new_position=0),
                )
            ),
            LockedNodeError,
        ),
        (
            "moving a beat somewhere a beat cannot live",
            Diff(ops=(MoveNode(node_id=beat_a, new_parent_id=fixture.story, new_position=0),)),
            InvalidStructureError,
        ),
        (
            "a second fact with an id already in use",
            Diff(
                ops=(
                    AddFact(
                        fact=Fact(
                            id=fixture.fact_f,
                            subject=fixture.kael,
                            predicate=Predicate.goal,
                            object_text="x",
                            valid_from_beat=beat_a,
                            established_by_beat=beat_a,
                        )
                    ),
                )
            ),
            DuplicateIdError,
        ),
        (
            "a fact pointing at an object entity that does not exist",
            Diff(
                ops=(
                    AddFact(
                        fact=Fact(
                            id=fact_id,
                            subject=fixture.kael,
                            predicate=Predicate.location,
                            object_entity=unknown_entity,
                            valid_from_beat=beat_a,
                            established_by_beat=beat_a,
                        )
                    ),
                )
            ),
            UnknownEntityError,
        ),
        (
            "removing an entity that was never in the bible",
            Diff(ops=(RemoveEntity(entity_id=unknown_entity),)),
            UnknownEntityError,
        ),
        (
            "merging away an entity that does not exist",
            Diff(ops=(MergeEntities(source_id=unknown_entity, target_id=fixture.kael),)),
            UnknownEntityError,
        ),
        (
            "merging into an entity that does not exist",
            Diff(ops=(MergeEntities(source_id=fixture.kael, target_id=unknown_entity),)),
            UnknownEntityError,
        ),
        (
            "merging an entity into itself",
            Diff(ops=(MergeEntities(source_id=fixture.kael, target_id=fixture.kael),)),
            InvalidStructureError,
        ),
        (
            "opening a thread whose id is already open",
            Diff(ops=(OpenThread(thread=a_thread), OpenThread(thread=a_thread))),
            DuplicateIdError,
        ),
        (
            "removing a fact that does not exist",
            Diff(ops=(RemoveFact(fact_id=unknown_fact),)),
            UnknownFactError,
        ),
    ]


def test_every_guard_actually_fires(fixture: Fixture, ids: IdGenerator) -> None:
    """`apply` is the only way state changes, so its guards *are* the invariants.

    An unexercised `raise` is an invariant that is claimed and not proven. One of
    these guards was silently wrong when it was first written — a locked node could
    still have its status changed — and the symptom was a desync that would only
    show up much later, so the whole table is walked rather than a sample.
    """
    state = fixture.repo.state()
    for name, diff, error in _guards(fixture, ids):
        with pytest.raises(error):
            apply(state, diff)
        assert fixture.repo.state() == state, f"state changed while rejecting: {name}"


def test_two_siblings_claiming_one_slot_are_ordered_by_acceptance(fixture: Fixture) -> None:
    """A proposal carries the position the tree had when it was generated.

    So a writer who asks for the next beat before accepting the last one gets two
    candidates both claiming position 0, and their order then falls back to their ids. In
    a real session that put a payoff before its setup — in the tree, in the audit, and in
    the slice the next generation reads. A taken slot means append.
    """
    state = fixture.repo.state()
    first, second = NodeId("n_first"), NodeId("n_second")
    state = apply(
        state,
        Diff(
            ops=(
                AddNode(node=Beat(id=first, parent_id=fixture.scene, position=0, title="setup")),
                AddNode(node=Beat(id=second, parent_id=fixture.scene, position=0, title="payoff")),
            ),
            author=DiffAuthor.human,
        ),
    )
    order = [state.nodes[n].title for n in state.children[fixture.scene]]
    assert order.index("setup") < order.index("payoff"), order
    assert state.nodes[second].position > state.nodes[first].position
