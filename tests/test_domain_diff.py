"""Delta summaries: the sentences a writer actually reads under each candidate."""

from __future__ import annotations

from typing import get_args

from tests.conftest import Fixture

from storygit.domain import diff as ops
from storygit.domain.diff import (
    AddEntity,
    AddFact,
    AddNode,
    Diff,
    DiffAuthor,
    InvalidateFact,
    OpenThread,
    SetLock,
    UpdateNode,
    delta_summary,
)
from storygit.domain.ids import EntityId, FactId, NodeId, ThreadId
from storygit.domain.ledger import Criterion, StyleNote, WriterLedger
from storygit.domain.nodes import Beat, NodeStatus
from storygit.domain.threads import Thread
from storygit.domain.world import Entity, EntityKind, Fact, Knows, Predicate
from storygit.graph.propagation import preview


def test_delta_summary_reads_like_english(fixture: Fixture) -> None:
    state = fixture.repo.state()
    new_entity = EntityId("e_mara")
    diff = Diff(
        ops=(
            AddEntity(entity=Entity(id=new_entity, kind=EntityKind.character, name="Mara")),
            AddNode(node=Beat(id=NodeId("n_new"), parent_id=fixture.scene, title="Mara arrives")),
            AddFact(
                fact=Fact(
                    id=FactId("f_new"),
                    subject=new_entity,
                    predicate=Predicate.goal,
                    object_text="to reach the harbour",
                    valid_from_beat=fixture.beat_c,
                    established_by_beat=fixture.beat_c,
                )
            ),
            UpdateNode(node_id=fixture.beat_b, fields={"what_happens": "different"}),
            InvalidateFact(fact_id=fixture.fact_f, valid_until_beat=fixture.beat_c),
            OpenThread(
                thread=Thread(
                    id=ThreadId("t_1"),
                    description="Who sent Mara?",
                    opened_at_beat=fixture.beat_c,
                    last_touched_beat=fixture.beat_c,
                )
            ),
            SetLock(node_id=fixture.beat_a),
        ),
        author=DiffAuthor.ai,
        intent="bring Mara in",
    )

    lines = delta_summary(diff, state, stale_count=len(preview(state, diff)))
    assert "introduces character Mara" in lines
    assert "adds beat “Mara arrives”" in lines
    assert "establishes: Mara wants to reach the harbour." in lines
    assert "changes what_happens of Beat B" in lines
    assert "ends Kael's location as of Beat C" in lines
    assert "opens thread: Who sent Mara?" in lines
    assert "locks Beat A" in lines
    assert lines[-1] == "would mark 2 downstream nodes stale"


def test_delta_summary_is_empty_for_an_empty_diff(fixture: Fixture) -> None:
    assert delta_summary(Diff(), fixture.repo.state()) == []


def test_diff_composition() -> None:
    a = Diff(ops=(SetLock(node_id=NodeId("n_1")),))
    b = Diff(ops=(SetLock(node_id=NodeId("n_2")),))
    assert len(a.concat(b)) == 2
    assert len(a.then(SetLock(node_id=NodeId("n_3")))) == 2
    assert len(a) == 1, "diffs are immutable"


def _one_of_every_op() -> dict[str, ops.Op]:
    """One instance of every operation in the union, with minimal valid arguments."""
    node, other = NodeId("n_1"), NodeId("n_2")
    entity, target = EntityId("e_1"), EntityId("e_2")
    fact, thread = FactId("f_1"), ThreadId("t_1")
    beat = Beat(id=node, parent_id=other, position=0, title="Beat One")
    person = Entity(id=entity, kind=EntityKind.character, name="Mara")
    a_fact = Fact(
        id=fact,
        subject=entity,
        predicate=Predicate.goal,
        object_text="reach the harbour",
        valid_from_beat=node,
        established_by_beat=node,
    )
    a_thread = Thread(
        id=thread, description="Who sent Mara?", opened_at_beat=node, last_touched_beat=node
    )
    return {
        type(op).__name__: op
        for op in [
            ops.AddNode(node=beat),
            ops.UpdateNode(node_id=node, fields={"title": "x"}),
            ops.RemoveNode(node_id=node),
            ops.MoveNode(node_id=node, new_parent_id=other, new_position=0),
            ops.SetProse(node_id=node, text="Two words."),
            ops.SetNodeStatus(node_id=node, status=NodeStatus.accepted),
            ops.SetLock(node_id=node),
            ops.ClearLock(node_id=node),
            ops.AddEntity(entity=person),
            ops.UpdateEntity(entity_id=entity, fields={"name": "Mara Vey"}),
            ops.RemoveEntity(entity_id=entity),
            ops.MergeEntities(source_id=entity, target_id=target),
            ops.AddFact(fact=a_fact),
            ops.UpdateFact(fact_id=fact, fields={"object_text": "x"}),
            ops.InvalidateFact(fact_id=fact, valid_until_beat=other),
            ops.RemoveFact(fact_id=fact),
            ops.AddKnows(knows=Knows(character=entity, fact=fact, since_beat=node)),
            ops.RemoveKnows(character=entity, fact_id=fact),
            ops.OpenThread(thread=a_thread),
            ops.TouchThread(thread_id=thread, beat_id=node),
            ops.CloseThread(thread_id=thread),
            ops.UpdateThread(thread_id=thread, fields={"description": "x"}),
            ops.RemoveThread(thread_id=thread),
            ops.SetDial(value=0.5),
            ops.AddStyleNote(note=StyleNote(text="shorter sentences")),
            ops.RemoveStyleNote(text="shorter sentences"),
            ops.AddCriterion(criterion=Criterion(name="dread", description="unease that builds")),
            ops.RemoveCriterion(name="dread"),
            ops.AddHardConstraint(text="nobody dies off-page"),
            ops.AddRejectedDirection(text="a dream sequence"),
            ops.ReplaceLedger(ledger=WriterLedger()),
        ]
    }


def test_every_operation_is_visible_in_the_summary(fixture: Fixture) -> None:
    """No operation may be silent.

    The delta summary is the whole review surface: it is what the writer reads
    before accepting. An op with no line is a change that gets committed without
    ever being shown, which is the failure this system exists to prevent. Five ops
    were silent when this test was written.
    """
    every = _one_of_every_op()
    union = {member.__name__ for member in get_args(get_args(ops.Op)[0])}
    assert set(every) == union, "an op was added to the union but not to this test"

    state = fixture.repo.state()
    for name, op in every.items():
        lines = delta_summary(Diff(ops=(op,)), state)
        assert lines, f"{name} produces no line in the summary"
        assert lines[0].strip() == lines[0] and lines[0], f"{name} produces a blank line"
