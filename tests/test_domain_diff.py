"""Delta summaries: the sentences a writer actually reads under each candidate."""

from __future__ import annotations

from tests.conftest import Fixture

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
from storygit.domain.nodes import Beat
from storygit.domain.threads import Thread
from storygit.domain.world import Entity, EntityKind, Fact, Predicate
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
