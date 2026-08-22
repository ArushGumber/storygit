"""Propagation: what an edit marks, what it must never mark, and why."""

from __future__ import annotations

from tests.conftest import Fixture

from storygit.domain.apply import apply
from storygit.domain.diff import (
    AddNode,
    Diff,
    DiffAuthor,
    RemoveFact,
    SetLock,
    SetProse,
    UpdateFact,
)
from storygit.domain.ids import NodeId
from storygit.domain.nodes import NodeStatus, Prose
from storygit.domain.provenance import Authorship, ProvenanceSpan
from storygit.graph.dependency import dependents_of_facts, hard_constraints
from storygit.graph.propagation import (
    MarkKind,
    apply_marks,
    marks_to_diff,
    preview,
    propagate_change,
)


def _change_fact_f(fixture: Fixture) -> Diff:
    """Move Kael somewhere else — the canonical upstream edit."""
    return Diff(
        ops=(
            UpdateFact(
                fact_id=fixture.fact_f, fields={"object_entity": None, "object_text": "Kell"}
            ),
        ),
        author=DiffAuthor.human,
        intent="Kael starts in Kell instead",
    )


def test_change_propagates_transitively_with_citations(fixture: Fixture) -> None:
    state = fixture.repo.state()
    after = apply(state, _change_fact_f(fixture))
    marks = propagate_change(state, after)

    marked = {m.node_id: m for m in marks}
    assert set(marked) == {fixture.beat_b, fixture.beat_c}
    assert fixture.beat_d not in marked, "an unrelated beat must never be marked"
    assert marked[fixture.beat_b].kind is MarkKind.stale
    assert marked[fixture.beat_b].origin_fact == fixture.fact_f
    assert marked[fixture.beat_b].origin_beat == fixture.beat_a
    assert "Beat A" in marked[fixture.beat_b].reason
    # C is reached through the fact B produced, and cites that fact.
    assert marked[fixture.beat_c].origin_beat == fixture.beat_b


def test_locked_nodes_are_never_marked_and_stop_traversal(fixture: Fixture) -> None:
    state = apply(fixture.repo.state(), Diff(ops=(SetLock(node_id=fixture.beat_b),)))
    after = apply(state, _change_fact_f(fixture))
    marks = propagate_change(state, after)
    assert marks == [], "a locked beat blocks the walk; nothing downstream of it changes"


def test_human_prose_is_flagged_for_review_not_staled(fixture: Fixture) -> None:
    state = fixture.repo.state()
    prose_id = NodeId("n_prose_b")
    state = apply(
        state,
        Diff(
            ops=(
                AddNode(node=Prose(id=prose_id, parent_id=fixture.beat_b, title="prose")),
                SetProse(
                    node_id=prose_id,
                    text="Kael ran. The market closed behind him.",
                    spans=(ProvenanceSpan(start=0, end=2, source=Authorship.human),),
                ),
            ),
            author=DiffAuthor.human,
        ),
    )
    after = apply(state, _change_fact_f(fixture))
    marks = {m.node_id: m for m in propagate_change(state, after)}

    assert marks[fixture.beat_b].kind is MarkKind.review
    assert "Your prose" in marks[fixture.beat_b].reason
    final = apply_marks(after, list(marks.values()))
    assert final.nodes[fixture.beat_b].status is not NodeStatus.stale
    assert final.nodes[fixture.beat_b].stale_reason is not None
    assert final.nodes[fixture.beat_c].status is NodeStatus.stale


def test_striking_a_fact_propagates(fixture: Fixture) -> None:
    state = fixture.repo.state()
    after = apply(state, Diff(ops=(RemoveFact(fact_id=fixture.fact_f),)))
    marks = propagate_change(state, after)
    assert {m.node_id for m in marks} == {fixture.beat_b, fixture.beat_c}
    assert "struck" in marks[0].reason


def test_adding_a_fact_marks_nothing(fixture: Fixture) -> None:
    from storygit.domain.diff import AddFact
    from storygit.domain.world import Fact, Predicate

    state = fixture.repo.state()
    after = apply(
        state,
        Diff(
            ops=(
                AddFact(
                    fact=Fact(
                        id=fixture.ids.fact(),
                        subject=fixture.kael,
                        predicate=Predicate.trait,
                        object_text="stubborn",
                        valid_from_beat=fixture.beat_d,
                        established_by_beat=fixture.beat_d,
                    )
                ),
            )
        ),
    )
    assert propagate_change(state, after) == []


def test_preview_matches_what_commit_would_do(fixture: Fixture) -> None:
    state = fixture.repo.state()
    diff = _change_fact_f(fixture)
    predicted = preview(state, diff)
    assert {m.node_id for m in predicted} == {fixture.beat_b, fixture.beat_c}
    # Previewing must not have changed anything.
    assert fixture.repo.state().facts[fixture.fact_f].object_entity == fixture.ashfall


def test_marks_to_diff_is_idempotent(fixture: Fixture) -> None:
    state = fixture.repo.state()
    after = apply(state, _change_fact_f(fixture))
    marks = propagate_change(state, after)
    once = apply(after, marks_to_diff(after, marks))
    twice_diff = marks_to_diff(once, propagate_change(after, once))
    assert len(twice_diff) == 0


def test_dependents_closure_and_hard_constraints(fixture: Fixture) -> None:
    state = fixture.repo.state()
    closure = dependents_of_facts(state, {fixture.fact_f})
    assert set(closure) == {fixture.beat_b, fixture.beat_c}

    locked = apply(state, Diff(ops=(SetLock(node_id=fixture.beat_a),)))
    lines = hard_constraints(locked)
    assert any("Kael is at Ashfall" in line for line in lines)
    assert any(line.startswith("Locked —") for line in lines)


def test_soft_edges_are_marked_distinctly(fixture: Fixture) -> None:
    class NeighbourEdges:
        """Stand-in for chunk 3's embedding-similarity edge provider."""

        def extra_dependents(self, state: object, node_id: NodeId) -> tuple[NodeId, ...]:
            return (fixture.beat_d,)

    state = fixture.repo.state()
    after = apply(state, _change_fact_f(fixture))
    marks = {m.node_id: m for m in propagate_change(state, after, edge_provider=NeighbourEdges())}
    assert marks[fixture.beat_d].kind is MarkKind.maybe_affected
    assert marks[fixture.beat_b].kind is MarkKind.stale
    final = apply_marks(after, list(marks.values()))
    assert final.nodes[fixture.beat_d].status is not NodeStatus.stale
