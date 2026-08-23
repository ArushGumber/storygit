"""Temporal facts, epistemic edges, threads, ledger, provenance, and slices."""

from __future__ import annotations

from tests.conftest import Fixture

from storygit.domain.apply import apply
from storygit.domain.diff import (
    AddCriterion,
    AddFact,
    AddKnows,
    AddNode,
    AddStyleNote,
    CloseThread,
    Diff,
    DiffAuthor,
    InvalidateFact,
    MergeEntities,
    MoveNode,
    OpenThread,
    SetDial,
    SetProse,
    TouchThread,
)
from storygit.domain.ids import EntityId, IdGenerator, NodeId, ThreadId
from storygit.domain.ledger import Criterion, StyleNote, StyleNoteSource
from storygit.domain.nodes import Beat, Prose
from storygit.domain.provenance import Authorship, ProvenanceSpan, authorship_ratio
from storygit.domain.threads import Thread, ThreadStatus
from storygit.domain.world import Fact, Knows, Predicate
from storygit.graph.slices import entities_in_scope, entity_slice


def _location_fact(fixture: Fixture, ids: IdGenerator, place: EntityId, at: NodeId) -> Fact:
    return Fact(
        id=ids.fact(),
        subject=fixture.kael,
        predicate=Predicate.location,
        object_entity=place,
        valid_from_beat=at,
        established_by_beat=at,
    )


def test_temporal_validity_has_no_overlap_at_the_boundary(
    fixture: Fixture, ids: IdGenerator
) -> None:
    state = fixture.repo.state()
    moved = _location_fact(fixture, ids, fixture.kell, fixture.beat_c)
    state = apply(
        state,
        Diff(
            ops=(
                InvalidateFact(fact_id=fixture.fact_f, valid_until_beat=fixture.beat_c),
                AddFact(fact=moved),
            ),
            author=DiffAuthor.human,
            intent="Kael leaves Ashfall",
        ),
    )

    at_b = state.facts_valid_at(fixture.beat_b, subject=fixture.kael, predicate=Predicate.location)
    at_c = state.facts_valid_at(fixture.beat_c, subject=fixture.kael, predicate=Predicate.location)
    assert [f.id for f in at_b] == [fixture.fact_f]
    assert [f.id for f in at_c] == [moved.id]
    assert len(at_c) == 1, "the old and new locations must not both be valid at the boundary"


def test_entity_slice_reflects_the_beat_it_is_taken_at(fixture: Fixture, ids: IdGenerator) -> None:
    state = fixture.repo.state()
    moved = _location_fact(fixture, ids, fixture.kell, fixture.beat_c)
    state = apply(
        state,
        Diff(
            ops=(
                InvalidateFact(fact_id=fixture.fact_f, valid_until_beat=fixture.beat_c),
                AddFact(fact=moved),
            )
        ),
    )
    early = entity_slice(state, {fixture.kael, fixture.ashfall, fixture.kell}, fixture.beat_b)
    late = entity_slice(state, {fixture.kael, fixture.ashfall, fixture.kell}, fixture.beat_d)

    assert "Kael is at Ashfall." in early.render()
    assert "Kael is at Ashfall." not in late.render()
    assert "Kael is at Warden of Kell." in late.render() or "Kael is at Kell" in late.render()
    assert early.path[0].node_type == "story"
    assert early.target_beat == fixture.beat_b


def test_seq_is_recomputed_on_insert_and_move(fixture: Fixture) -> None:
    state = fixture.repo.state()
    order = [b.id for b in state.beats_in_order()]
    assert order == [fixture.beat_a, fixture.beat_b, fixture.beat_c, fixture.beat_d]

    inserted = NodeId("n_inserted")
    state = apply(
        state,
        Diff(ops=(AddNode(node=Beat(id=inserted, parent_id=fixture.scene, position=1)),)),
    )
    # Ties on position break on id; the inserted beat lands beside beat B.
    assert inserted in [b.id for b in state.beats_in_order()]

    state = apply(
        state,
        Diff(ops=(MoveNode(node_id=fixture.beat_d, new_parent_id=fixture.scene, new_position=0),)),
    )
    assert state.beats_in_order()[0].id == fixture.beat_d
    assert state.seq_of(fixture.beat_d) < state.seq_of(fixture.beat_a)

    # Validity comparisons follow the new order.
    fact = state.facts[fixture.fact_f]
    assert state.valid_at(fact, fixture.beat_b) is True
    assert state.valid_at(fact, fixture.beat_d) is False


def test_epistemic_edges(fixture: Fixture, ids: IdGenerator) -> None:
    state = fixture.repo.state()
    state = apply(
        state,
        Diff(
            ops=(
                AddKnows(
                    knows=Knows(
                        character=fixture.kael, fact=fixture.fact_f, since_beat=fixture.beat_c
                    )
                ),
            )
        ),
    )
    assert state.knows_at(fixture.kael, fixture.fact_f, fixture.beat_c) is True
    assert state.knows_at(fixture.kael, fixture.fact_f, fixture.beat_d) is True
    assert state.knows_at(fixture.kael, fixture.fact_f, fixture.beat_a) is False
    assert state.knows_at(fixture.kell, fixture.fact_f, fixture.beat_d) is False


def test_threads_open_touch_close(fixture: Fixture) -> None:
    thread_id = ThreadId("t_sister")
    state = fixture.repo.state()
    state = apply(
        state,
        Diff(
            ops=(
                OpenThread(
                    thread=Thread(
                        id=thread_id,
                        description="Where is Kael's sister?",
                        opened_at_beat=fixture.beat_a,
                        last_touched_beat=fixture.beat_a,
                    )
                ),
                TouchThread(thread_id=thread_id, beat_id=fixture.beat_c),
            )
        ),
    )
    assert state.threads[thread_id].last_touched_beat == fixture.beat_c
    assert state.threads[thread_id].status is ThreadStatus.open

    state = apply(state, Diff(ops=(CloseThread(thread_id=thread_id, status=ThreadStatus.dropped),)))
    assert state.threads[thread_id].status is ThreadStatus.dropped
    assert entity_slice(state, {fixture.kael}, fixture.beat_d).threads == ()


def test_ledger_ops_and_mined_rule_gating(fixture: Fixture) -> None:
    state = fixture.repo.state()
    mined = StyleNote(text="shorter sentences", source=StyleNoteSource.mined)
    state = apply(
        state,
        Diff(
            ops=(
                SetDial(value=0.75),
                AddStyleNote(note=StyleNote(text="no adverbs")),
                AddStyleNote(note=mined),
                AddCriterion(
                    criterion=Criterion(name="menace", description="every scene should threaten")
                ),
            ),
            author=DiffAuthor.human,
        ),
    )
    assert state.ledger.dial == 0.75
    assert {n.text for n in state.ledger.active_style_notes()} == {"no adverbs"}

    state = apply(state, Diff(ops=(AddStyleNote(note=mined),)))
    assert {n.text for n in state.ledger.active_style_notes()} == {
        "no adverbs",
        "shorter sentences",
    }
    assert state.ledger.criteria[0].name == "menace"


def test_authorship_ratio_on_mixed_prose(fixture: Fixture) -> None:
    prose_id = NodeId("n_prose")
    state = apply(
        fixture.repo.state(),
        Diff(
            ops=(
                AddNode(node=Prose(id=prose_id, parent_id=fixture.beat_a)),
                SetProse(
                    node_id=prose_id,
                    text="One. Two. Three. Four.",
                    spans=(
                        ProvenanceSpan(start=0, end=4, source=Authorship.ai),
                        ProvenanceSpan(start=2, end=3, source=Authorship.ai_edited_by_human),
                        ProvenanceSpan(start=3, end=4, source=Authorship.human),
                    ),
                ),
            )
        ),
    )
    prose = state.prose_for(fixture.beat_a)
    assert prose is not None
    ratio = authorship_ratio(prose.spans)
    assert ratio[Authorship.ai] == 0.5
    assert ratio[Authorship.ai_edited_by_human] == 0.25
    assert ratio[Authorship.human] == 0.25


def test_entity_merge_rewrites_facts_and_aliases(fixture: Fixture, ids: IdGenerator) -> None:
    state = fixture.repo.state()
    duplicate = EntityId("e_dupe")
    subject_fact, object_fact = ids.fact(), ids.fact()
    from storygit.domain.diff import AddEntity, AddKnows
    from storygit.domain.world import Entity, EntityKind, Knows

    state = apply(
        state,
        Diff(
            ops=(
                AddEntity(
                    entity=Entity(id=duplicate, kind=EntityKind.character, name="the Warden")
                ),
                AddFact(
                    fact=Fact(
                        id=subject_fact,
                        subject=duplicate,
                        predicate=Predicate.goal,
                        object_text="hold the gate",
                        valid_from_beat=fixture.beat_a,
                        established_by_beat=fixture.beat_a,
                    )
                ),
                # The duplicate is also the *object* of somebody else's fact, and
                # somebody knows that fact. Both point at an id that is about to
                # stop existing.
                AddFact(
                    fact=Fact(
                        id=object_fact,
                        subject=fixture.kael,
                        predicate=Predicate.relationship,
                        object_entity=duplicate,
                        valid_from_beat=fixture.beat_a,
                        established_by_beat=fixture.beat_a,
                    )
                ),
                AddKnows(
                    knows=Knows(character=duplicate, fact=fixture.fact_f, since_beat=fixture.beat_a)
                ),
            )
        ),
    )
    state = apply(state, Diff(ops=(MergeEntities(source_id=duplicate, target_id=fixture.kell),)))

    assert duplicate not in state.entities
    assert "the Warden" in state.entities[fixture.kell].aliases
    assert state.entity_by_name("The Warden") is not None
    assert state.entity_by_name("The Warden").id == fixture.kell  # type: ignore[union-attr]
    # Nothing anywhere in the world may still point at the id that was merged away:
    # a fact whose subject or object is a dead entity is a bible that cannot be read.
    assert state.facts[subject_fact].subject == fixture.kell
    assert state.facts[object_fact].object_entity == fixture.kell
    assert all(f.subject != duplicate for f in state.facts.values())
    assert all(f.object_entity != duplicate for f in state.facts.values())
    assert all(character != duplicate for character, _ in state.knows)
    # And the knowledge moves rather than vanishing: an epistemic edge quietly
    # disappearing is the exact failure the epistemic layer exists to catch.
    assert (fixture.kell, fixture.fact_f) in state.knows


def test_entities_in_scope_uses_facts_and_names(fixture: Fixture) -> None:
    state = fixture.repo.state()
    assert entities_in_scope(state, fixture.beat_a) >= {fixture.kael, fixture.ashfall}
    assert entities_in_scope(state, fixture.beat_d) == set()


def test_hard_constraints_reach_the_slice(fixture: Fixture) -> None:
    from storygit.domain.diff import AddHardConstraint, SetLock

    state = apply(
        fixture.repo.state(),
        Diff(
            ops=(
                SetLock(node_id=fixture.beat_a),
                AddHardConstraint(text="Kael never kills anyone."),
            ),
            author=DiffAuthor.human,
        ),
    )
    rendered = entity_slice(state, {fixture.kael}, fixture.beat_c).render()
    assert "Kael never kills anyone." in rendered
    assert "Locked fact" in rendered
