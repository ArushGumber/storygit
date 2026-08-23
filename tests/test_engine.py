"""The engine loop: propose, accept, extract, propagate, record the signal."""

from __future__ import annotations

import pytest
from tests.conftest import Fixture
from tests.mockprovider import MockProvider, canned

from storygit.agents.schemas import Level
from storygit.continuity.bible_diff import compute as bible_diff
from storygit.domain.ids import IdGenerator
from storygit.domain.nodes import NodeStatus
from storygit.domain.provenance import Authorship
from storygit.domain.world import Predicate
from storygit.engine import Engine
from storygit.graph.propagation import MarkKind
from storygit.providers.router import Router
from storygit.selection.select import SelectionConfig, Selector
from storygit.store.signals import SignalKind

PROSE = {
    "text": "The ash came down like flour. Kael did not look up. He counted three guards.",
    "rationale": "Short sentences.",
    "delta_summary": ["writes the opening"],
}
EXTRACTION = {
    "facts": [
        {
            "subject": "Kael",
            "predicate": "location",
            "object": "Ashfall",
            "object_is_entity": True,
            "known_by": ["Kael"],
        }
    ],
    "new_characters": [],
    "threads_opened": [],
    "threads_touched": [],
}
BEAT = {
    "title": "The offer",
    "what_happens": "The Warden offers Kael a way out.",
    "audience_learns": "the Warden wants something",
    "audience_feels": "wary",
    "location": "Ashfall",
    "time": "night",
    "produces": [
        {
            "subject": "Warden of Kell",
            "predicate": "goal",
            "object": "recruit Kael",
            "object_is_entity": False,
            "known_by": [],
        }
    ],
    "consumes": [],
    "threads_touched": [],
    "new_characters": [],
    "rationale": "Gives Kael something to refuse.",
    "delta_summary": ["the Warden makes an offer"],
}


# One candidate, no judge, no dial, the baseline selector: these tests are about the
# engine loop, not about selection, and this configuration needs no encoder or judge.
PLAIN = SelectionConfig(
    n=1, k=1, selector=Selector.topk_temperature, use_judge=False, use_dial=False
)


def engine_for(
    fixture: Fixture,
    responses: list[str],
    *,
    selection: SelectionConfig = PLAIN,
    **kwargs: object,
) -> tuple[Engine, MockProvider]:
    """An engine wired to one mock provider serving canned JSON."""
    provider = MockProvider(responses)
    router = Router({"gemini": provider, "groq": provider})
    engine = Engine(
        fixture.repo,
        router,
        ids=IdGenerator(seed=99),
        selection=selection,
        use_nli=False,
        **kwargs,  # type: ignore[arg-type]
    )
    return engine, provider


async def test_propose_then_accept_commits_and_records_a_preference(fixture: Fixture) -> None:
    engine, _ = engine_for(
        fixture,
        [canned(BEAT)],
        selection=SelectionConfig(
            n=3, k=3, selector=Selector.topk_temperature, use_judge=False, use_dial=False
        ),
    )
    before_head = fixture.repo.head()

    candidates = await engine.propose_at(Level.beat, node_id=fixture.scene)
    assert len(candidates) == 3
    assert {c.axis_key for c in candidates} == {"raise_stakes", "slow_down", "subvert"}, (
        "each candidate is generated along a different named axis"
    )
    assert fixture.repo.head() == before_head, "proposing must not change state"

    chosen, *rest = candidates
    result = await engine.accept(chosen.proposal.id, shown_with=engine.shown())

    assert fixture.repo.head() != before_head
    assert result.snapshot_id == fixture.repo.head() or result.propagation_snapshot_id
    signals = engine.signals.all(kind=SignalKind.accept)
    assert len(signals) == 1
    assert signals[0].proposal_id == chosen.proposal.id
    assert set(signals[0].shown_with) == {c.proposal.id for c in rest}, (
        "the alternatives are recorded, which is what makes this a preference"
    )


async def test_accept_returns_a_writer_readable_bible_diff(fixture: Fixture) -> None:
    engine, _ = engine_for(fixture, [canned(BEAT)])
    candidates = await engine.propose_at(Level.beat, node_id=fixture.scene)
    result = await engine.accept(candidates[0].proposal.id)

    assert not result.bible_diff.is_empty
    assert len(result.bible_diff.added) == 1
    assert result.bible_diff.lines[0].startswith("+ ")
    assert "recruit Kael" in result.bible_diff.lines[0]


async def test_accepting_prose_extracts_facts_from_it(fixture: Fixture) -> None:
    engine, provider = engine_for(fixture, [canned(PROSE), canned(EXTRACTION)])
    candidates = await engine.propose_at(Level.prose, node_id=fixture.beat_d)
    result = await engine.accept(candidates[0].proposal.id)

    assert result.extracted is True
    state = engine.state()
    prose = state.prose_for(fixture.beat_d)
    assert prose is not None and prose.spans[0].source is Authorship.ai

    located = [
        f
        for f in state.facts.values()
        if f.predicate is Predicate.location and f.established_by_beat == fixture.beat_d
    ]
    assert len(located) == 1, "the world graph is populated from what was actually written"
    purposes = [r.purpose for r in provider.requests]
    assert "extract.facts" in purposes


async def test_edit_records_the_before_and_after_pair(fixture: Fixture) -> None:
    engine, _ = engine_for(fixture, [canned(PROSE), canned(EXTRACTION)])
    candidates = await engine.propose_at(Level.prose, node_id=fixture.beat_d)
    await engine.edit(candidates[0].proposal.id, "Ash fell. Kael counted guards.")

    edits = engine.signals.all(kind=SignalKind.edit)
    assert len(edits) == 1
    assert edits[0].payload["before"].startswith("The ash came down")
    assert edits[0].payload["after"] == "Ash fell. Kael counted guards."

    prose = engine.state().prose_for(fixture.beat_d)
    assert prose is not None
    assert prose.text == "Ash fell. Kael counted guards."
    assert prose.spans[0].source is Authorship.ai_edited_by_human, (
        "an edited proposal is neither purely AI nor purely human"
    )


async def test_reject_records_the_direction_in_the_ledger(fixture: Fixture) -> None:
    engine, _ = engine_for(fixture, [canned(BEAT)])
    candidates = await engine.propose_at(Level.beat, node_id=fixture.scene)
    engine.reject(candidates[0].proposal.id, reason="no more shadowy benefactors")

    rejects = engine.signals.all(kind=SignalKind.reject)
    assert len(rejects) == 1 and rejects[0].payload["reason"]
    ledger = engine.state().ledger
    assert ledger.rejected_directions[0].text == "no more shadowy benefactors"
    assert ledger.rejected_directions[0].proposal_id == candidates[0].proposal.id
    assert engine.pending(candidates[0].proposal.id) is None


async def test_hand_written_beats_skip_generation_but_run_everything_else(
    fixture: Fixture,
) -> None:
    engine, provider = engine_for(fixture, [canned(EXTRACTION)])
    result = await engine.write_beat(fixture.beat_d, "Kael walked into Ashfall at dusk.")

    assert [r.purpose for r in provider.requests] == ["extract.facts"], (
        "no generation call is made for prose the writer wrote"
    )
    assert result.extracted is True
    prose = engine.state().prose_for(fixture.beat_d)
    assert prose is not None and prose.spans[0].source is Authorship.human


async def test_accepting_a_change_propagates_and_marks_downstream(fixture: Fixture) -> None:
    engine, _ = engine_for(fixture, [canned(EXTRACTION)])
    from storygit.domain.diff import Diff, DiffAuthor, UpdateFact

    before = engine.state()
    fixture.repo.commit_diff(
        Diff(
            ops=(
                UpdateFact(
                    fact_id=fixture.fact_f, fields={"object_text": "Kell", "object_entity": None}
                ),
            ),
            author=DiffAuthor.human,
            intent="Kael starts in Kell",
        )
    )
    from storygit.graph.propagation import marks_to_diff, propagate_change

    after = engine.state()
    marks = propagate_change(before, after)
    assert {m.kind for m in marks} == {MarkKind.stale}
    final = fixture.repo.preview_apply(marks_to_diff(after, marks))
    assert final.nodes[fixture.beat_b].status is NodeStatus.stale
    assert final.nodes[fixture.beat_d].status is not NodeStatus.stale


async def test_ledger_controls_write_through_the_repository(fixture: Fixture) -> None:
    engine, _ = engine_for(fixture, [])
    engine.lock(fixture.beat_a)
    engine.set_dial(0.8)
    engine.add_style_note("shorter sentences")
    engine.add_criterion("menace", "every scene should threaten something")

    state = engine.state()
    assert fixture.beat_a in state.ledger.locks
    assert state.nodes[fixture.beat_a].status is NodeStatus.locked
    assert state.ledger.dial == 0.8
    assert [n.text for n in state.ledger.style_notes] == ["shorter sentences"]
    assert state.ledger.criteria[0].name == "menace"
    assert engine.signals.counts()["lock"] == 1

    engine.unlock(fixture.beat_a)
    assert fixture.beat_a not in engine.state().ledger.locks


async def test_accepting_an_unknown_proposal_raises(fixture: Fixture) -> None:
    engine, _ = engine_for(fixture, [])
    from storygit.domain.ids import ProposalId

    with pytest.raises(KeyError):
        await engine.accept(ProposalId("p_nope"))


def test_bible_diff_reports_additions_endings_and_strikes(fixture: Fixture) -> None:
    from storygit.domain.apply import apply
    from storygit.domain.diff import Diff, InvalidateFact

    before = fixture.repo.state()
    after = apply(
        before,
        Diff(ops=(InvalidateFact(fact_id=fixture.fact_f, valid_until_beat=fixture.beat_c),)),
    )
    diff = bible_diff(before, after)
    assert len(diff.ended) == 1 and not diff.added and not diff.removed
    assert diff.lines[0].startswith("~ ")
    assert "no longer true" in diff.lines[0]
