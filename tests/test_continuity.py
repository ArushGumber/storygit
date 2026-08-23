"""The three-layer continuity checker, one fixture per failure it is supposed to catch."""

from __future__ import annotations

import pytest
from tests.conftest import Fixture
from tests.mockprovider import MockProvider, canned

from storygit.continuity import audit, bible_diff, layer1, layer2_nli, layer3_judge
from storygit.continuity.flags import FlagKind, Severity, sort_flags
from storygit.domain.apply import apply
from storygit.domain.diff import (
    AddFact,
    AddKnows,
    Diff,
    DiffAuthor,
    InvalidateFact,
    OpenThread,
    UpdateNode,
)
from storygit.domain.ids import EntityId, FactId, IdGenerator, NodeId, ThreadId
from storygit.domain.state import StoryState
from storygit.domain.threads import Thread
from storygit.domain.world import Fact, Knows, Predicate
from storygit.graph.slices import StateSlice
from storygit.providers.router import Router


def fact(
    fixture: Fixture,
    ids: IdGenerator,
    *,
    subject: EntityId | None = None,
    predicate: Predicate = Predicate.location,
    entity: EntityId | None = None,
    text: str = "",
    at: NodeId | None = None,
) -> Fact:
    """Build a fact with sensible fixture defaults."""
    return Fact(
        id=ids.fact(),
        subject=subject or fixture.kael,
        predicate=predicate,
        object_entity=entity,
        object_text=text,
        valid_from_beat=at or fixture.beat_a,
        established_by_beat=at or fixture.beat_a,
    )


# --- layer 1: what it catches -------------------------------------------------


def test_two_locations_at_once_is_a_contradiction(fixture: Fixture, ids: IdGenerator) -> None:
    state = apply(
        fixture.repo.state(),
        Diff(ops=(AddFact(fact=fact(fixture, ids, entity=fixture.kell, at=fixture.beat_b)),)),
    )
    flags = layer1.check_state(state)
    contradictions = [f for f in flags if f.kind is FlagKind.contradiction]
    assert len(contradictions) == 1
    flag = contradictions[0]
    assert flag.severity is Severity.hard and flag.layer == 1
    assert flag.established_by == fixture.beat_a, "the flag cites where the first one came from"
    assert "Beat A" in flag.message
    assert fixture.kael in flag.entity_ids


def test_a_properly_ended_location_change_does_not_flag(fixture: Fixture, ids: IdGenerator) -> None:
    state = apply(
        fixture.repo.state(),
        Diff(
            ops=(
                InvalidateFact(fact_id=fixture.fact_f, valid_until_beat=fixture.beat_b),
                AddFact(fact=fact(fixture, ids, entity=fixture.kell, at=fixture.beat_b)),
            ),
            author=DiffAuthor.human,
            intent="Kael moves to Kell",
        ),
    )
    assert [f for f in layer1.check_state(state) if f.kind is FlagKind.contradiction] == []


def test_re_establishing_the_same_value_does_not_flag(fixture: Fixture, ids: IdGenerator) -> None:
    state = apply(
        fixture.repo.state(),
        Diff(ops=(AddFact(fact=fact(fixture, ids, entity=fixture.ashfall, at=fixture.beat_b)),)),
    )
    assert [f for f in layer1.check_state(state) if f.kind is FlagKind.contradiction] == []


def test_two_goals_at_once_is_a_character_not_a_bug(fixture: Fixture, ids: IdGenerator) -> None:
    state = apply(
        fixture.repo.state(),
        Diff(
            ops=(
                AddFact(
                    fact=fact(
                        fixture,
                        ids,
                        predicate=Predicate.goal,
                        text="protect the city",
                        at=fixture.beat_b,
                    )
                ),
            )
        ),
    )
    assert [f for f in layer1.check_state(state) if f.kind is FlagKind.contradiction] == []


def test_a_dead_character_who_acts_is_flagged(fixture: Fixture, ids: IdGenerator) -> None:
    death = fact(fixture, ids, predicate=Predicate.alive, text="dead", at=fixture.beat_b)
    later = fact(fixture, ids, predicate=Predicate.goal, text="escape", at=fixture.beat_c)
    state = apply(
        fixture.repo.state(),
        Diff(ops=(AddFact(fact=death), AddFact(fact=later))),
    )
    flags = [f for f in layer1.check_state(state) if f.kind is FlagKind.dead_actor]
    assert len(flags) == 1
    assert flags[0].established_by == fixture.beat_b, "cites where the death was established"
    assert "dead" in flags[0].message
    assert flags[0].node_id == fixture.beat_c, "flagged at the beat where the dead act"


def test_two_holders_of_one_object_is_flagged(fixture: Fixture, ids: IdGenerator) -> None:
    state = apply(
        fixture.repo.state(),
        Diff(
            ops=(
                AddFact(
                    fact=fact(
                        fixture,
                        ids,
                        predicate=Predicate.possesses,
                        text="the Warden's token",
                        at=fixture.beat_a,
                    )
                ),
                AddFact(
                    fact=fact(
                        fixture,
                        ids,
                        subject=fixture.kell,
                        predicate=Predicate.possesses,
                        text="the Warden's token",
                        at=fixture.beat_b,
                    )
                ),
            )
        ),
    )
    flags = [f for f in layer1.check_state(state) if f.kind is FlagKind.possession]
    assert len(flags) == 1
    assert "token" in flags[0].message
    assert set(flags[0].entity_ids) == {fixture.kael, fixture.kell}


def test_acting_on_a_secret_before_learning_it_is_flagged(
    fixture: Fixture, ids: IdGenerator
) -> None:
    secret = fact(
        fixture,
        ids,
        subject=fixture.kael,
        predicate=Predicate.secret,
        text="the ash answers him",
        at=fixture.beat_a,
    )
    warden_acts = fact(
        fixture,
        ids,
        subject=fixture.kell,
        predicate=Predicate.goal,
        text="recruit Kael",
        at=fixture.beat_c,
    )
    state = apply(
        fixture.repo.state(),
        Diff(
            ops=(
                AddFact(fact=secret),
                AddFact(fact=warden_acts),
                UpdateNode(node_id=fixture.beat_c, fields={"consumes": [str(secret.id)]}),
            )
        ),
    )

    flags = [f for f in layer1.check_state(state) if f.kind is FlagKind.epistemic]
    assert len(flags) == 1
    assert "never learns it" in flags[0].message
    assert flags[0].established_by == fixture.beat_a, "cites where the secret was established"
    assert flags[0].node_id == fixture.beat_c

    # Tell the Warden later than they act: the flag says exactly when they learn it.
    told_late = apply(
        state,
        Diff(
            ops=(
                AddKnows(
                    knows=Knows(character=fixture.kell, fact=secret.id, since_beat=fixture.beat_d)
                ),
            )
        ),
    )
    late = [f for f in layer1.check_state(told_late) if f.kind is FlagKind.epistemic]
    assert len(late) == 1
    assert "learns it in Beat D" in late[0].message

    # Tell them before: no flag.
    told_early = apply(
        state,
        Diff(
            ops=(
                AddKnows(
                    knows=Knows(character=fixture.kell, fact=secret.id, since_beat=fixture.beat_b)
                ),
            )
        ),
    )
    assert [f for f in layer1.check_state(told_early) if f.kind is FlagKind.epistemic] == []


def test_a_character_always_knows_their_own_secret(fixture: Fixture, ids: IdGenerator) -> None:
    secret = fact(
        fixture, ids, predicate=Predicate.secret, text="he can hear the ash", at=fixture.beat_a
    )
    acting = fact(fixture, ids, predicate=Predicate.goal, text="hide it", at=fixture.beat_c)
    state = apply(
        fixture.repo.state(),
        Diff(
            ops=(
                AddFact(fact=secret),
                AddFact(fact=acting),
                UpdateNode(node_id=fixture.beat_c, fields={"consumes": [str(secret.id)]}),
            )
        ),
    )
    assert [f for f in layer1.check_state(state) if f.kind is FlagKind.epistemic] == []


def test_layer1_cannot_call_a_model() -> None:
    """Layer 1's determinism is a structural property, so test it structurally."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(layer1))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden = [m for m in imported if "providers" in m or "agents" in m or "router" in m]
    assert forbidden == [], (
        f"layer 1 must be deterministic by construction, but imports {forbidden}"
    )
    assert all(not m.startswith(("torch", "transformers", "httpx")) for m in imported)


# --- layer 2: NLI -------------------------------------------------------------


def test_layer2_only_considers_pairs_layer1_cannot_decide(
    fixture: Fixture, ids: IdGenerator
) -> None:
    same_predicate_multi = apply(
        fixture.repo.state(),
        Diff(
            ops=(
                AddFact(
                    fact=fact(
                        fixture,
                        ids,
                        predicate=Predicate.note,
                        text="the ash obeys him",
                        at=fixture.beat_a,
                    )
                ),
                AddFact(
                    fact=fact(
                        fixture,
                        ids,
                        predicate=Predicate.note,
                        text="the ash never obeys anyone",
                        at=fixture.beat_b,
                    )
                ),
            )
        ),
    )
    pairs = layer2_nli.candidate_pairs(same_predicate_multi)
    assert len(pairs) == 1, "one note pair to check"

    # Single-valued predicates are layer 1's job and must not reach the cross-encoder.
    location_conflict = apply(
        fixture.repo.state(),
        Diff(ops=(AddFact(fact=fact(fixture, ids, entity=fixture.kell, at=fixture.beat_b)),)),
    )
    assert all(
        a.predicate is not Predicate.location
        for a, _ in layer2_nli.candidate_pairs(location_conflict)
    )


@pytest.mark.slow
def test_layer2_scores_a_contradiction_higher_than_a_paraphrase(
    fixture: Fixture, ids: IdGenerator
) -> None:
    from storygit.providers.local import nli_scores

    contradiction = nli_scores(
        [("Kael can command the ash whenever he chooses.", "The ash never obeys anyone.")]
    )[0]
    paraphrase = nli_scores(
        [("Kael can command the ash whenever he chooses.", "The ash does what Kael tells it.")]
    )[0]
    assert contradiction["contradiction"] > paraphrase["contradiction"]
    assert paraphrase["contradiction"] < 0.1, "a paraphrase must be nowhere near the threshold"
    assert contradiction["contradiction"] > layer2_nli.CONTRADICTION_THRESHOLD

    # A blunter contradiction should be scored far more confidently, which is what makes
    # the threshold a real separator rather than a coin flip.
    blunt = nli_scores(
        [
            (
                "Kael can command the ash whenever he chooses.",
                "Kael has never been able to affect the ash at all.",
            )
        ]
    )[0]
    assert blunt["contradiction"] > 0.9


@pytest.mark.slow
def test_layer2_flags_a_contradictory_note_pair(fixture: Fixture, ids: IdGenerator) -> None:
    state = apply(
        fixture.repo.state(),
        Diff(
            ops=(
                AddFact(
                    fact=fact(
                        fixture,
                        ids,
                        predicate=Predicate.note,
                        text="can command the ash whenever he chooses",
                        at=fixture.beat_a,
                    )
                ),
                AddFact(
                    fact=fact(
                        fixture,
                        ids,
                        predicate=Predicate.note,
                        text="has never been able to affect the ash at all",
                        at=fixture.beat_b,
                    )
                ),
            )
        ),
    )
    flags = layer2_nli.check(state)
    assert len(flags) == 1
    assert flags[0].severity is Severity.soft and flags[0].layer == 2
    assert flags[0].established_by == fixture.beat_a
    assert flags[0].score is not None and flags[0].score > 0.5


# --- layer 3: the soft judge --------------------------------------------------


async def test_layer3_returns_soft_flags_and_a_quality_score() -> None:
    verdict = {
        "scores": [
            {"name": "momentum", "argument": "It moves.", "score": 4},
            {"name": "specificity", "argument": "Named places.", "score": 5},
        ],
        "motivation_concern": "Kael gives up the token he has spent two episodes protecting.",
        "tone_concern": "",
    }
    provider = MockProvider([canned(verdict)])
    router = Router({"gemini": provider, "groq": provider})

    quality, flags, scores = await layer3_judge.judge(
        router, StateSlice(), "some candidate text", node_id=NodeId("n_1")
    )

    assert quality == pytest.approx((4.5 - 1) / 4)
    assert scores == {"momentum": 4.0, "specificity": 5.0}
    assert len(flags) == 1, "an empty concern is not a flag"
    assert flags[0].kind is FlagKind.motivation
    assert flags[0].severity is Severity.soft and flags[0].layer == 3

    prompt = provider.prompts_for("judge.soft")[0]
    assert "arguing the case FIRST" in prompt, "argue-before-rate is stated in the prompt"
    assert "then give a score" in prompt


async def test_layer3_failure_is_neutral_not_zero() -> None:
    provider = MockProvider(["not json", "still not json"])
    router = Router({"gemini": provider, "groq": provider})
    quality, flags, scores = await layer3_judge.judge(router, StateSlice(), "text")
    assert quality == 0.5, "an unreachable judge must not rank every candidate as bad"
    assert flags == [] and scores == {}


def test_soft_flags_never_outrank_hard_ones() -> None:
    from storygit.continuity.flags import Flag

    flags = [
        Flag(kind=FlagKind.tone, severity=Severity.soft, layer=3, message="soft"),
        Flag(kind=FlagKind.contradiction, severity=Severity.hard, layer=1, message="hard"),
    ]
    assert [f.message for f in sort_flags(flags)] == ["hard", "soft"]


# --- bible diff ---------------------------------------------------------------


def test_bible_diff_counts_additions_endings_and_strikes(
    fixture: Fixture, ids: IdGenerator
) -> None:
    before = fixture.repo.state()
    added_one = fact(fixture, ids, predicate=Predicate.trait, text="stubborn", at=fixture.beat_b)
    added_two = fact(
        fixture, ids, predicate=Predicate.goal, text="find the harbour", at=fixture.beat_b
    )
    after = apply(
        before,
        Diff(
            ops=(
                AddFact(fact=added_one),
                AddFact(fact=added_two),
                InvalidateFact(fact_id=fixture.fact_f, valid_until_beat=fixture.beat_c),
            )
        ),
    )
    diff = bible_diff.compute(before, after)
    assert diff.counts() == {"added": 2, "ended": 1, "removed": 0}
    assert sum(1 for line in diff.lines if line.startswith("+ ")) == 2
    assert sum(1 for line in diff.lines if line.startswith("~ ")) == 1
    assert not diff.is_empty


def test_striking_a_fact_ends_it_or_removes_it(fixture: Fixture) -> None:
    state = fixture.repo.state()

    ended = bible_diff.strike(state, fixture.fact_f, at_beat=fixture.beat_c)
    assert isinstance(ended.ops[0], InvalidateFact)
    assert apply(state, ended).facts[fixture.fact_f].valid_until_beat == fixture.beat_c

    removed = bible_diff.strike(state, fixture.fact_f)
    after = apply(state, removed)
    assert fixture.fact_f not in after.facts

    with pytest.raises(KeyError):
        bible_diff.strike(state, FactId("f_nope"))


def test_recheck_after_strike_looks_at_what_depended_on_it(fixture: Fixture) -> None:
    state = fixture.repo.state()
    after = apply(state, bible_diff.strike(state, fixture.fact_f))
    # The consumers are gone from the post-state, so nothing to re-check there.
    assert bible_diff.recheck_after_strike(state, fixture.fact_f) is not None
    assert bible_diff.recheck_after_strike(after, fixture.fact_f) == []


# --- audit --------------------------------------------------------------------


def test_audit_walks_the_whole_graph(fixture: Fixture, ids: IdGenerator) -> None:
    state = apply(
        fixture.repo.state(),
        Diff(ops=(AddFact(fact=fact(fixture, ids, entity=fixture.kell, at=fixture.beat_b)),)),
    )
    report = audit.run_audit(state, use_nli=False)
    assert report.facts_checked == len(state.facts)
    assert report.entities_checked == len(state.entities)
    assert any(f.kind is FlagKind.contradiction for f in report.flags)
    assert report.by_layer["layer1"] >= 1
    assert "audit:" in report.summary()


def test_audit_notices_a_dropped_thread(fixture: Fixture) -> None:
    thread_id = ThreadId("t_sister")
    state = apply(
        fixture.repo.state(),
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
            )
        ),
    )
    assert audit.check_dropped_threads(state, max_gap=2) != []
    flags = audit.check_dropped_threads(state, max_gap=2)
    assert flags[0].kind is FlagKind.dropped_thread
    assert flags[0].severity is Severity.soft
    assert "sister" in flags[0].message
    assert audit.check_dropped_threads(state, max_gap=100) == []


# --- soft edges (ablation 2d'') -----------------------------------------------


def test_soft_edges_fire_below_the_threshold_and_not_above(fixture: Fixture) -> None:
    import numpy as np

    from storygit.graph.soft_edges import EmbeddingEdgeProvider

    # Beats A and C read alike; B and D do not.
    vectors = {
        "Beat A": np.array([1.0, 0.0], dtype=np.float32),
        "Beat B": np.array([0.0, 1.0], dtype=np.float32),
        "Beat C": np.array([0.95, 0.31], dtype=np.float32),
        "Beat D": np.array([0.0, 1.0], dtype=np.float32),
    }

    def fake_embed(texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            for title, vector in vectors.items():
                if text.startswith(title):
                    rows.append(vector / np.linalg.norm(vector))
                    break
            else:  # pragma: no cover - fixture titles are exhaustive
                rows.append(np.zeros(2, dtype=np.float32))
        return np.vstack(rows)

    state = fixture.repo.state()
    loose = EmbeddingEdgeProvider(threshold=0.5, embed_fn=fake_embed)
    tight = EmbeddingEdgeProvider(threshold=0.99, embed_fn=fake_embed)

    assert fixture.beat_c in loose.extra_dependents(state, fixture.beat_a)
    assert tight.extra_dependents(state, fixture.beat_a) == ()


def test_soft_edge_marks_are_weaker_than_stale(fixture: Fixture) -> None:
    from storygit.domain.diff import UpdateFact
    from storygit.graph.propagation import MarkKind, propagate_change

    class Always:
        def extra_dependents(self, state: StoryState, node_id: NodeId) -> tuple[NodeId, ...]:
            return (fixture.beat_d,)

    before = fixture.repo.state()
    after = apply(
        before,
        Diff(
            ops=(
                UpdateFact(
                    fact_id=fixture.fact_f, fields={"object_entity": None, "object_text": "Kell"}
                ),
            )
        ),
    )
    marks = {m.node_id: m for m in propagate_change(before, after, edge_provider=Always())}
    assert marks[fixture.beat_b].kind is MarkKind.stale
    assert marks[fixture.beat_d].kind is MarkKind.maybe_affected
    assert "Not a declared dependency" in marks[fixture.beat_d].reason


def test_a_duplicate_entity_is_flagged_on_the_candidate_that_creates_it() -> None:
    """The hot path dropped the flag in the one case that produces duplicates.

    ``check_new_facts`` filtered to flags mentioning one of the new facts. A
    duplicate-entity flag carries entity ids and no fact ids, so it survived only when the
    change introduced no facts at all -- the opposite of the case that matters, since a
    duplicate is created by extracting a fact about a name under a slightly different
    spelling. The filter cannot simply pass every duplicate through, either: the pair is a
    property of the story, so all three candidates in a shown set would carry the same flag.
    Both halves are asserted here.
    """
    from storygit.continuity import layer1
    from storygit.continuity.flags import FlagKind
    from storygit.domain.ids import EntityId, FactId, NodeId
    from storygit.domain.nodes import Beat, Scene, Story
    from storygit.domain.state import StoryState
    from storygit.domain.world import Entity, EntityKind, Fact, Predicate

    story = Story(id=NodeId("n_root"), title="S")
    scene = Scene(id=NodeId("n_scene"), parent_id=story.id, title="One", position=0)
    beat = Beat(id=NodeId("n_beat"), parent_id=scene.id, title="B", position=0)
    marguerite = Entity(id=EntityId("e_full"), name="Marguerite Osei", kind=EntityKind.character)
    short = Entity(id=EntityId("e_short"), name="Marguerite", kind=EntityKind.character)
    other = Entity(id=EntityId("e_other"), name="Ronnie Fenn", kind=EntityKind.character)

    new_fact = Fact(
        id=FactId("f_new"),
        subject=short.id,
        predicate=Predicate.note,
        object_text="announced the pledges",
        valid_from_beat=beat.id,
        established_by_beat=beat.id,
    )
    unrelated = Fact(
        id=FactId("f_other"),
        subject=other.id,
        predicate=Predicate.note,
        object_text="said nothing at all",
        valid_from_beat=beat.id,
        established_by_beat=beat.id,
    )
    state = StoryState.build(
        nodes={story.id: story, scene.id: scene, beat.id: beat},
        entities={e.id: e for e in (marguerite, short, other)},
        facts={new_fact.id: new_fact, unrelated.id: unrelated},
    )

    about_the_duplicate = layer1.check_new_facts(state, {new_fact.id})
    assert any(f.kind is FlagKind.duplicate_entity for f in about_the_duplicate), (
        "a fact about the short name is what creates the duplicate; the flag must survive"
    )

    about_someone_else = layer1.check_new_facts(state, {unrelated.id})
    assert not any(f.kind is FlagKind.duplicate_entity for f in about_someone_else), (
        "a candidate that does not touch either name must not carry the story's duplicate"
    )


async def test_a_hard_constraint_violation_becomes_a_cited_soft_flag() -> None:
    """A hard constraint that is checked nowhere is a style note with a confident name.

    That is the writer's own sentence for it, and it was accurate: constraints reached the
    generation prompt and nothing looked at the result. After adding "Present day. No
    ten-shilling notes, no shillings, no Austin Cambridges" they were offered four pounds
    ten, a Morris Minor and three-shilling bits.

    The check rides the judge call that already runs per candidate, rather than adding one
    call per constraint -- which would multiply the most expensive stage of the pipeline by
    the size of the ledger. It is never automatic: the flag is soft, and the writer decides.
    """
    import json

    from tests.mockprovider import MockProvider

    from storygit.continuity.layer3_judge import judge
    from storygit.graph.slices import StateSlice
    from storygit.providers.router import Router

    constraint = "Present day. No ten-shilling notes, no shillings, no Austin Cambridges."
    slice_ = StateSlice(hard_constraints=(constraint,))

    verdict = json.dumps(
        {
            "scores": [{"name": "momentum", "argument": "It moves.", "score": 4}],
            "constraint_violations": [
                {"constraint": constraint, "how": "Ronnie pays in three-shilling bits."}
            ],
        }
    )
    router = Router({"gemini": MockProvider([verdict])})
    _, flags, _ = await judge(router, slice_, "Ronnie counts out three-shilling bits.")

    violation = next(f for f in flags if f.kind.value == "hard_constraint")
    assert violation.severity.value == "soft", "never automatic; the writer decides"
    assert constraint in violation.message, "the citation has to be checkable"
    assert "three-shilling bits" in violation.message


async def test_a_constraint_the_writer_never_set_is_not_flagged() -> None:
    """A citation the writer cannot find teaches them the citations are decorative."""
    import json

    from tests.mockprovider import MockProvider

    from storygit.continuity.layer3_judge import judge
    from storygit.graph.slices import StateSlice
    from storygit.providers.router import Router

    slice_ = StateSlice(hard_constraints=("Present day. No shillings.",))
    verdict = json.dumps(
        {
            "scores": [{"name": "momentum", "argument": "It moves.", "score": 4}],
            "constraint_violations": [
                {"constraint": "No dogs in the show.", "how": "There is a dog."}
            ],
        }
    )
    router = Router({"gemini": MockProvider([verdict])})
    _, flags, _ = await judge(router, slice_, "A dog walks past the bus shelter.")
    assert not [f for f in flags if f.kind.value == "hard_constraint"]


async def test_a_clean_candidate_produces_no_constraint_flag() -> None:
    """The normal case, which has to stay quiet or the flag list stops being read."""
    import json

    from tests.mockprovider import MockProvider

    from storygit.continuity.layer3_judge import judge
    from storygit.graph.slices import StateSlice
    from storygit.providers.router import Router

    slice_ = StateSlice(hard_constraints=("Present day. No shillings.",))
    verdict = json.dumps({"scores": [{"name": "momentum", "argument": "It moves.", "score": 4}]})
    router = Router({"gemini": MockProvider([verdict])})
    _, flags, _ = await judge(router, slice_, "Ronnie taps his card on the reader.")
    assert not [f for f in flags if f.kind.value == "hard_constraint"]
