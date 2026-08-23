"""The preference layer: signals, exemplars, edit mining, voice, the head, and the bandit.

Everything here is synthetic and seeded, so it runs in seconds and reruns identically.
"""

from __future__ import annotations

import random
from itertools import pairwise

import pytest
from tests.conftest import Fixture
from tests.mockprovider import MockProvider, canned

from storygit.domain.ids import NodeId, ProposalId
from storygit.preference import edit_mining, pretrain
from storygit.preference.bandit import (
    Arm,
    BanditState,
    ThompsonBandit,
    pseudo_regret_curve,
)
from storygit.preference.bt_head import (
    BTWeights,
    fit,
    pairwise_accuracy,
    weight_correlation,
)
from storygit.preference.exemplars import ExemplarPool
from storygit.preference.features import (
    BASE_FEATURES,
    FeatureVector,
    build,
    continuity_score,
    dialogue_ratio,
    length_score,
)
from storygit.preference.layer import PreferenceLayer
from storygit.preference.signals import SignalReader
from storygit.preference.state import PreferenceState, PreferenceStateStore
from storygit.providers.router import Router
from storygit.store.signals import Signal, SignalKind, SignalStore


def vector(**overrides: float) -> FeatureVector:
    """A feature vector at 0.5 everywhere except the named features."""
    values = dict.fromkeys(BASE_FEATURES, 0.5)
    values.update(overrides)
    return FeatureVector(values=values)


# --- signals ------------------------------------------------------------------


def test_pairs_are_formed_only_within_one_shown_set(fixture: Fixture) -> None:
    store = SignalStore(fixture.repo.conn)
    a, b, c, d = (ProposalId(f"p_{x}") for x in "abcd")

    store.record(
        Signal(
            kind=SignalKind.accept,
            proposal_id=a,
            shown_with=(b, c),
            node_id=fixture.beat_a,
            payload={"level": "beat"},
        )
    )
    store.record(
        Signal(kind=SignalKind.accept, proposal_id=d, shown_with=(), node_id=fixture.beat_b)
    )

    pairs = SignalReader(store).preference_pairs()
    assert {(p.winner, p.loser) for p in pairs} == {(a, b), (a, c)}, (
        "d was accepted alone, so it produces no comparison"
    )
    assert all(p.node_id == fixture.beat_a for p in pairs)
    assert pairs[0].level == "beat"


def test_an_edited_win_is_marked_as_weaker_evidence(fixture: Fixture) -> None:
    store = SignalStore(fixture.repo.conn)
    a, b = ProposalId("p_a"), ProposalId("p_b")
    store.record(
        Signal(
            kind=SignalKind.edit,
            proposal_id=a,
            payload={"before": "a long and winding sentence with many clauses", "after": "short"},
        )
    )
    store.record(Signal(kind=SignalKind.accept, proposal_id=a, shown_with=(b,)))

    reader = SignalReader(store)
    assert reader.preference_pairs()[0].was_edited is True
    edits = reader.edit_pairs()
    assert len(edits) == 1 and edits[0].shortened is True


def test_unchosen_candidates_are_recoverable_as_weak_negatives(fixture: Fixture) -> None:
    store = SignalStore(fixture.repo.conn)
    a, b, c = (ProposalId(f"p_{x}") for x in "abc")
    store.record(Signal(kind=SignalKind.accept, proposal_id=a, shown_with=(b, c)))
    store.record(Signal(kind=SignalKind.reject, proposal_id=b))

    reader = SignalReader(store)
    assert reader.accepted_ids() == [a]
    assert reader.rejected_ids() == [b]
    assert set(reader.unchosen_ids()) == {b, c}
    assert reader.counts()["preference_pairs"] == 2


# --- features -----------------------------------------------------------------


def test_feature_helpers_behave_sensibly() -> None:
    assert dialogue_ratio('He said. "Go." She left.') == pytest.approx(1 / 3)
    assert dialogue_ratio("") == 0.0
    assert length_score("word " * 220) == pytest.approx(1.0, abs=0.02)
    assert length_score("word " * 40) < length_score("word " * 200)
    assert length_score("word " * 900) < length_score("word " * 200), (
        "longer is not automatically better; the head learns the direction"
    )
    assert continuity_score(0, 0) == 1.0
    assert continuity_score(1, 0) < continuity_score(0, 3)


def test_feature_vector_is_exactly_the_designed_list() -> None:
    features = build(text="A sentence.")
    assert set(features.values) == set(BASE_FEATURES)
    assert len(BASE_FEATURES) == 10
    assert all(0.0 <= v <= 1.0 for v in features.values.values())
    # A missing judge score is neutral, not zero: an unreachable judge must not make every
    # candidate look bad.
    assert features.values["judge_momentum"] == 0.5


# --- Bradley-Terry ------------------------------------------------------------


def test_the_head_recovers_a_hidden_weight_vector() -> None:
    rng = random.Random(7)
    hidden = {name: rng.uniform(-1.0, 1.0) for name in BASE_FEATURES}

    def sample() -> FeatureVector:
        return FeatureVector(values={name: rng.random() for name in BASE_FEATURES})

    def prefers(a: FeatureVector, b: FeatureVector) -> bool:
        score = sum(hidden[k] * (a.values[k] - b.values[k]) for k in BASE_FEATURES)
        return score + rng.gauss(0.0, 0.2) > 0.0

    pairs = []
    for _ in range(200):
        a, b = sample(), sample()
        pairs.append((a, b) if prefers(a, b) else (b, a))
    held_out = []
    for _ in range(200):
        a, b = sample(), sample()
        held_out.append((a, b) if prefers(a, b) else (b, a))

    weights = fit(pairs, prior=BTWeights.uniform(), l2=0.5)
    assert pairwise_accuracy(weights, held_out) >= 0.85
    assert weight_correlation(weights, hidden) >= 0.8, (
        "recovering the ordering is not enough; the learned weights must mean something"
    )


def test_the_head_is_deterministic_and_explainable() -> None:
    pairs = [(vector(length=0.9), vector(length=0.1)) for _ in range(20)]
    first = fit(pairs, prior=BTWeights.uniform())
    second = fit(pairs, prior=BTWeights.uniform())
    assert first.weights == second.weights

    assert first.top_weights(1)[0][0] == "length", (
        "the only feature that varied must be the one that moved"
    )
    assert first.probability(vector(length=0.9), vector(length=0.1)) > 0.5


def test_the_head_with_no_data_is_the_prior() -> None:
    prior = BTWeights.uniform()
    assert fit([], prior=prior).weights == prior.weights
    assert prior.score(vector()) == pytest.approx(0.5)
    assert weight_correlation(prior, dict.fromkeys(BASE_FEATURES, 1.0)) == 0.0


# --- voice model --------------------------------------------------------------


def _punchy(rng: random.Random, n: int) -> list[str]:
    """Short, blunt, verb-first sentences."""
    subjects = ["Kael", "Mara", "The Warden", "She", "He"]
    verbs = ["ran", "stopped", "waited", "struck", "left", "listened"]
    return [
        " ".join(f"{rng.choice(subjects)} {rng.choice(verbs)}." for _ in range(4)) for _ in range(n)
    ]


def _flowery(rng: random.Random, n: int) -> list[str]:
    """Long, subordinate-clause-heavy, adjective-heavy sentences."""
    openers = [
        "In the long grey hours before the ash began to settle over the broken rooftops",
        "Beneath a sky the colour of old iron and older grief, where nothing moved",
        "Through the slow and terrible patience of a city that had learned to wait",
    ]
    tails = [
        "she considered, with a weariness that had become almost companionable, the many "
        "ways in which the evening might yet betray her",
        "he allowed himself the small and dangerous luxury of imagining another life, one "
        "in which the war had never reached this far north",
    ]
    return [f"{rng.choice(openers)}, {rng.choice(tails)}." for _ in range(n)]


@pytest.mark.slow
def test_the_voice_model_separates_two_styles_on_held_out_text() -> None:
    from storygit.preference.voice import train

    rng = random.Random(3)
    anchors = _punchy(rng, 10)
    positives = _punchy(rng, 10)
    negatives = _flowery(rng, 10)

    model = train(anchors, positives, negatives, seed=0)
    assert model.is_trained

    same_style = model.score(_punchy(rng, 12))
    cross_style = model.score(_flowery(rng, 12))

    # AUC of the two score distributions.
    wins = sum(1 for a in same_style for b in cross_style if a > b)
    ties = sum(1 for a in same_style for b in cross_style if a == b)
    auc = (wins + 0.5 * ties) / (len(same_style) * len(cross_style))
    assert auc >= 0.8, f"held-out AUC {auc:.2f} is too low to be useful"


@pytest.mark.slow
def test_the_edit_direction_separates_shortening_from_lengthening() -> None:
    from storygit.preference.voice import VoiceModel, edit_direction

    rng = random.Random(11)
    long_versions = _flowery(rng, 8)
    short_versions = _punchy(rng, 8)

    direction = edit_direction(long_versions, short_versions)
    assert direction

    model = VoiceModel(direction=direction, edits_seen=len(long_versions))
    held_out_short = model.direction_scores(_punchy(rng, 6) + _flowery(rng, 6))
    shortened = held_out_short[:6]
    lengthened = held_out_short[6:]
    assert sum(shortened) / 6 > sum(lengthened) / 6, (
        "candidates in the direction the writer keeps pulling must score higher"
    )


def test_an_untrained_voice_model_is_neutral_not_zero() -> None:
    from storygit.preference.voice import VoiceModel

    model = VoiceModel()
    assert model.is_trained is False
    assert model.score(["anything", "at all"]) == [0.5, 0.5]
    assert model.direction_scores(["anything"]) == [0.5]


# --- exemplars ----------------------------------------------------------------


def test_exemplar_block_renders_and_is_toggleable() -> None:
    pool = ExemplarPool(["Kael ran. He did not look back."], ["In the long grey hours..."])
    block = pool.render("Kael escapes", m=1)
    assert "THE WRITER KEPT THESE" in block
    assert "THE WRITER TURNED THESE DOWN" in block
    assert pool.render("Kael escapes", enabled=False) == "", "the ablation switch"
    assert ExemplarPool().render("anything") == "", "a new writer contributes nothing"


@pytest.mark.slow
def test_exemplar_retrieval_returns_the_nearest_items() -> None:
    pool = ExemplarPool(
        [
            "The ash fell over the market and Kael counted the guards.",
            "Mara steered the boat through the harbour mouth at dawn.",
            "The Warden's token was cold in his palm.",
        ],
        [],
    )
    positives, _ = pool.retrieve("a boat leaving the harbour", m=1)
    assert positives == ["Mara steered the boat through the harbour mouth at dawn."]


# --- edit mining --------------------------------------------------------------


async def test_mined_rules_are_deduped_against_the_ledger(fixture: Fixture) -> None:
    from storygit.domain.apply import apply
    from storygit.domain.diff import AddStyleNote, Diff
    from storygit.domain.ledger import StyleNote, StyleNoteSource
    from storygit.preference.signals import EditPair

    provider = MockProvider([canned({"rules": ["shorter sentences", "less exposition"]})])
    router = Router({"gemini": provider, "groq": provider})
    pairs = [EditPair(proposal_id=ProposalId("p_1"), before="a long thing", after="short")]

    rules = await edit_mining.mine(router, pairs)
    assert rules == ["shorter sentences", "less exposition"]
    assert provider.requests[0].purpose == "classify.edit", "mining runs on the cheap model"

    state = fixture.repo.state()
    diff = edit_mining.to_diff(state.ledger, rules)
    after = apply(state, diff)
    assert {n.text for n in after.ledger.style_notes} == {"shorter sentences", "less exposition"}
    assert all(n.source is StyleNoteSource.mined for n in after.ledger.style_notes)

    # Mined rules stay out of prompts until corroborated.
    assert after.ledger.active_style_notes() == ()
    twice = apply(after, edit_mining.to_diff(after.ledger, ["shorter sentences"]))
    assert {n.text for n in twice.ledger.active_style_notes()} == {"shorter sentences"}

    # A writer-written note is never gated.
    with_writer = apply(state, Diff(ops=(AddStyleNote(note=StyleNote(text="no adverbs")),)))
    assert {n.text for n in with_writer.ledger.active_style_notes()} == {"no adverbs"}


async def test_mining_failure_is_an_empty_list_not_a_crash() -> None:
    from storygit.preference.signals import EditPair

    provider = MockProvider(["not json", "still not json"])
    router = Router({"gemini": provider, "groq": provider})
    pairs = [EditPair(proposal_id=ProposalId("p_1"), before="a", after="b")]
    assert await edit_mining.mine(router, pairs) == []


@pytest.mark.slow
def test_semantically_identical_rules_collapse() -> None:
    rules = ["shorter sentences", "keep the sentences short", "more sensory detail"]
    deduped = edit_mining.dedupe(rules, threshold=0.7)
    assert len(deduped) == 2, f"expected two distinct rules, got {deduped}"
    assert "more sensory detail" in deduped


# --- bandit -------------------------------------------------------------------


def test_thompson_sampling_concentrates_on_the_better_arm() -> None:
    rng = random.Random(5)
    true_rate = {Arm.safe: 0.4, Arm.explore: 0.7}
    bandit = ThompsonBandit(seed=1)
    history: list[tuple[Arm, float]] = []

    for _ in range(400):
        arm = bandit.choose()
        reward = 1.0 if rng.random() < true_rate[arm] else 0.0
        bandit.update(arm, reward)
        history.append((arm, reward))

    assert bandit.state.mean_explore() > bandit.state.mean_safe()
    assert bandit.state.pulls["explore"] > bandit.state.pulls["safe"] * 2, (
        "the sampler must spend most of its pulls on the better arm"
    )

    regret = pseudo_regret_curve([arm for arm, _ in history], true_rate)
    assert all(b >= a for a, b in pairwise(regret)), (
        "pseudo-regret is non-decreasing by construction"
    )
    # Sublinear: the second half accrues much less regret than the first.
    first_half = regret[199]
    second_half = regret[-1] - regret[199]
    assert second_half < first_half * 0.5, (
        f"regret is not flattening: {first_half:.1f} then {second_half:.1f}"
    )


def test_the_bandit_composes_a_shown_set_and_falls_back_honestly() -> None:
    bandit = ThompsonBandit(seed=2)
    shown, arm = bandit.compose([0, 1, 2, 3, 4], [4, 3], k=3)
    assert len(shown) == 3
    if arm is Arm.explore:
        assert shown[:2] == [0, 1] and shown[2] in (3, 4)
    else:
        assert shown == [0, 1, 2]

    # Nothing surprising left to show: the arm reported is safe, so the bandit is never
    # credited for exploration that did not happen.
    shown, arm = bandit.compose([0, 1, 2], [0, 1], k=3)
    assert shown == [0, 1, 2] and arm is Arm.safe
    assert bandit.compose([], [], k=3) == ([], Arm.safe)


def test_bandit_state_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    state = BanditState().update(Arm.explore, 1.0).update(Arm.safe, 0.0)
    path = tmp_path / "bandit.json"
    state.save(path)
    assert BanditState.load(path) == state
    assert BanditState.load(tmp_path / "missing.json") == BanditState()


# --- pretraining --------------------------------------------------------------


def test_the_prior_beats_uniform_on_a_fresh_writers_first_decisions() -> None:
    prior = pretrain.fit_prior(n_personas=8, per_persona=200, seed=0)
    result = pretrain.evaluate_prior(prior, n_decisions=10, seed=99)
    assert result["prior"] > result["uniform"], (
        f"pretraining must earn its place: prior {result['prior']:.2f} "
        f"vs uniform {result['uniform']:.2f}"
    )
    assert result["lift"] > 0.02


def test_proto_personas_are_deterministic_and_structured() -> None:
    first = pretrain.make_personas(4, seed=0)
    assert [p.weights for p in first] == [p.weights for p in pretrain.make_personas(4, seed=0)]
    # Nobody prefers contradictions.
    assert all(p.weights["continuity"] > 0 for p in first)


# --- the layer ----------------------------------------------------------------


def test_every_learner_degrades_gracefully_at_zero_data(fixture: Fixture) -> None:
    layer = PreferenceLayer()
    assert layer.score([]) is None or layer.score([]) == []
    assert layer.state.weights.weights == BTWeights.uniform().weights
    assert layer.state.voice.is_trained is False
    assert layer.exemplars.is_empty
    assert layer.state.bandit.mean_safe() == 0.5

    reader = SignalReader(SignalStore(fixture.repo.conn))
    state = layer.learn(reader, {})
    assert state.pairs_seen == 0
    assert state.weights.weights == BTWeights.uniform().weights


def test_a_disabled_layer_contributes_nothing(fixture: Fixture) -> None:
    layer = PreferenceLayer(enabled=False)
    assert layer.score([object()]) is None
    assert layer.compose([0, 1, 2, 3], [3], k=3) == ([0, 1, 2], Arm.safe)
    reader = SignalReader(SignalStore(fixture.repo.conn))
    assert layer.learn(reader, {}) is layer.state


def test_the_layer_fits_from_recorded_comparisons(fixture: Fixture) -> None:
    store = SignalStore(fixture.repo.conn)
    features: dict[str, FeatureVector] = {}
    for index in range(12):
        winner, loser = ProposalId(f"p_w{index}"), ProposalId(f"p_l{index}")
        features[str(winner)] = vector(judge_specificity=0.9, length=0.2)
        features[str(loser)] = vector(judge_specificity=0.1, length=0.8)
        store.record(
            Signal(
                kind=SignalKind.accept,
                proposal_id=winner,
                shown_with=(loser,),
                node_id=NodeId("n_x"),
            )
        )

    layer = PreferenceLayer(seed=0)
    state = layer.learn(SignalReader(store), features)

    assert state.pairs_seen == 12
    top = dict(state.weights.top_weights(10))
    assert top["judge_specificity"] > top["length"], (
        "the head learned that this writer chooses specificity over length"
    )
    assert state.weights.probability(features["p_w0"], features["p_l0"]) > 0.6


def test_preference_state_round_trips_per_branch(fixture: Fixture) -> None:
    store = PreferenceStateStore(fixture.repo.conn)
    assert store.load("main") == PreferenceState()

    layer = PreferenceLayer(seed=0)
    layer.reward(Arm.explore, 1.0)
    layer.save(store, "main")

    reloaded = PreferenceLayer.load(store, "main")
    assert reloaded.state.bandit.mean_explore() > 0.5
    assert store.load("what-if") == PreferenceState(), (
        "taste learned on a what-if branch must not leak back to main"
    )
    assert store.branches() == ["main"]


def test_state_summary_is_renderable(fixture: Fixture) -> None:
    summary = PreferenceState().summary()
    assert summary["pairs_seen"] == 0
    assert summary["voice_trained"] is False
    assert summary["bandit_safe"] == 0.5
    assert len(summary["top_weights"]) == 4


async def test_turning_the_layer_off_reproduces_chunk_3_exactly(fixture: Fixture) -> None:
    """The no-preference ablation must be identical, not merely similar.

    An ablation that changes two things at once measures nothing, so this compares the
    engine with a disabled preference layer against the selector called directly with no
    quality override, on the same seeds and the same canned responses.
    """
    from storygit.agents.propose import Proposer
    from storygit.agents.schemas import Level
    from storygit.domain.ids import IdGenerator
    from storygit.engine import Engine
    from storygit.selection.select import CandidateSelector, SelectionConfig, Selector

    payload = {
        "title": "t",
        "what_happens": "w",
        "audience_learns": "",
        "audience_feels": "",
        "location": "",
        "time": "",
        "produces": [],
        "consumes": [],
        "threads_touched": [],
        "new_characters": [],
        "rationale": "r",
        "delta_summary": ["d"],
    }
    config = SelectionConfig(
        n=4, k=2, selector=Selector.topk_temperature, use_judge=False, use_dial=False
    )

    def make_router() -> Router:
        provider = MockProvider(lambda req: canned(dict(payload, title=f"beat {req.sample_index}")))
        return Router({"gemini": provider, "groq": provider})

    baseline_router = make_router()
    baseline = await CandidateSelector(
        Proposer(baseline_router, IdGenerator(seed=42)), baseline_router, config
    ).select(fixture.repo.state(), Level.beat, target_node_id=fixture.scene)

    engine = Engine(
        fixture.repo,
        make_router(),
        ids=IdGenerator(seed=42),
        selection=config,
        use_nli=False,
        preference=PreferenceLayer(enabled=False),
    )
    ablated = await engine.propose_at(Level.beat, node_id=fixture.scene)

    assert [c.proposal.id for c in ablated] == [c.proposal.id for c in baseline]
    assert [c.axis_key for c in ablated] == [c.axis_key for c in baseline]
    assert [c.selected for c in ablated] == [c.selected for c in baseline]
    assert [c.effective_quality for c in ablated] == [c.effective_quality for c in baseline]
    assert engine.preference.state == PreferenceState(), "a disabled layer learns nothing"


async def test_an_enabled_layer_with_no_data_also_defers_to_the_judge(
    fixture: Fixture,
) -> None:
    """A brand new writer must not have their ranking skewed by an unfitted head."""
    from storygit.agents.schemas import Level
    from storygit.domain.ids import IdGenerator
    from storygit.engine import Engine
    from storygit.selection.select import SelectionConfig, Selector

    payload = {
        "title": "t",
        "what_happens": "w",
        "audience_learns": "",
        "audience_feels": "",
        "location": "",
        "time": "",
        "produces": [],
        "consumes": [],
        "threads_touched": [],
        "new_characters": [],
        "rationale": "r",
        "delta_summary": ["d"],
    }
    provider = MockProvider(lambda req: canned(dict(payload, title=f"b{req.sample_index}")))
    engine = Engine(
        fixture.repo,
        Router({"gemini": provider, "groq": provider}),
        ids=IdGenerator(seed=42),
        selection=SelectionConfig(
            n=3, k=2, selector=Selector.topk_temperature, use_judge=False, use_dial=False
        ),
        use_nli=False,
        preference=PreferenceLayer(enabled=True),
    )
    candidates = await engine.propose_at(Level.beat, node_id=fixture.scene)
    assert len(candidates) == 3
    assert engine.preference.state.pairs_seen == 0
    # Features were still captured, so the very first accept produces usable training data.
    assert len(engine._features) == 3
