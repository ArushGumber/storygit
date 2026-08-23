"""The evaluation harness: personas, injection, metrics, ablations, and the recorder.

Everything here is offline. The live runs are the deliverable of chunk 5, but nothing in
the harness itself should need a network to be tested — a metrics module you can only
exercise by spending quota is a metrics module nobody checks.
"""

from __future__ import annotations

import pytest
from eval import ablations, inject, metrics, offline
from eval.gallery_record import Recorder, Session, replay
from eval.personas import PERSONAS, ForbiddenMove, get
from eval.simulate import RunLog, SimulatedWriter, run_writer, token_edit_distance
from tests.conftest import Fixture
from tests.mockprovider import MockProvider, canned

from storygit.agents.schemas import Level
from storygit.continuity import layer1
from storygit.domain.apply import apply
from storygit.domain.diff import Diff, DiffAuthor, UpdateFact
from storygit.domain.ids import IdGenerator
from storygit.engine import Engine
from storygit.graph.propagation import propagate_change
from storygit.preference.features import BASE_FEATURES, FeatureVector
from storygit.preference.layer import PreferenceLayer
from storygit.providers.router import Router
from storygit.selection.select import SelectionConfig, Selector

BEAT = {
    "title": "The offer",
    "what_happens": "The Warden offers Kael a way out of the city.",
    "audience_learns": "the Warden wants something from him",
    "audience_feels": "wary",
    "location": "Ashfall",
    "time": "night",
    "produces": [],
    "consumes": [],
    "threads_touched": [],
    "new_characters": [],
    "rationale": "Gives Kael something to refuse.",
    "delta_summary": ["the Warden makes an offer"],
}
TINY = SelectionConfig(
    n=2, k=2, selector=Selector.topk_temperature, use_judge=False, use_dial=False
)


def mock_engine(
    fixture: Fixture, *, router: Router | None = None, **kwargs: object
) -> tuple[Engine, MockProvider]:
    """An engine whose every call returns the same canned beat.

    Args:
        fixture: The story to run against.
        router: Share a router (and therefore one call log) across two engines, the way
            an evaluation invocation shares one across four personas.
        **kwargs: Passed through to the engine.
    """
    provider = MockProvider(lambda req: canned(dict(BEAT, title=f"b{req.sample_index}")))
    router = router or Router({"gemini": provider, "groq": provider})
    engine = Engine(
        fixture.repo,
        router,
        ids=IdGenerator(seed=3, stream="eval-test"),
        selection=TINY,
        use_nli=False,
        **kwargs,  # type: ignore[arg-type]
    )
    return engine, provider


# --- personas -----------------------------------------------------------------


def test_personas_are_defined_over_the_heads_feature_space() -> None:
    """The whole evaluation rests on this: same space, so recovery is measurable."""
    for persona in PERSONAS.values():
        assert set(persona.weights) == set(BASE_FEATURES), persona.name
    assert len(PERSONAS) == 4


def test_persona_decisions_are_deterministic_under_seed() -> None:
    import random

    persona = get("the Minimalist")
    features = FeatureVector(values=dict.fromkeys(BASE_FEATURES, 0.5))
    first = [persona.score(features, random.Random(1)) for _ in range(5)]
    second = [persona.score(features, random.Random(1)) for _ in range(5)]
    assert first == second


def test_forbidden_moves_are_always_vetoed() -> None:
    controller = get("the Controller")
    assert ForbiddenMove.kill_the_mentor in controller.forbidden
    assert controller.vetoes("The Warden dies at the gate.") is ForbiddenMove.kill_the_mentor
    assert controller.vetoes("A prophecy names him.") is ForbiddenMove.prophecy
    assert controller.vetoes("Kael walks to the harbour.") is None

    minimalist = get("the Minimalist")
    assert minimalist.vetoes("The Warden dies at the gate.") is None, (
        "one writer's absolute is not another's"
    )


def test_the_edit_instruction_never_leaks_the_hidden_weights() -> None:
    for persona in PERSONAS.values():
        instruction = persona.edit_instruction().lower()
        for feature in BASE_FEATURES:
            assert feature not in instruction, (
                f"{persona.name} leaks {feature} into the text the engine will learn from"
            )


# --- injection ----------------------------------------------------------------


def test_every_injected_class_actually_contradicts() -> None:
    for cls in inject.ContradictionClass:
        state, scenario = inject.build_scenario(classes=(cls,))
        if cls is inject.ContradictionClass.note:
            continue  # only layer 2 can decide this one; covered by the offline metrics
        flags = layer1.check_state(state)
        injected = {str(f) for f in scenario.injections[0].fact_ids}
        assert any({str(x) for x in flag.fact_ids} & injected for flag in flags), (
            f"{cls.value} was injected but layer 1 did not flag it"
        )


def test_a_clean_story_produces_no_flags() -> None:
    state, scenario = inject.clean_story(IdGenerator(seed=1, stream="t"))
    assert layer1.check_state(state) == [], (
        "a checker that flags a clean story has no usable precision"
    )
    assert len(scenario.beats) == 6


def test_the_stale_case_has_a_genuinely_undeclared_dependency() -> None:
    state, case = inject.build_stale_case()
    after = apply(
        state,
        Diff(
            ops=(
                UpdateFact(
                    fact_id=case.changed_fact,
                    fields={"object_entity": None, "object_text": "Kell"},
                ),
            ),
            author=DiffAuthor.human,
        ),
    )
    declared = {m.node_id for m in propagate_change(state, after)}
    assert declared == set(case.truly_affected)
    assert case.undeclared, "without an undeclared dependency the soft-edge sweep is unfair"
    assert not (declared & set(case.undeclared))
    assert not (declared & set(case.unaffected))


# --- metrics ------------------------------------------------------------------


def test_checker_recall_on_a_known_answer() -> None:
    class FakeInjection:
        def __init__(self, fact_ids: tuple[str, ...]) -> None:
            self.fact_ids = fact_ids

    class FakeFlag:
        def __init__(self, fact_ids: tuple[str, ...], layer: int = 1) -> None:
            self.fact_ids = fact_ids
            self.layer = layer

    injected = [FakeInjection((f"f{i}",)) for i in range(6)]
    flags = [FakeFlag((f"f{i}",)) for i in range(4)]
    result = metrics.checker_recall(injected, flags)
    assert result.recall == pytest.approx(4 / 6)
    assert result.precision == 1.0
    assert result.false_negatives == 2

    with_noise = [*flags, FakeFlag(("f_not_injected",))]
    noisy = metrics.checker_recall(injected, with_noise)
    assert noisy.false_positives == 1
    assert noisy.precision == pytest.approx(4 / 5)


def test_acceptance_and_edit_distance_curves() -> None:
    kinds = ["reject", "reject", "accept", "accept", "edit"]
    curve = metrics.acceptance_curve(kinds, window=5)
    assert curve[0] == 0.0
    assert curve[-1] == pytest.approx(3 / 5)
    assert curve[1] == 0.0 and curve[2] == pytest.approx(1 / 3)

    distances = [0.0, 0.0, 0.0, 0.0, 0.4]
    edits = metrics.edit_distance_curve(kinds, distances, window=5)
    assert edits[0] == 0.0, "no edits yet"
    assert edits[-1] == pytest.approx(0.4)


def test_token_edit_distance() -> None:
    assert token_edit_distance("a b c", "a b c") == 0.0
    assert token_edit_distance("a b c", "x y z") == 1.0
    assert token_edit_distance("a b c d", "a b c") == pytest.approx(0.25)
    assert token_edit_distance("", "") == 0.0
    assert token_edit_distance("a", "") == 1.0


def test_extraction_scores_match_loosely_on_the_object() -> None:
    expected = [("kael", "injury", "blistered hand"), ("mara", "possesses", "the token")]
    extracted = [
        ("Kael", "injury", "a blistered left hand"),
        ("Mara", "possesses", "the token"),
        ("Warden", "goal", "something invented"),
    ]
    result = metrics.extraction_scores(expected, extracted)
    assert result.recall == 1.0, "a rephrased object is a hit, not a miss"
    assert result.false_positives == 1
    assert result.precision == pytest.approx(2 / 3)


def test_stale_scores_and_weight_recovery() -> None:
    result = metrics.stale_scores(["a", "b", "z"], ["a", "b", "c"], ["z"])
    assert result.true_positives == 2 and result.false_positives == 1
    assert result.recall == pytest.approx(2 / 3)

    hidden = {"x": 1.0, "y": -1.0, "z": 0.0}
    assert metrics.weight_recovery({"x": 2.0, "y": -2.0, "z": 0.0}, hidden) == pytest.approx(1.0)
    assert metrics.weight_recovery({"x": -1.0, "y": 1.0, "z": 0.0}, hidden) == pytest.approx(-1.0)
    assert metrics.weight_recovery({"x": 1.0, "y": 1.0, "z": 1.0}, hidden) == 0.0


def test_mean_pairwise_distance() -> None:
    assert metrics.mean_pairwise_distance([[1.0, 0.0], [1.0, 0.0]]) == pytest.approx(0.0)
    assert metrics.mean_pairwise_distance([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(1.0)
    assert metrics.mean_pairwise_distance([[1.0, 0.0]]) == 0.0


def test_cost_per_action() -> None:
    summary = {"total_tokens": 1000, "cost_usd": 0.5, "calls": 20, "cache_hit_rate": 0.4}
    result = metrics.cost_per_action(summary, 10)
    assert result["tokens"] == 100.0 and result["usd"] == 0.05 and result["calls"] == 2.0
    assert metrics.cost_per_action(summary, 0)["tokens"] == 0.0


# --- ablations ----------------------------------------------------------------


def test_each_ablation_removes_exactly_one_thing() -> None:
    full = ablations.FULL
    for config in ablations.ABLATIONS:
        if config.name == "full":
            continue
        differences = [
            field
            for field in ("preference", "propagation", "checker", "selection", "soft_edges")
            if getattr(config, field) != getattr(full, field)
        ]
        # no_checker also disables NLI, which is part of the same component.
        assert len(differences) == 1, f"{config.name} changes {differences}"
        assert config.expectation, f"{config.name} has no stated expectation"


def test_ablation_configs_produce_distinct_engine_wirings(fixture: Fixture) -> None:
    from eval.run import build_engine

    provider = MockProvider([canned(BEAT)])
    router = Router({"gemini": provider, "groq": provider})

    off = build_engine(fixture.repo, router, ablations.NO_PREFERENCE, seed=1)
    assert off.preference.enabled is False

    on = build_engine(fixture.repo, router, ablations.FULL, seed=1)
    assert on.preference.enabled is True

    soft = build_engine(fixture.repo, router, ablations.SOFT_EDGES, seed=1)
    assert soft.edge_provider is not None
    assert on.edge_provider is None

    no_checker = build_engine(fixture.repo, router, ablations.NO_CHECKER, seed=1)
    assert no_checker.use_nli is False
    assert no_checker.selector.config.use_judge is False


def test_call_budget_estimates_are_sane() -> None:
    assert ablations.SMOKE.decisions_per_run < ablations.FULL.decisions_per_run
    assert ablations.SMOKE.calls_per_run < 60
    assert ablations.NO_DIAL.calls_per_run < ablations.FULL.calls_per_run


# --- simulation ---------------------------------------------------------------


async def test_a_tiny_run_completes_and_round_trips(fixture: Fixture, tmp_path) -> None:  # type: ignore[no-untyped-def]
    engine, _ = mock_engine(fixture)
    log = await run_writer(
        get("the Serialist"),
        engine,
        episodes=1,
        scenes_per_episode=1,
        beats_per_scene=1,
        seed=2,
        use_llm_edits=False,
        checkpoint=tmp_path,
    )
    assert log.persona == "the Serialist"
    assert log.actions, "the run produced no decisions"
    assert log.hidden_weights == get("the Serialist").weights

    path = log.save(tmp_path / "run.json")
    assert RunLog.load(path) == log
    assert (tmp_path / "the_serialist.partial.json").exists(), "checkpointed after episode 1"


async def test_the_persona_never_sees_more_than_the_writer_would(
    fixture: Fixture,
) -> None:
    """The engine must never be told the hidden weights, in any form."""
    engine, _ = mock_engine(fixture)
    persona = get("the Controller")
    writer = SimulatedWriter(persona, engine, seed=1, use_llm_edits=False)

    candidates = await engine.propose_at(Level.beat, node_id=fixture.scene)
    action = await writer.decide(candidates)

    assert set(action.shown) <= {c.proposal.id for c in candidates}
    assert len(action.shown) == sum(1 for c in candidates if c.selected) or action.shown
    # Nothing anywhere in the engine's reachable state carries the weights.
    assert not hasattr(engine, "persona")
    assert not hasattr(engine.preference, "hidden_weights")


async def test_a_forbidden_move_forces_a_rejection(fixture: Fixture) -> None:
    payload = dict(BEAT, what_happens="The Warden dies at the gate, and Kael watches.")
    provider = MockProvider([canned(payload)])
    router = Router({"gemini": provider, "groq": provider})
    engine = Engine(
        fixture.repo,
        router,
        ids=IdGenerator(seed=4, stream="veto"),
        selection=TINY,
        use_nli=False,
        preference=PreferenceLayer(enabled=False),
    )
    writer = SimulatedWriter(get("the Controller"), engine, seed=1, use_llm_edits=False)
    candidates = await engine.propose_at(Level.beat, node_id=fixture.scene)
    action = await writer.decide(candidates)

    assert action.kind == "reject"
    assert action.veto == ForbiddenMove.kill_the_mentor.value


# --- gallery ------------------------------------------------------------------


async def test_record_then_replay_is_identical_and_makes_no_calls(
    fixture: Fixture, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    engine, provider = mock_engine(fixture)
    recorder = Recorder(engine, "test_session", title="A test", summary="For the tests.")

    candidates = await engine.propose_at(Level.beat, node_id=fixture.scene)
    recorder.record(
        title="Propose a beat",
        note="Three labelled options.",
        candidates=candidates,
        node_id=fixture.scene,
        level="beat",
        intent="go on",
    )
    result = await engine.accept(candidates[0].proposal.id, shown_with=engine.shown())
    recorder.record(
        title="Accept the first",
        candidates=candidates,
        action="accept",
        chosen=candidates[0].proposal.id,
        result=result,
    )

    session = recorder.session()
    path = session.save(tmp_path)
    loaded = Session.load(path)
    assert loaded == session
    assert [s.title for s in loaded.steps] == ["Propose a beat", "Accept the first"]

    calls_before = len(provider.requests)
    replayed = replay(loaded, fixture.repo)
    assert len(provider.requests) == calls_before, "replay must make zero provider calls"
    assert len(replayed) == len(session.steps)
    assert replayed[0].tree, "replay reconstructs the plan tree from the snapshot"
    assert all(r.step.snapshot_id for r in replayed)


def test_recorded_candidates_keep_their_labels_and_flags(fixture: Fixture) -> None:
    from eval.gallery_record import shown_from

    from storygit.agents.propose import Proposal
    from storygit.continuity.flags import Flag, FlagKind, Severity
    from storygit.domain.ids import ProposalId
    from storygit.selection.select import Candidate

    candidate = Candidate(
        proposal=Proposal(
            id=ProposalId("p_1"),
            level=Level.beat,
            target_node_id=fixture.scene,
            diff=Diff(),
            rationale="because",
            delta_summary=("does a thing",),
        ),
        axis_key="raise_stakes",
        axis_label="raise the stakes",
        flags=(
            Flag(
                kind=FlagKind.contradiction,
                severity=Severity.hard,
                layer=1,
                message="conflict",
                established_by=fixture.beat_a,
            ),
        ),
        selected=True,
    )
    shown = shown_from([candidate])[0]
    assert shown.axis_label == "raise the stakes"
    assert shown.flags[0]["severity"] == "hard"
    assert shown.flags[0]["established_by"] == fixture.beat_a
    assert shown.selected is True


# --- offline metrics ----------------------------------------------------------


def test_offline_checker_metrics_are_deterministic_without_nli() -> None:
    first = offline.checker_layers(use_nli=False)
    second = offline.checker_layers(use_nli=False)
    assert first == second
    assert first["combined"]["recall"] >= 0.75, (
        "layer 1 alone must catch most of the injected classes"
    )
    assert first["false_positives_per_beat"] == 0.0


def test_offline_stale_sweep_reports_a_real_tradeoff() -> None:
    result = offline.stale_sweep()
    declared = result["points"][0]
    assert declared["precision"] == 1.0, "declared edges never guess"
    assert declared["recall"] < 1.0, "and cannot see the undeclared dependency"
    assert any(p["recall"] > declared["recall"] for p in result["points"][1:]), (
        "soft edges must be given a case where they can add something"
    )


def test_persona_thresholds_discriminate() -> None:
    """A threshold below the whole score distribution accepts everything and measures nothing.

    The first calibration did exactly that — 33 of 33 accepted, a flat acceptance curve —
    so the thresholds are now quantiles of each persona's own best-of-three score
    distribution, and this test asserts they land inside it rather than under it.
    """
    import random

    from eval.personas import BEST_OF, PERSONAS, reference_features

    for persona in PERSONAS.values():
        accept, edit = persona.thresholds()
        assert 0.0 < edit < accept < 1.0, persona.name

        rng = random.Random(7)
        draws = [
            max(persona._raw_score(reference_features(rng)) for _ in range(BEST_OF))
            for _ in range(2000)
        ]
        rate = sum(1 for d in draws if d >= accept) / len(draws)
        assert 0.02 < rate < 0.98, f"{persona.name} accepts {rate:.0%} of what it sees"
        assert abs(rate - (1 - persona.accept_quantile)) < 0.05, (
            f"{persona.name}: the realised rate should match the designed quantile"
        )


def test_personas_differ_in_how_picky_they_are() -> None:
    from eval.personas import PERSONAS

    quantiles = {name: p.accept_quantile for name, p in PERSONAS.items()}
    assert quantiles["the Controller"] > quantiles["the Serialist"], (
        "the Controller is defined as the hardest writer to satisfy"
    )
    assert len(set(quantiles.values())) == len(quantiles), "four distinct dispositions"


def test_thresholds_are_cached_and_deterministic() -> None:
    from eval.personas import get

    persona = get("the Minimalist")
    assert persona.thresholds() == persona.thresholds()


@pytest.mark.asyncio
async def test_a_run_is_billed_only_for_its_own_calls(fixture: Fixture, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """One router serves every persona in an invocation; the cost figures must not.

    Before this, each run snapshotted the running total at the end of its own run, so the
    fourth writer was charged for everything the first three did and the reported
    tokens-per-decision was inflated by up to four times.
    """
    engine, _ = mock_engine(fixture)
    first = await run_writer(
        get("the Serialist"),
        engine,
        episodes=1,
        scenes_per_episode=1,
        beats_per_scene=1,
        seed=2,
        use_llm_edits=False,
    )
    engine_two, _ = mock_engine(fixture, router=engine.router)
    second = await run_writer(
        get("the Minimalist"),
        engine_two,
        episodes=1,
        scenes_per_episode=1,
        beats_per_scene=1,
        seed=3,
        use_llm_edits=False,
    )

    whole_log = engine.router.summary()["calls"]
    assert first.call_summary["calls"] > 0 and second.call_summary["calls"] > 0
    assert second.call_summary["calls"] < whole_log, "the second run was billed for the first"
    assert first.call_summary["calls"] + second.call_summary["calls"] == whole_log


@pytest.mark.asyncio
async def test_every_decision_records_the_features_behind_its_scores(fixture: Fixture) -> None:
    """A score with no feature vector behind it cannot be autopsied without a rerun."""
    engine, _ = mock_engine(fixture)
    log = await run_writer(
        get("the Serialist"),
        engine,
        episodes=1,
        scenes_per_episode=1,
        beats_per_scene=1,
        seed=2,
        use_llm_edits=False,
    )
    for action in log.actions:
        assert len(action.features) == len(action.scores)
        for vector in action.features:
            assert set(vector) >= set(BASE_FEATURES), "a feature disappeared from the log"
