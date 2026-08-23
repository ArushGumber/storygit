"""The evaluation harness: personas, injection, metrics, ablations, and the recorder.

Everything here is offline. The live runs are the deliverable of chunk 5, but nothing in
the harness itself should need a network to be tested — a metrics module you can only
exercise by spending quota is a metrics module nobody checks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from eval import ablations, ceiling, costing, inject, metrics, offline, personas, probe
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
from storygit.preference.features import BASE_FEATURES, CRITERION_SLOTS, FeatureVector
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
    fixture: Fixture, *, router: Router | None = None, stream: str = "eval-test", **kwargs: object
) -> tuple[Engine, MockProvider]:
    """An engine whose every call returns the same canned beat.

    Args:
        fixture: The story to run against.
        router: Share a router (and therefore one call log) across two engines, the way
            an evaluation invocation shares one across four personas.
        stream: Id-generator stream, so two engines over one repository cannot collide.
        **kwargs: Passed through to the engine.
    """

    def reply(req: object) -> object:
        # Purpose-aware, because a beat payload does not validate as a prose proposal and
        # the run silently stops at the prose level otherwise -- which is why nothing
        # exercised the prose path until the voice model turned out never to have trained.
        purpose = getattr(req, "purpose", "")
        index = getattr(req, "sample_index", 0)
        if purpose.startswith("propose.prose"):
            return canned(
                {
                    "text": (
                        f"'Not this one,' she said, and meant it. He counted {index + 1} "
                        "guards and kept walking."
                    ),
                    "rationale": "short sentences",
                    "delta_summary": ["writes the beat"],
                }
            )
        if purpose.startswith("extract"):
            return canned(
                {"facts": [], "new_characters": [], "threads_opened": [], "threads_touched": []}
            )
        return canned(dict(BEAT, title=f"b{index}"))

    provider = MockProvider(reply)
    router = router or Router({"gemini": provider, "groq": provider})
    engine = Engine(
        fixture.repo,
        router,
        ids=IdGenerator(seed=3, stream=stream),
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
            max(
                persona._raw_score(reference_features(rng, criteria=len(persona.criteria)))
                for _ in range(BEST_OF)
            )
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
    engine_two, _ = mock_engine(fixture, router=engine.router, stream="eval-test-2")
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


@pytest.mark.asyncio
async def test_a_rejected_set_gets_exactly_one_informed_retry(fixture: Fixture) -> None:
    """A writer who dislikes all three asks again; they do not close the tool.

    The retry is what puts the system's own thesis under test — the rejection is already
    a turned-down direction in the ledger and a negative exemplar, so the second set is
    informed by the first. Exactly one: a writer who rejects twice means it, and an
    unbounded loop would burn a free tier on one stubborn decision.
    """
    engine, _ = mock_engine(fixture)
    proposals: list[str] = []
    accept_on_retry = get("the Serialist").model_copy(update={"noise": 0.0})

    original = engine.propose_at

    async def counting(level, **kwargs: object):  # type: ignore[no-untyped-def]
        proposals.append(f"{level.value}:{kwargs.get('node_id')}")
        return await original(level, **kwargs)  # type: ignore[arg-type]

    engine.propose_at = counting  # type: ignore[method-assign]

    # A bar nothing can clear: every set is rejected, so the retry always fires and the
    # second rejection has to stand.
    impossible = accept_on_retry.model_copy(update={"accept_quantile": 1.0, "edit_quantile": 1.0})
    personas._THRESHOLDS[impossible.name] = (2.0, 2.0)
    try:
        log = await run_writer(
            impossible,
            engine,
            episodes=1,
            scenes_per_episode=1,
            beats_per_scene=1,
            seed=5,
            use_llm_edits=False,
        )
    finally:
        personas._THRESHOLDS.pop(impossible.name, None)

    assert all(a.kind == "reject" for a in log.actions), "the bar was meant to be unclearable"
    assert log.actions, "the run produced no decisions"
    # Every rejected set is retried exactly once, so decisions arrive in same-level pairs
    # and never in threes.
    assert len(log.actions) % 2 == 0, "a rejection without its retry, or a retry of a retry"
    levels = [a.level for a in log.actions]
    assert all(levels[i] == levels[i + 1] for i in range(0, len(levels), 2)), levels
    # And never a third attempt at the same place: two proposals per node, no more.
    from collections import Counter

    assert max(Counter(proposals).values()) == 2, Counter(proposals)


@pytest.mark.asyncio
async def test_a_writer_who_accepts_on_the_retry_gets_what_they_asked_for(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of the retry: rejecting once must not end the decision.

    Scripted so the first set scores below the bar and the second above it, which is the
    case the retry exists for — the writer said "not these", the next set was informed by
    that, and the run continues instead of stopping.
    """
    engine, _ = mock_engine(fixture)
    persona = get("the Serialist")
    accept_at, _ = persona.thresholds()
    seen: list[int] = []

    def scripted(self: object, features: object, rng: object) -> float:
        seen.append(1)
        # Three candidates per set: the first set is under the bar, the second over it.
        return 0.0 if len(seen) <= 3 else accept_at + 0.05

    monkeypatch.setattr(personas.Persona, "score", scripted)
    log = await run_writer(
        persona,
        engine,
        episodes=1,
        scenes_per_episode=1,
        beats_per_scene=1,
        seed=5,
        use_llm_edits=False,
    )

    kinds = [a.kind for a in log.actions]
    assert kinds[0] == "reject", "the first set was scripted below the bar"
    assert kinds[1] == "accept", "the informed retry was scripted above it"
    assert kinds.count("reject") == 1, "one rejection, one retry, then on with the run"


def _synthetic_probe(seed: int = 3) -> tuple[probe.ProbeSet, dict[str, float]]:
    """A probe set and a hidden weight vector, both drawn from the same seed."""
    import random as _random

    rng = _random.Random(seed)
    hidden = {name: rng.uniform(-1.0, 1.0) for name in BASE_FEATURES}
    points = [
        probe.ProbePoint(
            source=f"synthetic/writer {i}",
            level="beat",
            features=tuple({name: rng.random() for name in BASE_FEATURES} for _ in range(3)),
        )
        for i in range(10)
    ]
    return probe.ProbeSet(points=tuple(points)), hidden


def test_probe_agreement_rises_as_the_head_sees_more_true_signal() -> None:
    """The probe is only a learning curve if it actually tracks learning.

    Fit the same head on progressively more comparisons drawn from a known hidden vector
    and the frozen probe set must reward it. The probe set never changes between
    measurements, which is the whole point: nothing here can move because a task got
    harder.
    """
    import random as _random

    from storygit.preference.bt_head import fit

    probe_set, hidden = _synthetic_probe()
    rng = _random.Random(11)

    # The persona's preference is expressed over every feature, but the probe's label
    # deliberately reads only the nine whose meaning is the same for every writer -- the
    # criterion slots are positional and the probe is cross-persona. Labelling the
    # training pairs the same way is what makes this a test of the *instrument* rather
    # than a test of how much of a writer's taste happens to live outside their own
    # criteria.
    readable = {k: v for k, v in hidden.items() if k not in CRITERION_SLOTS}

    def pair() -> tuple[FeatureVector, FeatureVector]:
        a = FeatureVector(values={n: rng.random() for n in BASE_FEATURES})
        b = FeatureVector(values={n: rng.random() for n in BASE_FEATURES})
        sa = sum(readable[k] * v for k, v in a.values.items() if k in readable)
        sb = sum(readable[k] * v for k, v in b.values.items() if k in readable)
        return (a, b) if sa >= sb else (b, a)

    pairs = [pair() for _ in range(160)]
    curve = [
        probe.agreement(probe_set.points, fit(pairs[:n], l2=0.5), hidden)["tau"]
        for n in (0, 10, 40, 160)
    ]
    assert curve[-1] > curve[0] + 0.2, f"the probe did not reward learning: {curve}"
    assert curve[-1] > 0.5, f"a head fitted on 160 true pairs should rank well: {curve}"
    # Monotone-ish from the point where the fit has data, and no further. The first step
    # is allowed to go *down*: curve[0] is an unfitted uniform head, and an all-positive
    # equal-weight vector is already a fair ranker, while a head fitted on ten pairs is a
    # small-sample regression that can be worse than doing nothing. The generated
    # pretraining curve shows the same dip (uniform: 0.795 at zero comparisons, 0.685 at
    # five), so a guard that forbade it here would be asserting something the deterministic
    # tier contradicts.
    assert all(curve[i + 1] >= curve[i] - 0.1 for i in range(1, len(curve) - 1)), curve


def test_a_persona_is_never_probed_on_its_own_decisions() -> None:
    """Leakage would make the probe measure memorisation instead of generalisation."""
    points = probe.ProbeSet(
        points=(
            probe.ProbePoint(source="full/the Serialist", level="beat"),
            probe.ProbePoint(source="full/the Minimalist", level="beat"),
        )
    )
    kept = points.for_persona("the Serialist")
    assert [p.source for p in kept] == ["full/the Minimalist"]


def test_the_committed_probe_fixture_is_leak_free_and_usable() -> None:
    """Every persona must have points to be probed on, and none of them its own."""
    probe_set = probe.ProbeSet.load()
    assert len(probe_set.points) >= 8, "too few points to average over"
    for persona in PERSONAS:
        kept = probe_set.for_persona(persona)
        assert kept, f"{persona} has no probe points left after excluding its own"
        assert all(persona not in p.source for p in kept)
    for point in probe_set.points:
        assert len(point.features) >= 2, "a probe point needs a ranking to score"
        assert all(set(f) >= set(BASE_FEATURES) for f in point.features)


def test_the_recovery_ceiling_is_deterministic_and_bounded() -> None:
    """A recovery number without a ceiling is a number without a scale.

    Same inputs, same answer, and the answer has to sit where a correlation can: the
    ceiling is what the estimator achieves when the model is correct by construction, so
    it is well below 1.0 at these sample sizes and well above chance.
    """
    import random as _random

    rng = _random.Random(4)
    matrices = [
        [{name: rng.random() for name in BASE_FEATURES} for _ in range(3)] for _ in range(25)
    ]
    first = ceiling.ceiling_for(matrices, noise=0.08, seeds=40)
    second = ceiling.ceiling_for(matrices, noise=0.08, seeds=40)
    assert first == second, "the ceiling must be seeded"
    assert 0.0 < first["mean"] < 1.0, first
    assert first["n_decisions"] == 25.0
    assert first["n_pairs"] > 0.0


def test_more_decisions_raise_the_ceiling() -> None:
    """If the ceiling did not rise with sample size it would not be a sample-size story."""
    import random as _random

    rng = _random.Random(5)

    def matrices(n: int) -> list[list[dict[str, float]]]:
        return [[{k: rng.random() for k in BASE_FEATURES} for _ in range(3)] for _ in range(n)]

    small = ceiling.ceiling_for(matrices(8), noise=0.08, seeds=40)["mean"]
    large = ceiling.ceiling_for(matrices(120), noise=0.08, seeds=40)["mean"]
    assert large > small + 0.05, f"small={small:.3f} large={large:.3f}"


def test_a_noisier_writer_has_a_lower_ceiling() -> None:
    """Decision noise is the other half of what caps recovery, and it must show."""
    import random as _random

    rng = _random.Random(6)
    matrices = [
        [{name: rng.random() for name in BASE_FEATURES} for _ in range(3)] for _ in range(40)
    ]
    quiet = ceiling.ceiling_for(matrices, noise=0.02, seeds=40)["mean"]
    loud = ceiling.ceiling_for(matrices, noise=0.40, seeds=40)["mean"]
    assert quiet > loud, f"quiet={quiet:.3f} loud={loud:.3f}"


def test_the_chunk_seven_costing_is_arithmetic_and_never_calls_anything() -> None:
    """The projection exists precisely so that no metered call has to be made to get it."""
    summary = {
        "call_summary": {"prompt_tokens": 4_000_000, "completion_tokens": 500_000},
        "runs": [{"decisions": 100}],
    }
    totals = costing.totals_from(summary)
    full = costing.project(totals, "anthropic/claude-3.5-sonnet")
    # 4M prompt at $3/M plus 0.5M completion at $15/M.
    assert full["prompt_usd"] == pytest.approx(12.0)
    assert full["completion_usd"] == pytest.approx(7.5)
    assert full["total_usd"] == pytest.approx(19.5)
    assert full["usd_per_decision"] == pytest.approx(0.195)

    half = costing.project(totals, "anthropic/claude-3.5-sonnet", fraction=0.5)
    assert half["total_usd"] == pytest.approx(9.75)

    body = costing.render(summary, cap_usd=12.0)
    assert "No OpenRouter call was made" in body
    assert "half only" in body, "a run over the cap must say so rather than just printing it"


@pytest.mark.asyncio
async def test_replaying_the_probe_makes_no_provider_calls(fixture: Fixture) -> None:
    """The probe runs after every episode, so it has to be free or it would not run.

    Asserted against the call log rather than by inspection: a probe that quietly
    regenerated a candidate would be both expensive and no longer a *frozen* decision.
    """
    engine, _ = mock_engine(fixture)
    probe_set, hidden = _synthetic_probe()
    before = engine.router.summary()["calls"]
    reading = probe.agreement(
        probe_set.points,
        engine.preference.state.weights,
        hidden,
        learner=engine.preference.state.voice,
    )
    assert engine.router.summary()["calls"] == before, "the probe called a provider"
    assert reading["points"] == float(len(probe_set.points))
    assert -1.0 <= reading["tau"] <= 1.0
    assert 0.0 <= reading["top1"] <= 1.0


def test_the_probe_points_discriminate_between_weight_vectors() -> None:
    """A point every weight vector gets right measures nothing.

    The first fixture was sampled uniformly at random and scored a fitted prior and an
    untrained uniform head *identically* on all four personas — because in a set where one
    candidate is better on every feature, the answer does not depend on the weights. Points
    are now chosen for the opposite property.
    """
    probe_set = probe.ProbeSet.load()
    scores = [probe.discrimination(list(p.features), seed=20260824) for p in probe_set.points]
    assert min(scores) > 0.25, f"a point no weight vector disagrees on: {min(scores):.2f}"


def test_a_probe_reading_carries_its_references() -> None:
    """A tau of 0.93 says nothing without knowing what an uninformed head scores."""
    probe_set, hidden = _synthetic_probe()
    from storygit.preference.bt_head import BTWeights

    out = probe.reading(
        list(probe_set.points),
        BTWeights.uniform(),
        hidden,
        baselines={"uniform": BTWeights.uniform()},
    )
    assert "tau_uniform" in out and "top1_uniform" in out
    assert out["tau"] == pytest.approx(out["tau_uniform"]), "same head, same reading"


def test_a_feature_that_never_varies_is_reported_as_unexercised() -> None:
    """A constant column gets no gradient, so the head keeps whatever the prior said.

    Harmless in session — it cannot reorder anything the writer sees — and precisely why it
    goes unnoticed. It stops being harmless the moment the head is applied off-distribution,
    which is what the cross-persona probe does. Reported per run so the claim is checkable
    in the artifact.
    """
    matrices = [
        [{"a": 0.1, "b": 0.5, "c": 0.0}, {"a": 0.9, "b": 0.5, "c": 0.0}],
        [{"a": 0.4, "b": 0.5, "c": 0.0}, {"a": 0.2, "b": 0.5, "c": 0.0}],
    ]
    assert metrics.unexercised_features(matrices) == ["b", "c"]
    assert metrics.unexercised_features([]) == []


def test_retry_rescues_are_counted_from_the_decision_sequence() -> None:
    """Reporting that runs stopped truncating is weaker than counting what was rescued."""
    levels = ["beat", "beat", "scene", "beat", "beat", "prose"]
    kinds = ["reject", "accept", "accept", "reject", "reject", "accept"]
    assert metrics.retry_rescues(levels, kinds) == {
        "rejected_sets": 2,
        "rescued": 1,
        "stood": 1,
    }
    assert metrics.retry_rescues([], []) == {"rejected_sets": 0, "rescued": 0, "stood": 0}


def test_the_invocation_call_summary_is_the_sum_of_its_runs() -> None:
    """Per-run summaries are deltas now, so the whole is a sum rather than the last one.

    Reading the last run's figure as the invocation total is the same mistake in reverse
    as billing every run for the whole invocation.
    """
    from eval.run import _combined_call_summary

    logs = [
        RunLog(persona="a", call_summary={"calls": 10, "prompt_tokens": 100, "cache_hits": 2}),
        RunLog(persona="b", call_summary={"calls": 30, "prompt_tokens": 300, "cache_hits": 6}),
    ]
    combined = _combined_call_summary(logs)
    assert combined["calls"] == 40
    assert combined["prompt_tokens"] == 400
    assert combined["cache_hit_rate"] == pytest.approx(0.2)


def test_every_probe_point_names_a_run_that_is_never_itself_probed() -> None:
    """Provenance is the leak defence a reader can check without running anything.

    The cross-persona filter is code; this is the label. A point sourced from a
    configuration the probe is later replayed against would be controlled for rather than
    excluded, and "unknown" would leave a reader unable to tell which.
    """
    for point in probe.ProbeSet.load().points:
        config, _, persona = point.source.partition("/")
        assert config == "probesample", f"probe point from {config!r}, which is probed"
        assert persona, "a probe point must name the writer it came from"


@pytest.mark.asyncio
async def test_the_writer_sometimes_writes_the_beat_itself(fixture: Fixture) -> None:
    """The hand-write path is the only source of anchors for the voice model.

    Generated prose the writer merely accepted is a positive example, not an anchor: the
    voice model is meant to learn what *this writer* sounds like. Until the personas used
    the "write it myself" path, no live run ever trained it and voice_cosine sat at a flat
    0.5 in every recorded run.
    """
    engine, _ = mock_engine(fixture)
    persona = get("the Minimalist").model_copy(update={"hand_write_probability": 1.0})
    log = await run_writer(
        persona,
        engine,
        episodes=1,
        scenes_per_episode=1,
        beats_per_scene=1,
        seed=4,
        use_llm_edits=False,
    )
    prose = [a for a in log.actions if a.level == "prose"]
    assert prose, "the run never reached prose level"
    assert all(a.kind == "write" for a in prose), [a.kind for a in prose]
    # And a hand-write is not an acceptance: the writer used none of the three suggestions.
    assert log.acceptance_rate < 1.0


def test_dialogue_is_counted_whatever_quotes_the_model_uses() -> None:
    """This feature read exactly 0.0 across every run because it only knew double quotes.

    The model in use writes speech in single quotes, so prose that was a third dialogue
    scored zero, and the feature showed up as "never varied" as though the stories had no
    dialogue in them. A silently constant feature is worse than an absent one: it occupies
    a weight and teaches the head nothing.
    """
    from storygit.preference.features import dialogue_ratio

    assert dialogue_ratio("'Drop the book,' she hissed. He said nothing.") == pytest.approx(0.5)
    assert dialogue_ratio('"Drop the book," she hissed. He said nothing.') == pytest.approx(0.5)
    assert dialogue_ratio("“Drop it,” she said.") == pytest.approx(1.0)
    # An apostrophe is not dialogue, and neither is a scare-quoted word.
    assert dialogue_ratio("He didn't move. She wouldn't either.") == 0.0
    assert dialogue_ratio("They called it 'art'. Nobody agreed.") == 0.0
    # A single quote wrapping a real utterance still counts, even when the sentence
    # splitter has eaten its closing mark.
    assert dialogue_ratio("She said, 'Go home now.' He didn't.") == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_polishing_prose_produces_the_pair_edit_direction_needs(
    fixture: Fixture,
) -> None:
    """A hand-write gives the voice model an anchor; only an edit gives it a direction.

    Accepting a candidate in the writer's own words is the one action that records a
    before and an after, which is what the edit-direction feature is computed from. No
    persona ever did it at prose level, so the feature was a flat 0.5 in every run.
    """
    engine, _ = mock_engine(fixture)
    persona = get("the Minimalist").model_copy(
        update={"prose_polish_probability": 1.0, "hand_write_probability": 0.0}
    )
    # A bar anything clears, so this measures where a decision is routed rather than
    # whether the mock's canned prose happens to satisfy a calibrated persona.
    personas._THRESHOLDS[persona.name] = (0.0, 0.0)
    try:
        log = await run_writer(
            persona,
            engine,
            episodes=1,
            scenes_per_episode=1,
            beats_per_scene=1,
            seed=6,
            use_llm_edits=False,
        )
    finally:
        personas._THRESHOLDS.pop(persona.name, None)
    prose = [a for a in log.actions if a.level == "prose"]
    assert prose, "the run never reached prose level"
    assert all(a.kind == "edit" for a in prose), [a.kind for a in prose]
    # An edit is still an acceptance: the writer used the candidate, in their own words.
    assert log.acceptance_rate == 1.0
    # And plan levels are untouched -- writers do not polish an outline the same way.
    assert all(a.kind != "edit" for a in log.actions if a.level in ("episode", "scene"))


def test_the_probe_can_see_the_features_it_is_asked_about() -> None:
    """An instrument sampled from a system inherits that system's blind spots.

    A feature with no spread *within* a probe point cannot change how that point is
    ranked, so no head difference on that weight is visible through it. The first fixture
    had exactly zero within-point spread on all five features the shrinkage flag touches,
    which made the A/B provably unable to answer its own question — both variants scored
    identically to three decimals.

    This asserts the fixture can see most of the space. Two criterion slots are
    structurally zero (the personas define two criteria each), so they are exempt and the
    limit is reported rather than hidden.
    """
    probe_set = probe.ProbeSet.load()
    structurally_absent = {"criterion_3", "criterion_4"}
    blind = []
    for name in BASE_FEATURES:
        if name in structurally_absent:
            continue
        spread = max(
            (max(f[name] for f in p.features) - min(f[name] for f in p.features))
            for p in probe_set.points
        )
        if spread <= 1e-9:
            blind.append(name)
    assert blind == [], f"the probe cannot distinguish any head on: {', '.join(blind)}"


def test_the_evaluation_refuses_the_metered_provider_unless_it_was_asked_for() -> None:
    """An enabled flag is not consent to spend; naming the model is.

    The metered provider being enabled is normally a mistake, and an evaluation that
    quietly spent money because a flag was left set somewhere would be a bad way to find
    out. Both directions are refused: enabled without a model named, and a model named
    without the provider enabled, which would otherwise run to completion producing
    nothing because every call is refused at the provider.
    """
    import asyncio

    import pytest
    from eval import ablations, run
    from eval.personas import PERSONAS

    from storygit.config import Settings

    # Constructed explicitly rather than subclassed: Settings reads a workspace .env, so a
    # subclass default would be silently overridden by whatever is on disk, and the test
    # would depend on the machine it runs on. It would also have reached a real provider.
    enabled = Settings(openrouter_enabled="true")
    disabled = Settings(openrouter_enabled="false")
    assert enabled.openrouter_is_enabled and not disabled.openrouter_is_enabled

    def go(settings: Settings, strong: str | None, tmp: Path) -> None:
        original = run.get_settings
        run.get_settings = lambda: settings  # type: ignore[assignment]
        try:
            asyncio.run(
                run.run_matrix(
                    configs=[ablations.get("smoke")],
                    personas=[PERSONAS["the Serialist"]],
                    results=tmp,
                    max_calls=1,
                    strong_model=strong,
                )
            )
        finally:
            run.get_settings = original  # type: ignore[assignment]

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(RuntimeError, match="no --strong-model was given"):
            go(enabled, None, Path(d))
        with pytest.raises(RuntimeError, match="needs OPENROUTER_ENABLED=true"):
            go(disabled, "some/model", Path(d))
