"""Driving the real engine with a simulated writer.

The persona sees exactly what a human would: the candidates the engine chose to show, and
nothing else. It never sees the feature vectors the head is fitted on, the hidden weights
are never passed to the engine, and the edit model is told the persona's *style* rather
than its objective. Those three constraints are what stop the evaluation from measuring
itself; the first is asserted in the tests.

Every decision, every candidate set, and every resulting snapshot is written to a
``RunLog``, which is both the metrics input and the raw material for a replayable gallery
session.

Runs checkpoint after every episode. Rate limits *will* hit during a 15-episode run, and a
run that cannot resume is a run that never finishes.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from eval import probe
from eval.personas import ForbiddenMove, Persona
from storygit.agents.schemas import Level
from storygit.domain.ids import IdGenerator, NodeId, ProposalId, SnapshotId
from storygit.domain.nodes import NodeType
from storygit.engine import Engine
from storygit.preference.bt_head import BTWeights
from storygit.preference.features import FeatureVector
from storygit.providers.base import (
    LLMRequest,
    Message,
    ProviderError,
    RateLimited,
    Role,
)
from storygit.providers.base import Role as _Role
from storygit.providers.router import Router
from storygit.selection.select import Candidate, candidate_text

del _Role


class Action(BaseModel):
    """What the simulated writer did with one candidate set.

    Attributes:
        node_id: The node the candidates were for.
        level: Plan level.
        shown: Proposal ids that were shown, in order.
        axes: Axis label per shown candidate.
        scores: The persona's private score per shown candidate.
        chosen: The proposal acted on, if any.
        kind: ``accept``, ``edit``, ``write`` (the writer discarded the suggestion and
            wrote the beat themselves), or ``reject``.
        veto: The forbidden move that forced a rejection, if any.
        edit_distance: Normalized token edit distance when the writer rewrote it.
        flags_shown: Continuity flags on the chosen candidate.
        snapshot_id: The snapshot the action produced.
        stale_marked: How many nodes propagation marked.
        features: One feature vector per shown candidate, in the same order as ``scores``.
            Recorded because the first autopsy of a truncated run wanted to decompose a
            rejection into the features that caused it and could not: the log held the
            scalar score and nothing behind it, so the answer needed a rerun.
        texts: The shown candidates, in the same order. The held-out probe is built from
            these: replaying a frozen decision means re-scoring the same words with a
            later head, and the two learner-dependent features need the words.
    """

    model_config = ConfigDict(frozen=True)

    node_id: NodeId | None = None
    level: str = ""
    shown: tuple[ProposalId, ...] = ()
    axes: tuple[str, ...] = ()
    scores: tuple[float, ...] = ()
    chosen: ProposalId | None = None
    kind: str = "reject"
    veto: str | None = None
    edit_distance: float = 0.0
    flags_shown: int = 0
    hard_flags_shown: int = 0
    snapshot_id: SnapshotId | None = None
    stale_marked: int = 0
    features: tuple[dict[str, float], ...] = ()
    texts: tuple[str, ...] = ()


class RunLog(BaseModel):
    """Everything one simulated writer did over one run.

    Attributes:
        persona: Persona name.
        config: The configuration the run used, as a plain dict.
        actions: Every decision, in order.
        episodes_completed: How many episodes finished.
        call_summary: The provider call log summary at the end of the run.
        preference_summary: What the preference layer had learned by the end.
        hidden_weights: The persona's true weights, recorded *after* the run for scoring
            recoverability. The engine never saw them.
        probe: Held-out probe agreement after each episode, in order. The deconfounded
            learning curve: the same frozen decisions re-ranked by a later head, so
            nothing in it can move because the task got harder.
        errors: Anything that went wrong, for honesty about partial runs.
        waited_seconds: Total time spent backing off from rate limits. Reported because a
            run that took an hour of which fifty minutes was waiting is a different fact
            about the system than one that took an hour of work.
        rate_limit_waits: How many times the run had to wait.
    """

    model_config = ConfigDict(frozen=True)

    persona: str
    config: dict[str, Any] = Field(default_factory=dict)
    actions: tuple[Action, ...] = ()
    episodes_completed: int = 0
    call_summary: dict[str, Any] = Field(default_factory=dict)
    preference_summary: dict[str, Any] = Field(default_factory=dict)
    hidden_weights: dict[str, float] = Field(default_factory=dict)
    probe: tuple[dict[str, float], ...] = ()
    errors: tuple[str, ...] = ()
    waited_seconds: float = 0.0
    rate_limit_waits: int = 0

    def save(self, path: Path | str) -> Path:
        """Write the log as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2))
        return target

    @staticmethod
    def load(path: Path | str) -> RunLog:
        """Read a log back."""
        return RunLog.model_validate(json.loads(Path(path).read_text()))

    @property
    def acceptance_rate(self) -> float:
        """Fraction of candidate sets the writer took something from.

        A hand-write counts against it: the writer looked at three suggestions and used
        none of them, which is the system failing to be useful even though a beat got
        written. Counting it as an acceptance would let the rate rise by the writer doing
        more of the work.
        """
        if not self.actions:
            return 0.0
        return sum(1 for a in self.actions if a.kind in ("accept", "edit")) / len(self.actions)


def token_edit_distance(before: str, after: str) -> float:
    """Normalized Levenshtein distance over tokens.

    Tokens rather than characters because the question is how much the writer *rewrote*,
    not how much they retyped, and a synonym swap should cost the same as any other word
    change.

    Args:
        before: The model's version.
        after: The writer's version.

    Returns:
        Distance in ``[0, 1]``; 0 means identical, 1 means nothing survived.
    """
    a, b = before.split(), after.split()
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (token_a != token_b),
                )
            )
        previous = current
    return previous[-1] / max(len(a), len(b))


class SimulatedWriter:
    """A persona making decisions against a real engine.

    Attributes:
        persona: Who is writing.
        engine: The engine under test.
        rng: Decision noise source, seeded.
    """

    def __init__(
        self,
        persona: Persona,
        engine: Engine,
        *,
        seed: int = 0,
        use_llm_edits: bool = True,
    ) -> None:
        """Create the simulated writer.

        Args:
            persona: The persona to simulate.
            engine: The engine to drive. Must not have been told the persona's weights.
            seed: RNG seed, so a run reproduces exactly.
            use_llm_edits: Whether edits go through the cheap model. Off makes edits a
                deterministic truncation, which is what the offline tests use.
        """
        self.persona = persona
        self.engine = engine
        self.rng = random.Random(seed)
        self._use_llm_edits = use_llm_edits

    async def decide(self, candidates: list[Candidate]) -> Action:
        """Score a candidate set and act on it.

        The decision rule: veto anything crossing a forbidden move; otherwise take the
        best-scoring candidate if it clears the accept threshold, rewrite it if it clears
        the edit threshold, and reject everything otherwise.

        Args:
            candidates: What the engine chose to show. Unselected candidates are ignored,
                exactly as a human would ignore what they were never shown.

        Returns:
            The action taken.
        """
        shown = [c for c in candidates if c.selected] or candidates[:3]
        if not shown:
            return Action(kind="reject")

        features = self.engine.preference.features_for(shown)
        scores = [self.persona.score(f, self.rng) for f in features]
        texts = [candidate_text(c.proposal) for c in shown]

        base = Action(
            node_id=shown[0].proposal.target_node_id,
            level=shown[0].proposal.level.value,
            shown=tuple(c.proposal.id for c in shown),
            axes=tuple(c.axis_label for c in shown),
            scores=tuple(round(s, 4) for s in scores),
            features=tuple({k: round(v, 4) for k, v in f.values.items()} for f in features),
            texts=tuple(texts),
            flags_shown=sum(len(c.flags) for c in shown),
            hard_flags_shown=sum(1 for c in shown for f in c.flags if f.is_hard),
        )

        accept_at, edit_at = self.persona.thresholds()
        order = sorted(range(len(shown)), key=lambda i: -scores[i])

        # Sometimes the writer reads the suggestion and writes the beat themselves. This
        # is the interface's "write it myself" path, and it is the only thing that
        # produces *anchors* for the voice model: prose the writer authored, as opposed to
        # generated prose they merely approved. Before this existed the voice model was
        # untrained in every live run ever recorded, and voice_cosine sat at a flat 0.5.
        if (
            shown[0].proposal.level is Level.prose
            and self.persona.hand_write_probability > 0.0
            and self.rng.random() < self.persona.hand_write_probability
        ):
            best = order[0]
            if self.persona.vetoes(texts[best]) is None:
                return await self._hand_write(base, shown, best, texts[best])

        # Writers polish prose. They do not polish an outline the same way, so this is a
        # prose-level disposition -- and it is the only action that produces a before/after
        # pair, which the edit-direction feature is computed from.
        polish = (
            shown[0].proposal.level is Level.prose
            and self.rng.random() < self.persona.prose_polish_probability
        )
        for index in order:
            veto = self.persona.vetoes(texts[index])
            if veto is not None:
                continue
            if scores[index] >= accept_at:
                if polish:
                    return await self._edit(base, shown, index, texts[index])
                return await self._accept(base, shown, index)
            if scores[index] >= edit_at:
                return await self._edit(base, shown, index, texts[index])
            break

        vetoed = next(
            (self.persona.vetoes(t) for t in texts if self.persona.vetoes(t) is not None), None
        )
        return await self._reject(base, shown, order[0], vetoed)

    async def _accept(self, base: Action, shown: list[Candidate], index: int) -> Action:
        chosen = shown[index]
        result = await self.engine.accept(
            chosen.proposal.id, shown_with=tuple(c.proposal.id for c in shown)
        )
        self._maybe_lock(chosen)
        return base.model_copy(
            update={
                "chosen": chosen.proposal.id,
                "kind": "accept",
                "snapshot_id": result.snapshot_id,
                "stale_marked": len(result.marks),
            }
        )

    async def _edit(self, base: Action, shown: list[Candidate], index: int, text: str) -> Action:
        chosen = shown[index]
        rewritten = await self._rewrite(text)
        result = await self.engine.edit(chosen.proposal.id, rewritten)
        self._maybe_lock(chosen)
        return base.model_copy(
            update={
                "chosen": chosen.proposal.id,
                "kind": "edit",
                "edit_distance": round(token_edit_distance(text, rewritten), 4),
                "snapshot_id": result.snapshot_id,
                "stale_marked": len(result.marks),
            }
        )

    async def _hand_write(
        self, base: Action, shown: list[Candidate], index: int, text: str
    ) -> Action:
        """The writer discards the proposal and writes the beat in their own words."""
        chosen = shown[index]
        beat_id = chosen.proposal.target_node_id
        if beat_id is None:
            return await self._accept(base, shown, index)
        written = await self._rewrite(text)
        result = await self.engine.write_beat(beat_id, written)
        return base.model_copy(
            update={
                "chosen": chosen.proposal.id,
                "kind": "write",
                "edit_distance": round(token_edit_distance(text, written), 4),
                "snapshot_id": result.snapshot_id,
                "stale_marked": len(result.marks),
            }
        )

    async def _reject(
        self,
        base: Action,
        shown: list[Candidate],
        index: int,
        veto: ForbiddenMove | None,
    ) -> Action:
        chosen = shown[index]
        reason = (
            f"no {veto.value.replace('_', ' ')}"
            if veto is not None
            else "none of these are what I want here"
        )
        self.engine.reject(chosen.proposal.id, reason=reason)
        for other in shown:
            if other.proposal.id != chosen.proposal.id:
                self.engine.signals.record(_rejection_signal(self.engine.branch, other.proposal.id))
        return base.model_copy(
            update={
                "chosen": chosen.proposal.id,
                "kind": "reject",
                "veto": veto.value if veto is not None else None,
            }
        )

    def _maybe_lock(self, candidate: Candidate) -> None:
        """Lock the node sometimes, per the persona's disposition."""
        node_id = candidate.proposal.target_node_id
        if node_id is None or self.rng.random() >= self.persona.lock_probability:
            return
        state = self.engine.state()
        children = state.children.get(node_id, ())
        if children:
            self.engine.lock(children[-1])

    async def _rewrite(self, text: str) -> str:
        """Rewrite a candidate towards the persona's style.

        The model is told the persona's *style instruction*, never its weights. If it were
        told the objective, the engine could learn the answer from text the persona
        produced, and the evaluation would be measuring a leak.
        """
        if not self._use_llm_edits:
            words = text.split()
            return " ".join(words[: max(8, self.persona.target_words // 3)])
        request = LLMRequest(
            messages=(
                Message(
                    role=Role.system,
                    content=(
                        "You are a writer revising a draft to your own taste. Return only "
                        "the revised prose: no commentary, no preamble."
                    ),
                ),
                Message(
                    role=Role.user,
                    content=f"{self.persona.edit_instruction()}\n\nDRAFT\n{text}",
                ),
            ),
            purpose="simulate.edit",
            temperature=0.4,
            max_tokens=900,
        )
        try:
            response = await self.engine.router.complete(request)
        except ProviderError:
            words = text.split()
            return " ".join(words[: max(8, self.persona.target_words // 3)])
        return response.text.strip() or text


def _rejection_signal(branch: str, proposal_id: ProposalId) -> Any:
    """A reject signal for a candidate that was shown but not acted on."""
    from storygit.store.signals import Signal, SignalKind

    return Signal(kind=SignalKind.reject, branch=branch, proposal_id=proposal_id)


async def run_writer(
    persona: Persona,
    engine: Engine,
    *,
    episodes: int = 3,
    scenes_per_episode: int = 2,
    beats_per_scene: int = 2,
    seed: int = 0,
    use_llm_edits: bool = True,
    checkpoint: Path | None = None,
    on_episode: Any | None = None,
    max_wait_seconds: float = 900.0,
    max_retries: int = 12,
    probe_set: probe.ProbeSet | None = None,
    config_name: str = "",
) -> RunLog:
    """Drive a full run: episodes, scenes, beats, and prose.

    Args:
        persona: Who is writing.
        engine: The engine under test.
        episodes: How many episodes to develop.
        scenes_per_episode: Scenes per episode.
        beats_per_scene: Beats per scene.
        seed: RNG seed.
        use_llm_edits: Whether edits go through the cheap model.
        checkpoint: Directory to write a partial log to after every episode. Rate limits
            *will* hit during a long run; a run that cannot resume never finishes.
        on_episode: Optional callback ``(index, RunLog)`` after each episode.
        max_wait_seconds: Total backoff budget. A per-minute quota clears in seconds; a
            daily one does not, and there is no point waiting hours for it.
        max_retries: How many times one step may wait and retry.
        probe_set: The held-out probe. Replayed after every episode, costing nothing,
            because the acceptance curve on its own cannot separate a head that learned
            from a task that got harder.
        config_name: Recorded in the log so a probe point can name where it came from,
            which is how leakage stays checkable by eye rather than only by code.

    Returns:
        The complete run log. A run cut short by provider errors still returns everything
        it managed, with the failure recorded in ``errors`` — a partial run reported
        honestly is worth more than a crash.
    """
    writer = SimulatedWriter(persona, engine, seed=seed, use_llm_edits=use_llm_edits)
    actions: list[Action] = []
    probe_points = probe_set.for_persona(persona.name) if probe_set is not None else []
    probe_curve: list[dict[str, float]] = []
    # References the probe reading is read against. `uniform` is a head that has learned
    # nothing; `prior` is the population prior every writer starts from. The head begins at
    # `prior` by construction, so without these a curve that starts high and stays flat
    # cannot be told apart from a measurement that is not measuring.
    probe_baselines = (
        {
            "uniform": BTWeights.uniform(),
            "prior": engine.preference.state.weights,
        }
        if probe_points
        else {}
    )
    errors: list[str] = []
    completed = 0
    waited = 0.0
    waits = 0
    # One router serves every persona in an invocation, so the log has to be scoped to
    # this run or the fourth writer is billed for the first three.
    started_at = engine.router.mark()

    engine.set_dial(persona.dial)
    for note in persona.style_notes:
        engine.add_style_note(note)
    # Creation order fixes each criterion's feature slot, so this loop is also what makes
    # the persona's per-criterion hidden weights line up with the head's.
    for name, description in persona.criteria:
        engine.add_criterion(name, description)

    def snapshot_log() -> RunLog:
        return RunLog(
            persona=persona.name,
            config={
                "name": config_name,
                "episodes": episodes,
                "scenes_per_episode": scenes_per_episode,
                "beats_per_scene": beats_per_scene,
                "seed": seed,
                "selection": engine.selector.config.model_dump(mode="json"),
                "preference_enabled": engine.preference.enabled,
                "use_nli": engine.use_nli,
            },
            actions=tuple(actions),
            episodes_completed=completed,
            call_summary=engine.router.summary(since=started_at),
            preference_summary=engine.preference.state.summary(),
            hidden_weights=persona.weights,
            probe=tuple(probe_curve),
            errors=tuple(errors),
            waited_seconds=round(waited, 1),
            rate_limit_waits=waits,
        )

    async def step(level: Level, node_id: NodeId | None, *, retry: bool = False) -> None:
        """One decision, waiting out rate limits rather than abandoning the run.

        A free-tier per-minute quota clears in seconds, and the provider tells us exactly
        how many. Abandoning a 33-decision run because one call needed a four-second wait
        is the difference between an evaluation that finishes and one that does not -- and
        it is how the first four-persona run lost three of its four writers.

        Args:
            level: What to propose.
            node_id: The parent to propose under.
            retry: Whether this call *is* the one informed re-proposal, in which case a
                second rejection stands.
        """
        nonlocal waited, waits
        for attempt in range(max_retries + 1):
            try:
                candidates = await engine.propose_at(level, node_id=node_id)
            except RateLimited as exc:
                if attempt == max_retries or waited >= max_wait_seconds:
                    raise
                pause = max(1.0, min(float(exc.retry_after), 120.0))
                waited += pause
                waits += 1
                print(f"      rate limited; waiting {pause:.0f}s (total {waited:.0f}s)")
                await asyncio.sleep(pause)
                continue
            if not candidates:
                errors.append(f"no candidates parsed at {level.value}")
                return
            action = await writer.decide(candidates)
            actions.append(action)
            if action.kind == "reject" and not retry:
                # A writer who dislikes all three says "not these" and asks again; they do
                # not close the tool. The rejection is already in the ledger as a
                # turned-down direction and in the exemplar pool as a negative, so the
                # second set is *informed* by the first -- which is the system's own
                # thesis (feedback in, better proposal out) put under test rather than
                # asserted. Exactly one retry: a writer who rejects twice means it, and
                # an unbounded loop would burn a free tier on one stubborn decision.
                await step(level, node_id, retry=True)
            return

    try:
        for episode_index in range(episodes):
            state = engine.state()
            await step(Level.episode, state.root_id)

            episodes_now = engine.state().nodes_of_type(NodeType.episode)
            if len(episodes_now) <= episode_index:
                errors.append(f"episode {episode_index + 1} was not accepted; stopping")
                break
            episode_id = episodes_now[-1].id

            for _ in range(scenes_per_episode):
                await step(Level.scene, episode_id)
                scenes = engine.state().children.get(episode_id, ())
                if not scenes:
                    continue
                scene_id = scenes[-1]
                for _ in range(beats_per_scene):
                    await step(Level.beat, scene_id)
                    beats = engine.state().children.get(scene_id, ())
                    if beats:
                        await step(Level.prose, beats[-1])

            completed += 1
            if probe_points:
                # Zero provider calls: the candidates are frozen, and only the two
                # learner-dependent features are recomputed.
                reading = probe.reading(
                    probe_points,
                    engine.preference.state.weights,
                    persona.weights,
                    learner=engine.preference.state.voice,
                    baselines=probe_baselines,
                )
                reading["decisions"] = float(len(actions))
                reading["episode"] = float(completed)
                probe_curve.append(reading)
            if checkpoint is not None:
                snapshot_log().save(checkpoint / f"{_slug(persona.name)}.partial.json")
            if on_episode is not None:
                on_episode(episode_index, snapshot_log())
    except ProviderError as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    if waits:
        print(f"      waited {waited:.0f}s across {waits} rate limits")
    return snapshot_log()


def _slug(name: str) -> str:
    """Filesystem-safe persona name."""
    return name.lower().replace(" ", "_").replace("'", "")


def new_engine(
    repo: Any,
    router: Router,
    *,
    seed: int,
    **kwargs: Any,
) -> Engine:
    """Build an engine for a run, with a per-run id stream so runs cannot collide.

    Args:
        repo: The repository.
        router: Model access.
        seed: Run seed.
        **kwargs: Passed through to :class:`storygit.engine.Engine`.

    Returns:
        The engine.
    """
    return Engine(repo, router, ids=IdGenerator(seed=seed, stream="eval"), **kwargs)


def features_of(engine: Engine) -> dict[str, FeatureVector]:
    """The feature vectors the engine captured while showing candidates."""
    return dict(engine._features)
