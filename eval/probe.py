"""The held-out replay probe: a learning curve task difficulty cannot touch.

The problem it solves. Acceptance rate over a run measures two things at once. The writer's
bar does not move, but the task gets harder every episode — each new candidate has to stay
consistent with more established facts, more open threads, and more accumulated rules — so
a flat or falling acceptance curve is not evidence the preference layer failed, and a
rising one would not be evidence that it worked. The curve confounds *has the head learned*
with *has the job got harder*, and no statistic computed on that curve separates them.

The fix is a different experiment, not a better statistic. Freeze a set of decision points.
Replay them at intervals during a run, ranking with the head as it stands at that moment,
and score the ranking against what the persona would actually have chosen. The probe set
never changes, so difficulty is held fixed by construction and any movement in the curve is
the head.

Two properties make it honest:

**Cross-persona sourcing.** A probe point sampled from persona P's own run is a decision P
already trained on, and replaying it would measure memorisation. Every point is therefore
drawn from *other* personas' runs, and :func:`for_persona` enforces it.

**No provider calls.** Each point stores the candidate-intrinsic features that were already
computed when the decision happened — judge sub-scores, continuity, length, dialogue ratio,
the writer-criterion slots. Replay recomputes only the parts that depend on the *learner's*
current state (voice cosine, edit-direction projection) and re-ranks. Nothing is generated,
so a probe costs milliseconds and can run after every episode.

    python -m eval.probe --build       # rebuild the fixture from recorded runs
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from storygit.preference.bt_head import BTWeights
from storygit.preference.features import BASE_FEATURES, CRITERION_SLOTS, FeatureVector

FIXTURE = Path(__file__).parent / "fixtures" / "probe_points.json"
"""The committed probe set. Regenerating it changes the measuring stick, so it is a
deliberate, reviewable act rather than something a run does on its own."""

LEARNER_FEATURES = ("voice_cosine", "edit_direction")
"""Features that depend on the learner's state rather than on the candidate.

Everything else about a candidate was fixed the moment it was generated. These two are the
learner looking at it, so replay recomputes them and leaves the rest frozen — which is what
keeps a probe point the *same* question at episode one and episode three.
"""


class ProbePoint(BaseModel):
    """One frozen decision: the candidates shown, and the features that were computed.

    Attributes:
        source: The run this was taken from, so leakage is checkable by eye.
        level: What was being proposed.
        texts: The candidate texts, in the order they were shown. Needed to recompute the
            learner-dependent features; never sent anywhere.
        features: One feature vector per candidate, as recorded at decision time.
    """

    model_config = ConfigDict(frozen=True)

    source: str
    level: str
    texts: tuple[str, ...] = ()
    features: tuple[dict[str, float], ...] = ()


class ProbeSet(BaseModel):
    """The committed probe fixture."""

    model_config = ConfigDict(frozen=True)

    points: tuple[ProbePoint, ...] = ()

    def for_persona(self, persona: str) -> list[ProbePoint]:
        """Every point *not* drawn from this persona's own run.

        Args:
            persona: The persona about to be probed.

        Returns:
            The leak-free subset.
        """
        return [p for p in self.points if not p.source.endswith(persona)]

    @staticmethod
    def load(path: Path | str = FIXTURE) -> ProbeSet:
        """Read the committed fixture."""
        return ProbeSet.model_validate_json(Path(path).read_text())

    def save(self, path: Path | str = FIXTURE) -> Path:
        """Write the fixture."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2))
        return target


def kendall_tau(a: list[float], b: list[float]) -> float:
    """Rank correlation between two scorings of the same items.

    Args:
        a: One scoring.
        b: The other.

    Returns:
        Tau-a in ``[-1, 1]``; 0.0 when there are fewer than two items or no discordance
        is possible.
    """
    n = len(a)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            left = a[i] - a[j]
            right = b[i] - b[j]
            if left * right > 0:
                concordant += 1
            elif left * right < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def agreement(
    points: list[ProbePoint],
    head: BTWeights,
    hidden: dict[str, float],
    *,
    learner: Any | None = None,
) -> dict[str, float]:
    """Score the head's ranking against the persona's, over the frozen probe set.

    Args:
        points: The probe points to replay.
        head: The learner's current Bradley-Terry weights.
        hidden: The persona's hidden weight vector. Never visible to the head.
        learner: Optional object with ``score(texts)`` and ``direction_scores(texts)``
            (the preference layer's voice model), used to refresh the two
            learner-dependent features *for the head's ranking only*. Omitted, the frozen
            values are used, which is what the deterministic offline test wants.

            The persona's label is always computed from the recorded vectors. Refreshing
            both sides would move the ground truth between episode 1 and episode 3 for any
            persona weighting those axes, driven by the learner being measured --- which is
            exactly the leakage the held-out design exists to prevent, arriving through a
            side door. The head must be scored on what it can see now; the truth must not
            move.

    Returns:
        ``{"tau": ..., "top1": ..., "points": ...}``. Tau is the mean rank correlation over
        the set; top1 is how often the head's first choice is the persona's first choice.
    """
    if not points:
        return {"tau": 0.0, "top1": 0.0, "points": 0.0}

    taus: list[float] = []
    hits = 0
    for point in points:
        recorded = [FeatureVector(values=dict(v)) for v in point.features]
        vectors = [dict(v) for v in point.features]
        if learner is not None and point.texts:
            voice = learner.score(list(point.texts))
            direction = learner.direction_scores(list(point.texts))
            for index, values in enumerate(vectors):
                values["voice_cosine"] = float(voice[index])
                values["edit_direction"] = float(direction[index])
        built = [FeatureVector(values=v) for v in vectors]
        learned = [head.score(f) for f in built]
        truth = [_hidden_score(f, hidden) for f in recorded]
        taus.append(kendall_tau(learned, truth))
        if learned.index(max(learned)) == truth.index(max(truth)):
            hits += 1
    return {
        "tau": sum(taus) / len(taus),
        "top1": hits / len(points),
        "points": float(len(points)),
    }


def reading(
    points: list[ProbePoint],
    head: BTWeights,
    hidden: dict[str, float],
    *,
    learner: Any | None = None,
    baselines: dict[str, BTWeights] | None = None,
) -> dict[str, float]:
    """One probe measurement, with the references that make it readable.

    A tau of 0.93 says nothing on its own. The head starts from a population prior that
    already ranks well, so the interesting quantity is not the level but the distance from
    the references: how far above an uninformed head it sits, and whether it moves away
    from the prior it started at. Without those, a probe curve that starts high and stays
    flat is indistinguishable from a probe that is not measuring anything.

    Args:
        points: The probe points.
        head: The learner's current weights.
        hidden: The persona's hidden weights.
        learner: The voice model, for the two learner-dependent features.
        baselines: Named reference heads, each scored over the same points.

    Returns:
        The head's reading, plus ``tau_<name>`` and ``top1_<name>`` per baseline.
    """
    out = agreement(points, head, hidden, learner=learner)
    for name, reference in (baselines or {}).items():
        scored = agreement(points, reference, hidden, learner=learner)
        out[f"tau_{name}"] = scored["tau"]
        out[f"top1_{name}"] = scored["top1"]
    return out


def _hidden_score(features: FeatureVector, hidden: dict[str, float]) -> float:
    """What the persona would privately think of this candidate, noiselessly.

    The criterion slots are excluded, and that exclusion is the whole point of this
    function existing separately from ``BTWeights.score``. Criterion features are
    *positional*: ``criterion_1`` means whatever the first criterion this writer defined
    happens to be, and the personas' criteria are disjoint. Every point this label is
    computed on came from a *different* persona's run, so ``criterion_1`` on a Maximalist
    point carries the judge's score for *interiority* while the Serialist's hidden weight
    on that slot means *pull*. Scoring one through the other is not noise, it is a
    category error, and it lands on the two features that carry the largest hidden weight
    for every persona.

    The cross-persona holdout is what deconfounds learning from task difficulty; positional
    slots are what let a writer weight their own criteria independently. The two are in
    direct tension and this is where the tension is resolved: the probe reads the nine
    features whose meaning is the same for every writer, and the criterion slots --- which
    the head still learns and still uses --- are simply not part of the held-out label.
    """
    return sum(
        weight * features.values.get(name, 0.0)
        for name, weight in hidden.items()
        if name not in CRITERION_SLOTS
    )


def discrimination(features: list[dict[str, float]], *, trials: int = 200, seed: int = 0) -> float:
    """How much a decision point depends on *which* weight vector is asked.

    A candidate set where one option is better on every feature has the same answer under
    every weight vector with positive weights, so replaying it measures nothing: a head
    that has learned the writer and a head that has learned nothing both get it right. The
    probe is only a measurement over points where the answer turns on the *shape* of the
    weights.

    Args:
        features: One feature dict per candidate.
        trials: Random weight vectors to poll.
        seed: Draw seed, so fixture selection is reproducible.

    Returns:
        The fraction of trials that do **not** pick the most popular winner — 0.0 when
        every weight vector agrees, approaching ``1 - 1/k`` when they are split evenly.
    """
    if len(features) < 2:
        return 0.0
    rng = random.Random(seed)
    vectors = [FeatureVector(values=dict(f)) for f in features]
    winners = []
    for _ in range(trials):
        weights = {name: rng.uniform(-1.0, 1.0) for name in BASE_FEATURES}
        scores = [_hidden_score(v, weights) for v in vectors]
        winners.append(scores.index(max(scores)))
    top = max(winners.count(i) for i in set(winners))
    return 1.0 - top / len(winners)


DISCRIMINATION_FLOOR = 0.40
"""Below this a decision point is dropped: too many weight vectors agree on it.

A floor, not a ranking. Taking the *most* discriminating points instead — which the first
version did — concentrates the fixture on near-ties, where a hair of noise flips the
winner, and so maximises the variance of every reading it is used for. The floor removes
the points that measure nothing; the draw above it keeps the rest representative.
"""


def build(
    runs: list[dict[str, Any]],
    *,
    per_run: int = 6,
    seed: int = 20260824,
    floor: float = DISCRIMINATION_FLOOR,
) -> ProbeSet:
    """Sample probe points from recorded runs: a discrimination floor, then a level draw.

    Two rules, in order.

    **The floor.** A candidate set where one option is better on every feature has the same
    answer under every weight vector, so no head can get it wrong and replaying it measures
    nothing. Points below ``floor`` on :func:`discrimination` are dropped. The first fixture
    had no floor and scored a fitted prior and an untrained uniform head identically on
    every persona.

    **The stratified draw.** Above the floor, points are drawn per level rather than by
    taking the maximum. Argmax would fill the fixture with near-ties — the points where the
    answer is most sensitive to the weights *and* to noise — which is precisely how to make
    every reading as noisy as possible. Levels are drawn round-robin so a fixture is not
    all beats: the plan levels and the prose level exercise different features.

    Neither rule ever consults a head, fitted or otherwise, which is what keeps the
    instrument independent of what it measures.

    Args:
        runs: Loaded RunLog dictionaries.
        per_run: How many decisions to take from each run.
        seed: Sampling seed, so the fixture is reproducible from the same runs.
        floor: Minimum discrimination to be eligible.

    Returns:
        The probe set, ready to commit.
    """
    rng = random.Random(seed)
    points: list[ProbePoint] = []
    for run in runs:
        eligible: dict[str, list[dict[str, Any]]] = {}
        for action in run.get("actions", ()):
            features = action.get("features", ())
            if len(features) < 2 or not all(set(f) >= set(BASE_FEATURES) for f in features):
                continue
            if discrimination(list(features), seed=seed) < floor:
                continue
            eligible.setdefault(action.get("level", ""), []).append(action)

        for bucket in eligible.values():
            rng.shuffle(bucket)
        # Round-robin over levels, so the draw is spread rather than concentrated in
        # whichever level happens to have the most decisions.
        levels = sorted(eligible)
        taken: list[dict[str, Any]] = []
        while len(taken) < per_run and any(eligible[level] for level in levels):
            for level in levels:
                if len(taken) >= per_run:
                    break
                if eligible[level]:
                    taken.append(eligible[level].pop())

        for action in taken:
            points.append(
                ProbePoint(
                    source=f"{run.get('config', {}).get('name', 'unknown')}/{run['persona']}",
                    level=action.get("level", ""),
                    texts=tuple(action.get("texts", ())),
                    features=tuple(dict(f) for f in action["features"]),
                )
            )
    return ProbeSet(points=tuple(points))


def main() -> None:
    """Rebuild the fixture from whatever runs are on disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="rebuild the fixture")
    parser.add_argument("--runs", default="eval/results/runs", help="directory of RunLog JSON")
    parser.add_argument(
        "--glob",
        default="probesample__*.json",
        help="which runs to sample from; the sampling run is never itself probed",
    )
    parser.add_argument("--per-run", type=int, default=6)
    args = parser.parse_args()
    if not args.build:
        probe = ProbeSet.load()
        print(f"{len(probe.points)} probe points from {len({p.source for p in probe.points})} runs")
        return
    runs = [
        json.loads(path.read_text())
        for path in sorted(Path(args.runs).glob(args.glob))
        if "partial" not in path.name
    ]
    if not runs:
        raise SystemExit(f"no runs matched {args.glob} in {args.runs}")
    probe = build(runs, per_run=args.per_run)
    path = probe.save()
    print(f"wrote {path}: {len(probe.points)} points from {len(runs)} runs")


if __name__ == "__main__":
    main()
