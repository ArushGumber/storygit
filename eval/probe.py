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
from storygit.preference.features import BASE_FEATURES, FeatureVector

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
            learner-dependent features. Omitted, the frozen values are used, which is what
            the deterministic offline test wants.

    Returns:
        ``{"tau": ..., "top1": ..., "points": ...}``. Tau is the mean rank correlation over
        the set; top1 is how often the head's first choice is the persona's first choice.
    """
    if not points:
        return {"tau": 0.0, "top1": 0.0, "points": 0.0}

    taus: list[float] = []
    hits = 0
    for point in points:
        vectors = [dict(v) for v in point.features]
        if learner is not None and point.texts:
            voice = learner.score(list(point.texts))
            direction = learner.direction_scores(list(point.texts))
            for index, values in enumerate(vectors):
                values["voice_cosine"] = float(voice[index])
                values["edit_direction"] = float(direction[index])
        built = [FeatureVector(values=v) for v in vectors]
        learned = [head.score(f) for f in built]
        truth = [_hidden_score(f, hidden) for f in built]
        taus.append(kendall_tau(learned, truth))
        if learned.index(max(learned)) == truth.index(max(truth)):
            hits += 1
    return {
        "tau": sum(taus) / len(taus),
        "top1": hits / len(points),
        "points": float(len(points)),
    }


def _hidden_score(features: FeatureVector, hidden: dict[str, float]) -> float:
    """What the persona would privately think of this candidate, noiselessly."""
    return sum(hidden.get(k, 0.0) * v for k, v in features.values.items())


def build(runs: list[dict[str, Any]], *, per_run: int = 3, seed: int = 20260824) -> ProbeSet:
    """Sample probe points from recorded runs.

    Args:
        runs: Loaded RunLog dictionaries.
        per_run: How many decisions to take from each.
        seed: Sampling seed, so the fixture is reproducible from the same runs.

    Returns:
        The probe set, ready to commit.
    """
    rng = random.Random(seed)
    points: list[ProbePoint] = []
    for run in runs:
        usable = [
            a
            for a in run.get("actions", ())
            if len(a.get("features", ())) >= 2
            and all(set(f) >= set(BASE_FEATURES) for f in a["features"])
        ]
        rng.shuffle(usable)
        for action in usable[:per_run]:
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
    parser.add_argument("--per-run", type=int, default=3)
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
