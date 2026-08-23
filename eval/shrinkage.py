"""The shrinkage A/B, decided by replay of recorded decisions rather than by a new run.

The question. Fitting regularizes towards the population prior, which is what makes ten
comparisons useful instead of noise. But a feature direction with no variance inside a
writer's candidate sets receives no gradient at all, so it simply *keeps* whatever the
prior asserted — and the cross-persona probe showed that costs accuracy off-distribution.
The alternative is to shrink those directions towards zero: no data, no opinion.

Why this is answerable for free. Per-decision feature vectors have been recorded since the
fix pass, so both variants can be refitted from the decisions that actually happened and
scored on the held-out probe with the same references. No provider calls, no new run, and
no opportunity to keep rerunning until one variant wins.

The decision rule was written into ``arush/logs/final_pass.md`` before this module produced
a number: ship the flag on by default *iff* mean probe tau improves across the archived
runs **and** the cold-start lift at five comparisons does not fall by more than its own
seed-to-seed noise.

    python -m eval.shrinkage
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from eval.personas import PERSONAS
from eval.probe import ProbeSet, agreement
from storygit.preference.bt_head import fit
from storygit.preference.features import FeatureVector
from storygit.preference.pretrain import evaluate_prior, fit_prior

RUNS = Path("eval/results/runs")
ARCHIVE = Path("eval/results/archive")


def _pairs_from_run(run: dict[str, Any]) -> list[tuple[FeatureVector, FeatureVector]]:
    """The comparisons a run's decisions imply: what was taken beat what was not.

    Args:
        run: A loaded RunLog.

    Returns:
        ``(winner, loser)`` pairs, one per shown-but-not-chosen candidate.
    """
    pairs: list[tuple[FeatureVector, FeatureVector]] = []
    for action in run.get("actions", ()):
        features = action.get("features", ())
        shown = action.get("shown", ())
        chosen = action.get("chosen")
        if action.get("kind") == "reject" or chosen is None or len(features) < 2:
            continue
        if len(shown) != len(features) or chosen not in shown:
            continue
        winner = FeatureVector(values=dict(features[shown.index(chosen)]))
        for index, vector in enumerate(features):
            if shown[index] != chosen:
                pairs.append((winner, FeatureVector(values=dict(vector))))
    return pairs


def replay(run: dict[str, Any], probe: ProbeSet, *, shrink: bool) -> dict[str, float]:
    """Refit one run's head under one variant and score it on the probe.

    Args:
        run: A loaded RunLog.
        probe: The held-out probe.
        shrink: Whether unidentified directions are pulled towards zero.

    Returns:
        The probe reading, plus how many weights the variant zeroed.
    """
    persona = run.get("persona", "")
    pairs = _pairs_from_run(run)
    prior = fit_prior()
    head = fit(pairs, prior=prior, l2=2.0, shrink_unidentified=shrink) if pairs else prior
    reading = agreement(probe.for_persona(persona), head, PERSONAS[persona].weights)
    reading["pairs"] = float(len(pairs))
    reading["zeroed"] = float(sum(1 for w in head.weights if abs(w) < 1e-9))
    return reading


def cold_start_lift(*, seeds: tuple[int, ...] = (0, 1, 2, 3, 4), shrink: bool) -> list[float]:
    """The prior's lift at five comparisons, under one variant, over several seeds.

    The prior exists to be useful before the writer's own data is. Shrinking towards zero
    could plausibly damage exactly that, so it is measured rather than assumed — and over
    several seeds, because the comparison has to survive its own noise.

    Args:
        seeds: Seeds to average over.
        shrink: Whether unidentified directions are pulled towards zero.

    Returns:
        One lift per seed.
    """
    out: list[float] = []
    for seed in seeds:
        prior = fit_prior(seed=seed)
        # The fresh writer must not be one of the eight the prior was fitted on.
        # ``fit_prior(seed=s)`` draws ``make_personas(8, seed=s)`` and
        # ``evaluate_prior(seed=s)`` draws ``make_personas(1, seed=s)[0]`` -- both seed
        # ``random.Random(s)`` and draw in the same order, so sharing the seed makes the
        # held-out writer proto-persona #0 of the training set. Every cold-start number
        # behind the second half of the pre-stated rule would then be measured on
        # training data. ``offline.pretraining`` already gets this right (prior at the
        # default seed, writer at 99); this is the same separation, per seed.
        out.append(
            evaluate_prior(prior, n_decisions=5, seed=seed + 1000, shrink_unidentified=shrink)[
                "lift"
            ]
        )
    return out


def run_ab() -> dict[str, Any]:
    """Both variants over every recorded run, plus the cold-start check.

    Returns:
        Per-variant probe means, the per-run detail, and the cold-start lifts.
    """
    probe = ProbeSet.load()
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("full__*.json"))]
    runs += [
        json.loads(p.read_text())
        for p in sorted(ARCHIVE.glob("*/full__*.json"))
        if json.loads(p.read_text()).get("actions", [{}])[0].get("features")
    ]

    detail: dict[str, list[dict[str, float]]] = {"prior_anchored": [], "shrunk": []}
    for run in runs:
        detail["prior_anchored"].append(replay(run, probe, shrink=False) | {"run": run["persona"]})
        detail["shrunk"].append(replay(run, probe, shrink=True) | {"run": run["persona"]})

    return {
        "runs_replayed": len(runs),
        "probe_tau": {
            variant: statistics.fmean([r["tau"] for r in rows]) if rows else 0.0
            for variant, rows in detail.items()
        },
        "probe_top1": {
            variant: statistics.fmean([r["top1"] for r in rows]) if rows else 0.0
            for variant, rows in detail.items()
        },
        "detail": detail,
        "cold_start_lift": {
            "prior_anchored": cold_start_lift(shrink=False),
            "shrunk": cold_start_lift(shrink=True),
        },
    }


def main() -> None:
    """Print the A/B and the verdict against the pre-stated rule."""
    result = run_ab()
    tau = result["probe_tau"]
    lift = result["cold_start_lift"]
    anchored_lift = statistics.fmean(lift["prior_anchored"])
    shrunk_lift = statistics.fmean(lift["shrunk"])
    noise = statistics.pstdev(lift["prior_anchored"]) if len(lift["prior_anchored"]) > 1 else 0.0

    print(f"replayed {result['runs_replayed']} recorded runs\n")
    print(f"{'variant':16s} {'probe tau':>10s} {'probe top-1':>12s}")
    for variant in ("prior_anchored", "shrunk"):
        print(f"{variant:16s} {tau[variant]:10.3f} {result['probe_top1'][variant]:12.1%}")
    print(f"\ncold-start lift at 5 comparisons over {len(lift['prior_anchored'])} seeds:")
    print(f"  prior-anchored {anchored_lift:+.3f}  (seed sd {noise:.3f})")
    print(f"  shrunk         {shrunk_lift:+.3f}")

    improves = tau["shrunk"] > tau["prior_anchored"]
    keeps_cold_start = shrunk_lift >= anchored_lift - noise
    print("\nrule: ship ON iff probe tau improves AND cold-start lift holds within seed noise")
    print(f"  probe tau improves: {improves}")
    print(f"  cold start holds:   {keeps_cold_start}")
    print(f"  VERDICT: ship shrinkage {'ON' if improves and keeps_cold_start else 'OFF'}")


if __name__ == "__main__":
    main()
