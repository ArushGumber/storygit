r"""Emit every measured number as a TeX macro, so no figure is ever typed by hand.

A number typed into a document is a number that goes stale silently. This writes
``docs/results.tex`` full of ``\newcommand`` definitions, and ``presentable.tex`` uses those
macros -- so regenerating the paper's numbers is:

    python -m eval.offline && python -m eval.texnumbers && tectonic docs/presentable.tex

Macros are named after what they measure rather than after where they came from, so a reader
of the TeX source can tell what a figure is without opening this file. Anything the
evaluation has not produced yet renders as a visible placeholder rather than silently as
zero: a document that quietly reports 0.00 for an experiment nobody ran is worse than one
that says the experiment has not been run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

from storygit.domain.diff import Op
from storygit.domain.world import Predicate
from storygit.preference.features import BASE_FEATURES, MAX_CRITERION_SLOTS

RESULTS = Path(__file__).parent / "results"
OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "results.tex"

MISSING = r"\textit{[not measured yet]}"


def _pct(value: float | None) -> str:
    return MISSING if value is None else f"{round(value * 100)}\\%"


def _num(value: float | None, places: int = 2) -> str:
    return MISSING if value is None else f"{value:.{places}f}"


def _get(data: Any, *path: str | int, default: Any = None) -> Any:
    """Walk a nested structure, returning ``default`` if any step is missing."""
    current = data
    for key in path:
        try:
            current = current[key]
        except (KeyError, IndexError, TypeError):
            return default
    return current


def collect() -> dict[str, str]:
    """Read the results files and build the macro table.

    Returns:
        Macro name to rendered value.
    """
    offline: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    offline_path = RESULTS / "offline_metrics.json"
    summary_path = RESULTS / "summary.json"
    if offline_path.exists():
        offline = json.loads(offline_path.read_text())
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    if not offline:
        offline = _get(summary, "offline", default={}) or {}

    macros: dict[str, str] = {}

    ablation = _get(offline, "checker_ablation", default={}) or {}
    macros["CheckerLayerOneRecall"] = _pct(_get(ablation, "layer1_only", "recall"))
    macros["CheckerLayerOnePrecision"] = _pct(_get(ablation, "layer1_only", "precision"))
    macros["CheckerLayerOneFOne"] = _num(_get(ablation, "layer1_only", "f1"))
    macros["CheckerBothRecall"] = _pct(_get(ablation, "layer1_and_2", "recall"))
    macros["CheckerBothPrecision"] = _pct(_get(ablation, "layer1_and_2", "precision"))
    macros["CheckerBothFOne"] = _num(_get(ablation, "layer1_and_2", "f1"))
    macros["CheckerFalsePositivesPerBeat"] = _num(
        _get(offline, "checker", "false_positives_per_beat")
    )
    classes = _get(offline, "checker", "per_class", default={}) or {}
    macros["CheckerClassesInjected"] = str(len(classes)) if classes else MISSING

    points = _get(offline, "stale_sweep", "points", default=[]) or []
    if points:
        declared = points[0]
        macros["StaleDeclaredPrecision"] = _num(declared.get("precision"))
        macros["StaleDeclaredRecall"] = _num(declared.get("recall"))
        macros["StaleDeclaredFOne"] = _num(declared.get("f1"))
        best = max(points[1:], key=lambda p: p["f1"], default=None)
        if best is not None:
            macros["StaleSoftBestLabel"] = str(best["label"]).replace("_", "\\_")
            macros["StaleSoftBestPrecision"] = _num(best.get("precision"))
            macros["StaleSoftBestRecall"] = _num(best.get("recall"))
            macros["StaleSoftBestFOne"] = _num(best.get("f1"))
    for name in (
        "StaleDeclaredPrecision",
        "StaleDeclaredRecall",
        "StaleDeclaredFOne",
        "StaleSoftBestLabel",
        "StaleSoftBestPrecision",
        "StaleSoftBestRecall",
        "StaleSoftBestFOne",
    ):
        macros.setdefault(name, MISSING)

    for point in _get(offline, "selector_diversity", "points", default=[]) or []:
        label = str(point["label"])
        key = "TopK" if label.startswith("top-k") else "Mmr" if label.startswith("MMR") else "Dpp"
        macros[f"Diversity{key}"] = _num(point.get("diversity"), 3)
        macros[f"Quality{key}"] = _num(point.get("quality"), 3)
    for key in ("TopK", "Mmr", "Dpp"):
        macros.setdefault(f"Diversity{key}", MISSING)
        macros.setdefault(f"Quality{key}", MISSING)

    sweep = _get(offline, "mmr_lambda_sweep", "points", default=[]) or []
    conventional = next((p for p in sweep if abs(p["lambda"] - 0.7) < 1e-9), None)
    if conventional is not None:
        macros["MmrConventionalDiversity"] = _num(conventional.get("diversity"), 3)
        macros["MmrConventionalQuality"] = _num(conventional.get("quality"), 3)
    macros.setdefault("MmrConventionalDiversity", MISSING)
    macros.setdefault("MmrConventionalQuality", MISSING)

    macros["BanditThompsonRegret"] = _num(_get(offline, "bandit", "final", "thompson"), 1)
    macros["BanditEpsilonRegret"] = _num(_get(offline, "bandit", "final", "epsilon_greedy"), 1)
    macros["BanditPullsSafe"] = str(_get(offline, "bandit", "pulls", "safe", default=MISSING))
    macros["BanditPullsExplore"] = str(_get(offline, "bandit", "pulls", "explore", default=MISSING))
    macros["BanditPosteriorSafe"] = _num(_get(offline, "bandit", "posterior", "safe"), 3)
    macros["BanditPosteriorExplore"] = _num(_get(offline, "bandit", "posterior", "explore"), 3)

    # TeX control sequences cannot contain digits -- `\rLiftAt10` parses as `\rLiftAt`
    # followed by the characters "10", which then get typeset in the preamble and produce a
    # baffling "Missing \begin{document}". Spell the numbers.
    by_decisions = _get(offline, "pretraining", "by_decisions", default={}) or {}
    for decisions, word in (("5", "Five"), ("10", "Ten"), ("40", "Forty")):
        entry = by_decisions.get(decisions, {})
        macros[f"PriorAt{word}"] = _num(entry.get("prior"), 3)
        macros[f"UniformAt{word}"] = _num(entry.get("uniform"), 3)
        lift = entry.get("lift")
        macros[f"LiftAt{word}"] = MISSING if lift is None else f"{lift:+.3f}"

    ceiling_points = _get(offline, "recovery_ceiling", "points", default=[]) or []
    for point in ceiling_points:
        if point.get("decisions") == 25:
            macros["CeilingAtTwentyFive"] = _num(point.get("mean"))
        if point.get("decisions") == 200:
            macros["CeilingAtTwoHundred"] = _num(point.get("mean"))
    macros.setdefault("CeilingAtTwentyFive", MISSING)
    macros.setdefault("CeilingAtTwoHundred", MISSING)

    runs = _get(summary, "runs", default=[]) or []
    macros["LiveRunCount"] = str(len(runs)) if runs else MISSING
    # Emitted rather than worked around in the prose, because "1 runs" in a document that
    # claims to be careful about numbers undermines the numbers.
    macros["LiveRunPlural"] = "run" if len(runs) == 1 else "runs"
    if runs:
        macros["LiveDecisionsTotal"] = str(sum(r.get("decisions", 0) for r in runs))
        recoveries = [r["weight_recovery"] for r in runs if r.get("weight_recovery") is not None]
        macros["LiveWeightRecoveryMean"] = _num(
            sum(recoveries) / len(recoveries) if recoveries else None
        )
        macros["LiveWeightRecoveryBest"] = _num(max(recoveries) if recoveries else None)
        macros["LiveTokensPerAction"] = (
            f"{round(sum(r.get('tokens_per_action', 0) for r in runs) / len(runs)):,}"
        )
        macros["LiveAcceptanceMean"] = _pct(sum(r.get("acceptance", 0.0) for r in runs) / len(runs))

        # The ceiling: what the estimator could have scored on this much data. Recovery
        # without it is a number with no scale.
        caps = [
            r["weight_recovery_ceiling"]
            for r in runs
            if r.get("weight_recovery_ceiling") is not None
        ]
        macros["CeilingMean"] = _num(sum(caps) / len(caps) if caps else None)
        macros["CeilingBest"] = _num(max(caps) if caps else None)
        if recoveries and caps:
            macros["RecoveryFractionOfCeiling"] = _pct(
                (sum(recoveries) / len(recoveries)) / (sum(caps) / len(caps))
            )

        # The deconfounded learning curve.
        firsts = [r["probe_tau_first"] for r in runs if r.get("probe_tau_first") is not None]
        lasts = [r["probe_tau_last"] for r in runs if r.get("probe_tau_last") is not None]
        tops = [r["probe_top1_last"] for r in runs if r.get("probe_top1_last") is not None]
        if firsts and lasts:
            macros["ProbeTauFirst"] = f"{sum(firsts) / len(firsts):+.3f}"
            macros["ProbeTauLast"] = f"{sum(lasts) / len(lasts):+.3f}"
            macros["ProbeTauDelta"] = (
                f"{(sum(lasts) / len(lasts)) - (sum(firsts) / len(firsts)):+.3f}"
            )
            macros["ProbeRose"] = str(
                sum(
                    1
                    for r in runs
                    if (r.get("probe_tau_last") or 0) > (r.get("probe_tau_first") or 0)
                )
            )
        if tops:
            macros["ProbeTopOneLast"] = _pct(sum(tops) / len(tops))
        points = [r["probe"][0].get("points") for r in runs if r.get("probe")]
        if points:
            macros["ProbePoints"] = str(int(min(p for p in points if p is not None)))
    for name in (
        "LiveDecisionsTotal",
        "LiveWeightRecoveryMean",
        "LiveWeightRecoveryBest",
        "LiveTokensPerAction",
        "LiveAcceptanceMean",
        "CeilingMean",
        "CeilingBest",
        "RecoveryFractionOfCeiling",
        "ProbeTauFirst",
        "ProbeTauLast",
        "ProbeTauDelta",
        "ProbeTopOneLast",
        "ProbeRose",
        "ProbePoints",
    ):
        macros.setdefault(name, MISSING)

    macros["ProviderCalls"] = str(_get(summary, "call_summary", "calls", default=MISSING))
    macros["ProviderCacheHitRate"] = _pct(_get(summary, "call_summary", "cache_hit_rate"))

    # Two counts read from the code rather than from a results file, because a paper that
    # says "the 30 operations" while the union holds 31 is wrong in a way no rerun fixes.
    macros["DiffOps"] = str(len(get_args(get_args(Op)[0])))
    macros["Predicates"] = str(len([p for p in Predicate if p is not Predicate.note]))
    macros["Features"] = str(len(BASE_FEATURES))
    macros["CriterionSlots"] = str(MAX_CRITERION_SLOTS)
    return macros


def render(macros: dict[str, str]) -> str:
    """Render the macro table as a TeX file.

    Raises:
        ValueError: If a macro name contains a digit, which TeX cannot express as a control
            sequence and which fails with an error that points somewhere else entirely.
    """
    bad = sorted(name for name in macros if any(c.isdigit() for c in name))
    if bad:
        raise ValueError(f"TeX macro names cannot contain digits: {', '.join(bad)}")
    lines = [
        "% results.tex -- generated by `python -m eval.texnumbers`. Do not edit.",
        "%",
        "% Every measured number in presentable.tex comes from here, so no figure in the",
        "% document can go stale without the regeneration step noticing. Regenerate with:",
        "%",
        "%   python -m eval.offline && python -m eval.texnumbers && tectonic presentable.tex",
        "%",
        f"% {len(macros)} macros.",
        "",
    ]
    for name, value in sorted(macros.items()):
        lines.append(f"\\newcommand{{\\r{name}}}{{{value}}}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Write ``docs/results.tex``."""
    macros = collect()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(macros))
    missing = sum(1 for value in macros.values() if value == MISSING)
    print(f"wrote {OUTPUT} -- {len(macros)} macros, {missing} not measured yet")


if __name__ == "__main__":
    main()
