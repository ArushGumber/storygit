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
from storygit.graph.soft_edges import DEFAULT_THRESHOLD as SOFT_EDGE_DEFAULT
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
        # Not argmax. Six thresholds are swept on one six-beat case with three true
        # positives, and four of the six tie at a perfect score -- so "the best threshold"
        # would name a maximum that was never located, chosen on the only data there is.
        # What the sweep actually establishes is a *range* over which the mechanism is
        # indistinguishable from perfect, which is the stronger claim and the honest one.
        soft = points[1:]
        if soft:
            top = max(p["f1"] for p in soft)
            tied = [p for p in soft if p["f1"] >= top - 1e-9]
            macros["StaleSoftBestLabel"] = str(tied[0]["label"]).replace("_", "\\_")
            macros["StaleSoftBestPrecision"] = _num(tied[0].get("precision"))
            macros["StaleSoftBestRecall"] = _num(tied[0].get("recall"))
            macros["StaleSoftBestFOne"] = _num(tied[0].get("f1"))
            macros["StaleSoftTiedCount"] = str(len(tied))
            macros["StaleSoftSweptCount"] = str(len(soft))
            macros["StaleSoftTiedRange"] = (
                str(tied[0]["label"]).replace("_", "\\_")
                + "--"
                + str(tied[-1]["label"]).replace("_", "\\_")
            )
            macros["StaleSoftShippedThreshold"] = _num(SOFT_EDGE_DEFAULT)
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

    # Row 0 is the cold-start row, the one a reader most wants, and it is the row where the
    # lift is negative. A table that starts at five comparisons is choosing where to start.
    entry = by_decisions.get("0", {})
    macros["PriorAtZero"] = _num(entry.get("prior"), 3)
    macros["UniformAtZero"] = _num(entry.get("uniform"), 3)
    zero_lift = entry.get("lift")
    macros["LiftAtZero"] = MISSING if zero_lift is None else f"{zero_lift:+.3f}"

    ceiling_points = _get(offline, "recovery_ceiling", "points", default=[]) or []
    for point in ceiling_points:
        if point.get("decisions") == 25:
            macros["CeilingAtTwentyFive"] = _num(point.get("mean"))
            macros["CeilingAtTwentyFiveSd"] = _num(point.get("sd"))
        if point.get("decisions") == 200:
            macros["CeilingAtTwoHundred"] = _num(point.get("mean"))
            macros["CeilingAtTwoHundredSd"] = _num(point.get("sd"))
    macros.setdefault("CeilingAtTwentyFive", MISSING)
    macros.setdefault("CeilingAtTwoHundred", MISSING)
    macros.setdefault("CeilingAtTwentyFiveSd", MISSING)
    macros.setdefault("CeilingAtTwoHundredSd", MISSING)

    # The shrinkage A/B. These decided a change to the learner, so they are the last
    # numbers in the document that should have been typed by hand -- which is what they
    # were until this block existed.
    ab = _get(offline, "shrinkage_ab", default={}) or {}
    if ab:
        macros["ShrinkRunsReplayed"] = str(ab.get("runs_replayed", 0))
        macros["ShrinkTauAnchored"] = _num(_get(ab, "probe_tau", "prior_anchored"), 3)
        macros["ShrinkTauShrunk"] = _num(_get(ab, "probe_tau", "shrunk"), 3)
        macros["ShrinkTopOneAnchored"] = _pct(_get(ab, "probe_top1", "prior_anchored"))
        macros["ShrinkTopOneShrunk"] = _pct(_get(ab, "probe_top1", "shrunk"))
        anchored = _get(ab, "cold_start_lift", "prior_anchored", default=[]) or []
        shrunk = _get(ab, "cold_start_lift", "shrunk", default=[]) or []
        if anchored and shrunk:
            import statistics

            macros["ShrinkColdStartAnchored"] = f"{statistics.fmean(anchored):+.3f}"
            macros["ShrinkColdStartShrunk"] = f"{statistics.fmean(shrunk):+.3f}"
            macros["ShrinkColdStartSeedSd"] = _num(
                statistics.pstdev(anchored) if len(anchored) > 1 else 0.0, 3
            )
            macros["ShrinkColdStartSeeds"] = str(len(anchored))
    for name in (
        "ShrinkRunsReplayed",
        "ShrinkTauAnchored",
        "ShrinkTauShrunk",
        "ShrinkTopOneAnchored",
        "ShrinkTopOneShrunk",
        "ShrinkColdStartAnchored",
        "ShrinkColdStartShrunk",
        "ShrinkColdStartSeedSd",
        "ShrinkColdStartSeeds",
    ):
        macros.setdefault(name, MISSING)

    # The summary now carries every configuration that has been run, so the free-tier
    # figures must not silently average in the metered ones. "Live" means the free-tier
    # reference configuration; the strong-model rerun gets its own prefix.
    all_runs = _get(summary, "runs", default=[]) or []

    def _config_of(row: dict[str, Any]) -> str:
        name = str(row.get("run", ""))
        return name.split("/", 1)[0] if "/" in name else ""

    runs = [r for r in all_runs if _config_of(r) in ("", "full")] or all_runs
    metered = [r for r in all_runs if _config_of(r) == "half"]

    macros["MeteredRunCount"] = str(len(metered)) if metered else MISSING
    if metered:
        macros["MeteredDecisionsTotal"] = str(sum(r.get("decisions", 0) for r in metered))
        macros["MeteredAcceptanceMean"] = _pct(
            sum(r.get("acceptance", 0.0) for r in metered) / len(metered)
        )
        macros["MeteredTokensPerAction"] = (
            f"{round(sum(r.get('tokens_per_action', 0) for r in metered) / len(metered)):,}"
        )
        recov = [r["weight_recovery"] for r in metered if r.get("weight_recovery") is not None]
        if recov:
            macros["MeteredWeightRecoveryMean"] = _num(sum(recov) / len(recov))
        taus = [r["probe_tau_last"] for r in metered if r.get("probe_tau_last") is not None]
        if taus:
            macros["MeteredProbeTauLast"] = f"{sum(taus) / len(taus):+.3f}"
        spend = sum((r.get("usd_per_action") or 0.0) * r.get("decisions", 0) for r in metered)
        macros["MeteredSpend"] = f"\\${spend:.2f}"
        macros["MeteredUsdPerDecision"] = (
            f"\\${spend / max(1, sum(r.get('decisions', 0) for r in metered)):.3f}"
        )
    for name in (
        "MeteredDecisionsTotal",
        "MeteredAcceptanceMean",
        "MeteredTokensPerAction",
        "MeteredWeightRecoveryMean",
        "MeteredProbeTauLast",
        "MeteredSpend",
        "MeteredUsdPerDecision",
    ):
        macros.setdefault(name, MISSING)

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
        # The same correlation restricted to the coordinates the writer actually weights.
        # The headline is over all thirteen, five or six of which are exactly zero for
        # every persona -- and shrinkage drives the fitted values on those same dead
        # coordinates to zero, so it improves the headline without learning anything. Both
        # are reported so a reader can see the size of that effect rather than discover it.
        identified = [
            r["weight_recovery_identified"]
            for r in runs
            if r.get("weight_recovery_identified") is not None
        ]
        macros["LiveWeightRecoveryIdentified"] = _num(
            sum(identified) / len(identified) if identified else None
        )
        macros["LiveTokensPerAction"] = (
            f"{round(sum(r.get('tokens_per_action', 0) for r in runs) / len(runs)):,}"
        )
        macros["LiveAcceptanceMean"] = _pct(sum(r.get("acceptance", 0.0) for r in runs) / len(runs))

        # Everything the live-tier prose used to assert by hand. A narrative sentence with a
        # typed number in it is a sentence that goes stale the first time the run changes,
        # and this section's numbers changed under it more than once.
        macros["LiveRunsCompleted"] = str(sum(1 for r in runs if not r.get("errors")))
        macros["LiveRunsTruncated"] = str(sum(1 for r in runs if r.get("errors")))
        offered = sum(_get(r, "retry", "rejected_sets", default=0) or 0 for r in runs)
        rescued = sum(_get(r, "retry", "rescued", default=0) or 0 for r in runs)
        macros["RetryOffered"] = str(offered)
        macros["RetryRescued"] = str(rescued)
        firsts_acc = [
            r["acceptance_first_third"] for r in runs if r.get("acceptance_first_third") is not None
        ]
        lasts_acc = [
            r["acceptance_last_third"] for r in runs if r.get("acceptance_last_third") is not None
        ]
        if firsts_acc and lasts_acc:
            macros["AcceptanceFirstThird"] = _pct(sum(firsts_acc) / len(firsts_acc))
            macros["AcceptanceLastThird"] = _pct(sum(lasts_acc) / len(lasts_acc))
            fell = [(a, b) for a, b in zip(firsts_acc, lasts_acc, strict=True) if b < a - 1e-9]
            macros["AcceptanceFellCount"] = str(len(fell))
            if fell:
                worst = min(fell, key=lambda pair: pair[1] - pair[0])
                macros["AcceptanceFellFrom"] = _pct(worst[0])
                macros["AcceptanceFellTo"] = _pct(worst[1])
        # Which persona ended furthest below its references, named rather than assumed.
        probed = [r for r in runs if r.get("probe_tau_last") is not None]
        if probed:
            worst_run = min(
                probed, key=lambda r: r["probe_tau_last"] - (r.get("probe_tau_prior") or 0.0)
            )
            macros["ProbeWorstPersona"] = str(worst_run.get("persona", "")).replace("_", "\\_")
            best_recovery = max(
                (r for r in runs if r.get("weight_recovery") is not None),
                key=lambda r: r["weight_recovery"],
                default=None,
            )
            if best_recovery is not None:
                macros["RecoveryBestPersona"] = str(best_recovery.get("persona", ""))
        macros["LiveDecisionsPerRun"] = str(
            round(sum(r.get("decisions", 0) for r in runs) / max(1, len(runs)))
        )

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
            # The readable statement is not "did the line go up" but "where did it end,
            # relative to a head that learned nothing and to the prior it started from".
            macros["ProbeAtOrAbovePrior"] = str(
                sum(
                    1
                    for r in runs
                    if r.get("probe_tau_last") is not None
                    and r["probe_tau_last"] >= (r.get("probe_tau_prior") or 0) - 1e-9
                )
            )
            macros["ProbeAboveUniform"] = str(
                sum(
                    1
                    for r in runs
                    if r.get("probe_tau_last") is not None
                    and r["probe_tau_last"] > (r.get("probe_tau_uniform") or 0) + 1e-9
                )
            )
        if tops:
            macros["ProbeTopOneLast"] = _pct(sum(tops) / len(tops))
        uni = [r["probe_tau_uniform"] for r in runs if r.get("probe_tau_uniform") is not None]
        pri = [r["probe_tau_prior"] for r in runs if r.get("probe_tau_prior") is not None]
        if uni:
            macros["ProbeTauUniform"] = f"{sum(uni) / len(uni):+.3f}"
        if pri:
            macros["ProbeTauPrior"] = f"{sum(pri) / len(pri):+.3f}"
        # Every run must have been measured against the same fixture, or the averaged
        # ProbeTau macros above are silently mixing two measuring sticks. min() would print
        # the smaller and say nothing; this refuses to print a single number when the runs
        # disagree, which is the case the fixture was rebuilt mid-evaluation and half the
        # archive predates it.
        points = [r["probe"][0].get("points") for r in runs if r.get("probe")]
        sizes = {int(p) for p in points if p is not None}
        if len(sizes) == 1:
            macros["ProbePoints"] = str(sizes.pop())
        elif sizes:
            macros["ProbePoints"] = MISSING
            print(
                f"  warning: runs were probed against different fixtures ({sorted(sizes)}); "
                "ProbePoints and every averaged tau are not comparable"
            )
    for name in (
        "LiveDecisionsTotal",
        "LiveWeightRecoveryMean",
        "LiveWeightRecoveryBest",
        "LiveWeightRecoveryIdentified",
        "LiveTokensPerAction",
        "LiveAcceptanceMean",
        "LiveRunsCompleted",
        "LiveRunsTruncated",
        "RetryOffered",
        "RetryRescued",
        "AcceptanceFirstThird",
        "AcceptanceLastThird",
        "AcceptanceFellCount",
        "AcceptanceFellFrom",
        "AcceptanceFellTo",
        "ProbeWorstPersona",
        "RecoveryBestPersona",
        "LiveDecisionsPerRun",
        "CeilingMean",
        "CeilingBest",
        "RecoveryFractionOfCeiling",
        "ProbeTauFirst",
        "ProbeTauLast",
        "ProbeTauDelta",
        "ProbeTopOneLast",
        "ProbeRose",
        "ProbePoints",
        "ProbeTauUniform",
        "ProbeTauPrior",
        "ProbeAtOrAbovePrior",
        "ProbeAboveUniform",
    ):
        macros.setdefault(name, MISSING)

    macros["ProviderCalls"] = str(_get(summary, "call_summary", "calls", default=MISSING))
    macros["ProviderCacheHitRate"] = _pct(_get(summary, "call_summary", "cache_hit_rate"))
    # The token figure the free-tier claim rests on. It counts *routed* calls only:
    # embeddings and the NLI checkpoint run locally and are called directly, so they
    # produce no log row. The dollar figure is unaffected -- local models are free -- but
    # this is not a complete token accounting and the document says so.
    total_tokens = _get(summary, "call_summary", "total_tokens")
    macros["ProviderTokens"] = (
        MISSING if total_tokens is None else f"{round(total_tokens / 1_000_000, 1)} million"
    )
    macros["ProviderErrors"] = str(_get(summary, "call_summary", "errors", default=MISSING))

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
