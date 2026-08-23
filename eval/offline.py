"""Every metric that needs no generation, computed deterministically.

A surprising amount of what the system claims can be measured without calling a model at
all: checker recall against injected contradictions, staleness precision and recall against
a hand-written dependency chain, the soft-edge threshold sweep, selector diversity on real
embeddings, bandit regret, and whether pretraining earns its place.

Separating these from the live runs has three benefits. They are **exact** — no sampling
noise, so a regression is a real regression. They are **free**, so they run in CI and on
every quality pass. And they are **honest**: nothing here depends on a model happening to
produce a good candidate on the day, so these numbers cannot be improved by rerunning.

    python -m eval.offline
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from eval import ablations, inject, metrics
from eval.plots import MUTE, SERIES, configure, line_plot, save
from storygit.continuity import audit, layer1, layer2_nli
from storygit.domain.apply import apply
from storygit.domain.diff import Diff, DiffAuthor, UpdateFact
from storygit.domain.ids import IdGenerator
from storygit.graph.propagation import propagate_change
from storygit.graph.soft_edges import EmbeddingEdgeProvider
from storygit.preference.bandit import Arm, ThompsonBandit, pseudo_regret_curve
from storygit.preference.pretrain import evaluate_prior, fit_prior
from storygit.selection.select import SelectionConfig

RESULTS = Path(__file__).parent / "results"


# --- continuity ---------------------------------------------------------------


def checker_layers(*, use_nli: bool = True) -> dict[str, Any]:
    """Per-layer recall against injected contradictions, plus the clean-story false rate.

    Args:
        use_nli: Whether to run layer 2. Off makes this instant and fully deterministic.

    Returns:
        Per-layer precision/recall, the combined figure, per-class detail, and the
        false-positive rate on a story with nothing wrong with it.
    """
    state, scenario = inject.build_scenario()
    flags = list(layer1.check_state(state))
    if use_nli:
        flags.extend(layer2_nli.check(state))

    per_layer = {
        f"layer{layer}": metrics.checker_recall(
            scenario.injections, flags, layer=layer
        ).model_dump()
        for layer in (1, 2)
    }
    combined = metrics.checker_recall(scenario.injections, flags)

    per_class: dict[str, dict[str, Any]] = {}
    for injection in scenario.injections:
        single = metrics.checker_recall([injection], flags)
        per_class[injection.kind.value] = {
            "caught": single.true_positives == 1,
            "expected_layer": injection.expected_layer,
            "caught_by": sorted(
                {
                    f.layer
                    for f in flags
                    if {str(x) for x in f.fact_ids} & {str(x) for x in injection.fact_ids}
                }
            ),
            "description": injection.description,
        }

    clean_state, clean_scenario = inject.clean_story(IdGenerator(seed=5, stream="clean"))
    clean_flags = list(layer1.check_state(clean_state))
    if use_nli:
        clean_flags.extend(layer2_nli.check(clean_state))

    return {
        "per_layer": per_layer,
        "combined": combined.model_dump()
        | {"precision": combined.precision, "recall": combined.recall, "f1": combined.f1},
        "per_class": per_class,
        "false_positives_per_beat": metrics.false_positive_rate(
            clean_flags, len(clean_scenario.beats)
        ),
        "clean_flags": len(clean_flags),
    }


def checker_ablation() -> dict[str, Any]:
    """What each layer contributes, by running with and without it."""
    with_nli = checker_layers(use_nli=True)
    without = checker_layers(use_nli=False)
    return {
        "layer1_only": without["combined"],
        "layer1_and_2": with_nli["combined"],
        "layer2_adds": with_nli["combined"]["recall"] - without["combined"]["recall"],
    }


# --- propagation --------------------------------------------------------------


def stale_sweep() -> dict[str, Any]:
    """Stale-prediction precision and recall: declared edges vs declared + soft edges.

    The soft-edge threshold is swept, because the whole question is the trade-off and a
    single number would hide it.

    Returns:
        The declared-only point and one point per threshold.
    """
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
            intent="move Kael",
        ),
    )

    # Truth includes the beat that depends on the fact but never declared it, so the
    # soft-edge ablation has something real to find rather than only false alarms to add.
    truth = case.all_affected
    declared = propagate_change(state, after)
    points = [
        metrics.stale_scores(
            [m.node_id for m in declared], truth, case.unaffected, label="declared only"
        )
    ]
    for threshold in ablations.SOFT_EDGE_THRESHOLDS:
        marks = propagate_change(
            state,
            after,
            edge_provider=EmbeddingEdgeProvider(threshold=threshold),
        )
        points.append(
            metrics.stale_scores(
                [m.node_id for m in marks],
                truth,
                case.unaffected,
                label=f"+ soft edges @ {threshold:.2f}",
            )
        )

    return {
        "thresholds": [None, *ablations.SOFT_EDGE_THRESHOLDS],
        "points": [
            p.model_dump() | {"precision": p.precision, "recall": p.recall, "f1": p.f1}
            for p in points
        ],
    }


# --- selection ----------------------------------------------------------------

CANDIDATE_TEXTS = (
    "Kael runs from the market, the token still in his fist, and does not look back.",
    "Kael bolts from the market with the token clenched in his hand, never looking back.",
    "Kael sprints away from the stalls, the token gripped tight, refusing to turn around.",
    "The Warden lets him go, and only afterwards wonders why she did.",
    "Mara, three streets away, hears the shouting and starts walking towards it.",
    "The ash stops falling. Every single person in the square notices except Kael.",
)
"""Six candidates: three near-paraphrases and three genuinely different moves. If a
selector cannot tell those apart, nothing else about it matters."""

CANDIDATE_QUALITY = (0.92, 0.90, 0.88, 0.72, 0.70, 0.66)
"""Quality deliberately anti-correlated with diversity: the three paraphrases score highest,
so plain top-k will take all three and the diversity-aware selectors must give up quality
to avoid that. Without this the comparison would be free."""


MMR_LAMBDAS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.9)
"""Lambda values swept, to show where the diversity term stops binding."""


def mmr_lambda_sweep() -> dict[str, Any]:
    """How MMR's trade-off parameter behaves on the paraphrase case.

    The point of the sweep is that the conventional default (0.7) turns out to reproduce
    the top-k baseline exactly, which would have made the headline selector
    indistinguishable from the ablation it is supposed to beat.

    Returns:
        One point per lambda, plus the DPP for reference.
    """
    from storygit.selection.dpp import dpp_select
    from storygit.selection.embed import embed_texts, similarity
    from storygit.selection.mmr import mmr_select

    vectors = embed_texts(list(CANDIDATE_TEXTS))
    sim = similarity(vectors)
    points = []
    for lam in MMR_LAMBDAS:
        picked = mmr_select(CANDIDATE_QUALITY, sim, 3, lambda_=lam)
        points.append(
            {
                "lambda": lam,
                "picked": picked,
                "diversity": metrics.mean_pairwise_distance([vectors[i].tolist() for i in picked]),
                "quality": sum(CANDIDATE_QUALITY[i] for i in picked) / 3,
            }
        )
    dpp = dpp_select(CANDIDATE_QUALITY, sim, 3)
    return {
        "points": points,
        "dpp": {
            "picked": dpp,
            "diversity": metrics.mean_pairwise_distance([vectors[i].tolist() for i in dpp]),
            "quality": sum(CANDIDATE_QUALITY[i] for i in dpp) / 3,
        },
    }


def selector_diversity() -> dict[str, Any]:
    """Mean pairwise distance and mean quality for each selector, on real embeddings."""
    from storygit.selection.dpp import dpp_select, topk_select
    from storygit.selection.embed import embed_texts, similarity
    from storygit.selection.mmr import mmr_select

    vectors = embed_texts(list(CANDIDATE_TEXTS))
    sim = similarity(vectors)
    default_lambda = SelectionConfig().lambda_
    selectors = {
        f"MMR (lambda={default_lambda})": mmr_select,
        "DPP": dpp_select,
        "top-k (temperature only)": topk_select,
    }
    points = []
    for label, fn in selectors.items():
        picked = fn(CANDIDATE_QUALITY, sim, 3, lambda_=default_lambda)
        points.append(
            metrics.diversity_quality_point(
                [vectors[i].tolist() for i in picked],
                [CANDIDATE_QUALITY[i] for i in picked],
                label,
            )
            | {"picked": picked}
        )
    return {"points": points}


# --- learning -----------------------------------------------------------------


def bandit_regret(*, rounds: int = 400, seed: int = 1) -> dict[str, Any]:
    """Pseudo-regret for Thompson sampling against epsilon-greedy."""
    rates = {Arm.safe: 0.4, Arm.explore: 0.7}
    rng = random.Random(seed)
    bandit = ThompsonBandit(seed=seed)
    thompson_arms: list[Arm] = []
    for _ in range(rounds):
        arm = bandit.choose()
        bandit.update(arm, 1.0 if rng.random() < rates[arm] else 0.0)
        thompson_arms.append(arm)

    eps_rng = random.Random(seed)
    counts = dict.fromkeys(Arm, 0)
    totals = dict.fromkeys(Arm, 0.0)
    eps_arms: list[Arm] = []
    for _ in range(rounds):
        if eps_rng.random() < 0.1 or min(counts.values()) == 0:
            arm = eps_rng.choice(list(Arm))
        else:
            arm = max(Arm, key=lambda a: totals[a] / max(1, counts[a]))
        counts[arm] += 1
        totals[arm] += 1.0 if eps_rng.random() < rates[arm] else 0.0
        eps_arms.append(arm)

    thompson = pseudo_regret_curve(thompson_arms, rates)
    epsilon = pseudo_regret_curve(eps_arms, rates)
    return {
        "thompson": thompson,
        "epsilon_greedy": epsilon,
        "final": {"thompson": thompson[-1], "epsilon_greedy": epsilon[-1]},
        "pulls": dict(bandit.state.pulls),
        "posterior": {
            "safe": bandit.state.mean_safe(),
            "explore": bandit.state.mean_explore(),
        },
        "true_rates": {arm.value: rate for arm, rate in rates.items()},
    }


def pretraining() -> dict[str, Any]:
    """Whether the population prior beats uniform initialization early on."""
    prior = fit_prior()
    curve = {str(n): evaluate_prior(prior, n_decisions=n, seed=99) for n in (0, 5, 10, 20, 40)}
    return {
        "top_weights": prior.top_weights(5),
        "by_decisions": curve,
    }


# --- audit --------------------------------------------------------------------


def audit_report() -> dict[str, Any]:
    """A whole-graph audit of an injected story, for the report."""
    state, scenario = inject.build_scenario()
    report = audit.run_audit(state, use_nli=True)
    return {
        "summary": report.summary(),
        "by_layer": report.by_layer,
        "injected": len(scenario.injections),
        "flags": [
            {
                "kind": f.kind.value,
                "severity": f.severity.value,
                "layer": f.layer,
                "message": f.message,
            }
            for f in report.flags
        ],
    }


# --- driver -------------------------------------------------------------------


def run_all(*, results: Path = RESULTS) -> dict[str, Any]:
    """Compute every offline metric and write the figures.

    Args:
        results: Directory to write into.

    Returns:
        Everything computed, also written to ``offline_metrics.json``.
    """
    out: dict[str, Any] = {
        "checker": checker_layers(),
        "checker_ablation": checker_ablation(),
        "stale_sweep": stale_sweep(),
        "selector_diversity": selector_diversity(),
        "mmr_lambda_sweep": mmr_lambda_sweep(),
        "bandit": bandit_regret(),
        "pretraining": pretraining(),
        "audit": audit_report(),
    }
    _write_figures(out, results)
    results.mkdir(parents=True, exist_ok=True)
    (results / "offline_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def _write_figures(data: dict[str, Any], results: Path) -> None:
    """Write every offline figure."""
    plt = configure()

    line_plot(
        {
            "Thompson sampling": data["bandit"]["thompson"],
            "epsilon-greedy (0.1)": data["bandit"]["epsilon_greedy"],
        },
        title="Bandit pseudo-regret (arms at p = 0.40 and 0.70)",
        xlabel="candidate sets shown",
        ylabel="cumulative pseudo-regret",
        path=results / "bandit_regret.svg",
    )

    # Checker recall per layer.
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    labels = ["layer 1", "layer 1 + 2"]
    values = [
        data["checker_ablation"]["layer1_only"]["recall"],
        data["checker_ablation"]["layer1_and_2"]["recall"],
    ]
    ax.bar(labels, values, color=[SERIES[1], SERIES[0]], width=0.5)
    for index, value in enumerate(values):
        ax.text(index, value + 0.02, f"{value:.0%}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("injected contradictions caught")
    ax.set_title("What each checker layer contributes")
    save(fig, results / "checker_recall.svg")
    plt.close(fig)

    # Stale precision/recall sweep.
    points = data["stale_sweep"]["points"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(
        [p["recall"] for p in points[1:]],
        [p["precision"] for p in points[1:]],
        marker="o",
        color=SERIES[0],
        label="declared + soft edges",
    )
    # Several thresholds land on the same (recall, precision) once the sweep saturates,
    # and one label per threshold draws them on top of each other into an unreadable
    # smudge. Group by position and label each position once, as a range.
    grouped: dict[tuple[float, float], list[float]] = {}
    for point, threshold in zip(points[1:], ablations.SOFT_EDGE_THRESHOLDS, strict=True):
        grouped.setdefault((round(point["recall"], 4), round(point["precision"], 4)), []).append(
            threshold
        )
    for (recall, precision), thresholds in grouped.items():
        label = (
            f"{thresholds[0]:.2f}"
            if len(thresholds) == 1
            else f"{min(thresholds):.2f}\u2013{max(thresholds):.2f}"
        )
        ax.annotate(
            label,
            (recall, precision),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=7,
        )
    ax.scatter(
        [points[0]["recall"]],
        [points[0]["precision"]],
        color=SERIES[1],
        zorder=5,
        label="declared only",
        s=45,
    )
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Staleness prediction: declared edges vs embedding edges")
    ax.legend()
    save(fig, results / "stale_precision_recall.svg")
    plt.close(fig)

    # Diversity vs quality.
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for index, point in enumerate(data["selector_diversity"]["points"]):
        ax.scatter(
            point["diversity"],
            point["quality"],
            color=SERIES[index],
            s=70,
            label=point["label"],
        )
    ax.set_xlabel("mean pairwise distance among shown candidates")
    ax.set_ylabel("mean quality of shown candidates")
    ax.set_title("Diversity against quality, same six candidates")
    ax.legend()
    save(fig, results / "diversity_quality.svg")
    plt.close(fig)

    # MMR lambda sweep: where the diversity term stops binding.
    sweep = data["mmr_lambda_sweep"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(
        [p["lambda"] for p in sweep["points"]],
        [p["diversity"] for p in sweep["points"]],
        marker="o",
        color=SERIES[0],
        label="MMR diversity",
    )
    ax.plot(
        [p["lambda"] for p in sweep["points"]],
        [p["quality"] for p in sweep["points"]],
        marker="s",
        color=SERIES[1],
        label="MMR mean quality",
    )
    ax.axhline(
        sweep["dpp"]["diversity"],
        color=SERIES[2],
        linestyle="--",
        linewidth=1.0,
        label="DPP diversity (no parameter)",
    )
    ax.axvline(0.5, color=MUTE, linestyle=":", linewidth=1.0)
    ax.set_xlabel("MMR lambda (1.0 = pure quality)")
    ax.set_ylabel("value")
    ax.set_title("Where MMR's diversity term stops binding")
    ax.legend()
    save(fig, results / "mmr_lambda_sweep.svg")
    plt.close(fig)

    # Pretraining lift.
    by_decisions = data["pretraining"]["by_decisions"]
    xs = sorted(int(k) for k in by_decisions)
    line_plot(
        {
            "from the population prior": [by_decisions[str(x)]["prior"] for x in xs],
            "from uniform weights": [by_decisions[str(x)]["uniform"] for x in xs],
        },
        title="Preference head: held-out pairwise accuracy on a fresh writer",
        xlabel="comparisons the writer has made",
        ylabel="held-out pairwise accuracy",
        path=results / "pretraining_lift.svg",
        baseline=0.5,
        baseline_label="chance",
    )


def main() -> None:
    """Run everything and print a readable summary."""
    data = run_all()
    checker = data["checker"]
    print("=== continuity checker ===")
    print(
        f"  recall: layer1 {data['checker_ablation']['layer1_only']['recall']:.0%}, "
        f"+layer2 {data['checker_ablation']['layer1_and_2']['recall']:.0%}"
    )
    for name, detail in checker["per_class"].items():
        mark = "caught" if detail["caught"] else "MISSED"
        print(
            f"    {name:12} {mark:7} by layer(s) {detail['caught_by']} "
            f"(expected {detail['expected_layer']})"
        )
    print(f"  false positives on a clean story: {checker['false_positives_per_beat']:.2f}/beat")

    print("\n=== staleness prediction ===")
    for point in data["stale_sweep"]["points"]:
        print(
            f"  {point['label']:26} P={point['precision']:.2f} "
            f"R={point['recall']:.2f} F1={point['f1']:.2f}"
        )

    print("\n=== MMR lambda sweep ===")
    for point in data["mmr_lambda_sweep"]["points"]:
        print(
            f"  lambda={point['lambda']:.1f}  picked={point['picked']}  "
            f"diversity={point['diversity']:.3f} quality={point['quality']:.3f}"
        )
    dpp = data["mmr_lambda_sweep"]["dpp"]
    print(
        f"  DPP (no lambda) picked={dpp['picked']}  "
        f"diversity={dpp['diversity']:.3f} quality={dpp['quality']:.3f}"
    )

    print("\n=== selector diversity ===")
    for point in data["selector_diversity"]["points"]:
        print(
            f"  {point['label']:26} diversity={point['diversity']:.3f} "
            f"quality={point['quality']:.3f} picked={point['picked']}"
        )

    print("\n=== bandit ===")
    print(
        f"  pseudo-regret: thompson {data['bandit']['final']['thompson']:.1f}, "
        f"epsilon-greedy {data['bandit']['final']['epsilon_greedy']:.1f}"
    )
    print(
        f"  pulls: {data['bandit']['pulls']}  posterior: "
        f"{ {k: round(v, 3) for k, v in data['bandit']['posterior'].items()} }"
    )

    print("\n=== pretraining ===")
    for decisions, result in sorted(
        data["pretraining"]["by_decisions"].items(), key=lambda kv: int(kv[0])
    ):
        print(
            f"  after {decisions:>2} decisions: prior {result['prior']:.3f} "
            f"uniform {result['uniform']:.3f} lift {result['lift']:+.3f}"
        )

    print(f"\nwrote {RESULTS}/offline_metrics.json and 6 figures")


if __name__ == "__main__":
    main()
