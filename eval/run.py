"""The experiment driver.

    python -m eval.run --config smoke          # one tiny live run, proves the harness
    python -m eval.run --config full           # all components on, every persona
    python -m eval.run --config matrix         # every ablation x every persona, budgeted
    python -m eval.run --offline-only          # everything that needs no model calls

Three things this does that a simpler runner would not, all for the same reason — a long
run on a free tier *will* be interrupted:

* **Checkpoints after every episode.** A run that cannot resume never finishes.
* **Budgets calls, not time.** The binding constraint is the daily request quota, so the
  driver counts calls and stops cleanly at the cap rather than dying at it.
* **Reports what actually ran.** ``summary.md`` names every configuration that was skipped
  and why. A results table that silently omits the runs that failed is a lie.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval import ablations, metrics, offline, simulate
from eval.ablations import RunConfig
from eval.personas import PERSONAS, Persona
from eval.plots import line_plot
from eval.simulate import RunLog
from storygit.config import get_settings
from storygit.domain.diff import AddEntity, Diff, DiffAuthor
from storygit.domain.ids import IdGenerator
from storygit.domain.nodes import Story
from storygit.domain.state import StoryState
from storygit.domain.world import Entity, EntityKind
from storygit.graph.soft_edges import EmbeddingEdgeProvider
from storygit.preference.layer import PreferenceLayer
from storygit.providers.router import Router, build_router
from storygit.store.repository import Repository

RESULTS = Path(__file__).parent / "results"

SEED_PREMISE = (
    "A powerless orphan discovers an ability that could change the balance of power "
    "in a world at war."
)


def seed_story(repo: Repository, ids: IdGenerator) -> None:
    """Put the orphan seed and its opening cast into a fresh repository."""
    story_id = ids.node()
    kael, ashfall, warden = ids.entity(), ids.entity(), ids.entity()
    repo.initialize(
        StoryState.build(
            nodes={
                story_id: Story(
                    id=story_id,
                    title="Ashfall",
                    seed=SEED_PREMISE,
                    premise=(
                        "Kael, an orphan in a city under siege, discovers that the ash "
                        "falling over Ashfall answers him."
                    ),
                    existential_question="Is power worth what it costs the powerless?",
                )
            }
        )
    )
    repo.commit_diff(
        Diff(
            ops=(
                AddEntity(entity=Entity(id=kael, kind=EntityKind.character, name="Kael")),
                AddEntity(entity=Entity(id=ashfall, kind=EntityKind.place, name="Ashfall")),
                AddEntity(
                    entity=Entity(
                        id=warden,
                        kind=EntityKind.character,
                        name="Warden of Kell",
                        aliases=("the Warden",),
                    )
                ),
            ),
            author=DiffAuthor.human,
            intent="seed the cast",
        )
    )


def build_engine(repo: Repository, router: Router, config: RunConfig, *, seed: int) -> Any:
    """Wire an engine for one configuration, with exactly the components it enables."""
    edge_provider = (
        EmbeddingEdgeProvider(threshold=config.soft_edge_threshold) if config.soft_edges else None
    )
    engine = simulate.new_engine(
        repo,
        router,
        seed=seed,
        selection=config.selection,
        use_nli=config.use_nli and config.checker,
        edge_provider=edge_provider,
        preference=PreferenceLayer.with_prior(enabled=config.preference, seed=seed),
    )
    if not config.checker:
        # The no-checker ablation must remove the checker, not merely hide its output:
        # its scores feed the ranking, so leaving it on would leak the component back in.
        engine.selector.config = config.selection.model_copy(update={"use_judge": False})
    return engine


async def run_one(
    persona: Persona,
    config: RunConfig,
    *,
    router: Router,
    results: Path,
    seed: int,
    use_llm_edits: bool = True,
) -> RunLog:
    """Run one persona through one configuration and write its log."""
    ids = IdGenerator(seed=seed, stream=f"{config.name}:{persona.name}")
    db = results / "runs" / f"{config.name}__{simulate._slug(persona.name)}.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()

    repo = Repository.open(db)
    seed_story(repo, ids)
    engine = build_engine(repo, router, config, seed=seed)

    log = await simulate.run_writer(
        persona,
        engine,
        episodes=config.episodes,
        scenes_per_episode=config.scenes_per_episode,
        beats_per_scene=config.beats_per_scene,
        seed=seed,
        use_llm_edits=use_llm_edits,
        checkpoint=results / "runs",
    )
    fitted = dict(
        zip(
            engine.preference.state.weights.names,
            engine.preference.state.weights.weights,
            strict=False,
        )
    )
    log = log.model_copy(
        update={
            "preference_summary": engine.preference.state.summary()
            | {
                "fitted_weights": fitted,
                "weight_recovery": metrics.weight_recovery(fitted, persona.weights),
            }
        }
    )
    log.save(results / "runs" / f"{config.name}__{simulate._slug(persona.name)}.json")
    repo.close()
    return log


async def run_matrix(
    *,
    configs: list[RunConfig],
    personas: list[Persona],
    results: Path,
    max_calls: int,
    seed: int = 7,
    use_llm_edits: bool = True,
) -> dict[str, Any]:
    """Run every (configuration, persona) pair that fits in the call budget.

    Args:
        configs: Configurations to run.
        personas: Personas to run them with.
        results: Output directory.
        max_calls: Stop cleanly once the provider call log passes this many calls.
        seed: Base seed.
        use_llm_edits: Whether the simulated writer's edits go through the cheap model.

    Returns:
        Logs by ``config/persona``, plus what was skipped and why.
    """
    settings = get_settings()
    if settings.openrouter_is_enabled:
        raise RuntimeError(
            "OpenRouter is enabled. The evaluation never uses it; enable it only for "
            "chunk 7, with Arush present."
        )

    # A per-invocation call log, so "calls so far" means this run rather than every run
    # ever made on this machine. The *cache* stays shared and persistent -- that is the
    # whole point of it -- but the cost figures and the budget must be about this run.
    log_db = results / "runs" / "calllog.db"
    log_db.parent.mkdir(parents=True, exist_ok=True)
    if log_db.exists():
        log_db.unlink()
    router = build_router(settings, calllog_path=str(log_db))
    logs: dict[str, RunLog] = {}
    skipped: list[dict[str, str]] = []

    try:
        for config in configs:
            for index, persona in enumerate(personas):
                calls = int(router.summary().get("calls", 0) or 0)
                if calls >= max_calls:
                    skipped.append(
                        {
                            "config": config.name,
                            "persona": persona.name,
                            "reason": f"call budget reached ({calls}/{max_calls})",
                        }
                    )
                    continue
                print(f"--- {config.name} / {persona.name}  (calls so far: {calls})")
                log = await run_one(
                    persona,
                    config,
                    router=router,
                    results=results,
                    seed=seed + index,
                    use_llm_edits=use_llm_edits,
                )
                logs[f"{config.name}/{persona.name}"] = log
                if log.errors:
                    print(f"    errors: {log.errors}")
                print(
                    f"    decisions={len(log.actions)} "
                    f"acceptance={log.acceptance_rate:.0%} "
                    f"episodes={log.episodes_completed}"
                )
    finally:
        summary = router.summary()
        await router.aclose()

    return {"logs": logs, "skipped": skipped, "call_summary": summary}


def summarize(
    logs: dict[str, RunLog],
    skipped: list[dict[str, str]],
    call_summary: dict[str, Any],
    offline_metrics: dict[str, Any],
    results: Path,
) -> dict[str, Any]:
    """Compute cross-run metrics, write the plots, and write ``summary.md``."""
    rows: list[dict[str, Any]] = []
    acceptance: dict[str, list[float]] = {}
    edit_distance: dict[str, list[float]] = {}

    for key, log in sorted(logs.items()):
        kinds = [a.kind for a in log.actions]
        distances = [a.edit_distance for a in log.actions]
        curve = metrics.acceptance_curve(kinds, window=8)
        acceptance[key] = curve
        edit_distance[key] = metrics.edit_distance_curve(kinds, distances, window=8)
        cost = metrics.cost_per_action(log.call_summary, len(log.actions))
        rows.append(
            {
                "run": key,
                "persona": log.persona,
                "decisions": len(log.actions),
                "acceptance": log.acceptance_rate,
                "acceptance_first_third": _mean(curve[: max(1, len(curve) // 3)]),
                "acceptance_last_third": _mean(curve[-max(1, len(curve) // 3) :]),
                "mean_edit_distance": _mean(
                    [d for d, k in zip(distances, kinds, strict=True) if k == "edit"]
                ),
                "hard_flags_shown": sum(a.hard_flags_shown for a in log.actions),
                "stale_marked": sum(a.stale_marked for a in log.actions),
                "vetoes": sum(1 for a in log.actions if a.veto),
                "weight_recovery": log.preference_summary.get("weight_recovery"),
                "tokens_per_action": cost["tokens"],
                "usd_per_action": cost["usd"],
                "episodes": log.episodes_completed,
                "errors": list(log.errors),
            }
        )

    if acceptance:
        line_plot(
            acceptance,
            title="Acceptance rate over decisions (rolling, window 8)",
            xlabel="writer decisions",
            ylabel="fraction accepted or edited-and-accepted",
            path=results / "acceptance.svg",
        )
        line_plot(
            edit_distance,
            title="How much the writer rewrote what they took",
            xlabel="writer decisions",
            ylabel="normalized token edit distance",
            path=results / "edit_distance.svg",
        )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "runs": rows,
        "skipped": skipped,
        "call_summary": call_summary,
        "offline": offline_metrics,
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (results / "summary.md").write_text(_render_markdown(summary))
    return summary


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _render_markdown(summary: dict[str, Any]) -> str:
    """Render the summary as a readable table, including what did not run."""
    lines = [
        "# Evaluation summary",
        "",
        f"Generated {summary['generated_at']}.",
        "",
        "## Offline metrics (deterministic, no model calls)",
        "",
    ]
    offline_data = summary["offline"]
    checker = offline_data["checker_ablation"]
    lines += [
        "### Continuity checker",
        "",
        "| configuration | recall | precision | F1 |",
        "|---|---|---|---|",
        f"| layer 1 only | {checker['layer1_only']['recall']:.0%} | "
        f"{checker['layer1_only']['precision']:.0%} | {checker['layer1_only']['f1']:.2f} |",
        f"| layer 1 + 2 | {checker['layer1_and_2']['recall']:.0%} | "
        f"{checker['layer1_and_2']['precision']:.0%} | {checker['layer1_and_2']['f1']:.2f} |",
        "",
        f"False positives on a clean story: "
        f"{offline_data['checker']['false_positives_per_beat']:.2f} per beat.",
        "",
        "### Staleness prediction",
        "",
        "| configuration | precision | recall | F1 |",
        "|---|---|---|---|",
    ]
    for point in offline_data["stale_sweep"]["points"]:
        lines.append(
            f"| {point['label']} | {point['precision']:.2f} | "
            f"{point['recall']:.2f} | {point['f1']:.2f} |"
        )
    lines += [
        "",
        "### Selector diversity",
        "",
        "| selector | diversity | mean quality |",
        "|---|---|---|",
    ]
    for point in offline_data["selector_diversity"]["points"]:
        lines.append(f"| {point['label']} | {point['diversity']:.3f} | {point['quality']:.3f} |")

    lines += [
        "",
        "### Bandit",
        "",
        f"Pseudo-regret after 400 rounds: Thompson "
        f"{offline_data['bandit']['final']['thompson']:.1f}, "
        f"epsilon-greedy {offline_data['bandit']['final']['epsilon_greedy']:.1f}. "
        f"Pulls: {offline_data['bandit']['pulls']}.",
        "",
        "### Pretraining",
        "",
        "| comparisons made | from prior | from uniform | lift |",
        "|---|---|---|---|",
    ]
    for decisions, result in sorted(
        offline_data["pretraining"]["by_decisions"].items(), key=lambda kv: int(kv[0])
    ):
        lines.append(
            f"| {decisions} | {result['prior']:.3f} | {result['uniform']:.3f} | "
            f"{result['lift']:+.3f} |"
        )

    lines += ["", "## Live runs", ""]
    if not summary["runs"]:
        lines.append("_No live runs in this invocation._")
    else:
        lines += [
            "| run | decisions | acceptance | first third | last third | "
            "mean edit dist | weight recovery | tokens/action |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in summary["runs"]:
            recovery = row["weight_recovery"]
            lines.append(
                f"| {row['run']} | {row['decisions']} | {row['acceptance']:.0%} | "
                f"{row['acceptance_first_third']:.0%} | {row['acceptance_last_third']:.0%} | "
                f"{row['mean_edit_distance']:.2f} | "
                f"{recovery:.2f} | {row['tokens_per_action']:.0f} |"
                if recovery is not None
                else f"| {row['run']} | {row['decisions']} | {row['acceptance']:.0%} | "
                f"{row['acceptance_first_third']:.0%} | {row['acceptance_last_third']:.0%} | "
                f"{row['mean_edit_distance']:.2f} | - | {row['tokens_per_action']:.0f} |"
            )

    if summary["skipped"]:
        lines += ["", "## Not run", "", "| configuration | persona | why |", "|---|---|---|"]
        for item in summary["skipped"]:
            lines.append(f"| {item['config']} | {item['persona']} | {item['reason']} |")

    errors = [(r["run"], e) for r in summary["runs"] for e in r["errors"]]
    if errors:
        lines += ["", "## Errors during runs", ""]
        lines += [f"- `{run}`: {error}" for run, error in errors]

    lines += [
        "",
        "## Provider cost",
        "",
        f"- calls: {summary['call_summary'].get('calls', 0)}",
        f"- tokens: {summary['call_summary'].get('total_tokens', 0)}",
        f"- cache hit rate: {summary['call_summary'].get('cache_hit_rate', 0):.0%}",
        f"- estimated cost: ${summary['call_summary'].get('cost_usd', 0):.4f} "
        "(free tier; the price table would apply on a metered provider)",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description="storygit evaluation harness")
    parser.add_argument(
        "--config",
        default="smoke",
        help=(
            "smoke (tiny live run), matrix (every ablation), or one config: "
            + ", ".join(ablations.BY_NAME)
        ),
    )
    parser.add_argument("--personas", default="", help="comma-separated persona names")
    parser.add_argument("--max-calls", type=int, default=600, help="provider call budget")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--offline-only", action="store_true", help="skip every live run")
    parser.add_argument(
        "--no-llm-edits",
        action="store_true",
        help="simulated edits truncate instead of calling the cheap model",
    )
    args = parser.parse_args()

    results = Path(args.results)
    print("computing offline metrics...")
    offline_metrics = offline.run_all(results=results)

    logs: dict[str, RunLog] = {}
    skipped: list[dict[str, str]] = []
    call_summary: dict[str, Any] = {}

    if not args.offline_only:
        # "matrix" means every ablation; "full" is the single all-components-on config,
        # which is also the reference point the ablations are compared against.
        configs = (
            list(ablations.ABLATIONS) if args.config == "matrix" else [ablations.get(args.config)]
        )
        names = (
            [n.strip() for n in args.personas.split(",") if n.strip()]
            if args.personas
            else (["the Serialist"] if args.config == "smoke" else list(PERSONAS))
        )
        personas = [PERSONAS[n] for n in names]

        print(
            f"running {len(configs)} config(s) x {len(personas)} persona(s), "
            f"budget {args.max_calls} calls"
        )
        outcome = asyncio.run(
            run_matrix(
                configs=configs,
                personas=personas,
                results=results,
                max_calls=args.max_calls,
                seed=args.seed,
                use_llm_edits=not args.no_llm_edits,
            )
        )
        logs = outcome["logs"]
        skipped = outcome["skipped"]
        call_summary = outcome["call_summary"]

    summary = summarize(logs, skipped, call_summary, offline_metrics, results)
    print(f"\nwrote {results}/summary.json and summary.md")
    for row in summary["runs"]:
        print(
            f"  {row['run']:34} acceptance {row['acceptance']:.0%} "
            f"({row['acceptance_first_third']:.0%} -> {row['acceptance_last_third']:.0%})"
        )


if __name__ == "__main__":
    main()
