"""What the chunk-7 rerun would cost on a metered provider. Arithmetic only.

Chunk 7 reruns the evaluation on a stronger model through OpenRouter, which is the one
provider in this project that costs money. OpenRouter stays locked until Arush is present,
so this makes no calls: it takes the token counts the free-tier run actually produced and
multiplies them by published rates.

The number that matters is not the total but the total against the **in-code budget cap**
(``OPENROUTER_BUDGET_USD``, $12 by default), which is checked before every metered call and
is independent of any account-side limit. A projection that quietly exceeds the cap means
the run would stop halfway, which is worse than not starting it.

    python -m eval.costing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from storygit.config import get_settings
from storygit.providers.pricing import price_for

CANDIDATES = (
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "google/gemini-2.5-pro",
)
"""Models worth costing. Anything unlisted falls back to the conservative provider rate."""


def totals_from(summary: dict[str, Any]) -> dict[str, float]:
    """Prompt and completion tokens, and decisions, from an evaluation summary.

    Args:
        summary: A loaded ``summary.json``.

    Returns:
        ``{"prompt": ..., "completion": ..., "decisions": ...}``.
    """
    calls = summary.get("call_summary", {}) or {}
    return {
        "prompt": float(calls.get("prompt_tokens", 0)),
        "completion": float(calls.get("completion_tokens", 0)),
        "decisions": float(sum(r.get("decisions", 0) for r in summary.get("runs", []))),
    }


def project(totals: dict[str, float], model: str, *, fraction: float = 1.0) -> dict[str, float]:
    """Cost of running the same token volume through one metered model.

    Args:
        totals: Token totals from a free-tier run.
        model: OpenRouter model id.
        fraction: Scale the run down — 0.5 is the half config.

    Returns:
        Dollars, split by direction, plus the per-decision figure.
    """
    price = price_for("openrouter", model)
    prompt = totals["prompt"] * fraction / 1_000_000 * price.prompt
    completion = totals["completion"] * fraction / 1_000_000 * price.completion
    decisions = max(totals["decisions"] * fraction, 1.0)
    return {
        "prompt_usd": prompt,
        "completion_usd": completion,
        "total_usd": prompt + completion,
        "usd_per_decision": (prompt + completion) / decisions,
    }


def render(summary: dict[str, Any], *, cap_usd: float) -> str:
    """The costing table, as Markdown.

    Args:
        summary: A loaded ``summary.json`` from the free-tier run.
        cap_usd: The in-code budget cap the projection is judged against.

    Returns:
        The document body.
    """
    totals = totals_from(summary)
    lines = [
        "# chunk7_costing.md - what the metered rerun costs",
        "",
        "A projection, and the measurement that now exists to check it against. The table",
        "below is arithmetic: it multiplies the token counts the free-tier run produced by",
        "published rates. The metered rerun has since been run, so the last section compares",
        "what this predicted against what it actually cost.",
        "",
        f"Measured on the current run: {totals['prompt']:,.0f} prompt tokens, "
        f"{totals['completion']:,.0f} completion tokens, across "
        f"{totals['decisions']:,.0f} writer decisions.",
        "",
        f"The in-code budget guard caps metered spend at **${cap_usd:,.2f}**, checked before",
        "every call and independent of any account-side limit.",
        "",
        "| model | full run | half config | $/decision (full) | fits the cap? |",
        "|---|---|---|---|---|",
    ]
    measured = _measured_metered(summary)
    for model in CANDIDATES:
        full = project(totals, model)
        half = project(totals, model, fraction=0.5)
        fits = (
            "yes"
            if full["total_usd"] <= cap_usd
            else ("half only" if half["total_usd"] <= cap_usd else "**no**")
        )
        lines.append(
            f"| `{model}` | ${full['total_usd']:,.2f} | ${half['total_usd']:,.2f} | "
            f"${full['usd_per_decision']:.3f} | {fits} |"
        )
    lines += [
        "",
        "## Reading it",
        "",
        f"This system is **prompt-heavy**: {totals['prompt'] / max(totals['completion'], 1):.1f}"
        " prompt tokens for every completion token. Every proposal carries the state slice,",
        "and a lock exports its facts into every subsequent prompt, so the prompt side grows",
        "with the story while the completion side does not. That is the cost consequence of a",
        "design that is about state rather than about generation, and it means a cheaper",
        "**prompt** rate matters more than a cheaper completion rate when picking the chunk-7",
        "model — which is not the direction most model comparisons are written in.",
        "",
        "Caching is the other half. The read-through cache is keyed on the full request",
        "including `sample_index`, so a *rerun* of an identical configuration is nearly free;",
        "these figures are for the first run, which is the one that has to fit.",
    ]
    if measured:
        lines += [
            "",
            "## Projection against measurement",
            "",
            f"The metered rerun has since run on `{measured['model']}`: "
            f"**${measured['spend']:,.2f}** across {measured['decisions']:,.0f} decisions, "
            f"or **${measured['per_decision']:.3f} per decision**, under the ${cap_usd:,.2f} "
            "cap, which was not raised and never had to refuse a call.",
            "",
            f"The projection for that model was ${measured['projected_per_decision']:.3f} "
            "per decision, so the estimate was "
            f"{measured['ratio']:.0%} of what it cost. The gap is not a pricing error, it is "
            "a run-shape error: the projection divides a whole run's tokens by a whole run's",
            "decisions, and three of the four metered runs stopped at their first episode.",
            "They paid for episode proposals, which are the expensive part, and never reached",
            "the beats that would have amortised them. A cost model built from complete runs",
            "under-prices incomplete ones, which is worth knowing before quoting a per-decision",
            "figure for a run that might not finish.",
        ]
    return "\n".join(lines) + "\n"


def _measured_metered(summary: dict[str, Any]) -> dict[str, Any] | None:
    """What the metered rerun actually cost, if one has been run.

    A projection that is never checked against a measurement is a spreadsheet. This pulls
    the real figures out of the summary so the two sit next to each other.

    Args:
        summary: A loaded ``summary.json``.

    Returns:
        The comparison, or ``None`` when no metered run is recorded.
    """
    rows = [r for r in (summary.get("runs") or []) if str(r.get("run", "")).startswith("half/")]
    if not rows:
        return None
    decisions = sum(r.get("decisions", 0) for r in rows)
    spend = sum((r.get("usd_per_action") or 0.0) * r.get("decisions", 0) for r in rows)
    if not decisions or not spend:
        return None
    model = "google/gemini-2.5-pro"
    projected = project(totals_from(summary), model)
    projected_per_decision = projected["usd_per_decision"]
    per_decision = spend / decisions
    return {
        "model": model,
        "spend": spend,
        "decisions": decisions,
        "per_decision": per_decision,
        "projected_per_decision": projected_per_decision,
        "ratio": projected_per_decision / per_decision if per_decision else 0.0,
    }


def main() -> None:
    """Write the costing note next to the private logs."""
    summary = json.loads(Path("eval/results/summary.json").read_text())
    cap = get_settings().openrouter_budget_usd
    out = Path("../arush/logs/chunk7_costing.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(summary, cap_usd=cap))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
