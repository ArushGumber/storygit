"""Plotting style, shared by every figure the evaluation produces.

One palette, one typeface, no gridline clutter, no chartjunk — the same visual language as
the TikZ diagrams, so a plot dropped into the paper or the Eval tab looks like it belongs
there. The colour values are the ones in ``docs/diagrams/style.tex``.

Everything is written as SVG. The frontend serves SVG directly, and the TeX build converts
it, so one artifact serves both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

INK = "#2B2A28"
SLATE = "#5C5A55"
MUTE = "#9A968E"
RULE = "#D6D2C8"
PAPER = "#FAF8F4"
ACCENT = "#8C5A3C"

SERIES = ("#8C5A3C", "#3F5666", "#4F6B52", "#9C6B2F", "#6B4A6B", "#5C5A55")
"""Categorical series colours: the accent first, then muted blues, greens, and browns.

Chosen to stay distinguishable in greyscale — a printed take-home is a real possibility —
and to avoid red/green as the only difference between two series.
"""


def configure() -> Any:
    """Apply the house style to matplotlib and return the module.

    Returns:
        The ``matplotlib.pyplot`` module, already configured.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": RULE,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.family": "serif",
            "font.serif": ["Charter", "Bitstream Charter", "DejaVu Serif"],
            "figure.dpi": 110,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def save(fig: Any, path: Path | str) -> Path:
    """Write a figure as SVG, creating parent directories.

    Args:
        fig: A matplotlib figure.
        path: Destination.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(target, format="svg", bbox_inches="tight")
    return target


def line_plot(
    series: Mapping[str, Sequence[float]],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path | str,
    baseline: float | None = None,
    baseline_label: str = "",
    x_start: int = 0,
    x_values: Sequence[float] | None = None,
) -> Path:
    """A multi-series line plot in the house style.

    Args:
        series: Label to y-values. x is the index, offset by ``x_start``.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        path: Where to write the SVG.
        baseline: Optional horizontal reference line.
        baseline_label: Label for the reference line.
        x_start: What the first point is called on the x-axis. Curves counted in episodes
            start at one, and an axis reading 0 to 2 under the label "episodes completed"
            is off by one against its own caption.
        x_values: Explicit x coordinates, when the points are not evenly spaced. Without
            this a sweep over 0, 5, 10, 20 and 40 comparisons plots against 0-4 while the
            axis label still says "comparisons", which is not an approximation but a
            different graph.

    Returns:
        The path written.
    """
    plt = configure()
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    for index, (label, values) in enumerate(series.items()):
        xs = list(x_values) if x_values is not None else list(range(x_start, x_start + len(values)))
        ax.plot(xs, values, color=SERIES[index % len(SERIES)], label=label)
    # A count is a whole number; fractional ticks under "episodes completed" invite the
    # reader to look for a point that does not exist.
    if all(float(x).is_integer() for x in (x_values or [x_start])):
        from matplotlib.ticker import MaxNLocator

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if baseline is not None:
        ax.axhline(baseline, color=MUTE, linestyle="--", linewidth=1.0, label=baseline_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(series) > 1 or baseline is not None:
        ax.legend()
    written = save(fig, path)
    plt.close(fig)
    return written
