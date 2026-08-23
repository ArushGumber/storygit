"""The experiment matrix: which component is turned off, and what it should cost.

Each ablation removes exactly one thing, because an ablation that changes two things
measures neither. Each carries a stated **expectation** — what should get worse, and
roughly how — so the results can disagree with the design rather than merely describe it.
A prediction written after the numbers arrive is not a prediction.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from storygit.selection.select import SelectionConfig, Selector


class RunConfig(BaseModel):
    """One point in the experiment matrix.

    Attributes:
        name: Short identifier used in filenames and plot labels.
        description: What is being removed or changed.
        expectation: What should get worse if the component earns its place. Stated in
            advance; the summary reports whether it held.
        selection: Candidate-selection settings.
        preference: Whether the preference layer contributes.
        propagation: Whether staleness propagation runs.
        checker: Whether the continuity checker runs.
        use_nli: Whether layer 2 runs.
        soft_edges: Whether the embedding edge provider is attached.
        soft_edge_threshold: Its similarity threshold.
        episodes: Episodes per run.
        scenes_per_episode: Scenes per episode.
        beats_per_scene: Beats per scene.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    expectation: str = ""
    selection: SelectionConfig = SelectionConfig()
    preference: bool = True
    propagation: bool = True
    checker: bool = True
    use_nli: bool = True
    soft_edges: bool = False
    soft_edge_threshold: float = 0.72
    episodes: int = Field(default=3, ge=1)
    scenes_per_episode: int = Field(default=2, ge=1)
    beats_per_scene: int = Field(default=2, ge=1)

    @property
    def decisions_per_run(self) -> int:
        """Roughly how many writer decisions this configuration produces."""
        per_episode = 1 + self.scenes_per_episode * (1 + self.beats_per_scene * 2)
        return self.episodes * per_episode

    @property
    def calls_per_run(self) -> int:
        """Rough upper bound on provider calls, before caching.

        ``n`` proposals plus ``n`` judge calls per decision, plus one greedy continuation
        when the dial is live, plus one extraction per accepted prose node.
        """
        per_decision = self.selection.n * (2 if self.selection.use_judge else 1)
        per_decision += 1 if self.selection.use_dial else 0
        return self.decisions_per_run * per_decision


FULL = RunConfig(
    name="full",
    description="everything on: axes, MMR, dial, three-layer checker, preference layer",
    expectation="the reference point every ablation is compared against",
)

NO_PREFERENCE = RunConfig(
    name="no_preference",
    description="preference layer off; ranking is the judge score alone",
    expectation=(
        "acceptance should stay flat rather than rising, and the fitted weights should not "
        "correlate with the persona's hidden ones -- there is nothing fitting them"
    ),
    preference=False,
)

NO_PROPAGATION = RunConfig(
    name="no_propagation",
    description="accepted changes do not mark downstream nodes",
    expectation=(
        "contradiction count should rise over a run: nothing tells the writer that what "
        "they built on has moved"
    ),
    propagation=False,
)

NO_CHECKER = RunConfig(
    name="no_checker",
    description="continuity checker off; candidates are shown unflagged",
    expectation=(
        "contradictions should survive into accepted state, and the Controller persona "
        "-- who weights continuity highest -- should accept least"
    ),
    checker=False,
    use_nli=False,
)

TEMPERATURE_ONLY = RunConfig(
    name="temperature_only",
    description="no axes, no MMR: sample hot and keep the top k by quality",
    expectation=(
        "mean pairwise distance among the shown candidates should drop sharply at similar "
        "mean quality -- three good candidates that are the same idea"
    ),
    selection=SelectionConfig(selector=Selector.topk_temperature),
)

DPP_SELECTOR = RunConfig(
    name="dpp",
    description="DPP instead of MMR, everything else identical",
    expectation=(
        "diversity at least as good as MMR at similar quality. At n=6, k=3 the two should "
        "largely agree; a large gap either way would be the interesting result"
    ),
    selection=SelectionConfig(selector=Selector.dpp),
)

NO_DIAL = RunConfig(
    name="no_dial",
    description="dial ignored; ranking is quality alone",
    expectation=(
        "the Maximalist (dial 0.70) should notice most; low-dial personas should barely "
        "differ from full"
    ),
    selection=SelectionConfig(use_dial=False),
)

SOFT_EDGES = RunConfig(
    name="soft_edges",
    description="embedding-similarity edges added to declared ones (ablation 2d'')",
    expectation=(
        "stale recall up, precision down. The prediction is that the crossover is too poor "
        "to use as a primary mechanism; if it is not, the design should change"
    ),
    soft_edges=True,
)

ABLATIONS: tuple[RunConfig, ...] = (
    FULL,
    NO_PREFERENCE,
    NO_PROPAGATION,
    NO_CHECKER,
    TEMPERATURE_ONLY,
    DPP_SELECTOR,
    NO_DIAL,
    SOFT_EDGES,
)
"""The full matrix."""

SMOKE = RunConfig(
    name="smoke",
    description="one persona, tiny run, to prove the harness works end to end",
    expectation="completes without errors and produces a metrics file",
    selection=SelectionConfig(n=3, k=2),
    episodes=1,
    scenes_per_episode=1,
    beats_per_scene=1,
)
"""The configuration to run first, always. It costs almost nothing and catches everything
structural."""

HALF = RunConfig(
    name="half",
    description="the full configuration at half the decisions, for the metered rerun",
    expectation=(
        "the same mechanisms on a stronger model, at a cost that fits the in-code cap; "
        "the comparison against `full` is the one that separates the system from the model"
    ),
    episodes=2,
    scenes_per_episode=2,
    beats_per_scene=1,
)
"""Half the decision count of ``FULL``, and nothing else changed.

The metered rerun exists to separate "the system works" from "the model is good", which
only holds if the two runs differ in exactly one variable. So every component stays on:
same axes, same selector, same dial, same checker, same preference layer, same personas,
same seed. What changes is the decision count, because the full configuration on the
cheapest model that fits does not fit the $12 cap, and reporting a partial run of the full
configuration would confound the comparison with where it stopped.
"""

SOFT_EDGE_THRESHOLDS = (0.55, 0.62, 0.68, 0.72, 0.78, 0.85)
"""Threshold sweep for the 2d'' ablation. Reported as a precision/recall curve, not a
single number, because the whole question is the trade-off."""

PROBE_SAMPLE = RunConfig(
    name="probesample",
    description="two episodes per persona, to sample the held-out probe fixture from",
    expectation="produces decisions with features and texts; never itself probed",
    selection=SelectionConfig(n=6, k=3),
    episodes=2,
    scenes_per_episode=2,
    beats_per_scene=2,
)
"""The run the probe fixture is built from.

Kept separate from the evaluation runs on purpose. Sampling probe points from the same runs
they are used to measure would make the probe a memory test even with the cross-persona
rule; taking them from a run that is never itself probed removes the question entirely.
"""

BY_NAME: dict[str, RunConfig] = {c.name: c for c in (*ABLATIONS, SMOKE, PROBE_SAMPLE, HALF)}


def get(name: str) -> RunConfig:
    """Look up a configuration by name.

    Args:
        name: Configuration name.

    Returns:
        The configuration.

    Raises:
        KeyError: If no such configuration exists.
    """
    return BY_NAME[name]
