"""The selection façade: axes in, three labelled candidates out.

The pipeline, in order:

    axes -> parallel sampling -> embed -> quality -> continuity -> dial -> MMR/DPP

Each stage exists for a reason the previous one does not cover. Axes make the candidates
*nameably* different. Parallel sampling makes six of them cost about as much wall-clock as
one. Embedding makes "different" measurable. Quality comes from the soft judge, and in
chunk 4 from the preference head — the seam is a single argument. Continuity checking runs
on each candidate *before the writer sees it*, so a candidate that contradicts the bible
arrives with the contradiction already attached. The dial re-weights quality against
distance-from-obvious. And MMR or DPP picks the three that are good and mutually
different.

Selector choice is one config value, so the evaluation can ablate it honestly:
``mmr``, ``dpp``, or ``topk_temperature`` (the baseline — good candidates with nothing
stopping all three being the same idea).
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from storygit.agents.propose import Proposal, Proposer
from storygit.agents.schemas import Level
from storygit.continuity import layer1, layer3_judge
from storygit.continuity.flags import Flag, sort_flags
from storygit.domain.apply import apply
from storygit.domain.errors import ApplyError
from storygit.domain.ids import NodeId
from storygit.domain.state import StoryState
from storygit.graph.slices import StateSlice, entities_in_scope, entity_slice
from storygit.providers.router import Router
from storygit.selection import axes as axes_module
from storygit.selection.dial import effective_quality, surprise_scores
from storygit.selection.dpp import dpp_select, topk_select
from storygit.selection.embed import embed_texts, min_max, similarity
from storygit.selection.mmr import mmr_select

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


class Selector(StrEnum):
    """Which selection algorithm to use."""

    mmr = "mmr"
    dpp = "dpp"
    topk_temperature = "topk_temperature"


SELECTORS: dict[Selector, Callable[..., list[int]]] = {
    Selector.mmr: mmr_select,
    Selector.dpp: dpp_select,
    Selector.topk_temperature: topk_select,
}
"""All three share one signature, which is what makes the ablation a config change."""


class SelectionConfig(BaseModel):
    """Everything tunable about candidate selection.

    Attributes:
        n: How many candidates to sample.
        k: How many to show the writer. Three is Fabula's overwhelm lesson: more
            alternatives cost more to review than they save.
        selector: Which algorithm picks the k.
        lambda_: MMR's quality/diversity trade-off. 0.5 by measurement, not convention:
            at the usual 0.7 MMR reproduces the top-k baseline exactly on realistic
            quality spreads. See ``mmr.DEFAULT_LAMBDA``. Ignored by DPP, which gets the
            trade-off from its kernel.
        use_judge: Whether to score candidates with the layer-3 judge. Off falls back to
            a continuity-only quality signal, which is what the no-judge ablation wants.
        use_dial: Whether to apply the coherent/surprising re-weighting.
        temperature: Sampling temperature for candidate generation.
    """

    model_config = ConfigDict(frozen=True)

    n: int = Field(default=6, ge=1, le=12)
    k: int = Field(default=3, ge=1, le=6)
    selector: Selector = Selector.mmr
    lambda_: float = Field(default=0.5, ge=0.0, le=1.0)
    use_judge: bool = True
    use_dial: bool = True
    temperature: float = Field(default=0.95, ge=0.0, le=2.0)


class Candidate(BaseModel):
    """One sampled proposal with everything selection learned about it.

    Attributes:
        proposal: The proposal itself, carrying its diff.
        axis_key: Which axis it was generated along.
        axis_label: The label the writer reads.
        flags: Continuity flags found by checking it before it was shown.
        base_quality: Judge score in ``[0, 1]``, penalized by hard flags.
        surprise: Embedding distance from the greedy continuation.
        effective_quality: What the selector actually ranked on.
        judge_scores: Per-axis judge sub-scores; features for chunk 4.
        selected: Whether it made the shortlist.
    """

    model_config = ConfigDict(frozen=True)

    proposal: Proposal
    axis_key: str = ""
    axis_label: str = ""
    flags: tuple[Flag, ...] = ()
    base_quality: float = 0.5
    surprise: float = 0.0
    effective_quality: float = 0.5
    judge_scores: dict[str, float] = Field(default_factory=dict)
    selected: bool = False

    @property
    def text(self) -> str:
        """The candidate rendered as text, for embedding and judging."""
        return candidate_text(self.proposal)


def candidate_text(proposal: Proposal) -> str:
    """Render a proposal as the text used for embedding and judging.

    Args:
        proposal: The proposal.

    Returns:
        A single string built from what the diff actually says.
    """
    parts: list[str] = []
    for op in proposal.diff.ops:
        node = getattr(op, "node", None)
        if node is not None:
            parts.extend(
                p
                for p in (
                    getattr(node, "title", ""),
                    getattr(node, "what_happens", ""),
                    getattr(node, "audience_learns", ""),
                    getattr(node, "text", ""),
                )
                if p
            )
        text = getattr(op, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    if not parts:
        parts.extend(proposal.delta_summary)
    return " ".join(parts)


class CandidateSelector:
    """Runs the whole selection pipeline for one node.

    Attributes:
        proposer: Generation.
        router: Model access, for the judge and the greedy continuation.
        config: Tunables.
    """

    def __init__(
        self,
        proposer: Proposer,
        router: Router,
        config: SelectionConfig | None = None,
    ) -> None:
        """Wire the selector.

        Args:
            proposer: The proposer to sample with.
            router: Model access.
            config: Tunables; defaults to n=6, k=3, MMR.
        """
        self.proposer = proposer
        self.router = router
        self.config = config or SelectionConfig()

    async def select(
        self,
        state: StoryState,
        level: Level,
        *,
        target_node_id: NodeId | None = None,
        intent: str = "",
        exemplars: str = "",
        quality_fn: Callable[[Sequence[Candidate]], Awaitable[list[float] | None]] | None = None,
    ) -> list[Candidate]:
        """Generate, check, rank, and shortlist candidates for one node.

        Args:
            state: Current story state.
            level: What to propose.
            target_node_id: The node to attach to.
            intent: The writer's instruction.
            exemplars: The writer's own accepted and rejected work, rendered for the
                prompt. Empty when the preference layer is off.
            quality_fn: Overrides the base-quality *ranking*. This is the chunk 4 seam:
                the preference head is dropped in here and nothing else changes. It
                receives candidates that already carry their judge sub-scores and flags,
                and may return ``None`` to mean "no opinion, use the judge" — which is
                what makes the no-preference ablation exact rather than approximate.

        Returns:
            All sampled candidates, ordered with the selected ``k`` first. Returning the
            unselected ones matters — the evaluation needs them to measure diversity, and
            chunk 4 needs them as the negative side of a preference pair.
        """
        config = self.config
        parent_id = target_node_id
        chosen_axes = axes_module.rotate(
            config.n, offset=axes_module.offset_for(str(parent_id or level.value))
        )

        sampled = await self._sample(state, level, parent_id, intent, chosen_axes, exemplars)
        if not sampled:
            return []

        slice_ = self._slice(state, parent_id)
        candidates = [self._with_flags(state, proposal, axis) for proposal, axis in sampled]

        # The judge always runs when enabled, even if a preference head will override the
        # ranking: its sub-scores are *features* of that head, and its soft flags are shown
        # to the writer either way. An override replaces the ranking, not the analysis.
        if config.use_judge:
            judged, judge_flags, judge_scores = await self._judge_all(slice_, candidates, parent_id)
        else:
            judged = [c.base_quality for c in candidates]
            judge_scores = [{} for _ in candidates]
            judge_flags = [[] for _ in candidates]

        candidates = [
            candidates[i].model_copy(
                update={
                    "judge_scores": judge_scores[i],
                    "flags": tuple(sort_flags([*candidates[i].flags, *judge_flags[i]])),
                }
            )
            for i in range(len(candidates))
        ]

        override = await quality_fn(candidates) if quality_fn is not None else None
        base = list(override) if override is not None else judged
        base = [self._penalize(score, candidates[i].flags) for i, score in enumerate(base)]

        # Embeddings are only computed when something needs them. The top-k baseline
        # ignores similarity and a dial at zero ignores distance, so the ablation runs
        # do not pay for an encoder they never consult.
        wants_dial = config.use_dial and state.ledger.dial > 0.0
        needs_vectors = config.selector is not Selector.topk_temperature or wants_dial
        surprise = [0.0] * len(candidates)
        if needs_vectors:
            vectors = embed_texts([c.text for c in candidates])
            sim = similarity(vectors)
            if wants_dial:
                greedy = await self._greedy_embedding(state, level, parent_id, intent)
                if greedy is not None:
                    surprise = surprise_scores(vectors, greedy)
        else:
            sim = _identity(len(candidates))

        ranked = (
            effective_quality(base, surprise, state.ledger.dial)
            if config.use_dial
            else min_max(base)
        )
        picks = SELECTORS[config.selector](ranked, sim, config.k, lambda_=config.lambda_)
        picked = set(picks)

        enriched = [
            candidates[i].model_copy(
                update={
                    "base_quality": base[i],
                    "surprise": surprise[i],
                    "effective_quality": ranked[i],
                    "selected": i in picked,
                }
            )
            for i in range(len(candidates))
        ]
        order = [*picks, *(i for i in range(len(enriched)) if i not in picked)]
        return [enriched[i] for i in order]

    # --- stages -----------------------------------------------------------------

    async def _sample(
        self,
        state: StoryState,
        level: Level,
        parent_id: NodeId | None,
        intent: str,
        chosen_axes: tuple[axes_module.Axis, ...],
        exemplars: str = "",
    ) -> list[tuple[Proposal, axes_module.Axis]]:
        """Sample one candidate per axis, all at once."""
        results = await asyncio.gather(
            *(
                self.proposer.propose(
                    state,
                    level,
                    target_node_id=parent_id,
                    intent=_with_axis(intent, axis),
                    k=1,
                    temperature=self.config.temperature,
                    exemplars=exemplars,
                )
                for axis in chosen_axes
            )
        )
        out: list[tuple[Proposal, axes_module.Axis]] = []
        for axis, proposals in zip(chosen_axes, results, strict=True):
            for proposal in proposals:
                out.append((proposal.model_copy(update={"axis": axis.key}), axis))
        return out

    def _slice(self, state: StoryState, parent_id: NodeId | None) -> StateSlice:
        """The slice used for judging, matching what generation saw."""
        if parent_id is None:
            return entity_slice(state, set(state.entities), None)
        scope = entities_in_scope(state, parent_id) or set(state.entities)
        beats = state.beats_in_order()
        at_beat = (
            parent_id if parent_id in {b.id for b in beats} else (beats[-1].id if beats else None)
        )
        return entity_slice(state, scope, at_beat)

    def _with_flags(
        self, state: StoryState, proposal: Proposal, axis: axes_module.Axis
    ) -> Candidate:
        """Run layer 1 on what the candidate *would* make true.

        The diff is applied to a scratch state and checked there, so the writer sees the
        contradiction attached to the candidate rather than discovering it after
        accepting. A candidate whose diff will not even apply is kept, flagged, and
        scored at zero — dropping it silently would hide a real generation failure.
        """
        try:
            scratch = apply(state, proposal.diff)
        except ApplyError as exc:
            from storygit.continuity.flags import FlagKind, Severity

            return Candidate(
                proposal=proposal,
                axis_key=axis.key,
                axis_label=axis.label,
                base_quality=0.0,
                flags=(
                    Flag(
                        kind=FlagKind.contradiction,
                        severity=Severity.hard,
                        layer=1,
                        message=f"This change cannot be applied to the story as it stands: {exc}",
                        node_id=proposal.target_node_id,
                    ),
                ),
            )
        new_fact_ids = set(scratch.facts) - set(state.facts)
        flags = layer1.check_new_facts(scratch, new_fact_ids)
        return Candidate(
            proposal=proposal,
            axis_key=axis.key,
            axis_label=axis.label,
            flags=tuple(sort_flags(flags)),
        )

    async def _judge_all(
        self,
        slice_: StateSlice,
        candidates: list[Candidate],
        node_id: NodeId | None,
    ) -> tuple[list[float], list[list[Flag]], list[dict[str, float]]]:
        """Score every candidate with the soft judge, concurrently."""
        results = await asyncio.gather(
            *(
                layer3_judge.judge(self.router, slice_, c.text, node_id=node_id, sample_index=i)
                for i, c in enumerate(candidates)
            )
        )
        quality = [r[0] for r in results]
        flags = [r[1] for r in results]
        scores = [r[2] for r in results]
        return quality, flags, scores

    async def _greedy_embedding(
        self,
        state: StoryState,
        level: Level,
        parent_id: NodeId | None,
        intent: str,
    ) -> np.ndarray | None:
        """Embed the temperature-0 continuation — the definition of "the obvious move".

        One extra call per node, cached, so moving the dial afterwards is free.
        """
        greedy = await self.proposer.propose(
            state, level, target_node_id=parent_id, intent=intent, k=1, temperature=0.0
        )
        if not greedy:
            return None
        vectors = embed_texts([candidate_text(greedy[0])])
        return vectors[0] if vectors.shape[0] else None

    @staticmethod
    def _penalize(quality: float, flags: tuple[Flag, ...]) -> float:
        """Push candidates with hard contradictions down the ranking.

        Not to zero, and never removed: a contradiction can be exactly what the writer
        wants (a character is lying, a narrator is unreliable), so the decision stays
        theirs. The flag is on screen either way.
        """
        hard = sum(1 for f in flags if f.is_hard)
        return max(0.0, quality * (0.6**hard))


def _identity(n: int) -> np.ndarray:
    """An identity similarity matrix, for selectors that never look at it."""
    import numpy as np

    return np.eye(n, dtype=np.float32)


def _with_axis(intent: str, axis: axes_module.Axis) -> str:
    """Append an axis instruction to the writer's intent."""
    if intent:
        return f"{intent}\n\nFor this option specifically: {axis.fragment}"
    return axis.fragment
