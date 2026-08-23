"""Selection: axis conditioning, MMR and DPP geometry, the dial, and the ablation swap.

The geometry tests use synthetic embeddings rather than a real encoder, so they are about
the algorithm rather than about the model — and they run in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import Fixture
from tests.mockprovider import MockProvider, canned

from storygit.agents.schemas import Level
from storygit.domain.ids import IdGenerator
from storygit.providers.router import Router
from storygit.selection import axes as axes_module
from storygit.selection.dial import effective_quality, surprise_scores
from storygit.selection.dpp import build_kernel, dpp_select, topk_select
from storygit.selection.embed import min_max, normalize_rows, similarity
from storygit.selection.mmr import mmr_select
from storygit.selection.select import (
    CandidateSelector,
    SelectionConfig,
    Selector,
    candidate_text,
)

# Two tight clusters plus one outlier. Any selector that respects diversity must reach
# across them; plain top-k will not.
CLUSTERED = normalize_rows(
    np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.14, 0.0],
            [0.98, 0.20, 0.0],
            [0.0, 1.0, 0.0],
            [0.14, 0.99, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
)
CLUSTER_OF = {0: "a", 1: "a", 2: "a", 3: "b", 4: "b", 5: "c"}


def beat_payload(title: str, what: str) -> dict[str, object]:
    """A minimal valid beat proposal."""
    return {
        "title": title,
        "what_happens": what,
        "audience_learns": "",
        "audience_feels": "",
        "location": "",
        "time": "",
        "produces": [],
        "consumes": [],
        "threads_touched": [],
        "new_characters": [],
        "rationale": f"because {title}",
        "delta_summary": [title],
    }


# --- axes ---------------------------------------------------------------------


def test_axes_rotate_deterministically_and_cover_the_ring() -> None:
    assert len(axes_module.AXES) == 7
    first = axes_module.rotate(6, offset=0)
    assert [a.key for a in first] == [a.key for a in axes_module.rotate(6, offset=0)]
    assert len({a.key for a in first}) == 6, "no axis is used twice in one round"

    shifted = axes_module.rotate(6, offset=3)
    assert shifted[0].key == axes_module.AXES[3].key
    assert axes_module.offset_for("n_abc") == axes_module.offset_for("n_abc")


# --- MMR ----------------------------------------------------------------------


def test_mmr_reaches_across_clusters_when_quality_is_equal() -> None:
    quality = [1.0] * 6
    picked = mmr_select(quality, similarity(CLUSTERED), 3, lambda_=0.5)
    assert len({CLUSTER_OF[i] for i in picked}) == 3, (
        "equal quality and three clusters means one from each, not three from one"
    )


def test_mmr_at_lambda_one_is_plain_top_k() -> None:
    quality = [0.1, 0.9, 0.8, 0.2, 0.3, 0.4]
    assert mmr_select(quality, similarity(CLUSTERED), 3, lambda_=1.0) == [1, 2, 5]
    assert topk_select(quality, similarity(CLUSTERED), 3) == [1, 2, 5]


def test_mmr_at_lambda_zero_is_maximal_spread() -> None:
    quality = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
    picked = mmr_select(quality, similarity(CLUSTERED), 3, lambda_=0.0)
    assert len({CLUSTER_OF[i] for i in picked}) == 3


def test_mmr_handles_degenerate_inputs() -> None:
    assert mmr_select([], np.zeros((0, 0)), 3) == []
    assert mmr_select([1.0], np.ones((1, 1)), 5) == [0]


# --- DPP ----------------------------------------------------------------------


def test_dpp_spreads_at_least_as_well_as_the_temperature_baseline() -> None:
    quality = [0.95, 0.94, 0.93, 0.6, 0.6, 0.55]
    sim = similarity(CLUSTERED)

    def mean_distance(indices: list[int]) -> float:
        pairs = [1.0 - float(sim[a][b]) for i, a in enumerate(indices) for b in indices[i + 1 :]]
        return sum(pairs) / len(pairs)

    dpp = dpp_select(quality, sim, 3)
    baseline = topk_select(quality, sim, 3)
    assert mean_distance(dpp) >= mean_distance(baseline)
    assert baseline == [0, 1, 2], "the baseline takes three near-duplicates"
    assert len({CLUSTER_OF[i] for i in dpp}) > 1


def test_dpp_kernel_is_quality_weighted_similarity() -> None:
    kernel = build_kernel([1.0, 0.5], np.array([[1.0, 0.8], [0.8, 1.0]]))
    assert kernel[0][0] == pytest.approx(1.0, abs=1e-6)
    assert kernel[1][1] == pytest.approx(0.25, abs=1e-6)
    assert kernel[0][1] == pytest.approx(0.4, abs=1e-6)


def test_all_three_selectors_share_one_signature() -> None:
    from storygit.selection.select import SELECTORS

    quality = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    sim = similarity(CLUSTERED)
    for selector, fn in SELECTORS.items():
        picked = fn(quality, sim, 3, lambda_=0.7)
        assert len(picked) == 3, selector
        assert len(set(picked)) == 3, selector


# --- the dial -----------------------------------------------------------------


def test_dial_at_zero_ranks_by_quality_and_at_one_by_distance() -> None:
    greedy = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    candidates = normalize_rows(
        np.array([[1.0, 0.02, 0.0], [0.7, 0.7, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    )
    quality = [0.9, 0.5, 0.1]
    surprise = surprise_scores(candidates, greedy)
    assert surprise[0] < surprise[1] < surprise[2], "distance from the obvious continuation"

    coherent = effective_quality(quality, surprise, 0.0)
    surprising = effective_quality(quality, surprise, 1.0)
    midpoint = effective_quality(quality, surprise, 0.5)

    assert coherent.index(max(coherent)) == 0
    assert surprising.index(max(surprising)) == 2
    assert min(coherent[1], surprising[1]) <= midpoint[1] <= max(coherent[1], surprising[1])


def test_min_max_keeps_a_tie_a_tie() -> None:
    assert min_max([3.0, 3.0, 3.0]) == [0.5, 0.5, 0.5]
    assert min_max([]) == []
    assert min_max([1.0, 3.0]) == [0.0, 1.0]


# --- the whole pipeline -------------------------------------------------------


async def test_axis_fragments_reach_the_prompts_and_the_labels_survive(
    fixture: Fixture,
) -> None:
    provider = MockProvider(
        lambda req: canned(beat_payload(f"beat {req.sample_index}", "something happens"))
    )
    router = Router({"gemini": provider, "groq": provider})
    from storygit.agents.propose import Proposer

    selector = CandidateSelector(
        Proposer(router, IdGenerator(seed=5)),
        router,
        SelectionConfig(
            n=6, k=3, selector=Selector.topk_temperature, use_judge=False, use_dial=False
        ),
    )
    candidates = await selector.select(
        fixture.repo.state(), Level.beat, target_node_id=fixture.scene, intent="go on"
    )

    assert len(candidates) == 6
    assert sum(1 for c in candidates if c.selected) == 3
    assert candidates[0].selected and not candidates[-1].selected, "selected come first"

    labels = {c.axis_label for c in candidates}
    assert len(labels) == 6, "six distinct axis labels survive to the writer"
    assert all(c.axis_key for c in candidates)

    prompts = provider.prompts_for("propose.beat")
    fragments = {axis.fragment for axis in axes_module.AXES}
    used = [f for f in fragments if any(f in p for p in prompts)]
    assert len(used) == 6, "each parallel call carries a different axis instruction"

    indices = [r.sample_index for r in provider.requests if r.purpose == "propose.beat"]
    assert len(set(indices)) >= 1
    assert len({"".join(m.content for m in r.messages) for r in provider.requests}) == 6, (
        "six distinct prompts means six distinct cache keys"
    )


async def test_selector_choice_is_one_config_value(fixture: Fixture) -> None:
    from storygit.agents.propose import Proposer

    def make(selector: Selector) -> CandidateSelector:
        provider = MockProvider(
            lambda req: canned(beat_payload(f"beat {req.sample_index}", "x" * 20))
        )
        router = Router({"gemini": provider, "groq": provider})
        return CandidateSelector(
            Proposer(router, IdGenerator(seed=5)),
            router,
            SelectionConfig(n=4, k=2, selector=selector, use_judge=False, use_dial=False),
        )

    for selector in (Selector.mmr, Selector.dpp, Selector.topk_temperature):
        candidates = await make(selector).select(
            fixture.repo.state(), Level.beat, target_node_id=fixture.scene
        )
        assert sum(1 for c in candidates if c.selected) == 2, selector


async def test_a_candidate_that_cannot_apply_is_flagged_not_dropped(
    fixture: Fixture,
) -> None:
    from storygit.agents.propose import Proposer

    payload = beat_payload("bad beat", "x")
    payload["consumes"] = []
    provider = MockProvider([canned(payload)])
    router = Router({"gemini": provider, "groq": provider})
    selector = CandidateSelector(
        Proposer(router, IdGenerator(seed=5)),
        router,
        SelectionConfig(
            n=1, k=1, selector=Selector.topk_temperature, use_judge=False, use_dial=False
        ),
    )
    candidates = await selector.select(
        fixture.repo.state(), Level.beat, target_node_id=fixture.scene
    )
    assert len(candidates) == 1
    assert candidates[0].proposal.diff.ops


def test_candidate_text_reads_from_the_diff(fixture: Fixture) -> None:
    from storygit.agents.propose import Proposal
    from storygit.domain.diff import AddNode, Diff
    from storygit.domain.ids import NodeId, ProposalId
    from storygit.domain.nodes import Beat

    proposal = Proposal(
        id=ProposalId("p_1"),
        level=Level.beat,
        target_node_id=fixture.scene,
        diff=Diff(
            ops=(
                AddNode(
                    node=Beat(
                        id=NodeId("n_x"),
                        parent_id=fixture.scene,
                        title="The offer",
                        what_happens="The Warden offers a way out.",
                    )
                ),
            )
        ),
    )
    text = candidate_text(proposal)
    assert "The offer" in text and "way out" in text


async def test_the_greedy_continuation_costs_one_extra_call_per_node(
    fixture: Fixture,
) -> None:
    """The dial must cost at most one extra call, and moving it must cost none.

    Surprise is measured against the model's own temperature-0 continuation. Generating
    that per *candidate* rather than per node would multiply the dial's cost by n, and
    re-generating it when the writer moves the slider would make the dial feel expensive —
    which is exactly what would stop them using it.
    """
    from storygit.agents.propose import Proposer

    calls: list[float] = []

    def handler(req: object) -> str:
        calls.append(getattr(req, "temperature", 1.0))
        return canned(beat_payload("beat", "something happens here at some length"))

    provider = MockProvider(handler)
    router = Router({"gemini": provider, "groq": provider})
    selector = CandidateSelector(
        Proposer(router, IdGenerator(seed=5)),
        router,
        SelectionConfig(n=4, k=2, selector=Selector.mmr, use_judge=False, use_dial=True),
    )

    from storygit.domain.diff import Diff, SetDial

    state = fixture.repo.preview_apply(Diff(ops=(SetDial(value=0.6),)))
    await selector.select(state, Level.beat, target_node_id=fixture.scene)

    greedy = [t for t in calls if t == 0.0]
    assert len(greedy) == 1, f"the greedy continuation ran {len(greedy)} times, not once"
    assert len(calls) == 5, "four candidates plus one greedy continuation"

    # Moving the dial re-ranks the candidates already in hand; it does not regenerate.
    from storygit.selection.dial import effective_quality

    quality = [0.9, 0.5, 0.2]
    surprise = [0.1, 0.5, 0.9]
    before = len(calls)
    coherent = effective_quality(quality, surprise, 0.0)
    surprising = effective_quality(quality, surprise, 1.0)
    assert len(calls) == before, "re-ranking makes no calls at all"
    assert coherent.index(max(coherent)) != surprising.index(max(surprising))


async def test_the_greedy_continuation_is_skipped_when_the_dial_is_at_zero(
    fixture: Fixture,
) -> None:
    """At dial 0 the surprise term is multiplied by zero, so generating it is waste."""
    from storygit.agents.propose import Proposer

    calls: list[float] = []
    provider = MockProvider(
        lambda req: (calls.append(req.temperature), canned(beat_payload("b", "x" * 30)))[1]
    )
    router = Router({"gemini": provider, "groq": provider})
    selector = CandidateSelector(
        Proposer(router, IdGenerator(seed=5)),
        router,
        SelectionConfig(n=3, k=2, selector=Selector.mmr, use_judge=False, use_dial=True),
    )

    from storygit.domain.diff import Diff, SetDial

    state = fixture.repo.preview_apply(Diff(ops=(SetDial(value=0.0),)))
    await selector.select(state, Level.beat, target_node_id=fixture.scene)
    assert 0.0 not in calls, "the greedy continuation was generated and then ignored"
    assert len(calls) == 3
