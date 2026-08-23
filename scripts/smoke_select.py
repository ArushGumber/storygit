#!/usr/bin/env python
"""One live `select_candidates` on the orphan seed, to see the labelled options for real.

    .venv/bin/python scripts/smoke_select.py

Six axis-conditioned candidates on Gemini, checked by layer 1, judged by layer 3, ranked
through the dial and MMR, three shown. Prints what the writer would see. Nothing touches
OpenRouter.
"""

from __future__ import annotations

import asyncio
import sys

from storygit.agents.propose import Proposer
from storygit.agents.schemas import Level
from storygit.config import get_settings
from storygit.domain.diff import AddEntity, AddNode, Diff, DiffAuthor, SetDial
from storygit.domain.ids import IdGenerator
from storygit.domain.nodes import Beat, Episode, NodeType, Scene, Story
from storygit.domain.state import StoryState
from storygit.domain.world import Entity, EntityKind
from storygit.providers.router import build_router
from storygit.selection.select import CandidateSelector, SelectionConfig
from storygit.store.repository import Repository

SEED = (
    "A powerless orphan discovers an ability that could change the balance of power "
    "in a world at war."
)


def build_story(ids: IdGenerator) -> Repository:
    """A one-scene story with a beat already written, to propose the next beat after."""
    repo = Repository.open(":memory:")
    story_id, episode_id, scene_id, beat_id = (ids.node() for _ in range(4))
    kael, ashfall, warden = ids.entity(), ids.entity(), ids.entity()

    repo.initialize(
        StoryState.build(
            nodes={
                story_id: Story(
                    id=story_id,
                    title="Ashfall",
                    seed=SEED,
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
                AddNode(
                    node=Episode(
                        id=episode_id,
                        parent_id=story_id,
                        title="Episode 1: Ashfall",
                        what_happens="Kael is caught stealing and something answers him.",
                        hook="A boy with nothing steals from the one person who would notice.",
                    )
                ),
                AddNode(
                    node=Scene(
                        id=scene_id,
                        parent_id=episode_id,
                        title="The market",
                        what_happens="Kael is cornered behind the fish stalls.",
                        location="Ashfall market",
                    )
                ),
                AddNode(
                    node=Beat(
                        id=beat_id,
                        parent_id=scene_id,
                        title="Caught",
                        what_happens=(
                            "Two of the Warden's men corner Kael with a stolen waterskin."
                        ),
                    )
                ),
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
                SetDial(value=0.4),
            ),
            author=DiffAuthor.human,
            intent="seed the story",
        )
    )
    return repo


async def main() -> int:
    """Run one selection and print the labelled candidates."""
    settings = get_settings()
    if settings.openrouter_is_enabled:
        print("REFUSING TO RUN: OpenRouter is enabled and this script does not use it.")
        return 2

    ids = IdGenerator(seed=31337, stream="smoke_select")
    repo = build_story(ids)
    # Per-invocation call log: the shared one accumulates across every run of everything,
    # and the number this script exists to show is what *this* selection cost. The cache
    # stays shared, so a repeat run is free.
    router = build_router(settings, calllog_path=":memory:")
    selector = CandidateSelector(
        Proposer(router, ids), router, SelectionConfig(n=6, k=3, use_judge=True, use_dial=True)
    )

    state = repo.state()
    scene_id = state.nodes_of_type(NodeType.scene)[0].id
    print(f"dial: {state.ledger.dial}   selector: {selector.config.selector.value}\n")

    try:
        candidates = await selector.select(
            state,
            Level.beat,
            target_node_id=scene_id,
            intent="Kael's ability shows itself, but only he understands what happened.",
        )
        for c in candidates:
            mark = "SHOWN" if c.selected else "     "
            print(f"{mark}  [{c.axis_label}]")
            print(
                f"       quality={c.base_quality:.2f} surprise={c.surprise:.2f} "
                f"effective={c.effective_quality:.2f}"
            )
            for line in c.proposal.delta_summary[:4]:
                print(f"       - {line}")
            if c.proposal.rationale:
                print(f"       why: {c.proposal.rationale[:150]}")
            for note in c.proposal.notes:
                print(f"       note: {note}")
            for flag in c.flags:
                print(f"       [{flag.severity.value}/{flag.layer}] {flag.message[:150]}")
            print()
    finally:
        print("--- call log ---------------------------------------------------")
        print(router.calllog.render_summary())
        await router.aclose()
        repo.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
