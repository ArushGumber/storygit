#!/usr/bin/env python
"""One tiny live call to each provider, to prove the rotation and cache work for real.

Run it twice. The first run makes two network calls; the second makes none, because both
are served from the disk cache. That second run is the point: it is what makes a
15-episode evaluation affordable on a free tier.

    .venv/bin/python scripts/smoke_live.py

Nothing here touches OpenRouter, and nothing prints a key.
"""

from __future__ import annotations

import asyncio
import sys

from storygit.agents.propose import Proposer
from storygit.agents.schemas import Level
from storygit.config import get_settings
from storygit.domain.diff import AddEntity, AddNode, Diff, DiffAuthor
from storygit.domain.ids import IdGenerator
from storygit.domain.nodes import Beat, Episode, NodeType, Scene, Story
from storygit.domain.state import StoryState
from storygit.domain.world import Entity, EntityKind
from storygit.providers.router import build_router
from storygit.store.repository import Repository

SEED = (
    "A powerless orphan discovers an ability that could change the balance of power "
    "in a world at war."
)


def build_story(ids: IdGenerator) -> tuple[Repository, StoryState]:
    """An in-memory story with one episode, one scene, and one beat to write into.

    Args:
        ids: The session's id generator. One generator is shared with the proposer, so
            nothing can mint a colliding id.

    Returns:
        The repository and its current state.
    """
    repo = Repository.open(":memory:")
    story_id, episode_id, scene_id, beat_id = (ids.node() for _ in range(4))
    kael = ids.entity()

    repo.initialize(
        StoryState.build(nodes={story_id: Story(id=story_id, title="Ashfall", seed=SEED)})
    )
    repo.commit_diff(
        Diff(
            ops=(
                AddNode(node=Episode(id=episode_id, parent_id=story_id, title="Episode 1")),
                AddNode(node=Scene(id=scene_id, parent_id=episode_id, title="The market")),
                AddNode(
                    node=Beat(
                        id=beat_id,
                        parent_id=scene_id,
                        title="Caught",
                        what_happens="Kael is caught stealing.",
                    )
                ),
                AddEntity(entity=Entity(id=kael, kind=EntityKind.character, name="Kael")),
            ),
            author=DiffAuthor.human,
            intent="smoke fixture",
        )
    )
    return repo, repo.state()


async def main() -> int:
    """Run one Gemini proposal and one Groq extraction, then print the call log."""
    settings = get_settings()
    print(f"gemini keys configured: {len(settings.gemini_keys)}")
    print(f"gemini model: {settings.gemini_model}  fallbacks: {settings.gemini_fallbacks}")
    print(f"groq model: {settings.groq_model}")
    print(f"openrouter enabled: {settings.openrouter_is_enabled}  (must be False)")
    if settings.openrouter_is_enabled:
        print("REFUSING TO RUN: OpenRouter is enabled and this script does not use it.")
        return 2

    ids = IdGenerator(seed=2026, stream="smoke")
    repo, state = build_story(ids)
    # A fresh call log per invocation. The shared one on disk accumulates across every run
    # of everything, so printing it here would report a couple of hundred calls for a
    # script whose entire point is that it makes two. The cache stays shared, because the
    # second run being free is the thing this script demonstrates.
    router = build_router(settings, calllog_path=":memory:")
    proposer = Proposer(router, ids)

    try:
        scene_id = state.nodes_of_type(NodeType.scene)[0].id
        beat_id = state.nodes_of_type(NodeType.beat)[0].id

        print("\n--- propose.beat on Gemini -------------------------------------")
        proposals = await proposer.propose(
            state,
            Level.beat,
            target_node_id=scene_id,
            intent="Kael discovers what he can do, but only he notices.",
            k=1,
        )
        if not proposals:
            print("no proposal parsed; see the call log below")
        for proposal in proposals:
            print(f"title:     {proposal.diff.ops[0].node.title!r}")  # type: ignore[union-attr]
            print(f"rationale: {proposal.rationale}")
            for line in proposal.delta_summary:
                print(f"  - {line}")

        print("\n--- extract.facts on Groq --------------------------------------")
        diff = await proposer.extract(
            state,
            beat_id,
            "Kael pressed his blistered hand against his coat and said nothing. "
            "The Warden watched him the whole way across the square.",
        )
        for op in diff.ops:
            print(f"  {type(op).__name__}")

    finally:
        print("\n--- call log ---------------------------------------------------")
        print(router.calllog.render_summary())
        print(f"\ncache: {router.cache.stats()}")
        await router.aclose()
        repo.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
