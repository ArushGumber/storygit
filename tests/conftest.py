"""Shared fixtures: a deterministic, offline story to test the state layer against."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from storygit.domain.diff import AddEntity, AddFact, AddNode, Diff, DiffAuthor, UpdateNode
from storygit.domain.ids import EntityId, FactId, IdGenerator, NodeId
from storygit.domain.nodes import Beat, Episode, Scene, Story
from storygit.domain.state import StoryState
from storygit.domain.world import Entity, EntityKind, Fact, FactSource, Predicate
from storygit.store.repository import Repository

FIXED_TIME = datetime(2026, 8, 23, 4, 0, 0, tzinfo=UTC)


def fixed_clock() -> datetime:
    """A clock that never moves, so snapshot ids are reproducible."""
    return FIXED_TIME


@dataclass
class Fixture:
    """A small four-beat story plus the ids the tests need to reach into it."""

    repo: Repository
    ids: IdGenerator
    story: NodeId
    episode: NodeId
    scene: NodeId
    beat_a: NodeId
    beat_b: NodeId
    beat_c: NodeId
    beat_d: NodeId
    kael: EntityId
    ashfall: EntityId
    kell: EntityId
    fact_f: FactId


@pytest.fixture
def repo() -> Repository:
    """An empty in-memory repository with a frozen clock."""
    return Repository.open(":memory:", clock=fixed_clock)


@pytest.fixture
def ids() -> IdGenerator:
    """A seeded id generator."""
    return IdGenerator(seed=1234)


@pytest.fixture
def fixture(repo: Repository, ids: IdGenerator) -> Fixture:
    """A story where beat B consumes what beat A produced, and C consumes B's output.

    Shape::

        story -> episode -> scene -> [A, B, C, D]
        A produces F (Kael is at Ashfall)
        B consumes F, produces G (Kael's goal)
        C consumes G
        D is unrelated
    """
    story_id = ids.node()
    episode_id = ids.node()
    scene_id = ids.node()
    beat_ids = [ids.node() for _ in range(4)]
    kael, ashfall, kell = ids.entity(), ids.entity(), ids.entity()
    fact_f, fact_g = ids.fact(), ids.fact()

    story = Story(id=story_id, title="The Ashfall Orphan", seed="A powerless orphan...")
    repo.initialize(StoryState.build(nodes={story_id: story}))

    structure = Diff(
        ops=(
            AddNode(node=Episode(id=episode_id, parent_id=story_id, title="Episode 1", position=0)),
            AddNode(node=Scene(id=scene_id, parent_id=episode_id, title="The market", position=0)),
            *(
                AddNode(
                    node=Beat(
                        id=beat_id,
                        parent_id=scene_id,
                        title=f"Beat {label}",
                        position=index,
                        what_happens=f"Something happens in beat {label}.",
                    )
                )
                for index, (beat_id, label) in enumerate(zip(beat_ids, "ABCD", strict=True))
            ),
            AddEntity(entity=Entity(id=kael, kind=EntityKind.character, name="Kael")),
            AddEntity(entity=Entity(id=ashfall, kind=EntityKind.place, name="Ashfall")),
            AddEntity(
                entity=Entity(
                    id=kell,
                    kind=EntityKind.character,
                    name="Warden of Kell",
                    aliases=("the Warden",),
                )
            ),
        ),
        author=DiffAuthor.human,
        intent="build the fixture story",
    )
    repo.commit_diff(structure)

    wiring = Diff(
        ops=(
            AddFact(
                fact=Fact(
                    id=fact_f,
                    subject=kael,
                    predicate=Predicate.location,
                    object_entity=ashfall,
                    valid_from_beat=beat_ids[0],
                    established_by_beat=beat_ids[0],
                    source=FactSource.ai,
                )
            ),
            AddFact(
                fact=Fact(
                    id=fact_g,
                    subject=kael,
                    predicate=Predicate.goal,
                    object_text="find his sister",
                    valid_from_beat=beat_ids[1],
                    established_by_beat=beat_ids[1],
                    source=FactSource.ai,
                )
            ),
        ),
        author=DiffAuthor.ai,
        intent="establish facts",
    )
    repo.commit_diff(wiring)

    consume = Diff(
        ops=(
            UpdateNode(node_id=beat_ids[1], fields={"consumes": [fact_f]}),
            UpdateNode(node_id=beat_ids[2], fields={"consumes": [fact_g]}),
        ),
        author=DiffAuthor.system,
        intent="declare dependencies",
    )
    repo.commit_diff(consume)

    return Fixture(
        repo=repo,
        ids=ids,
        story=story_id,
        episode=episode_id,
        scene=scene_id,
        beat_a=beat_ids[0],
        beat_b=beat_ids[1],
        beat_c=beat_ids[2],
        beat_d=beat_ids[3],
        kael=kael,
        ashfall=ashfall,
        kell=kell,
        fact_f=fact_f,
    )
