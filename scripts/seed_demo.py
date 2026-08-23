#!/usr/bin/env python
"""Create the demo story's starting point: Arush's premise and its opening cast.

The evaluation's seed is the assignment's own one-liner (an orphan, a city at war), which
is the right story to measure against and the wrong one to demo with — a fantasy premise
takes a minute of setup before anything is funny, and a demo has thirty seconds. This one
is a comedy that explains itself in a sentence.

    scripts/seed_demo.py [--db demo/story.db]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from storygit.domain.diff import AddEntity, Diff, DiffAuthor
from storygit.domain.ids import IdGenerator
from storygit.domain.nodes import Story
from storygit.domain.state import StoryState
from storygit.domain.world import Entity, EntityKind
from storygit.store.repository import Repository

SEED = (
    "A small-time thief runs the broken-leg sympathy con -- cast, crutches, the whole "
    "performance -- while being perfectly fine, and the lie keeps almost falling apart."
)
"""The writer's one line."""

PREMISE = (
    "Ronnie Fenn has worn the same fibreglass cast for eleven weeks. It has made him a "
    "living, a reputation, and exactly one problem: everybody who matters now believes "
    "he cannot run."
)

EXISTENTIAL_QUESTION = "What do you owe the people who believed you?"


def seed(db: Path) -> None:
    """Write the demo story's root and opening cast.

    Args:
        db: Where to create the database.
    """
    db.parent.mkdir(parents=True, exist_ok=True)
    ids = IdGenerator(seed=11, stream="demo")
    repo = Repository.open(db)
    story_id = ids.node()
    ronnie, marguerite, dev, pharmacy = (ids.entity() for _ in range(4))

    repo.initialize(
        StoryState.build(
            nodes={
                story_id: Story(
                    id=story_id,
                    title="A Leg to Stand On",
                    seed=SEED,
                    premise=PREMISE,
                    existential_question=EXISTENTIAL_QUESTION,
                )
            }
        )
    )
    repo.commit_diff(
        Diff(
            ops=(
                AddEntity(
                    entity=Entity(
                        id=ronnie,
                        kind=EntityKind.character,
                        name="Ronnie Fenn",
                        description=(
                            "Thirty-four, charming in a way that does not survive scrutiny. "
                            "Has not needed the crutches since March."
                        ),
                    )
                ),
                AddEntity(
                    entity=Entity(
                        id=marguerite,
                        kind=EntityKind.character,
                        name="Marguerite Osei",
                        aliases=("Mrs Osei",),
                        description=(
                            "Runs the community centre. Organised the fundraiser. Believes "
                            "in Ronnie with a sincerity that is becoming a problem."
                        ),
                    )
                ),
                AddEntity(
                    entity=Entity(
                        id=dev,
                        kind=EntityKind.character,
                        name="Dev",
                        description=(
                            "Ronnie's oldest friend and a physiotherapist. Has offered, "
                            "warmly and repeatedly, to look at the leg."
                        ),
                    )
                ),
                AddEntity(
                    entity=Entity(
                        id=pharmacy,
                        kind=EntityKind.place,
                        name="Bellamy Road",
                        description=(
                            "Three shops, a bus stop, and everyone who has ever given Ronnie money."
                        ),
                    )
                ),
            ),
            author=DiffAuthor.human,
            intent="the opening cast",
        )
    )
    repo.close()


def main() -> None:
    """Seed the demo database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="demo/story.db")
    args = parser.parse_args()
    path = Path(args.db)
    if path.exists():
        path.unlink()
    seed(path)
    print(f"seeded {path}")


if __name__ == "__main__":
    main()
