"""The seed story: the premise from the assignment, and the cast it starts with.

Lives in the library rather than in the evaluation harness because both need it — the
interface opens onto it, and every simulated run starts from it — and because a deliverable
package should not import from its own test harness.
"""

from __future__ import annotations

from storygit.domain.diff import AddEntity, Diff, DiffAuthor
from storygit.domain.ids import IdGenerator
from storygit.domain.nodes import Story
from storygit.domain.state import StoryState
from storygit.domain.world import Entity, EntityKind
from storygit.store.repository import Repository

SEED = (
    "A powerless orphan discovers an ability that could change the balance of power "
    "in a world at war."
)
"""The writer's one line, verbatim from the assignment."""

PREMISE = (
    "Kael, an orphan in a city under siege, discovers that the ash falling over Ashfall "
    "answers him."
)

EXISTENTIAL_QUESTION = "Is power worth what it costs the powerless?"


def seed_story(repo: Repository, ids: IdGenerator, *, branch: str = "main") -> None:
    """Initialize a repository with the seed premise and its opening cast.

    A brand new story with no root makes every read return an empty tree, which is
    indistinguishable from a bug. Seeding on first open means the tool opens onto something.

    Args:
        repo: The repository to seed.
        ids: Id generator.
        branch: Branch to create.
    """
    story_id = ids.node()
    kael, ashfall, warden = ids.entity(), ids.entity(), ids.entity()
    repo.initialize(
        StoryState.build(
            nodes={
                story_id: Story(
                    id=story_id,
                    title="Ashfall",
                    seed=SEED,
                    premise=PREMISE,
                    existential_question=EXISTENTIAL_QUESTION,
                )
            }
        ),
        branch=branch,
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
        ),
        branch=branch,
    )
