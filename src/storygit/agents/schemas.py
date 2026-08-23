"""The shapes the model must fill, one per level of the plan tree.

Every generation call is schema-constrained: the model is handed a JSON Schema derived
from these classes and its output is parsed back through them. Two reasons. Free text
would have to be interpreted before it could become a diff, and interpretation is where
silent errors live. And a schema is where the *requirements* live — a beat proposal
cannot be returned without saying which facts it establishes and which it relies on,
because those fields are required, so the dependency graph cannot quietly go
unmaintained.

Every proposal schema also carries a ``rationale`` and a ``delta_summary``: the writer
never sees an unexplained candidate.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from storygit.domain.world import EntityKind, Predicate

# Field length bounds, forwarded into the model's response schema.
#
# They do two jobs. They keep proposals readable — a beat's `what_happens` should be one
# or two sentences, not an essay, and a writer scanning three candidates will not read
# more than that. And they are a hard stop on a real failure mode: a small model can fall
# into a repetition loop and fill a string field until it hits the token cap, which turns
# a perfectly good candidate into truncated JSON and costs a repair call to recover.
# Observed on gemini-3.1-flash-lite during the chunk 2 live smoke test.
NAME = 80
LINE = 200
PARAGRAPH = 600
PROSE_TEXT = 6000


class Level(StrEnum):
    """Which level of the plan tree a proposal targets.

    The progression is deliberate and is Fabula's lesson about progressive build-up:
    the writer settles the premise before episodes exist, and episodes before beats.
    """

    premise = "premise"
    episode = "episode"
    scene = "scene"
    beat = "beat"
    prose = "prose"


class CharacterDraft(BaseModel):
    """A character, place, object, or faction the proposal introduces."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(max_length=NAME, description="Canonical name, as the prose would use it.")
    kind: EntityKind = Field(default=EntityKind.character, description="What sort of thing.")
    description: str = Field(
        default="", max_length=LINE, description="One line: who or what this is."
    )
    aliases: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Other names the text might use for it.",
    )


class FactDraft(BaseModel):
    """One fact a beat establishes.

    ``object_is_entity`` tells the diff builder whether ``object`` names another entity
    (``Kael`` / ``location`` / ``Ashfall``) or is a literal (``Kael`` / ``goal`` /
    ``find his sister``).
    """

    model_config = ConfigDict(extra="ignore")

    subject: str = Field(max_length=NAME, description="Name of the entity the fact is about.")
    predicate: Predicate = Field(description="One of the allowed predicates.")
    object: str = Field(
        max_length=LINE, description="The other entity's name, or a short literal value."
    )
    object_is_entity: bool = Field(
        default=False, description="True when `object` names an entity rather than a literal."
    )
    known_by: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Characters who learn this fact at this beat. Leave empty if it is true but "
            "nobody in the scene is told. Getting this right is what stops a character "
            "acting on something they were never told."
        ),
    )


class ProposalBase(BaseModel):
    """Fields every proposal carries, whatever its level."""

    model_config = ConfigDict(extra="ignore")

    rationale: str = Field(
        default="",
        max_length=PARAGRAPH,
        description="One or two sentences: why this, for this story, now.",
    )
    delta_summary: list[str] = Field(
        default_factory=list,
        max_length=6,
        description="Short phrases naming what this changes, in the writer's language.",
    )


class PremiseProposal(ProposalBase):
    """The story's premise, its central question, and its opening cast."""

    premise: str = Field(
        max_length=PARAGRAPH, description="Two or three sentences developing the seed."
    )
    existential_question: str = Field(
        max_length=LINE, description="The question the story exists to ask, phrased as a question."
    )
    title: str = Field(default="", max_length=NAME, description="A working title.")
    main_characters: list[CharacterDraft] = Field(
        default_factory=list, max_length=6, description="Two to four characters to start with."
    )


class EpisodeProposal(ProposalBase):
    """One episode of the serial, with the mechanics that make it a serial."""

    title: str = Field(max_length=NAME)
    what_happens: str = Field(max_length=PARAGRAPH)
    audience_learns: str = Field(default="", max_length=LINE)
    audience_feels: str = Field(default="", max_length=LINE)
    location: str = Field(default="", max_length=NAME)
    time: str = Field(default="", max_length=NAME)
    hook: str = Field(
        default="", max_length=LINE, description="The opening pull that earns the first minute."
    )
    cliffhanger: str = Field(
        default="", max_length=LINE, description="The closing pull that earns the next episode."
    )
    recap_of_previous: str = Field(
        default="", max_length=LINE, description="What a returning listener needs re-established."
    )
    threads_opened: list[str] = Field(
        default_factory=list, max_length=4, description="New open questions this episode raises."
    )
    threads_touched: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Ids of existing threads this episode advances.",
    )
    threads_closed: list[str] = Field(
        default_factory=list,
        max_length=4,
        description="Ids of existing threads this episode pays off.",
    )
    target_length: int | None = Field(default=None, description="Target length in words.")


class SceneProposal(ProposalBase):
    """A continuous unit of place and time inside an episode."""

    title: str = Field(max_length=NAME)
    what_happens: str = Field(max_length=PARAGRAPH)
    audience_learns: str = Field(default="", max_length=LINE)
    audience_feels: str = Field(default="", max_length=LINE)
    location: str = Field(default="", max_length=NAME)
    time: str = Field(default="", max_length=NAME)


class BeatProposal(ProposalBase):
    """The smallest planned unit, and the only one that declares dependencies."""

    title: str = Field(max_length=NAME)
    what_happens: str = Field(max_length=PARAGRAPH)
    audience_learns: str = Field(default="", max_length=LINE)
    audience_feels: str = Field(default="", max_length=LINE)
    location: str = Field(default="", max_length=NAME)
    time: str = Field(default="", max_length=NAME)
    produces: list[FactDraft] = Field(
        default_factory=list,
        max_length=8,
        description="Facts this beat establishes. Everything the beat makes true.",
    )
    consumes: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Ids of already-established facts this beat relies on, copied from the "
            "ESTABLISHED FACTS list. If this beat only works because something earlier "
            "is true, say so here."
        ),
    )
    threads_touched: list[str] = Field(
        default_factory=list, max_length=4, description="Ids of open threads this beat advances."
    )
    new_characters: list[CharacterDraft] = Field(
        default_factory=list,
        max_length=4,
        description="Entities appearing here for the first time.",
    )


class ProseProposal(ProposalBase):
    """Written prose for one beat."""

    text: str = Field(
        max_length=PROSE_TEXT, description="The prose itself. No headings, no commentary."
    )


class ExtractionResult(BaseModel):
    """What a fact-extraction call returns for a passage of prose."""

    model_config = ConfigDict(extra="ignore")

    facts: list[FactDraft] = Field(
        default_factory=list, max_length=12, description="Facts the passage establishes."
    )
    new_characters: list[CharacterDraft] = Field(
        default_factory=list, max_length=6, description="Entities the passage introduces."
    )
    threads_opened: list[str] = Field(
        default_factory=list, max_length=4, description="New open questions the passage raises."
    )
    threads_touched: list[str] = Field(
        default_factory=list, max_length=4, description="Ids of open threads the passage advances."
    )


SCHEMA_FOR_LEVEL: dict[Level, type[ProposalBase]] = {
    Level.premise: PremiseProposal,
    Level.episode: EpisodeProposal,
    Level.scene: SceneProposal,
    Level.beat: BeatProposal,
    Level.prose: ProseProposal,
}
"""Which response model each level expects."""
