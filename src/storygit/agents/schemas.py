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

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


def _declared_max_length(field: Any) -> int | None:
    """The ``max_length`` a field declares, if any."""
    for constraint in getattr(field, "metadata", ()):
        limit = getattr(constraint, "max_length", None)
        if limit is not None:
            return int(limit)
    return None


SENTENCE_END = re.compile(r"[.!?][\"\u201d\')\]]*(?=\s|$)")
"""Terminal punctuation, plus any closing quote or bracket that belongs with it."""

MIN_SENTENCE_CHARS = 24
"""Below this a "sentence" is a fragment the model was still assembling."""

DEGENERATE_MIN_LENGTH = 40
"""Only text long enough for a pattern to be visible is checked for one."""

DISTINCT_CHARS_FLOOR = 12
"""Below this many distinct characters, text of any length is padding rather than prose."""


def looks_degenerate(text: str) -> bool:
    """Whether a field is padding rather than writing.

    The writer's session produced a ``what_happens`` that was literal keyboard mash padded
    out to a minimum length, and it was committed and then read back as context by every
    later prompt. Two cheap signals catch that without touching real prose: too few
    distinct characters for the length, and one short chunk repeated to fill the space.

    Args:
        text: The field value.

    Returns:
        True when the text is padding.
    """
    stripped = text.strip()
    if len(stripped) < DEGENERATE_MIN_LENGTH:
        return False
    # Absolute counts, not ratios. A distinct-character *ratio* falls as text gets longer
    # whatever the text is -- the alphabet is bounded and the length is not -- so the first
    # version of this flagged four sentences of ordinary prose. English prose past forty
    # characters is never below a dozen distinct characters; "asdfasdf..." is four.
    if len(set(stripped.lower())) < DISTINCT_CHARS_FLOOR:
        return True
    if re.search(r"(.)\1{9,}", stripped):
        return True
    # One sentence repeated to fill the space. Four is enough to be deliberate; a writer
    # repeating a line three times for rhythm is doing something on purpose.
    sentences = [part.strip() for part in SENTENCE_END.split(stripped) if part.strip()]
    return len(sentences) >= 4 and len(set(sentences)) == 1


def _cut_at_sentence(text: str, limit: int) -> str | None:
    """Trim to ``limit`` at the last sentence boundary inside it.

    Args:
        text: The over-long value.
        limit: Its declared bound.

    Returns:
        The trimmed text, or ``None`` when nothing inside the bound ends a sentence --
        which means every cut available is mid-clause, and the caller should re-ask rather
        than commit a fragment.
    """
    window = text[:limit]
    ends = [m.end() for m in SENTENCE_END.finditer(window)]
    if not ends or ends[-1] < MIN_SENTENCE_CHARS:
        return None
    return window[: ends[-1]].rstrip()


class BoundedModel(BaseModel):
    """A response model whose length bounds truncate rather than reject.

    The bounds exist to keep proposals readable and to stop a small model running away
    inside one string field. They are forwarded into the provider's response schema, but
    **Gemini does not enforce ``maxLength``** -- verified against the live API, where a
    ``time`` field bounded at 80 characters came back with 300.

    Rejecting on that costs a whole repair call to recover a candidate that was fine
    except for being twenty characters long. During the first evaluation run it was
    happening on roughly 80% of proposals, doubling the cost of the entire experiment.
    That history is why the contract is truncate-then-re-ask and not reject.

    But truncation as originally written cut mid-word, and the fragment became permanent
    state that every later prompt read back as context -- roughly a third of one writer's
    candidates ended mid-clause. So the contract has three steps, in order:

    1. **Truncate at a sentence boundary.** A bounded field ends where a sentence ends.
    2. **Re-ask once** when nothing inside the bound ends a sentence, or when the value is
       degenerate -- padding, repetition, keyboard mash. Raising here reaches
       ``complete_structured``, which repairs once by construction.
    3. **Fail the candidate** if the second attempt is no better. A dropped candidate costs
       one sample out of six; a committed fragment costs every prompt that reads it after.
    """

    @model_validator(mode="before")
    @classmethod
    def _truncate_to_bounds(cls, data: Any) -> Any:
        """Trim over-long strings to a sentence boundary and over-long lists to bounds.

        Raises:
            ValueError: When a field is degenerate, or when trimming it to its bound would
                leave no complete sentence. ``parse_into`` turns this into a
                ``SchemaParseError``, which is the signal to re-ask.
        """
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for name, field in cls.model_fields.items():
            limit = _declared_max_length(field)
            if limit is None or name not in out:
                continue
            value = out[name]
            if isinstance(value, list) and len(value) > limit:
                out[name] = value[:limit]
                continue
            if not isinstance(value, str):
                continue
            if looks_degenerate(value):
                raise ValueError(f"{name!r} is padding rather than writing; write it again")
            if len(value) <= limit:
                continue
            trimmed = _cut_at_sentence(value, limit)
            if trimmed is None:
                raise ValueError(
                    f"{name!r} runs past its {limit}-character bound with no sentence "
                    "ending inside it; say it in fewer words"
                )
            out[name] = trimmed
        return out


class CharacterDraft(BoundedModel):
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


class FactDraft(BoundedModel):
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


class ProposalBase(BoundedModel):
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


class ExtractionResult(BoundedModel):
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
