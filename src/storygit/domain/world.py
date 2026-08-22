"""The world state: entities, temporally-scoped facts, and who knows what.

This is the half of the state that stops the story contradicting itself. Facts are
edges in a graph with validity intervals over the beat sequence, so "Kael is in
Ashfall" can be true for beats 3–7 and false afterwards without either version
being wrong. Epistemic edges (`Knows`) are separate on purpose: the worst
continuity failure in generated fiction is a character acting on information they
have not been told.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from storygit.domain.ids import EntityId, FactId, NodeId


class EntityKind(StrEnum):
    """What sort of thing an entity is."""

    character = "character"
    place = "place"
    object = "object"
    faction = "faction"


class Predicate(StrEnum):
    """Closed vocabulary of fact predicates, plus a free-text escape hatch.

    A closed vocabulary is what makes layer 1 of the continuity checker a dict
    lookup and an equality test instead of a language-model call. ``note`` is the
    escape hatch for everything that will not fit, and it is exactly the set of
    facts that has to be checked with NLI in layer 2.
    """

    location = "location"
    alive = "alive"
    injury = "injury"
    possesses = "possesses"
    relationship = "relationship"
    relationship_status = "relationship_status"
    goal = "goal"
    secret = "secret"
    trait = "trait"
    faction = "faction"
    note = "note"


ENTITY_VALUED_PREDICATES: frozenset[Predicate] = frozenset(
    {Predicate.location, Predicate.possesses, Predicate.relationship, Predicate.faction}
)
"""Predicates whose object is normally another entity rather than a literal."""

SINGLE_VALUED_PREDICATES: frozenset[Predicate] = frozenset(
    {Predicate.location, Predicate.alive, Predicate.faction, Predicate.relationship_status}
)
"""Predicates where two simultaneously valid values are a contradiction.

A character is in one place at a time and is either alive or not; they may hold
several goals, traits, or possessions at once.
"""


class FactSource(StrEnum):
    """Where a fact came from."""

    human = "human"
    ai = "ai"
    ai_edited = "ai_edited"


class Entity(BaseModel):
    """A character, place, object, or faction.

    Attributes:
        id: Stable identifier.
        kind: What sort of thing it is.
        name: Canonical name.
        aliases: Other names the text uses for it; drives canonicalization when a
            proposal mentions "the Warden" and the bible knows "Warden of Kell".
        description: Free-text description shown in the world-state pane.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: EntityId
    kind: EntityKind
    name: str
    aliases: tuple[str, ...] = ()
    description: str = ""

    @property
    def names(self) -> tuple[str, ...]:
        """Canonical name followed by every alias."""
        return (self.name, *self.aliases)


class Fact(BaseModel):
    """One edge of the world graph, valid over a half-open interval of beats.

    Exactly one of ``object_entity`` and ``object_text`` is set: a fact points
    either at another entity ("Kael is in Ashfall") or at a literal ("Kael's goal
    is to find his sister").

    Attributes:
        id: Stable identifier.
        subject: Entity the fact is about.
        predicate: Closed-vocabulary relation.
        object_entity: The object, when it is another entity.
        object_text: The object, when it is a literal.
        valid_from_beat: First beat at which the fact holds.
        valid_until_beat: First beat at which it no longer holds; ``None`` means
            still true.
        established_by_beat: The beat that put it on the page. Every continuity
            flag cites this, which is what makes flags actionable.
        source: Whether a human, the AI, or a human edit of AI output asserted it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: FactId
    subject: EntityId
    predicate: Predicate
    object_entity: EntityId | None = None
    object_text: str = ""
    valid_from_beat: NodeId
    valid_until_beat: NodeId | None = None
    established_by_beat: NodeId
    source: FactSource = FactSource.ai

    @model_validator(mode="after")
    def _exactly_one_object(self) -> Fact:
        has_entity = self.object_entity is not None
        has_text = bool(self.object_text)
        if has_entity == has_text:
            raise ValueError(
                f"fact {self.id}: set exactly one of object_entity / object_text "
                f"(got entity={self.object_entity!r}, text={self.object_text!r})"
            )
        return self

    @property
    def object(self) -> str:
        """The object as a string, whichever form it took."""
        return self.object_entity if self.object_entity is not None else self.object_text

    @property
    def key(self) -> tuple[EntityId, Predicate]:
        """The ``(subject, predicate)`` pair layer 1 of the checker indexes on."""
        return (self.subject, self.predicate)

    def sentence(self, name_of: dict[EntityId, str] | None = None) -> str:
        """Render the fact as an English sentence.

        Used for NLI pairs in layer 2 and for the writer-facing bible diff.

        Args:
            name_of: Entity id to display name; ids are shown raw when absent.

        Returns:
            A sentence such as ``"Kael location Ashfall"`` humanized per predicate.
        """
        names = name_of or {}
        subject = names.get(self.subject, self.subject)
        obj = self.object
        if self.object_entity is not None:
            obj = names.get(self.object_entity, self.object_entity)
        templates = {
            Predicate.location: "{s} is at {o}.",
            Predicate.alive: "{s} is {o}.",
            Predicate.injury: "{s} is injured: {o}.",
            Predicate.possesses: "{s} has {o}.",
            Predicate.relationship: "{s} is related to {o}.",
            Predicate.relationship_status: "{s}'s relationships stand as: {o}.",
            Predicate.goal: "{s} wants {o}.",
            Predicate.secret: "{s} keeps a secret: {o}.",
            Predicate.trait: "{s} is {o}.",
            Predicate.faction: "{s} belongs to {o}.",
            Predicate.note: "{s}: {o}",
        }
        return templates[self.predicate].format(s=subject, o=obj)


class Knows(BaseModel):
    """An epistemic edge: a character learned a fact at a beat.

    Attributes:
        character: Who knows.
        fact: What they know.
        since_beat: The beat at which they learned it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    character: EntityId
    fact: FactId
    since_beat: NodeId

    @property
    def key(self) -> tuple[EntityId, FactId]:
        """Logical key of the edge in a snapshot manifest."""
        return (self.character, self.fact)
