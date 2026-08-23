"""Canonicalization: deciding whether "the Warden" is somebody we already know.

Every proposal mentions entities by name. Some of those names are already in the bible
under a different form; some are genuinely new. Getting this wrong in either direction is
bad — a duplicate entity fragments a character's facts across two nodes, and a wrong
merge welds two characters together — so the rule here is deliberately conservative:
match exactly (after normalization), otherwise create, and *tell the writer* that a new
entity was created so they can merge it in one click if it was a duplicate.

No embedding similarity is used for this. A fuzzy match that is occasionally wrong would
corrupt the world graph silently, which is exactly the failure this whole system exists
to avoid.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from storygit.domain.ids import EntityId, IdGenerator
from storygit.domain.state import StoryState, normalize_name
from storygit.domain.world import Entity, EntityKind


class Resolution(BaseModel):
    """The outcome of resolving one name.

    Attributes:
        name: The name as it appeared in the proposal.
        entity_id: The entity it resolved to, existing or newly minted.
        is_new: Whether this created an entity rather than matching one.
        note: A writer-facing line, set only when the writer should look at it.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    entity_id: EntityId
    is_new: bool
    note: str | None = None


class AliasResolver:
    """Resolves names to entity ids against a state, minting new ones as needed.

    New entities minted during one resolution pass are remembered, so a proposal that
    mentions "Mara" three times creates one entity, not three.
    """

    def __init__(self, state: StoryState, ids: IdGenerator) -> None:
        """Create a resolver.

        Args:
            state: The state to resolve against.
            ids: Id generator for new entities.
        """
        self._state = state
        self._ids = ids
        self._minted: dict[str, Entity] = {}

    @property
    def new_entities(self) -> tuple[Entity, ...]:
        """Entities minted so far, in creation order."""
        return tuple(self._minted.values())

    def resolve(
        self,
        name: str,
        *,
        kind: EntityKind = EntityKind.character,
        description: str = "",
        aliases: tuple[str, ...] = (),
    ) -> Resolution:
        """Resolve a name to an entity, creating one if it is genuinely unknown.

        Args:
            name: The name as written in the proposal.
            kind: What sort of entity to create if this is new.
            description: Description for a new entity.
            aliases: Aliases for a new entity.

        Returns:
            The resolution, with a writer-facing note when a new entity was created and
            an existing entity looks similar enough to be worth checking.
        """
        key = normalize_name(name)
        existing = self._state.alias_index.get(key)
        if existing is not None:
            return Resolution(name=name, entity_id=existing, is_new=False)
        if key in self._minted:
            return Resolution(name=name, entity_id=self._minted[key].id, is_new=True)

        entity = Entity(
            id=self._ids.entity(),
            kind=kind,
            name=name.strip(),
            aliases=tuple(dict.fromkeys(a.strip() for a in aliases if a.strip())),
            description=description,
        )
        self._minted[key] = entity
        return Resolution(
            name=name,
            entity_id=entity.id,
            is_new=True,
            note=self._similarity_note(name),
        )

    def _similarity_note(self, name: str) -> str:
        """A line telling the writer a new entity was created, and what it may duplicate.

        The suggestion is a substring check, not a similarity score: it is a prompt for
        the writer to look, never an automatic action.
        """
        key = normalize_name(name)
        near = [
            entity.name
            for entity in self._state.entities.values()
            if any(key in normalize_name(n) or normalize_name(n) in key for n in entity.names)
        ]
        base = f"introduces new entity “{name}”"
        if near:
            options = ", ".join(f"“{n}”" for n in sorted(set(near))[:3])
            return f"{base} — same as {options}? merge in the world-state pane"
        return base
