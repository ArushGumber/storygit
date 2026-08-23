"""Entity-scoped state slices — the only thing generation prompts ever see.

Sending the whole story to a model does not scale past a few episodes and does not
help: for a beat about Kael and the Warden in Ashfall, the model needs those three
entities' current facts, who knows what, the open threads that touch them, and the
plan path down to the beat. That is a few hundred tokens instead of tens of
thousands, and it is *retrieval over a typed graph* rather than over text chunks,
so what comes back is exactly true rather than approximately relevant.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from storygit.domain.ids import EntityId, NodeId
from storygit.domain.ledger import Criterion, StyleNote
from storygit.domain.state import StoryState
from storygit.domain.threads import Thread, ThreadStatus
from storygit.domain.world import Entity, Fact
from storygit.graph.dependency import hard_constraints

RECENT_REJECTIONS = 4
"""How many turned-down directions reach a prompt.

All of them would crowd out the state and, worse, would keep telling the model to avoid
something the story moved past ten beats ago. The newest few are the ones the writer still
means.
"""


class PathStep(BaseModel):
    """One ancestor of the target node, rendered for the prompt."""

    model_config = ConfigDict(frozen=True)

    node_id: NodeId
    node_type: str
    title: str
    what_happens: str


class KnowledgeItem(BaseModel):
    """One character's knowledge of one fact."""

    model_config = ConfigDict(frozen=True)

    character: EntityId
    character_name: str
    fact_id: str
    since_beat: NodeId


class StateSlice(BaseModel):
    """Everything generation is allowed to know about one moment in the story.

    Attributes:
        target_beat: The beat being written or planned.
        path: Story → episode → scene ancestors of the target.
        entities: Entities in scope.
        facts: Facts about those entities valid at the target beat.
        knowledge: Epistemic edges for those entities.
        threads: Every open thread. Not filtered to these entities: a dropped thread is
            invisible to every other check, so the slice carries all of them and lets the
            model see what is outstanding.
        hard_constraints: Locked facts and writer constraints; never violate.
        style_notes: Soft prose rules from the ledger.
        criteria: Writer-defined scoring axes, so the judge scores what the writer
            actually cares about.
        rejected_directions: The most recent directions the writer turned down, in their
            own words. The ledger has always described these as "fed to generation";
            until the fix pass nothing carried them into a prompt, so a writer who said
            "not this" got the same proposal again. Bounded to the newest few: a
            standing list of everything ever rejected would crowd out the state.
        recent_prose: The last few paragraphs, for voice continuity.
    """

    model_config = ConfigDict(frozen=True)

    target_beat: NodeId | None = None
    path: tuple[PathStep, ...] = ()
    entities: tuple[Entity, ...] = ()
    facts: tuple[Fact, ...] = ()
    knowledge: tuple[KnowledgeItem, ...] = ()
    threads: tuple[Thread, ...] = ()
    hard_constraints: tuple[str, ...] = ()
    style_notes: tuple[StyleNote, ...] = ()
    criteria: tuple[Criterion, ...] = ()
    rejected_directions: tuple[str, ...] = ()
    recent_prose: tuple[str, ...] = ()

    def render(self, names: dict[EntityId, str] | None = None) -> str:
        """Render the slice as the compact block that goes into a prompt.

        Args:
            names: Entity display names; derived from ``entities`` when omitted.

        Returns:
            A plain-text block, deliberately terse.
        """
        display = names or {e.id: e.name for e in self.entities}
        out: list[str] = []
        if self.path:
            out.append("STORY SO FAR")
            for step in self.path:
                label = step.title or step.node_type
                summary = f": {step.what_happens}" if step.what_happens else ""
                out.append(f"  [{step.node_type}] {label}{summary}")
        if self.entities:
            out.append("ENTITIES")
            for entity in self.entities:
                alias = f" (also: {', '.join(entity.aliases)})" if entity.aliases else ""
                desc = f" — {entity.description}" if entity.description else ""
                out.append(f"  {entity.name} [{entity.kind.value}]{alias}{desc}")
        if self.facts:
            # Fact ids are printed so a proposal can cite exactly which facts it
            # consumes; without them the model can only gesture at "the thing about Kael".
            out.append("ESTABLISHED FACTS (true right now, cite these ids in `consumes`)")
            out.extend(f"  [{fact.id}] {fact.sentence(display)}" for fact in self.facts)
        if self.knowledge:
            out.append("WHO KNOWS WHAT")
            out.extend(
                f"  {item.character_name} knows [{item.fact_id}] (since {item.since_beat})"
                for item in self.knowledge
            )
        if self.threads:
            out.append("OPEN THREADS (cite these ids when a beat advances one)")
            out.extend(f"  [{t.id}] {t.description} ({t.status.value})" for t in self.threads)
        if self.hard_constraints:
            out.append("HARD CONSTRAINTS (must not be violated)")
            out.extend(f"  {line}" for line in self.hard_constraints)
        if self.style_notes:
            out.append("STYLE NOTES")
            out.extend(f"  {note.text}" for note in self.style_notes)
        if self.criteria:
            out.append("THE WRITER'S OWN CRITERIA")
            out.extend(f"  {c.name}: {c.description}" for c in self.criteria)
        if self.recent_prose:
            out.append("RECENT PROSE (for voice)")
            out.extend(f"  {para}" for para in self.recent_prose)
        return "\n".join(out)


def entity_slice(
    state: StoryState,
    entity_ids: set[EntityId],
    at_beat: NodeId | None,
    *,
    recent_prose: int = 2,
    mined_threshold: int = 2,
) -> StateSlice:
    """Build the slice for a set of entities at a point in the story.

    Args:
        state: The state to slice.
        entity_ids: Entities in scope. Facts about anything else are excluded.
        at_beat: The beat to evaluate fact validity at; ``None`` means the end of
            the story so far.
        recent_prose: How many preceding prose nodes to include for voice.
        mined_threshold: Observation count a mined style rule needs before it is
            allowed to influence generation.

    Returns:
        A serializable slice.
    """
    beats = state.beats_in_order()
    target = at_beat if at_beat is not None else (beats[-1].id if beats else None)

    path: list[PathStep] = []
    if target is not None and target in state.nodes:
        for ancestor in (*state.ancestors_of(target), state.nodes[target]):
            path.append(
                PathStep(
                    node_id=ancestor.id,
                    node_type=ancestor.node_type.value,
                    title=ancestor.title,
                    what_happens=ancestor.what_happens,
                )
            )

    entities = tuple(state.entities[eid] for eid in sorted(entity_ids) if eid in state.entities)
    facts: tuple[Fact, ...] = ()
    if target is not None:
        facts = tuple(
            fact
            for fact in state.facts_valid_at(target)
            if fact.subject in entity_ids or fact.object_entity in entity_ids
        )
    fact_ids = {fact.id for fact in facts}

    names = state.entity_names()
    knowledge = tuple(
        KnowledgeItem(
            character=edge.character,
            character_name=names.get(edge.character, str(edge.character)),
            fact_id=str(edge.fact),
            since_beat=edge.since_beat,
        )
        for edge in sorted(state.knows.values(), key=lambda e: (e.character, e.fact))
        if edge.character in entity_ids and edge.fact in fact_ids
    )

    threads = tuple(
        thread
        for thread in sorted(state.threads.values(), key=lambda t: t.id)
        if thread.status is ThreadStatus.open
    )

    prose: list[str] = []
    if target is not None and target in state.seq:
        target_seq = state.seq[target]
        for beat in reversed(beats):
            if state.seq.get(beat.id, 0) >= target_seq:
                continue
            written = state.prose_for(beat.id)
            if written is not None and written.text:
                prose.append(written.text)
            if len(prose) >= recent_prose:
                break

    return StateSlice(
        target_beat=target,
        path=tuple(path),
        entities=entities,
        facts=facts,
        knowledge=knowledge,
        threads=threads,
        hard_constraints=tuple(hard_constraints(state)),
        style_notes=state.ledger.active_style_notes(mined_threshold),
        criteria=state.ledger.criteria,
        rejected_directions=tuple(
            d.text for d in state.ledger.rejected_directions[-RECENT_REJECTIONS:]
        ),
        recent_prose=tuple(reversed(prose)),
    )


def entities_in_scope(state: StoryState, node_id: NodeId) -> set[EntityId]:
    """Guess which entities a node is about, without asking a model.

    Uses the facts the node consumes and produces, plus name matches in its planning
    fields. Chunk 2's extraction refines this; here it is the default scoping rule.

    Args:
        state: The state to search.
        node_id: The node in question.

    Returns:
        Entity ids in scope. Possibly empty -- a node that declares no facts and whose
        title names nobody has nothing in scope, and callers that cannot work with an empty
        set fall back to the whole cast rather than pretending otherwise.
    """
    node = state.nodes.get(node_id)
    if node is None:
        return set()
    scoped: set[EntityId] = set()
    for fact_id in (*getattr(node, "produces", ()), *getattr(node, "consumes", ())):
        fact = state.facts.get(fact_id)
        if fact is not None:
            scoped.add(fact.subject)
            if fact.object_entity is not None:
                scoped.add(fact.object_entity)
    haystack = " ".join(
        (node.title, node.what_happens, node.audience_learns, node.location)
    ).lower()
    for entity in state.entities.values():
        if any(name.lower() in haystack for name in entity.names if name):
            scoped.add(entity.id)
    return scoped
