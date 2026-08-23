"""Turning model output into typed diffs.

This is where a language model's answer stops being text and becomes a proposed change
to the story. The conversion is deliberately mechanical: the model fills a schema, and
this module maps each field onto operations — a node here, an entity there, facts,
epistemic edges, thread movements. Nothing is interpreted twice, and nothing reaches the
state without passing through :func:`storygit.domain.apply.apply`'s validation.

Sampling ``k`` candidates is a plain loop here. Chunk 3 replaces the loop with
axis-conditioned parallel sampling and MMR/DPP selection; the seam is exactly this
module's :meth:`Proposer.propose`, and nothing in ``providers/`` has to change for it.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict, Field

from storygit.agents import prompts
from storygit.agents.aliases import AliasResolver
from storygit.agents.parse import complete_structured
from storygit.agents.schemas import (
    SCHEMA_FOR_LEVEL,
    BeatProposal,
    CharacterDraft,
    EpisodeProposal,
    FactDraft,
    Level,
    PremiseProposal,
    ProposalBase,
    ProseProposal,
    SceneProposal,
)
from storygit.domain.diff import (
    AddEntity,
    AddFact,
    AddKnows,
    AddNode,
    CloseThread,
    Diff,
    DiffAuthor,
    Op,
    OpenThread,
    TouchThread,
    UpdateNode,
    delta_summary,
)
from storygit.domain.ids import FactId, IdGenerator, NodeId, ProposalId, ThreadId
from storygit.domain.nodes import Beat, Episode, NodeType, Prose, Scene
from storygit.domain.provenance import Authorship, ProvenanceSpan
from storygit.domain.state import StoryState
from storygit.domain.threads import Thread, ThreadStatus
from storygit.domain.world import (
    ENTITY_VALUED_PREDICATES,
    EntityKind,
    Fact,
    FactSource,
    Knows,
    Predicate,
)
from storygit.graph.propagation import preview
from storygit.graph.slices import StateSlice, entities_in_scope, entity_slice
from storygit.providers.base import LLMRequest, Message, SchemaParseError
from storygit.providers.router import Router


class Proposal(BaseModel):
    """One candidate change, ready for the writer to accept, edit, or reject.

    Attributes:
        id: Stable identifier; signals reference it.
        level: Which level of the plan tree this targets.
        target_node_id: The node it attaches to (the parent, for a new child).
        diff: The change itself. Applying this is the only thing "accept" does.
        rationale: The model's one- or two-sentence justification.
        delta_summary: Writer-facing lines describing what it changes.
        stale_preview: How many downstream nodes it would mark.
        axis: The conditioning axis it was sampled along; set in chunk 3.
        scores: Named quality scores; filled by chunks 3 and 4.
        raw_text: The model's raw output, kept for debugging and for edit mining.
        notes: Things the writer should look at, such as a possibly duplicate entity.
    """

    model_config = ConfigDict(frozen=True)

    id: ProposalId
    level: Level
    target_node_id: NodeId | None
    diff: Diff
    rationale: str = ""
    delta_summary: tuple[str, ...] = ()
    stale_preview: int = 0
    axis: str | None = None
    scores: dict[str, float] = Field(default_factory=dict)
    raw_text: str = ""
    notes: tuple[str, ...] = ()


class Proposer:
    """Generates proposals at any level of the plan tree.

    Attributes:
        router: Where model calls go.
        ids: Id generator; seed it for reproducible proposals.
    """

    def __init__(self, router: Router, ids: IdGenerator | None = None) -> None:
        """Create a proposer.

        Args:
            router: The router to call through.
            ids: Id generator. A seeded one makes whole runs reproducible.
        """
        self.router = router
        self.ids = ids or IdGenerator()

    # --- public API -------------------------------------------------------------

    async def propose(
        self,
        state: StoryState,
        level: Level,
        *,
        target_node_id: NodeId | None = None,
        intent: str = "",
        k: int = 3,
        temperature: float = 0.95,
    ) -> list[Proposal]:
        """Generate ``k`` candidate diffs at one level.

        Args:
            state: Current story state.
            level: What to propose.
            target_node_id: The parent to attach to. Defaults to the story root for
                episodes, and to the last node of the appropriate level otherwise.
            intent: The writer's instruction, in their words.
            k: How many candidates to return.
            temperature: Sampling temperature.

        Returns:
            Up to ``k`` proposals. Candidates whose output fails to validate twice are
            dropped rather than returned broken; if every one fails, the list is empty
            and the caller resamples.

        Raises:
            ProviderError: If the model could not be reached at all.
        """
        parent_id = target_node_id or self._default_parent(state, level)
        slice_ = self._slice_for(state, level, parent_id)
        model_cls = SCHEMA_FOR_LEVEL[level]
        messages = self._messages(state, level, slice_, intent, parent_id, model_cls)

        results = await asyncio.gather(
            *(
                self._one(state, level, parent_id, messages, model_cls, intent, index, temperature)
                for index in range(k)
            )
        )
        return [p for p in results if p is not None]

    async def extract(
        self,
        state: StoryState,
        beat_id: NodeId,
        text: str,
    ) -> Diff:
        """Read accepted prose and land what it establishes as a diff.

        This is the path a hand-written beat takes: the engine does not generate
        anything, it only reads what the writer wrote and updates the world graph so
        that continuity checking and propagation keep working.

        Args:
            state: Current story state.
            beat_id: The beat the prose belongs to.
            text: The prose.

        Returns:
            A diff adding the extracted entities, facts, epistemic edges, and thread
            movements. Empty if the model returned nothing usable.
        """
        from storygit.agents.schemas import ExtractionResult

        slice_ = entity_slice(state, entities_in_scope(state, beat_id), beat_id)
        system, user = prompts.extraction_prompt(slice_, text, ExtractionResult)
        request = LLMRequest(
            messages=(system, user),
            purpose="extract.facts",
            temperature=0.0,
            max_tokens=1500,
            json_schema=ExtractionResult.model_json_schema(),
        )
        try:
            result, _ = await complete_structured(self.router, request, ExtractionResult)
        except SchemaParseError:
            return Diff(author=DiffAuthor.system, intent="extract facts (failed)")

        resolver = AliasResolver(state, self.ids)
        ops: list[Op] = []
        for character in result.new_characters:
            resolver.resolve(
                character.name,
                kind=character.kind,
                description=character.description,
                aliases=tuple(character.aliases),
            )
        fact_ops, knows_ops, _ = self._fact_ops(
            state, resolver, result.facts, beat_id, source=FactSource.human
        )
        ops.extend(AddEntity(entity=e) for e in resolver.new_entities)
        ops.extend(fact_ops)
        ops.extend(knows_ops)
        ops.extend(self._thread_ops(state, beat_id, result.threads_opened, result.threads_touched))
        return Diff(
            ops=tuple(ops),
            author=DiffAuthor.system,
            intent=f"extract facts from prose at {beat_id}",
        )

    # --- internals --------------------------------------------------------------

    async def _one(
        self,
        state: StoryState,
        level: Level,
        parent_id: NodeId | None,
        messages: tuple[Message, Message],
        model_cls: type[ProposalBase],
        intent: str,
        sample_index: int,
        temperature: float,
    ) -> Proposal | None:
        """Sample one candidate and convert it into a proposal.

        Returns ``None`` when the model's output could not be validated even after the
        repair retry — a bad sample is dropped, not returned broken. Every other failure
        propagates: a bug in diff construction must not look like a flaky model.
        """
        request = LLMRequest(
            messages=messages,
            purpose=f"propose.{level.value}",
            temperature=temperature,
            # Proposal schemas have many fields and prose proposals are long; a cap that
            # truncates mid-JSON turns a good candidate into a parse failure, which costs
            # a repair call to recover. Headroom is cheaper than repairs.
            max_tokens=4096,
            json_schema=model_cls.model_json_schema(),
            sample_index=sample_index,
        )
        try:
            parsed, response = await complete_structured(self.router, request, model_cls)
        except SchemaParseError:
            return None
        return self._to_proposal(state, level, parent_id, parsed, intent, response.text)

    def _default_parent(self, state: StoryState, level: Level) -> NodeId | None:
        """Where a proposal attaches when the caller did not say."""
        if level is Level.premise:
            return state.root_id
        wanted = {
            Level.episode: NodeType.story,
            Level.scene: NodeType.episode,
            Level.beat: NodeType.scene,
            Level.prose: NodeType.beat,
        }[level]
        candidates = state.nodes_of_type(wanted)
        return candidates[-1].id if candidates else state.root_id

    def _slice_for(self, state: StoryState, level: Level, parent_id: NodeId | None) -> StateSlice:
        """Build the entity-scoped slice this level's prompt should see."""
        if parent_id is None:
            return entity_slice(state, set(), None)
        scope = entities_in_scope(state, parent_id)
        if not scope:
            scope = set(state.entities)
        beats = state.beats_in_order()
        at_beat = (
            parent_id if parent_id in {b.id for b in beats} else (beats[-1].id if beats else None)
        )
        return entity_slice(state, scope, at_beat)

    def _messages(
        self,
        state: StoryState,
        level: Level,
        slice_: StateSlice,
        intent: str,
        parent_id: NodeId | None,
        model_cls: type[ProposalBase],
    ) -> tuple[Message, Message]:
        """Dispatch to the right prompt builder for this level."""
        parent = state.nodes.get(parent_id) if parent_id else None
        parent_label = (parent.title or parent.node_type.value) if parent else "the story"
        match level:
            case Level.premise:
                seed = getattr(parent, "seed", "") if parent else ""
                return prompts.premise_prompt(slice_, seed, intent, model_cls)
            case Level.episode:
                position = len(state.nodes_of_type(NodeType.episode)) + 1
                return prompts.episode_prompt(slice_, intent, model_cls, position)
            case Level.scene:
                return prompts.scene_prompt(slice_, intent, model_cls, parent_label)
            case Level.beat:
                return prompts.beat_prompt(slice_, intent, model_cls, parent_label)
            case Level.prose:
                beat_text = parent.what_happens if parent else ""
                return prompts.prose_prompt(slice_, intent, model_cls, beat_text or parent_label)

    # --- schema -> diff ---------------------------------------------------------

    def _to_proposal(
        self,
        state: StoryState,
        level: Level,
        parent_id: NodeId | None,
        parsed: ProposalBase,
        intent: str,
        raw: str,
    ) -> Proposal:
        """Convert a validated schema instance into a typed diff."""
        proposal_id = self.ids.proposal()
        resolver = AliasResolver(state, self.ids)
        ops: list[Op] = []
        notes: list[str] = []

        match parsed:
            case PremiseProposal():
                ops.extend(self._premise_ops(state, resolver, parsed, notes))
            case EpisodeProposal():
                ops.extend(self._episode_ops(state, parsed, parent_id))
            case SceneProposal():
                ops.extend(self._scene_ops(state, parsed, parent_id))
            case BeatProposal():
                ops.extend(self._beat_ops(state, resolver, parsed, parent_id, notes))
            case ProseProposal():
                ops.extend(self._prose_ops(state, parsed, parent_id, proposal_id))

        diff = Diff(
            ops=tuple(ops),
            author=DiffAuthor.ai,
            intent=intent or f"propose {level.value}",
            proposal_id=proposal_id,
        )
        marks = preview(state, diff)
        summary = tuple(parsed.delta_summary) or tuple(
            delta_summary(diff, state, stale_count=len(marks))
        )
        if not parsed.delta_summary:
            model_lines: tuple[str, ...] = ()
        else:
            model_lines = tuple(delta_summary(diff, state, stale_count=len(marks)))
        return Proposal(
            id=proposal_id,
            level=level,
            target_node_id=parent_id,
            diff=diff,
            rationale=parsed.rationale,
            delta_summary=tuple(dict.fromkeys((*summary, *model_lines))),
            stale_preview=len(marks),
            raw_text=raw,
            notes=tuple(notes),
        )

    def _premise_ops(
        self,
        state: StoryState,
        resolver: AliasResolver,
        parsed: PremiseProposal,
        notes: list[str],
    ) -> list[Op]:
        """Premise: update the story root and introduce the opening cast."""
        ops: list[Op] = []
        root_id = state.root_id
        if root_id is not None:
            fields: dict[str, object] = {
                "premise": parsed.premise,
                "existential_question": parsed.existential_question,
                "what_happens": parsed.premise,
            }
            if parsed.title:
                fields["title"] = parsed.title
            ops.append(UpdateNode(node_id=root_id, fields=fields))
        self._resolve_characters(resolver, parsed.main_characters, notes)
        ops.extend(AddEntity(entity=e) for e in resolver.new_entities)
        return ops

    def _episode_ops(
        self, state: StoryState, parsed: EpisodeProposal, parent_id: NodeId | None
    ) -> list[Op]:
        """Episode: a new node plus its thread movements."""
        parent = parent_id or state.root_id
        if parent is None:
            return []
        node_id = self.ids.node()
        position = len(state.children.get(parent, ()))
        previous = state.nodes_of_type(NodeType.episode)
        carried: tuple[ThreadId, ...] = ()
        if previous:
            last = previous[-1]
            carried = tuple(getattr(last, "threads_out", ()))

        ops: list[Op] = [
            AddNode(
                node=Episode(
                    id=node_id,
                    parent_id=parent,
                    title=parsed.title,
                    what_happens=parsed.what_happens,
                    audience_learns=parsed.audience_learns,
                    audience_feels=parsed.audience_feels,
                    location=parsed.location,
                    time=parsed.time,
                    position=position,
                    hook=parsed.hook,
                    cliffhanger=parsed.cliffhanger,
                    recap_of_previous=parsed.recap_of_previous,
                    threads_in=carried,
                    target_length=parsed.target_length,
                )
            )
        ]
        opened: list[ThreadId] = []
        for description in parsed.threads_opened:
            thread_id = self.ids.thread()
            opened.append(thread_id)
            ops.append(
                OpenThread(
                    thread=Thread(
                        id=thread_id,
                        description=description,
                        opened_at_beat=node_id,
                        last_touched_beat=node_id,
                    )
                )
            )
        for raw_id in parsed.threads_touched:
            thread_id = ThreadId(raw_id.strip())
            if thread_id in state.threads:
                ops.append(TouchThread(thread_id=thread_id, beat_id=node_id))
        closed: set[ThreadId] = set()
        for raw_id in parsed.threads_closed:
            thread_id = ThreadId(raw_id.strip())
            if thread_id in state.threads:
                closed.add(thread_id)
                ops.append(CloseThread(thread_id=thread_id, status=ThreadStatus.paid_off))

        still_open = [t for t in (*carried, *opened) if t not in closed]
        if still_open:
            ops.append(UpdateNode(node_id=node_id, fields={"threads_out": list(still_open)}))
        return ops

    def _scene_ops(
        self, state: StoryState, parsed: SceneProposal, parent_id: NodeId | None
    ) -> list[Op]:
        """Scene: one new node."""
        if parent_id is None:
            return []
        return [
            AddNode(
                node=Scene(
                    id=self.ids.node(),
                    parent_id=parent_id,
                    title=parsed.title,
                    what_happens=parsed.what_happens,
                    audience_learns=parsed.audience_learns,
                    audience_feels=parsed.audience_feels,
                    location=parsed.location,
                    time=parsed.time,
                    position=len(state.children.get(parent_id, ())),
                )
            )
        ]

    def _beat_ops(
        self,
        state: StoryState,
        resolver: AliasResolver,
        parsed: BeatProposal,
        parent_id: NodeId | None,
        notes: list[str],
    ) -> list[Op]:
        """Beat: the node, its new entities, its facts, and its dependency edges."""
        if parent_id is None:
            return []
        beat_id = self.ids.node()
        consumes = tuple(
            FactId(raw.strip()) for raw in parsed.consumes if FactId(raw.strip()) in state.facts
        )
        ops: list[Op] = [
            AddNode(
                node=Beat(
                    id=beat_id,
                    parent_id=parent_id,
                    title=parsed.title,
                    what_happens=parsed.what_happens,
                    audience_learns=parsed.audience_learns,
                    audience_feels=parsed.audience_feels,
                    location=parsed.location,
                    time=parsed.time,
                    position=len(state.children.get(parent_id, ())),
                    consumes=consumes,
                )
            )
        ]
        self._resolve_characters(resolver, parsed.new_characters, notes)
        fact_ops, knows_ops, fact_notes = self._fact_ops(
            state, resolver, parsed.produces, beat_id, source=FactSource.ai
        )
        notes.extend(fact_notes)
        ops.extend(AddEntity(entity=e) for e in resolver.new_entities)
        ops.extend(fact_ops)
        ops.extend(knows_ops)
        ops.extend(self._thread_ops(state, beat_id, [], parsed.threads_touched))
        return ops

    def _prose_ops(
        self,
        state: StoryState,
        parsed: ProseProposal,
        parent_id: NodeId | None,
        proposal_id: ProposalId,
    ) -> list[Op]:
        """Prose: a prose node under the beat, attributed to the AI sentence by sentence."""
        if parent_id is None:
            return []
        existing = state.prose_of_beat.get(parent_id)
        sentences = max(1, parsed.text.count(".") + parsed.text.count("?") + parsed.text.count("!"))
        span = ProvenanceSpan(start=0, end=sentences, source=Authorship.ai, proposal_id=proposal_id)
        if existing is not None:
            from storygit.domain.diff import SetProse

            return [SetProse(node_id=existing, text=parsed.text, spans=(span,))]
        return [
            AddNode(
                node=Prose(
                    id=self.ids.node(),
                    parent_id=parent_id,
                    title="prose",
                    text=parsed.text,
                    spans=(span,),
                )
            )
        ]

    # --- shared conversions -----------------------------------------------------

    def _resolve_characters(
        self, resolver: AliasResolver, drafts: list[CharacterDraft], notes: list[str]
    ) -> None:
        """Resolve every drafted entity, collecting writer-facing notes."""
        for draft in drafts:
            if not draft.name.strip():
                continue
            resolution = resolver.resolve(
                draft.name,
                kind=draft.kind,
                description=draft.description,
                aliases=tuple(draft.aliases),
            )
            if resolution.note:
                notes.append(resolution.note)

    def _fact_ops(
        self,
        state: StoryState,
        resolver: AliasResolver,
        drafts: list[FactDraft],
        beat_id: NodeId,
        *,
        source: FactSource,
    ) -> tuple[list[Op], list[Op], list[str]]:
        """Convert drafted facts into ``AddFact`` and ``AddKnows`` operations."""
        fact_ops: list[Op] = []
        knows_ops: list[Op] = []
        notes: list[str] = []
        for draft in drafts:
            if not draft.subject.strip() or not draft.object.strip():
                continue
            subject = resolver.resolve(draft.subject)
            if subject.note:
                notes.append(subject.note)
            object_entity = None
            object_text = ""
            if draft.object_is_entity:
                resolved = resolver.resolve(draft.object, kind=_kind_for(draft))
                if resolved.note:
                    notes.append(resolved.note)
                object_entity = resolved.entity_id
            else:
                object_text = draft.object.strip()
            fact = Fact(
                id=self.ids.fact(),
                subject=subject.entity_id,
                predicate=draft.predicate,
                object_entity=object_entity,
                object_text=object_text,
                valid_from_beat=beat_id,
                established_by_beat=beat_id,
                source=source,
            )
            fact_ops.append(AddFact(fact=fact))
            for name in draft.known_by:
                if not name.strip():
                    continue
                knower = resolver.resolve(name)
                knows_ops.append(
                    AddKnows(
                        knows=Knows(character=knower.entity_id, fact=fact.id, since_beat=beat_id)
                    )
                )
        return fact_ops, knows_ops, notes

    def _thread_ops(
        self,
        state: StoryState,
        node_id: NodeId,
        opened: list[str],
        touched: list[str],
    ) -> list[Op]:
        """Open and advance threads named by a proposal."""
        ops: list[Op] = []
        for description in opened:
            if not description.strip():
                continue
            ops.append(
                OpenThread(
                    thread=Thread(
                        id=self.ids.thread(),
                        description=description.strip(),
                        opened_at_beat=node_id,
                        last_touched_beat=node_id,
                    )
                )
            )
        for raw_id in touched:
            thread_id = ThreadId(raw_id.strip())
            if thread_id in state.threads:
                ops.append(TouchThread(thread_id=thread_id, beat_id=node_id))
        return ops


def _kind_for(draft: FactDraft) -> EntityKind:
    """Guess what kind of entity a fact's object is, from its predicate."""
    if draft.predicate is Predicate.location:
        return EntityKind.place
    if draft.predicate is Predicate.faction:
        return EntityKind.faction
    if draft.predicate is Predicate.possesses:
        return EntityKind.object
    if draft.predicate in ENTITY_VALUED_PREDICATES:
        return EntityKind.character
    return EntityKind.character
