"""Diffs: every change to the story, as reviewable typed operations.

Nothing in storygit writes state directly. The AI proposes a ``Diff``; the writer
accepts, edits, or rejects it; only then is it applied and committed. That is the
whole answer to Fabula's two structural complaints — edits that silently rewrite
everything, and a writer reduced to judging walls of replacement text. A diff is
small, it is typed, and it can be summarized in English before anyone commits to it.

Ops are grouped by what they touch: the plan tree, entities, facts, epistemic
edges, threads, and the writer ledger.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from storygit.domain.ids import EntityId, FactId, NodeId, ProposalId, ThreadId
from storygit.domain.ledger import Criterion, StyleNote, WriterLedger
from storygit.domain.nodes import AnyNode, NodeStatus
from storygit.domain.provenance import ProvenanceSpan
from storygit.domain.threads import Thread, ThreadStatus
from storygit.domain.world import Entity, Fact, Knows


class DiffAuthor(StrEnum):
    """Who authored a diff."""

    human = "human"
    ai = "ai"
    system = "system"


class _Op(BaseModel):
    """Base for every operation; frozen and forbidding extra fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --- plan tree ----------------------------------------------------------------


class AddNode(_Op):
    """Insert a node into the plan tree."""

    op: Literal["add_node"] = "add_node"
    node: AnyNode


class UpdateNode(_Op):
    """Set fields on an existing node.

    Attributes:
        node_id: Node to change.
        fields: Field name to new value; validated against the node's model.
    """

    op: Literal["update_node"] = "update_node"
    node_id: NodeId
    fields: dict[str, Any]


class RemoveNode(_Op):
    """Delete a node and, by default, its whole subtree."""

    op: Literal["remove_node"] = "remove_node"
    node_id: NodeId
    recursive: bool = True


class MoveNode(_Op):
    """Reparent or reorder a node; recomputes the global beat sequence."""

    op: Literal["move_node"] = "move_node"
    node_id: NodeId
    new_parent_id: NodeId
    new_position: int = Field(ge=0)


class SetProse(_Op):
    """Replace a prose node's text and provenance spans."""

    op: Literal["set_prose"] = "set_prose"
    node_id: NodeId
    text: str
    spans: tuple[ProvenanceSpan, ...] = ()


class SetNodeStatus(_Op):
    """Set a node's review status and stale reason.

    Propagation emits these, which is what keeps staleness flowing through the same
    snapshot mechanism as everything else instead of being a side channel.
    """

    op: Literal["set_node_status"] = "set_node_status"
    node_id: NodeId
    status: NodeStatus
    stale_reason: str | None = None


class SetLock(_Op):
    """Freeze a node: propagation skips it, its facts become hard constraints."""

    op: Literal["set_lock"] = "set_lock"
    node_id: NodeId


class ClearLock(_Op):
    """Unfreeze a node and return it to ``accepted``."""

    op: Literal["clear_lock"] = "clear_lock"
    node_id: NodeId


# --- entities -----------------------------------------------------------------


class AddEntity(_Op):
    """Introduce a new character, place, object, or faction."""

    op: Literal["add_entity"] = "add_entity"
    entity: Entity


class UpdateEntity(_Op):
    """Set fields on an existing entity (typically to add an alias)."""

    op: Literal["update_entity"] = "update_entity"
    entity_id: EntityId
    fields: dict[str, Any]


class RemoveEntity(_Op):
    """Delete an entity and everything asserted about it.

    Emitted by structural diffs and merges; the UI offers ``MergeEntities`` instead,
    because a writer almost always means "these two are the same person".
    """

    op: Literal["remove_entity"] = "remove_entity"
    entity_id: EntityId


class MergeEntities(_Op):
    """Fold one entity into another.

    Used when canonicalization discovers that "the Warden" and "Warden of Kell" are
    the same character. Facts and epistemic edges are rewritten onto the survivor
    and the source's names become aliases.
    """

    op: Literal["merge_entities"] = "merge_entities"
    source_id: EntityId
    target_id: EntityId


# --- facts --------------------------------------------------------------------


class AddFact(_Op):
    """Assert a new fact into the world graph."""

    op: Literal["add_fact"] = "add_fact"
    fact: Fact


class UpdateFact(_Op):
    """Set fields on an existing fact."""

    op: Literal["update_fact"] = "update_fact"
    fact_id: FactId
    fields: dict[str, Any]


class InvalidateFact(_Op):
    """End a fact's validity at a beat, keeping it in the history.

    This is how a fact *changes*: the old one stops being true where the new one
    starts, so both remain retrievable at their own times.
    """

    op: Literal["invalidate_fact"] = "invalidate_fact"
    fact_id: FactId
    valid_until_beat: NodeId


class RemoveFact(_Op):
    """Delete a fact outright — the writer striking it from the bible."""

    op: Literal["remove_fact"] = "remove_fact"
    fact_id: FactId


# --- epistemic ----------------------------------------------------------------


class AddKnows(_Op):
    """Record that a character learned a fact at a beat."""

    op: Literal["add_knows"] = "add_knows"
    knows: Knows


class RemoveKnows(_Op):
    """Remove an epistemic edge."""

    op: Literal["remove_knows"] = "remove_knows"
    character: EntityId
    fact_id: FactId


# --- threads ------------------------------------------------------------------


class OpenThread(_Op):
    """Open a narrative thread."""

    op: Literal["open_thread"] = "open_thread"
    thread: Thread


class TouchThread(_Op):
    """Record that a beat advanced a thread."""

    op: Literal["touch_thread"] = "touch_thread"
    thread_id: ThreadId
    beat_id: NodeId


class CloseThread(_Op):
    """Mark a thread paid off or dropped."""

    op: Literal["close_thread"] = "close_thread"
    thread_id: ThreadId
    status: ThreadStatus = ThreadStatus.paid_off


class UpdateThread(_Op):
    """Set fields on a thread."""

    op: Literal["update_thread"] = "update_thread"
    thread_id: ThreadId
    fields: dict[str, Any]


class RemoveThread(_Op):
    """Delete a thread; emitted by structural diffs, not by the UI."""

    op: Literal["remove_thread"] = "remove_thread"
    thread_id: ThreadId


# --- writer ledger ------------------------------------------------------------


class SetDial(_Op):
    """Move the coherent-to-surprising dial."""

    op: Literal["set_dial"] = "set_dial"
    value: float = Field(ge=0.0, le=1.0)


class AddStyleNote(_Op):
    """Add a style rule, or bump the count of an identical existing one."""

    op: Literal["add_style_note"] = "add_style_note"
    note: StyleNote


class RemoveStyleNote(_Op):
    """Delete a style rule by its text — including a mined one."""

    op: Literal["remove_style_note"] = "remove_style_note"
    text: str


class AddCriterion(_Op):
    """Add a writer-defined scoring axis."""

    op: Literal["add_criterion"] = "add_criterion"
    criterion: Criterion


class RemoveCriterion(_Op):
    """Delete a writer-defined scoring axis by name."""

    op: Literal["remove_criterion"] = "remove_criterion"
    name: str


class AddHardConstraint(_Op):
    """Add a rule that generation must never violate."""

    op: Literal["add_hard_constraint"] = "add_hard_constraint"
    text: str


class AddRejectedDirection(_Op):
    """Record a direction the writer turned down."""

    op: Literal["add_rejected_direction"] = "add_rejected_direction"
    text: str
    proposal_id: ProposalId | None = None


class ReplaceLedger(_Op):
    """Replace the whole ledger; emitted by structural diffs and merges."""

    op: Literal["replace_ledger"] = "replace_ledger"
    ledger: WriterLedger


Op = Annotated[
    AddNode
    | UpdateNode
    | RemoveNode
    | MoveNode
    | SetProse
    | SetNodeStatus
    | SetLock
    | ClearLock
    | AddEntity
    | UpdateEntity
    | RemoveEntity
    | MergeEntities
    | AddFact
    | UpdateFact
    | InvalidateFact
    | RemoveFact
    | AddKnows
    | RemoveKnows
    | OpenThread
    | TouchThread
    | CloseThread
    | UpdateThread
    | RemoveThread
    | SetDial
    | AddStyleNote
    | RemoveStyleNote
    | AddCriterion
    | RemoveCriterion
    | AddHardConstraint
    | AddRejectedDirection
    | ReplaceLedger,
    Field(discriminator="op"),
]
"""Discriminated union of every operation, keyed on ``op``."""


class Diff(BaseModel):
    """An ordered list of operations plus who proposed it and why.

    Attributes:
        ops: Operations, applied in order.
        author: Human, AI, or the system (propagation marks).
        intent: The natural-language intent this diff answers.
        proposal_id: The proposal it came from, when it came from one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ops: tuple[Op, ...] = ()
    author: DiffAuthor = DiffAuthor.ai
    intent: str = ""
    proposal_id: ProposalId | None = None

    def __len__(self) -> int:
        """Number of operations."""
        return len(self.ops)

    def then(self, *ops: Op) -> Diff:
        """Return a copy with more operations appended."""
        return self.model_copy(update={"ops": (*self.ops, *ops)})

    def concat(self, other: Diff) -> Diff:
        """Return a copy with another diff's operations appended."""
        return self.model_copy(update={"ops": (*self.ops, *other.ops)})


def _as_entity_names(names: dict[str, str]) -> dict[EntityId, str]:
    """Re-key a display-name map for :meth:`storygit.domain.world.Fact.sentence`."""
    return {EntityId(k): v for k, v in names.items()}


def delta_summary(
    diff: Diff,
    state: Any,
    *,
    stale_count: int | None = None,
) -> list[str]:
    """Describe a diff in the sentences a writer would use.

    This is what appears under each candidate in the UI, so it deliberately reads
    as English rather than as an op log: "introduces Kael", not "AddEntity(e_1a2b)".

    Args:
        diff: The diff to describe.
        state: The :class:`~storygit.domain.state.StoryState` the diff applies to,
            used to resolve ids to names. Typed loosely to keep this module free of
            an import cycle.
        stale_count: Number of nodes propagation would mark, when the caller has
            run :func:`storygit.graph.propagation.preview`.

    Returns:
        One line per meaningful change, in op order, ending with the staleness
        line when a count was supplied.
    """
    names: dict[str, str] = {}
    if getattr(state, "entities", None):
        names = {str(eid): entity.name for eid, entity in state.entities.items()}
    # Entities the diff itself introduces are not in the state yet, but the writer
    # still expects to read "Mara wants ..." rather than an opaque id.
    for op in diff.ops:
        if isinstance(op, AddEntity):
            names[str(op.entity.id)] = op.entity.name
    nodes = dict(getattr(state, "nodes", {}) or {})
    # Same for nodes the diff creates: "Kael learns this at Beat 4" beats an opaque id.
    for op in diff.ops:
        if isinstance(op, AddNode):
            nodes[op.node.id] = op.node

    def entity_name(entity_id: str) -> str:
        return names.get(str(entity_id), str(entity_id))

    def node_label(node_id: NodeId) -> str:
        node = nodes.get(node_id)
        if node is None:
            return str(node_id)
        return node.title or f"{node.node_type.value} {node_id}"

    lines: list[str] = []
    for op in diff.ops:
        match op:
            case AddNode():
                label = op.node.title or op.node.node_type.value
                lines.append(f"adds {op.node.node_type.value} “{label}”")
            case UpdateNode():
                fields = ", ".join(sorted(op.fields))
                lines.append(f"changes {fields} of {node_label(op.node_id)}")
            case RemoveNode():
                lines.append(f"removes {node_label(op.node_id)}")
            case MoveNode():
                lines.append(f"moves {node_label(op.node_id)} under {node_label(op.new_parent_id)}")
            case SetProse():
                words = len(op.text.split())
                lines.append(f"writes {words} words of prose for {node_label(op.node_id)}")
            case SetNodeStatus():
                lines.append(f"marks {node_label(op.node_id)} {op.status.value}")
            case SetLock():
                lines.append(f"locks {node_label(op.node_id)}")
            case ClearLock():
                lines.append(f"unlocks {node_label(op.node_id)}")
            case AddEntity():
                lines.append(f"introduces {op.entity.kind.value} {op.entity.name}")
            case UpdateEntity():
                lines.append(
                    f"updates {entity_name(op.entity_id)} ({', '.join(sorted(op.fields))})"
                )
            case RemoveEntity():
                lines.append(f"removes {entity_name(op.entity_id)} from the bible")
            case MergeEntities():
                lines.append(f"merges {entity_name(op.source_id)} into {entity_name(op.target_id)}")
            case AddFact():
                lines.append(f"establishes: {op.fact.sentence(_as_entity_names(names))}")
            case UpdateFact():
                lines.append(f"revises a fact about {', '.join(sorted(op.fields))}")
            case InvalidateFact():
                fact = state.facts.get(op.fact_id)
                subject = entity_name(fact.subject) if fact else str(op.fact_id)
                pred = fact.predicate.value if fact else "fact"
                lines.append(f"ends {subject}'s {pred} as of {node_label(op.valid_until_beat)}")
            case RemoveFact():
                fact = state.facts.get(op.fact_id)
                lines.append(
                    f"strikes: {fact.sentence(_as_entity_names(names))}"
                    if fact
                    else f"strikes fact {op.fact_id}"
                )
            case AddKnows():
                lines.append(
                    f"{entity_name(op.knows.character)} learns this "
                    f"at {node_label(op.knows.since_beat)}"
                )
            case RemoveKnows():
                lines.append(f"{entity_name(op.character)} no longer knows a fact")
            case OpenThread():
                lines.append(f"opens thread: {op.thread.description}")
            case TouchThread():
                thread = state.threads.get(op.thread_id)
                desc = thread.description if thread else str(op.thread_id)
                lines.append(f"advances thread: {desc}")
            case CloseThread():
                thread = state.threads.get(op.thread_id)
                desc = thread.description if thread else str(op.thread_id)
                lines.append(f"{op.status.value.replace('_', ' ')} thread: {desc}")
            case SetDial():
                lines.append(f"sets the dial to {op.value:.2f}")
            case AddStyleNote():
                lines.append(f"adds style note: {op.note.text}")
            case AddCriterion():
                lines.append(f"adds criterion: {op.criterion.name}")
            case AddHardConstraint():
                lines.append(f"adds hard constraint: {op.text}")
            case AddRejectedDirection():
                lines.append(f"records a rejected direction: {op.text}")
            case _:
                continue
    if stale_count:
        noun = "node" if stale_count == 1 else "nodes"
        lines.append(f"would mark {stale_count} downstream {noun} stale")
    return lines
