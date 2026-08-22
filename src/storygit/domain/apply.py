"""``apply``: the one function that turns a diff into new state.

Pure by construction — it copies the aggregate's collections, works on the copies,
and returns a freshly indexed :class:`~storygit.domain.state.StoryState`. The input
is never touched, which is what makes "apply the same diff twice and get identical
hashes" true, and what lets the engine preview a diff on a scratch state before the
writer has agreed to anything.

Every op validates its preconditions before it changes anything, and every failure
is a typed error from :mod:`storygit.domain.errors`.
"""

from __future__ import annotations

from typing import Any

from storygit.domain.diff import (
    AddCriterion,
    AddEntity,
    AddFact,
    AddHardConstraint,
    AddKnows,
    AddNode,
    AddRejectedDirection,
    AddStyleNote,
    ClearLock,
    CloseThread,
    Diff,
    InvalidateFact,
    MergeEntities,
    MoveNode,
    Op,
    OpenThread,
    RemoveCriterion,
    RemoveEntity,
    RemoveFact,
    RemoveKnows,
    RemoveNode,
    RemoveStyleNote,
    RemoveThread,
    ReplaceLedger,
    SetDial,
    SetLock,
    SetNodeStatus,
    SetProse,
    TouchThread,
    UpdateEntity,
    UpdateFact,
    UpdateNode,
    UpdateThread,
)
from storygit.domain.errors import (
    CycleError,
    DuplicateIdError,
    InvalidStructureError,
    LockedNodeError,
    UnknownEntityError,
    UnknownFactError,
    UnknownNodeError,
    UnknownThreadError,
)
from storygit.domain.ids import EntityId, FactId, NodeId, ThreadId
from storygit.domain.ledger import RejectedDirection, StyleNote, WriterLedger
from storygit.domain.nodes import (
    ALLOWED_CHILDREN,
    AnyNode,
    BaseNode,
    Beat,
    NodeStatus,
    NodeType,
    Prose,
)
from storygit.domain.state import KnowsKey, StoryState
from storygit.domain.threads import Thread, ThreadStatus
from storygit.domain.world import Entity, Fact, Knows


class _Working:
    """Mutable scratch copies of a state's collections, used only inside ``apply``."""

    def __init__(self, state: StoryState) -> None:
        self.nodes: dict[NodeId, BaseNode] = dict(state.nodes)
        self.entities: dict[EntityId, Entity] = dict(state.entities)
        self.facts: dict[FactId, Fact] = dict(state.facts)
        self.knows: dict[KnowsKey, Knows] = dict(state.knows)
        self.threads: dict[ThreadId, Thread] = dict(state.threads)
        self.ledger: WriterLedger = state.ledger

    def freeze(self, state: StoryState) -> StoryState:
        """Rebuild an indexed state from the scratch copies."""
        return state.evolve(
            nodes=self.nodes,
            entities=self.entities,
            facts=self.facts,
            knows=self.knows,
            threads=self.threads,
            ledger=self.ledger,
        )


def apply(state: StoryState, diff: Diff) -> StoryState:
    """Apply a diff to a state, returning a new state.

    Args:
        state: The state to apply to. Never modified.
        diff: The operations to apply, in order.

    Returns:
        A new, fully indexed state.

    Raises:
        ApplyError: Any subclass, when an operation's preconditions do not hold.
            The input state is unchanged in every case.
    """
    work = _Working(state)
    for op in diff.ops:
        _apply_op(work, op)
    return work.freeze(state)


def _apply_op(work: _Working, op: Op) -> None:
    """Dispatch one operation onto the working copies."""
    match op:
        case AddNode():
            _add_node(work, op.node)
        case UpdateNode():
            _update_node(work, op.node_id, op.fields)
        case RemoveNode():
            _remove_node(work, op.node_id, op.recursive)
        case MoveNode():
            _move_node(work, op.node_id, op.new_parent_id, op.new_position)
        case SetProse():
            node = _require_node(work, op.node_id)
            if not isinstance(node, Prose):
                raise InvalidStructureError(f"{op.node_id} is not a prose node")
            work.nodes[op.node_id] = node.model_copy(
                update={"text": op.text, "spans": tuple(op.spans)}
            )
        case SetNodeStatus():
            node = _require_node(work, op.node_id)
            if op.node_id in work.ledger.locks and op.status is NodeStatus.stale:
                raise LockedNodeError(f"{op.node_id} is locked and cannot be marked stale")
            work.nodes[op.node_id] = node.model_copy(
                update={"status": op.status, "stale_reason": op.stale_reason}
            )
        case SetLock():
            node = _require_node(work, op.node_id)
            work.nodes[op.node_id] = node.model_copy(
                update={"status": NodeStatus.locked, "stale_reason": None}
            )
            work.ledger = work.ledger.model_copy(update={"locks": work.ledger.locks | {op.node_id}})
        case ClearLock():
            node = _require_node(work, op.node_id)
            if node.status is NodeStatus.locked:
                work.nodes[op.node_id] = node.model_copy(update={"status": NodeStatus.accepted})
            work.ledger = work.ledger.model_copy(update={"locks": work.ledger.locks - {op.node_id}})
        case AddEntity():
            if op.entity.id in work.entities:
                raise DuplicateIdError(f"entity {op.entity.id} already exists")
            work.entities[op.entity.id] = op.entity
        case UpdateEntity():
            entity = work.entities.get(op.entity_id)
            if entity is None:
                raise UnknownEntityError(f"no entity {op.entity_id}")
            work.entities[op.entity_id] = _validated_copy(entity, op.fields)
        case RemoveEntity():
            _remove_entity(work, op.entity_id)
        case MergeEntities():
            _merge_entities(work, op.source_id, op.target_id)
        case AddFact():
            _add_fact(work, op.fact)
        case UpdateFact():
            fact = _require_fact(work, op.fact_id)
            work.facts[op.fact_id] = _validated_copy(fact, op.fields)
        case InvalidateFact():
            fact = _require_fact(work, op.fact_id)
            _require_node(work, op.valid_until_beat)
            work.facts[op.fact_id] = fact.model_copy(
                update={"valid_until_beat": op.valid_until_beat}
            )
        case RemoveFact():
            _remove_facts(work, {op.fact_id})
        case AddKnows():
            if op.knows.fact not in work.facts:
                raise UnknownFactError(f"no fact {op.knows.fact}")
            if op.knows.character not in work.entities:
                raise UnknownEntityError(f"no entity {op.knows.character}")
            _require_node(work, op.knows.since_beat)
            work.knows[op.knows.key] = op.knows
        case RemoveKnows():
            work.knows.pop((op.character, op.fact_id), None)
        case OpenThread():
            if op.thread.id in work.threads:
                raise DuplicateIdError(f"thread {op.thread.id} already exists")
            _require_node(work, op.thread.opened_at_beat)
            work.threads[op.thread.id] = op.thread
        case TouchThread():
            thread = _require_thread(work, op.thread_id)
            _require_node(work, op.beat_id)
            work.threads[op.thread_id] = thread.model_copy(update={"last_touched_beat": op.beat_id})
        case CloseThread():
            thread = _require_thread(work, op.thread_id)
            work.threads[op.thread_id] = thread.model_copy(update={"status": op.status})
        case UpdateThread():
            thread = _require_thread(work, op.thread_id)
            work.threads[op.thread_id] = _validated_copy(thread, op.fields)
        case RemoveThread():
            work.threads.pop(op.thread_id, None)
        case SetDial():
            work.ledger = work.ledger.model_copy(update={"dial": op.value})
        case AddStyleNote():
            _add_style_note(work, op.note.text, op.note)
        case RemoveStyleNote():
            work.ledger = work.ledger.model_copy(
                update={
                    "style_notes": tuple(n for n in work.ledger.style_notes if n.text != op.text)
                }
            )
        case AddCriterion():
            existing = tuple(c for c in work.ledger.criteria if c.name != op.criterion.name)
            work.ledger = work.ledger.model_copy(update={"criteria": (*existing, op.criterion)})
        case RemoveCriterion():
            work.ledger = work.ledger.model_copy(
                update={"criteria": tuple(c for c in work.ledger.criteria if c.name != op.name)}
            )
        case AddHardConstraint():
            if op.text not in work.ledger.hard_constraints:
                work.ledger = work.ledger.model_copy(
                    update={"hard_constraints": (*work.ledger.hard_constraints, op.text)}
                )
        case AddRejectedDirection():
            work.ledger = work.ledger.model_copy(
                update={
                    "rejected_directions": (
                        *work.ledger.rejected_directions,
                        RejectedDirection(text=op.text, proposal_id=op.proposal_id),
                    )
                }
            )
        case ReplaceLedger():
            work.ledger = op.ledger
        case _:  # pragma: no cover - the union is exhaustive
            raise InvalidStructureError(f"unhandled op {op!r}")


# --- helpers ------------------------------------------------------------------


def _require_node(work: _Working, node_id: NodeId) -> BaseNode:
    node = work.nodes.get(node_id)
    if node is None:
        raise UnknownNodeError(f"no node {node_id}")
    return node


def _require_fact(work: _Working, fact_id: FactId) -> Fact:
    fact = work.facts.get(fact_id)
    if fact is None:
        raise UnknownFactError(f"no fact {fact_id}")
    return fact


def _require_thread(work: _Working, thread_id: ThreadId) -> Thread:
    thread = work.threads.get(thread_id)
    if thread is None:
        raise UnknownThreadError(f"no thread {thread_id}")
    return thread


def _validated_copy(model: Any, fields: dict[str, Any]) -> Any:
    """Copy a frozen model with new field values, re-running its validators."""
    unknown = set(fields) - set(type(model).model_fields)
    if unknown:
        raise InvalidStructureError(
            f"{type(model).__name__} has no field(s) {', '.join(sorted(unknown))}"
        )
    data = model.model_dump()
    data.update(fields)
    return type(model).model_validate(data)


def _add_node(work: _Working, node: AnyNode) -> None:
    if node.id in work.nodes:
        raise DuplicateIdError(f"node {node.id} already exists")
    if node.parent_id is None:
        if node.node_type is not NodeType.story:
            raise InvalidStructureError(f"only a story node may be the root, got {node.node_type}")
        if any(n.parent_id is None for n in work.nodes.values()):
            raise InvalidStructureError("the story already has a root")
    else:
        parent = _require_node(work, node.parent_id)
        allowed = ALLOWED_CHILDREN[parent.node_type]
        if node.node_type not in allowed:
            raise InvalidStructureError(
                f"a {parent.node_type.value} cannot contain a {node.node_type.value}"
            )
        if node.node_type is NodeType.prose:
            existing = [
                n
                for n in work.nodes.values()
                if n.parent_id == node.parent_id and n.node_type is NodeType.prose
            ]
            if existing:
                raise InvalidStructureError(f"beat {node.parent_id} already has prose")
    work.nodes[node.id] = node


def _update_node(work: _Working, node_id: NodeId, fields: dict[str, Any]) -> None:
    node = _require_node(work, node_id)
    if node_id in work.ledger.locks:
        raise LockedNodeError(f"{node_id} is locked; unlock it before editing")
    if "id" in fields or "node_type" in fields:
        raise InvalidStructureError("id and node_type are immutable")
    if "parent_id" in fields:
        raise InvalidStructureError("use MoveNode to reparent")
    work.nodes[node_id] = _validated_copy(node, fields)


def _subtree_ids(work: _Working, node_id: NodeId) -> set[NodeId]:
    out = {node_id}
    frontier = [node_id]
    while frontier:
        current = frontier.pop()
        for candidate in work.nodes.values():
            if candidate.parent_id == current and candidate.id not in out:
                out.add(candidate.id)
                frontier.append(candidate.id)
    return out


def _remove_node(work: _Working, node_id: NodeId, recursive: bool) -> None:
    _require_node(work, node_id)
    if node_id in work.ledger.locks:
        raise LockedNodeError(f"{node_id} is locked; unlock it before removing")
    children = [n.id for n in work.nodes.values() if n.parent_id == node_id]
    if children and not recursive:
        raise InvalidStructureError(f"{node_id} has children; pass recursive=True to remove them")
    doomed = _subtree_ids(work, node_id) if recursive else {node_id}
    doomed_facts = {fid for fid, fact in work.facts.items() if fact.established_by_beat in doomed}
    for nid in doomed:
        work.nodes.pop(nid, None)
    if doomed_facts:
        _remove_facts(work, doomed_facts)
    for tid, thread in list(work.threads.items()):
        if thread.opened_at_beat in doomed:
            work.threads.pop(tid)
        elif thread.last_touched_beat in doomed:
            work.threads[tid] = thread.model_copy(
                update={"last_touched_beat": thread.opened_at_beat, "status": ThreadStatus.open}
            )


def _move_node(work: _Working, node_id: NodeId, new_parent_id: NodeId, position: int) -> None:
    node = _require_node(work, node_id)
    parent = _require_node(work, new_parent_id)
    if node_id in work.ledger.locks:
        raise LockedNodeError(f"{node_id} is locked; unlock it before moving")
    # Cycles are checked first: the type rule below would also reject this move, but
    # "you cannot move a node into itself" is the useful message.
    if new_parent_id in _subtree_ids(work, node_id):
        raise CycleError(f"cannot move {node_id} under its own descendant {new_parent_id}")
    if node.node_type not in ALLOWED_CHILDREN[parent.node_type]:
        raise InvalidStructureError(
            f"a {parent.node_type.value} cannot contain a {node.node_type.value}"
        )
    work.nodes[node_id] = node.model_copy(update={"parent_id": new_parent_id, "position": position})
    siblings = sorted(
        (n for n in work.nodes.values() if n.parent_id == new_parent_id and n.id != node_id),
        key=lambda n: n.position,
    )
    index = 0
    for sibling in siblings:
        if index == position:
            index += 1
        work.nodes[sibling.id] = sibling.model_copy(update={"position": index})
        index += 1


def _add_fact(work: _Working, fact: Fact) -> None:
    if fact.id in work.facts:
        raise DuplicateIdError(f"fact {fact.id} already exists")
    if fact.subject not in work.entities:
        raise UnknownEntityError(f"no entity {fact.subject} (fact subject)")
    if fact.object_entity is not None and fact.object_entity not in work.entities:
        raise UnknownEntityError(f"no entity {fact.object_entity} (fact object)")
    _require_node(work, fact.valid_from_beat)
    _require_node(work, fact.established_by_beat)
    if fact.valid_until_beat is not None:
        _require_node(work, fact.valid_until_beat)
    work.facts[fact.id] = fact
    beat = work.nodes.get(fact.established_by_beat)
    if isinstance(beat, Beat) and fact.id not in beat.produces:
        work.nodes[beat.id] = beat.model_copy(update={"produces": (*beat.produces, fact.id)})


def _remove_facts(work: _Working, fact_ids: set[FactId]) -> None:
    for fid in fact_ids:
        if fid not in work.facts:
            raise UnknownFactError(f"no fact {fid}")
        work.facts.pop(fid)
    for key in [k for k in work.knows if k[1] in fact_ids]:
        work.knows.pop(key)
    for nid, node in list(work.nodes.items()):
        if not isinstance(node, Beat):
            continue
        produces = tuple(f for f in node.produces if f not in fact_ids)
        consumes = tuple(f for f in node.consumes if f not in fact_ids)
        if produces != node.produces or consumes != node.consumes:
            work.nodes[nid] = node.model_copy(update={"produces": produces, "consumes": consumes})


def _remove_entity(work: _Working, entity_id: EntityId) -> None:
    if entity_id not in work.entities:
        raise UnknownEntityError(f"no entity {entity_id}")
    doomed = {
        fid
        for fid, fact in work.facts.items()
        if fact.subject == entity_id or fact.object_entity == entity_id
    }
    if doomed:
        _remove_facts(work, doomed)
    for key in [k for k in work.knows if k[0] == entity_id]:
        work.knows.pop(key)
    work.entities.pop(entity_id)


def _merge_entities(work: _Working, source_id: EntityId, target_id: EntityId) -> None:
    source = work.entities.get(source_id)
    target = work.entities.get(target_id)
    if source is None:
        raise UnknownEntityError(f"no entity {source_id} (merge source)")
    if target is None:
        raise UnknownEntityError(f"no entity {target_id} (merge target)")
    if source_id == target_id:
        raise InvalidStructureError("cannot merge an entity into itself")
    aliases = tuple(dict.fromkeys((*target.aliases, source.name, *source.aliases)))
    work.entities[target_id] = target.model_copy(update={"aliases": aliases})
    work.entities.pop(source_id)
    for fid, fact in list(work.facts.items()):
        update: dict[str, Any] = {}
        if fact.subject == source_id:
            update["subject"] = target_id
        if fact.object_entity == source_id:
            update["object_entity"] = target_id
        if update:
            work.facts[fid] = fact.model_copy(update=update)
    for key, edge in list(work.knows.items()):
        if edge.character == source_id:
            work.knows.pop(key)
            moved = edge.model_copy(update={"character": target_id})
            work.knows[moved.key] = moved


def _add_style_note(work: _Working, text: str, note: StyleNote) -> None:
    existing = {n.text: n for n in work.ledger.style_notes}
    if text in existing:
        bumped = existing[text].model_copy(update={"count": existing[text].count + note.count})
        notes = tuple(bumped if n.text == text else n for n in work.ledger.style_notes)
    else:
        notes = (*work.ledger.style_notes, note)
    work.ledger = work.ledger.model_copy(update={"style_notes": notes})
