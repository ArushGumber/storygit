"""Branch pointers, structural diff between versions, and three-way merge.

Branching is the writer's "what if I killed the mentor instead" — an exploration
that can be compared against the main line and either merged or thrown away. All
three operations here are pure set arithmetic over snapshot manifests: no model
call decides what changed.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict

from storygit.domain.apply import apply
from storygit.domain.diff import (
    AddEntity,
    AddFact,
    AddKnows,
    AddNode,
    ClearLock,
    Diff,
    DiffAuthor,
    MoveNode,
    Op,
    OpenThread,
    RemoveEntity,
    RemoveFact,
    RemoveKnows,
    RemoveNode,
    RemoveThread,
    ReplaceLedger,
    UpdateEntity,
    UpdateFact,
    UpdateNode,
    UpdateThread,
)
from storygit.domain.errors import SnapshotNotFoundError
from storygit.domain.ids import EntityId, FactId, NodeId, SnapshotId, ThreadId
from storygit.domain.state import StoryState
from storygit.store.snapshots import LEDGER_KEY, SnapshotStore

DEFAULT_BRANCH = "main"

_IMMUTABLE_NODE_FIELDS = frozenset({"id", "node_type", "parent_id"})


class BranchStore:
    """CRUD for named pointers into the snapshot chain."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Create a branch store over an open connection."""
        self.conn = conn

    def set(self, name: str, snapshot_id: SnapshotId) -> None:
        """Point a branch at a snapshot, creating the branch if needed."""
        self.conn.execute(
            "INSERT INTO branches(name, snapshot_id) VALUES (?, ?) "
            "ON CONFLICT(name) DO UPDATE SET snapshot_id = excluded.snapshot_id",
            (name, snapshot_id),
        )

    def head(self, name: str) -> SnapshotId:
        """Snapshot a branch currently points at.

        Raises:
            SnapshotNotFoundError: If the branch does not exist.
        """
        row = self.conn.execute(
            "SELECT snapshot_id FROM branches WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            raise SnapshotNotFoundError(f"no branch {name!r}")
        return SnapshotId(row["snapshot_id"])

    def exists(self, name: str) -> bool:
        """Whether a branch exists."""
        return (
            self.conn.execute("SELECT 1 FROM branches WHERE name = ?", (name,)).fetchone()
            is not None
        )

    def list(self) -> dict[str, SnapshotId]:
        """All branches and their heads, by name."""
        rows = self.conn.execute("SELECT name, snapshot_id FROM branches ORDER BY name").fetchall()
        return {row["name"]: SnapshotId(row["snapshot_id"]) for row in rows}


# --- structural diff ----------------------------------------------------------


def _depth(state: StoryState, node_id: NodeId) -> int:
    depth = 0
    current = state.nodes[node_id].parent_id
    while current is not None:
        depth += 1
        current = state.nodes[current].parent_id
    return depth


def _top_most_removed(before: StoryState, removed: set[NodeId]) -> list[NodeId]:
    """Removed nodes whose parent survives, so one recursive op covers the subtree."""
    return sorted(
        (nid for nid in removed if (before.nodes[nid].parent_id or NodeId("")) not in removed),
        key=lambda nid: (_depth(before, nid), nid),
    )


def structural_diff(before: StoryState, after: StoryState) -> Diff:
    """Compute the diff that turns ``before`` into ``after``.

    Object-granular and deterministic: it compares ids and content, emits removals
    before additions and parents before children, and finishes with a fixup pass so
    that ``apply(before, structural_diff(before, after))`` is content-identical to
    ``after``.

    Args:
        before: Source state.
        after: Target state.

    Returns:
        A diff authored by ``system``.
    """
    ops: list[Op] = []

    changed_nodes = {
        nid
        for nid in before.nodes.keys() & after.nodes.keys()
        if before.nodes[nid] != after.nodes[nid]
    }
    removed_nodes = before.nodes.keys() - after.nodes.keys()
    touched = changed_nodes | removed_nodes
    unlocked = sorted(before.ledger.locks & touched)
    ops.extend(ClearLock(node_id=nid) for nid in unlocked)

    # Facts and epistemic edges go before entity removals so cascades never fire.
    for fid in sorted(before.facts.keys() - after.facts.keys()):
        ops.append(RemoveFact(fact_id=fid))
    for character, fact_id in sorted(before.knows.keys() - after.knows.keys()):
        ops.append(RemoveKnows(character=character, fact_id=fact_id))
    for tid in sorted(before.threads.keys() - after.threads.keys()):
        ops.append(RemoveThread(thread_id=tid))

    for nid in _top_most_removed(before, set(removed_nodes)):
        ops.append(RemoveNode(node_id=nid, recursive=True))
    for eid in sorted(before.entities.keys() - after.entities.keys()):
        ops.append(RemoveEntity(entity_id=eid))

    for eid in sorted(after.entities.keys() - before.entities.keys()):
        ops.append(AddEntity(entity=after.entities[eid]))
    for eid in sorted(before.entities.keys() & after.entities.keys()):
        old, new = before.entities[eid], after.entities[eid]
        if old != new:
            ops.append(UpdateEntity(entity_id=eid, fields=_field_delta(old, new, {"id"})))

    added_nodes = sorted(
        after.nodes.keys() - before.nodes.keys(), key=lambda nid: (_depth(after, nid), nid)
    )
    for nid in added_nodes:
        ops.append(AddNode(node=after.nodes[nid]))
    for nid in sorted(changed_nodes, key=lambda n: (_depth(after, n), n)):
        old_node, new_node = before.nodes[nid], after.nodes[nid]
        if old_node.parent_id != new_node.parent_id:
            ops.append(
                MoveNode(
                    node_id=nid,
                    new_parent_id=new_node.parent_id,
                    new_position=new_node.position,
                )
            )
        fields = _field_delta(old_node, new_node, _IMMUTABLE_NODE_FIELDS)
        if fields:
            ops.append(UpdateNode(node_id=nid, fields=fields))

    for fid in sorted(after.facts.keys() - before.facts.keys()):
        ops.append(AddFact(fact=after.facts[fid]))
    for fid in sorted(before.facts.keys() & after.facts.keys()):
        old_fact, new_fact = before.facts[fid], after.facts[fid]
        if old_fact != new_fact:
            ops.append(UpdateFact(fact_id=fid, fields=_field_delta(old_fact, new_fact, {"id"})))

    for key in sorted(after.knows.keys() - before.knows.keys()):
        ops.append(AddKnows(knows=after.knows[key]))
    for key in sorted(before.knows.keys() & after.knows.keys()):
        if before.knows[key] != after.knows[key]:
            ops.append(AddKnows(knows=after.knows[key]))

    for tid in sorted(after.threads.keys() - before.threads.keys()):
        ops.append(OpenThread(thread=after.threads[tid]))
    for tid in sorted(before.threads.keys() & after.threads.keys()):
        old_thread, new_thread = before.threads[tid], after.threads[tid]
        if old_thread != new_thread:
            ops.append(
                UpdateThread(thread_id=tid, fields=_field_delta(old_thread, new_thread, {"id"}))
            )

    # Locks temporarily cleared above must be restored, so the ledger is rewritten
    # whenever it differs *or* whenever we had to unlock something to get here.
    if before.ledger != after.ledger or unlocked:
        ops.append(ReplaceLedger(ledger=after.ledger))

    diff = Diff(ops=tuple(ops), author=DiffAuthor.system, intent="structural diff")
    return _with_fixups(before, after, diff)


def _field_delta(old: Any, new: Any, skip: Iterable[str]) -> dict[str, Any]:
    """Fields whose values differ, as a JSON-safe mapping."""
    skipped = set(skip)
    old_data = old.model_dump(mode="json")
    new_data = new.model_dump(mode="json")
    return {k: v for k, v in new_data.items() if k not in skipped and old_data.get(k) != v}


def _with_fixups(before: StoryState, after: StoryState, diff: Diff) -> Diff:
    """Append corrective node updates so the diff reproduces ``after`` exactly.

    ``AddFact`` maintains the establishing beat's ``produces`` list as a
    convenience, which can leave a node one field away from the target when facts
    and nodes were both edited. Rather than special-case the interaction, apply the
    diff and correct whatever still differs — one deterministic extra pass.
    """
    result = apply(before, diff)
    extra: list[UpdateNode] = []
    for nid, node in after.nodes.items():
        current = result.nodes.get(nid)
        if current is not None and current != node:
            fields = _field_delta(current, node, _IMMUTABLE_NODE_FIELDS)
            if fields:
                extra.append(UpdateNode(node_id=nid, fields=fields))
    if not extra:
        return diff
    locked = sorted(result.ledger.locks & {op.node_id for op in extra})
    tail: tuple[Op, ...] = (*(ClearLock(node_id=nid) for nid in locked), *extra)
    if locked:
        tail = (*tail, ReplaceLedger(ledger=after.ledger))
    return diff.model_copy(update={"ops": (*diff.ops, *tail)})


# --- three-way merge ----------------------------------------------------------


class MergeConflict(BaseModel):
    """One object that both branches changed differently.

    Attributes:
        kind: Object kind (``node``, ``fact``, ...).
        logical_id: The object's stable id.
        base_hash: Content hash in the common ancestor, ``None`` if newly added.
        ours_hash: Content hash on the branch we are merging into.
        theirs_hash: Content hash on the branch we are merging from.
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    logical_id: str
    base_hash: str | None
    ours_hash: str | None
    theirs_hash: str | None


class MergeResult(BaseModel):
    """The outcome of a three-way merge.

    Attributes:
        diff: Changes to apply to *ours* to take theirs' non-conflicting work.
        conflicts: Objects that need a human decision. Never auto-resolved.
    """

    model_config = ConfigDict(frozen=True)

    diff: Diff
    conflicts: tuple[MergeConflict, ...] = ()

    @property
    def clean(self) -> bool:
        """Whether the merge completed without conflicts."""
        return not self.conflicts


def merge(
    store: SnapshotStore,
    base_id: SnapshotId,
    ours_id: SnapshotId,
    theirs_id: SnapshotId,
) -> MergeResult:
    """Three-way merge two snapshots at object granularity.

    For every object: if both sides agree, or only one side changed it, the answer
    is obvious and is taken. If both changed it differently, it is a conflict and is
    returned for the writer to resolve — the system never guesses which version of a
    scene the author meant.

    Args:
        store: Snapshot store to read manifests and objects from.
        base_id: Common ancestor.
        ours_id: The branch being merged into.
        theirs_id: The branch being merged from.

    Returns:
        A :class:`MergeResult` whose diff applies cleanly to *ours*.
    """
    base, ours, theirs = (store.manifest(s) for s in (base_id, ours_id, theirs_id))
    ours_state = store.load_state(ours_id)
    theirs_state = store.load_state(theirs_id)

    conflicts: list[MergeConflict] = []
    take_theirs: dict[str, set[str]] = {}
    for kind in sorted(set(base) | set(ours) | set(theirs)):
        base_kind, ours_kind, theirs_kind = (
            base.get(kind, {}),
            ours.get(kind, {}),
            theirs.get(kind, {}),
        )
        for logical_id in sorted(set(base_kind) | set(ours_kind) | set(theirs_kind)):
            b = base_kind.get(logical_id)
            o = ours_kind.get(logical_id)
            t = theirs_kind.get(logical_id)
            if o == t:
                continue
            if o == b:
                take_theirs.setdefault(kind, set()).add(logical_id)
            elif t == b:
                continue
            else:
                conflicts.append(
                    MergeConflict(
                        kind=kind, logical_id=logical_id, base_hash=b, ours_hash=o, theirs_hash=t
                    )
                )

    target = _state_with(ours_state, theirs_state, take_theirs)
    return MergeResult(
        diff=structural_diff(ours_state, target).model_copy(
            update={"intent": f"merge {theirs_id[:8]} into {ours_id[:8]}"}
        ),
        conflicts=tuple(conflicts),
    )


def _state_with(ours: StoryState, theirs: StoryState, take: dict[str, set[str]]) -> StoryState:
    """Build the merged target: ours, with theirs' non-conflicting objects."""
    nodes = dict(ours.nodes)
    entities = dict(ours.entities)
    facts = dict(ours.facts)
    knows = dict(ours.knows)
    threads = dict(ours.threads)
    ledger = ours.ledger

    for logical_id in take.get("node", set()):
        node_id = NodeId(logical_id)
        if node_id in theirs.nodes:
            nodes[node_id] = theirs.nodes[node_id]
        else:
            nodes.pop(node_id, None)
    for logical_id in take.get("entity", set()):
        entity_id = EntityId(logical_id)
        if entity_id in theirs.entities:
            entities[entity_id] = theirs.entities[entity_id]
        else:
            entities.pop(entity_id, None)
    for logical_id in take.get("fact", set()):
        fact_id = FactId(logical_id)
        if fact_id in theirs.facts:
            facts[fact_id] = theirs.facts[fact_id]
        else:
            facts.pop(fact_id, None)
    for logical_id in take.get("knows", set()):
        character_part, fact_part = logical_id.split("|", 1)
        key = (EntityId(character_part), FactId(fact_part))
        if key in theirs.knows:
            knows[key] = theirs.knows[key]
        else:
            knows.pop(key, None)
    for logical_id in take.get("thread", set()):
        thread_id = ThreadId(logical_id)
        if thread_id in theirs.threads:
            threads[thread_id] = theirs.threads[thread_id]
        else:
            threads.pop(thread_id, None)
    if LEDGER_KEY in take.get("ledger", set()):
        ledger = theirs.ledger

    return StoryState.build(
        nodes=nodes,
        entities=entities,
        facts=facts,
        knows=knows,
        threads=threads,
        ledger=ledger,
    )
