"""Content-addressed snapshots: git, for story state.

Every object — a node, an entity, a fact, an epistemic edge, a thread, the ledger —
is stored once under the sha256 of its canonical JSON. A snapshot is a *manifest*:
a mapping from logical id to content hash. Committing a change that touches three
nodes therefore writes three new object rows and one manifest, not a fresh copy of
the story.

Two consequences worth knowing:

* Two states with identical content produce identical manifests, which is why
  "apply the same diff twice, get the same hashes" is testable.
* Structural diff between any two versions is a set comparison over manifests — no
  model call, no heuristics.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from storygit.domain.errors import SnapshotNotFoundError
from storygit.domain.ids import EntityId, FactId, NodeId, SnapshotId, ThreadId
from storygit.domain.ledger import WriterLedger
from storygit.domain.nodes import BaseNode, node_from_payload
from storygit.domain.state import StoryState
from storygit.domain.threads import Thread
from storygit.domain.world import Entity, Fact, Knows

Manifest = dict[str, dict[str, str]]
"""``{object kind: {logical id: content hash}}``."""

LEDGER_KEY = "ledger"
"""The ledger is a singleton; it lives under this logical id."""


def canonical_json(payload: Any) -> str:
    """Serialize to the exact string the store hashes.

    Sorted keys and no whitespace, so the same content always yields the same bytes
    and therefore the same hash.

    Args:
        payload: Any JSON-serializable value.

    Returns:
        Canonical JSON text.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def object_hash(kind: str, payload_json: str) -> str:
    """Content hash of a stored object.

    The kind is part of the hash so that two different object types can never
    collide on identical payloads.

    Args:
        kind: Object kind, e.g. ``"node"`` or ``"fact"``.
        payload_json: Canonical JSON of the object.

    Returns:
        Hex sha256 digest.
    """
    return hashlib.sha256(f"{kind}:{payload_json}".encode()).hexdigest()


def _model_json(model: Any) -> str:
    return canonical_json(model.model_dump(mode="json"))


class SnapshotStore:
    """Reads and writes snapshots against an open connection.

    Attributes:
        conn: The SQLite connection; owned by the caller.
    """

    def __init__(self, conn: sqlite3.Connection, *, clock: Callable[[], datetime] | None = None):
        """Create a store.

        Args:
            conn: An open connection from :func:`storygit.store.db.connect`.
            clock: Returns "now"; injectable so tests get deterministic snapshot ids.
        """
        self.conn = conn
        self._clock = clock or (lambda: datetime.now(UTC))

    # --- objects ----------------------------------------------------------------

    def put_object(self, kind: str, model: Any) -> str:
        """Store an object if it is not already present.

        Args:
            kind: Object kind.
            model: A pydantic model.

        Returns:
            The object's content hash.
        """
        payload = _model_json(model)
        digest = object_hash(kind, payload)
        self.conn.execute(
            "INSERT OR IGNORE INTO objects(hash, kind, payload) VALUES (?, ?, ?)",
            (digest, kind, payload),
        )
        return digest

    def get_object(self, digest: str) -> tuple[str, dict[str, Any]]:
        """Load an object by hash.

        Args:
            digest: Content hash.

        Returns:
            ``(kind, payload)``.

        Raises:
            SnapshotNotFoundError: If no object has that hash.
        """
        row = self.conn.execute(
            "SELECT kind, payload FROM objects WHERE hash = ?", (digest,)
        ).fetchone()
        if row is None:
            raise SnapshotNotFoundError(f"no object {digest}")
        return row["kind"], json.loads(row["payload"])

    # --- snapshots --------------------------------------------------------------

    def manifest_of_state(self, state: StoryState) -> Manifest:
        """Compute the manifest for a state, writing any new objects.

        Args:
            state: The state to persist.

        Returns:
            The manifest to store on the snapshot row.
        """
        manifest: Manifest = {
            "node": {},
            "entity": {},
            "fact": {},
            "knows": {},
            "thread": {},
            "ledger": {},
        }
        for node_id, node in state.nodes.items():
            manifest["node"][str(node_id)] = self.put_object("node", node)
        for entity_id, entity in state.entities.items():
            manifest["entity"][str(entity_id)] = self.put_object("entity", entity)
        for fact_id, fact in state.facts.items():
            manifest["fact"][str(fact_id)] = self.put_object("fact", fact)
        for (character, fact_id), edge in state.knows.items():
            manifest["knows"][f"{character}|{fact_id}"] = self.put_object("knows", edge)
        for thread_id, thread in state.threads.items():
            manifest["thread"][str(thread_id)] = self.put_object("thread", thread)
        manifest["ledger"][LEDGER_KEY] = self.put_object("ledger", state.ledger)
        return manifest

    def write_snapshot(
        self,
        state: StoryState,
        *,
        parent_id: SnapshotId | None,
        author: str,
        intent: str,
    ) -> SnapshotId:
        """Persist a state as a new snapshot.

        Args:
            state: The state to persist.
            parent_id: The snapshot this one descends from, if any.
            author: ``human``, ``ai``, or ``system``.
            intent: The intent line the change answers.

        Returns:
            The new snapshot's id.
        """
        manifest = self.manifest_of_state(state)
        manifest_json = canonical_json(manifest)
        created_at = self._clock().isoformat()
        snapshot_id = SnapshotId(
            hashlib.sha256(
                canonical_json(
                    {
                        "manifest": manifest_json,
                        "parent": parent_id,
                        "author": author,
                        "intent": intent,
                        "created_at": created_at,
                    }
                ).encode()
            ).hexdigest()[:32]
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO snapshots(id, parent_id, created_at, author, intent, manifest)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (snapshot_id, parent_id, created_at, author, intent, manifest_json),
        )
        return snapshot_id

    def manifest(self, snapshot_id: SnapshotId) -> Manifest:
        """Load a snapshot's manifest.

        Args:
            snapshot_id: The snapshot.

        Returns:
            Its manifest.

        Raises:
            SnapshotNotFoundError: If the snapshot does not exist.
        """
        row = self.conn.execute(
            "SELECT manifest FROM snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise SnapshotNotFoundError(f"no snapshot {snapshot_id}")
        return dict(json.loads(row["manifest"]))

    def info(self, snapshot_id: SnapshotId) -> dict[str, Any]:
        """Metadata for one snapshot (parent, author, intent, timestamp)."""
        row = self.conn.execute(
            "SELECT id, parent_id, created_at, author, intent FROM snapshots WHERE id = ?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise SnapshotNotFoundError(f"no snapshot {snapshot_id}")
        return dict(row)

    def load_state(self, snapshot_id: SnapshotId) -> StoryState:
        """Rebuild the full in-memory aggregate for a snapshot.

        Args:
            snapshot_id: The snapshot to load.

        Returns:
            A fully indexed :class:`~storygit.domain.state.StoryState`.
        """
        manifest = self.manifest(snapshot_id)
        nodes: dict[NodeId, BaseNode] = {}
        for logical_id, digest in manifest.get("node", {}).items():
            _, payload = self.get_object(digest)
            nodes[NodeId(logical_id)] = node_from_payload(str(payload["node_type"]), payload)
        entities: dict[EntityId, Entity] = {}
        for logical_id, digest in manifest.get("entity", {}).items():
            _, payload = self.get_object(digest)
            entities[EntityId(logical_id)] = Entity.model_validate(payload)
        facts: dict[FactId, Fact] = {}
        for logical_id, digest in manifest.get("fact", {}).items():
            _, payload = self.get_object(digest)
            facts[FactId(logical_id)] = Fact.model_validate(payload)
        knows: dict[tuple[EntityId, FactId], Knows] = {}
        for _, digest in manifest.get("knows", {}).items():
            _, payload = self.get_object(digest)
            edge = Knows.model_validate(payload)
            knows[edge.key] = edge
        threads: dict[ThreadId, Thread] = {}
        for logical_id, digest in manifest.get("thread", {}).items():
            _, payload = self.get_object(digest)
            threads[ThreadId(logical_id)] = Thread.model_validate(payload)
        ledger_hash = manifest.get("ledger", {}).get(LEDGER_KEY)
        ledger = WriterLedger()
        if ledger_hash:
            _, payload = self.get_object(ledger_hash)
            ledger = WriterLedger.model_validate(payload)
        return StoryState.build(
            nodes=nodes,
            entities=entities,
            facts=facts,
            knows=knows,
            threads=threads,
            ledger=ledger,
        )

    def history(self, snapshot_id: SnapshotId, limit: int = 100) -> list[dict[str, Any]]:
        """Walk the parent chain from a snapshot back toward the root.

        Args:
            snapshot_id: Where to start.
            limit: Maximum number of snapshots to return.

        Returns:
            Snapshot metadata, newest first.
        """
        out: list[dict[str, Any]] = []
        current: SnapshotId | None = snapshot_id
        while current is not None and len(out) < limit:
            info = self.info(current)
            out.append(info)
            parent = info["parent_id"]
            current = SnapshotId(parent) if parent else None
        return out
