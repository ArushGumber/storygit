"""The learning signal: every writer action, recorded as it happens.

The preference layer in chunk 4 trains on nothing but this table. Recording it from
chunk 2 onward means that by the time there is a model to fit, there is already data to
fit it on — and it means the signal is captured where it is *unambiguous*, at the moment
of the action, rather than reconstructed later from snapshot diffs.

Every signal carries the proposal it refers to and the set of candidates it was shown
alongside. That second field is what makes paired preferences possible: "accepted A over
B and C" is a much stronger statement than "accepted A".
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from storygit.domain.ids import NodeId, ProposalId

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    branch      TEXT NOT NULL,
    node_id     TEXT,
    proposal_id TEXT,
    shown_with  TEXT NOT NULL DEFAULT '[]',
    payload     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_signals_kind ON signals(kind);
CREATE INDEX IF NOT EXISTS idx_signals_proposal ON signals(proposal_id);
"""


class SignalKind(StrEnum):
    """What the writer did."""

    accept = "accept"
    reject = "reject"
    edit = "edit"
    lock = "lock"
    unlock = "unlock"
    style_note = "style_note"
    style_note_removed = "style_note_removed"
    criterion_added = "criterion_added"
    criterion_removed = "criterion_removed"
    dial_moved = "dial_moved"
    dismiss_stale = "dismiss_stale"
    strike_fact = "strike_fact"


class Signal(BaseModel):
    """One recorded writer action.

    Attributes:
        kind: What they did.
        branch: Which branch they did it on.
        node_id: The node it concerned.
        proposal_id: The proposal it concerned.
        shown_with: The other candidates on screen at the time. This is what turns an
            accept into a *preference*.
        payload: Kind-specific extras — for an edit, the before and after text.
        ts: When, ISO-8601.
    """

    model_config = ConfigDict(frozen=True)

    kind: SignalKind
    branch: str = "main"
    node_id: NodeId | None = None
    proposal_id: ProposalId | None = None
    shown_with: tuple[ProposalId, ...] = ()
    payload: dict[str, Any] = Field(default_factory=dict)
    ts: str = ""


class SignalStore:
    """Append-and-read access to the signals table."""

    def __init__(
        self, conn: sqlite3.Connection, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        """Wrap an open connection and ensure the table exists.

        Args:
            conn: A connection, typically the repository's.
            clock: Injectable "now".
        """
        self.conn = conn
        self.conn.executescript(SCHEMA)
        self._clock = clock or (lambda: datetime.now(UTC))

    def record(self, signal: Signal) -> None:
        """Append one signal."""
        self.conn.execute(
            "INSERT INTO signals(ts, kind, branch, node_id, proposal_id, shown_with, payload)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                signal.ts or self._clock().isoformat(),
                signal.kind.value,
                signal.branch,
                signal.node_id,
                signal.proposal_id,
                json.dumps(list(signal.shown_with)),
                json.dumps(signal.payload),
            ),
        )

    def all(self, *, kind: SignalKind | None = None, branch: str | None = None) -> list[Signal]:
        """Read signals back, oldest first.

        Args:
            kind: Restrict to one kind.
            branch: Restrict to one branch.

        Returns:
            The matching signals in chronological order.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind.value)
        if branch is not None:
            clauses.append("branch = ?")
            params.append(branch)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM signals {where} ORDER BY id ASC",
            params,
        ).fetchall()
        return [
            Signal(
                kind=SignalKind(row["kind"]),
                branch=row["branch"],
                node_id=NodeId(row["node_id"]) if row["node_id"] else None,
                proposal_id=ProposalId(row["proposal_id"]) if row["proposal_id"] else None,
                shown_with=tuple(ProposalId(p) for p in json.loads(row["shown_with"])),
                payload=json.loads(row["payload"]),
                ts=row["ts"],
            )
            for row in rows
        ]

    def counts(self) -> dict[str, int]:
        """How many of each kind of signal exist."""
        rows = self.conn.execute("SELECT kind, COUNT(*) AS n FROM signals GROUP BY kind").fetchall()
        return {row["kind"]: int(row["n"]) for row in rows}
