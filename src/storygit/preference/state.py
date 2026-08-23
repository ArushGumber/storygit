"""Persisting learned per-writer state, in the same file as the story.

The learned state — preference weights, the voice head and centroid, the bandit posteriors
— lives in a ``preference_state`` table in the story's own SQLite file, keyed by branch
name.

Two reasons. **One file.** The whole point of the store is that a story is a single
copyable artifact with its full history; a sidecar directory of model weights that has to
travel with it would undo that. **Branch scoping falls out for free.** A what-if branch may
explore a direction the writer would reject on main, so the taste learned there should not
leak back — and with the state keyed by branch, it cannot.

The state is a JSON blob rather than columns. It is small, it is only ever read and written
whole, and its shape changes every time a learner is added; a schema migration per feature
would be pure friction for no benefit.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from pydantic import BaseModel, ConfigDict

from storygit.preference.bandit import BanditState
from storygit.preference.bt_head import BTWeights
from storygit.preference.voice import VoiceModel

SCHEMA = """
CREATE TABLE IF NOT EXISTS preference_state (
    branch     TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    payload    TEXT NOT NULL
);
"""


class PreferenceState(BaseModel):
    """Everything the preference layer has learned about one writer on one branch.

    Attributes:
        weights: The Bradley-Terry head.
        voice: The contrastive voice model and edit-direction vector.
        bandit: Posteriors over shown-set composition.
        pairs_seen: Comparisons the head has been fitted on.
        edits_seen: Edit pairs mined so far.
    """

    model_config = ConfigDict(frozen=True)

    weights: BTWeights = BTWeights.uniform()
    voice: VoiceModel = VoiceModel()
    bandit: BanditState = BanditState()
    pairs_seen: int = 0
    edits_seen: int = 0

    def summary(self) -> dict[str, Any]:
        """A small dict for logs and the interface's ledger pane."""
        return {
            "pairs_seen": self.pairs_seen,
            "edits_seen": self.edits_seen,
            "voice_trained": self.voice.is_trained,
            # A list of dicts rather than tuples: this goes through JSON on its way to
            # the run log and the interface, and tuples come back as lists, which breaks
            # round-trip equality.
            "top_weights": [
                {"feature": name, "weight": round(weight, 4)}
                for name, weight in self.weights.top_weights(4)
            ],
            "bandit_safe": round(self.bandit.mean_safe(), 3),
            "bandit_explore": round(self.bandit.mean_explore(), 3),
        }


class PreferenceStateStore:
    """Reads and writes preference state, keyed by branch."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        """Wrap an open connection and ensure the table exists.

        Args:
            conn: A connection, typically the repository's.
        """
        self.conn = conn
        self.conn.executescript(SCHEMA)

    def load(self, branch: str) -> PreferenceState:
        """Load a branch's state, returning a fresh one if it has none yet.

        Args:
            branch: Branch name.

        Returns:
            The state. A brand-new writer gets uniform weights, an untrained voice model,
            and uniform bandit posteriors — every learner degrades to "no opinion" rather
            than to an error.
        """
        row = self.conn.execute(
            "SELECT payload FROM preference_state WHERE branch = ?", (branch,)
        ).fetchone()
        if row is None:
            return PreferenceState()
        return PreferenceState.model_validate(json.loads(row["payload"]))

    def save(self, branch: str, state: PreferenceState) -> None:
        """Write a branch's state, replacing whatever was there."""
        self.conn.execute(
            "INSERT INTO preference_state(branch, payload) VALUES (?, ?) "
            "ON CONFLICT(branch) DO UPDATE SET payload = excluded.payload, "
            "updated_at = datetime('now')",
            (branch, state.model_dump_json()),
        )

    def branches(self) -> list[str]:
        """Branches that have learned state."""
        rows = self.conn.execute("SELECT branch FROM preference_state ORDER BY branch").fetchall()
        return [row["branch"] for row in rows]
