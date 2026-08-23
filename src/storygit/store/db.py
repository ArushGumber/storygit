"""SQLite schema and connection handling.

One file holds the whole story: every version of every object, the snapshot chain,
and the branch pointers. SQLite is the right database here because there is exactly
one writer (one human), the whole dataset is small, and a single file is trivially
copyable — which for a versioned artifact is a feature, not a limitation.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Content-addressed object store: hash -> canonical JSON payload.
-- Objects are shared between snapshots, so an unchanged node costs nothing to
-- keep in a new version.
CREATE TABLE IF NOT EXISTS objects (
    hash    TEXT PRIMARY KEY,
    kind    TEXT NOT NULL,
    payload TEXT NOT NULL
);

-- One row per version of the story. `manifest` maps kind -> {logical id -> hash}.
CREATE TABLE IF NOT EXISTS snapshots (
    id         TEXT PRIMARY KEY,
    parent_id  TEXT,
    created_at TEXT NOT NULL,
    author     TEXT NOT NULL,
    intent     TEXT NOT NULL,
    manifest   TEXT NOT NULL,
    FOREIGN KEY (parent_id) REFERENCES snapshots(id)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_parent ON snapshots(parent_id);

-- Named pointers into the snapshot chain.
CREATE TABLE IF NOT EXISTS branches (
    name        TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL,
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) a story database.

    Args:
        path: File path, or ``":memory:"`` for a throwaway database.

    Returns:
        A connection with WAL journaling, foreign keys on, and the schema applied.
    """
    target = str(path)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False because the HTTP layer runs synchronous routes in a thread
    # pool, so the connection is used from a different thread than the one that opened it.
    # That is only safe because writes are serialized -- see Repository's write lock, and
    # the single-writer assumption documented in api/deps.py.
    conn = sqlite3.connect(target, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if target != ":memory:":
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    return conn
