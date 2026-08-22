"""Read-through disk cache for model calls.

Keyed on the SHA-256 of everything that could change the answer: provider, model,
messages, temperature, token cap, schema, stop sequences, and ``sample_index``. The last
one is what lets six axis-conditioned candidates for the same beat be six different
cached entries instead of one.

The cache lives outside the repository (``<workspace>/.storygit_cache/``) because it
fills up with model output, and because a cache checked into a deliverable is a way to
accidentally ship somebody's API responses. Keys are hashes of *prompts*, never of keys:
no secret is an input to the digest.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from storygit.providers.base import LLMRequest, LLMResponse

SCHEMA = """
CREATE TABLE IF NOT EXISTS llmcache (
    key        TEXT PRIMARY KEY,
    provider   TEXT NOT NULL,
    model      TEXT NOT NULL,
    purpose    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    response   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llmcache_purpose ON llmcache(purpose);
"""


def cache_key(provider: str, model: str, request: LLMRequest) -> str:
    """Digest of everything that determines the answer.

    Args:
        provider: Backend name.
        model: Model that will serve the call.
        request: The request.

    Returns:
        A hex SHA-256 digest.
    """
    payload = {
        "provider": provider,
        "model": model,
        "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "schema": request.json_schema,
        "stop": list(request.stop),
        "sample_index": request.sample_index,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class LLMCache:
    """SQLite-backed read-through cache.

    Attributes:
        path: Database file, or ``":memory:"``.
        hits: Cache hits so far this process.
        misses: Cache misses so far this process.
    """

    def __init__(self, path: Path | str = ":memory:") -> None:
        """Open (creating if needed) the cache database.

        Args:
            path: Database file, or ``":memory:"`` for a throwaway cache.
        """
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> LLMResponse | None:
        """Look up a cached response.

        Args:
            key: A key from :func:`cache_key`.

        Returns:
            The response with ``cached=True``, or ``None`` on a miss.
        """
        row = self._conn.execute("SELECT response FROM llmcache WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        response = LLMResponse.model_validate_json(row["response"])
        return response.model_copy(update={"cached": True, "latency_s": 0.0})

    def put(self, key: str, purpose: str, response: LLMResponse) -> None:
        """Store a response.

        Args:
            key: A key from :func:`cache_key`.
            purpose: The request's purpose tag, for cache statistics by purpose.
            response: The response to store.
        """
        self._conn.execute(
            "INSERT OR REPLACE INTO llmcache(key, provider, model, purpose, response) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                key,
                response.provider,
                response.model,
                purpose,
                response.model_copy(update={"cached": False}).model_dump_json(),
            ),
        )

    def stats(self) -> dict[str, int]:
        """Hits, misses, and the number of stored entries."""
        row = self._conn.execute("SELECT COUNT(*) AS n FROM llmcache").fetchone()
        return {"hits": self.hits, "misses": self.misses, "entries": int(row["n"])}

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()
