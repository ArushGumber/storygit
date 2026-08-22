"""Every model call, recorded.

This is the source of the cost metric, the cache-efficiency numbers, and the
tokens-per-writer-action figure the evaluation reports. It is also the audit trail that
makes "no key was ever logged" checkable: the schema has a ``key_index`` column and no
column that could hold a key, and the writer scrubs every message through
:meth:`storygit.config.Settings.redact` before it is stored.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS calllog (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    purpose           TEXT NOT NULL,
    key_index         INTEGER,          -- rotation slot, never the key
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_s         REAL NOT NULL DEFAULT 0,
    cached            INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL NOT NULL DEFAULT 0,
    error             TEXT
);
CREATE INDEX IF NOT EXISTS idx_calllog_purpose ON calllog(purpose);
CREATE INDEX IF NOT EXISTS idx_calllog_provider ON calllog(provider);
"""


class CallLog:
    """Append-only log of provider calls, with aggregation for reporting."""

    def __init__(
        self,
        path: Path | str = ":memory:",
        *,
        clock: Callable[[], datetime] | None = None,
        redactor: Callable[[str], str] | None = None,
    ) -> None:
        """Open (creating if needed) the log database.

        Args:
            path: Database file, or ``":memory:"``.
            clock: Injectable "now", so tests are deterministic.
            redactor: Scrubs secrets out of error messages before they are stored.
                Defaults to the process settings' redactor.
        """
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._clock = clock or (lambda: datetime.now(UTC))
        if redactor is None:
            from storygit.config import get_settings

            redactor = get_settings().redact
        self._redact = redactor

    def record(
        self,
        *,
        provider: str,
        model: str,
        purpose: str,
        key_index: int | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_s: float = 0.0,
        cached: bool = False,
        cost_usd: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Append one row.

        Args:
            provider: Backend name.
            model: Model id.
            purpose: Purpose tag.
            key_index: Rotation slot that served the call. Never a key.
            prompt_tokens: Input tokens.
            completion_tokens: Output tokens.
            latency_s: Wall-clock seconds.
            cached: Whether the response came from cache.
            cost_usd: Estimated dollars.
            error: Error message, already redacted on the way in.
        """
        self._conn.execute(
            "INSERT INTO calllog(ts, provider, model, purpose, key_index, prompt_tokens,"
            " completion_tokens, latency_s, cached, cost_usd, error)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._clock().isoformat(),
                provider,
                model,
                purpose,
                key_index,
                prompt_tokens,
                completion_tokens,
                latency_s,
                int(cached),
                cost_usd,
                self._redact(error) if error else None,
            ),
        )

    def rows(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Recent rows, newest first."""
        cursor = self._conn.execute("SELECT * FROM calllog ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def summary(self) -> dict[str, Any]:
        """Aggregate the log for the evaluation and the Eval tab.

        Returns:
            Totals plus per-purpose and per-provider breakdowns.
        """
        total = self._conn.execute(
            "SELECT COUNT(*) AS calls, COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,"
            " COALESCE(SUM(completion_tokens),0) AS completion_tokens,"
            " COALESCE(SUM(cost_usd),0) AS cost_usd,"
            " COALESCE(SUM(cached),0) AS cache_hits,"
            " COALESCE(SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END),0) AS errors,"
            " COALESCE(AVG(latency_s),0) AS mean_latency_s"
            " FROM calllog"
        ).fetchone()
        by_purpose = self._conn.execute(
            "SELECT purpose, COUNT(*) AS calls,"
            " COALESCE(SUM(prompt_tokens + completion_tokens),0) AS tokens,"
            " COALESCE(SUM(cost_usd),0) AS cost_usd,"
            " COALESCE(SUM(cached),0) AS cache_hits"
            " FROM calllog GROUP BY purpose ORDER BY tokens DESC"
        ).fetchall()
        by_provider = self._conn.execute(
            "SELECT provider, COUNT(*) AS calls,"
            " COALESCE(SUM(prompt_tokens + completion_tokens),0) AS tokens,"
            " COALESCE(SUM(cost_usd),0) AS cost_usd"
            " FROM calllog GROUP BY provider ORDER BY calls DESC"
        ).fetchall()
        summary = dict(total)
        summary["total_tokens"] = summary["prompt_tokens"] + summary["completion_tokens"]
        summary["cache_hit_rate"] = (
            summary["cache_hits"] / summary["calls"] if summary["calls"] else 0.0
        )
        summary["by_purpose"] = [dict(r) for r in by_purpose]
        summary["by_provider"] = [dict(r) for r in by_provider]
        return summary

    def render_summary(self) -> str:
        """The summary as a plain-text table, for logs and chunk write-ups."""
        s = self.summary()
        lines = [
            f"calls={s['calls']}  tokens={s['total_tokens']}"
            f" (in {s['prompt_tokens']} / out {s['completion_tokens']})"
            f"  cache_hits={s['cache_hits']} ({s['cache_hit_rate']:.0%})"
            f"  errors={s['errors']}  cost=${s['cost_usd']:.4f}"
            f"  mean_latency={s['mean_latency_s']:.2f}s",
            "",
            f"{'purpose':<24}{'calls':>7}{'tokens':>10}{'hits':>7}{'cost $':>10}",
        ]
        for row in s["by_purpose"]:
            lines.append(
                f"{row['purpose']:<24}{row['calls']:>7}{row['tokens']:>10}"
                f"{row['cache_hits']:>7}{row['cost_usd']:>10.4f}"
            )
        lines.append("")
        lines.append(f"{'provider':<24}{'calls':>7}{'tokens':>10}{'cost $':>10}")
        for row in s["by_provider"]:
            lines.append(
                f"{row['provider']:<24}{row['calls']:>7}{row['tokens']:>10}{row['cost_usd']:>10.4f}"
            )
        return "\n".join(lines)

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()
