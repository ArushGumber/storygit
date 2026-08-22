"""The spend guard.

Two jobs. It records estimated spend per call so the evaluation can report cost per
writer action. And it refuses any OpenRouter call that would cross the in-code cap,
*before* the call is made, using a conservative pre-estimate of the output size — a
guard that only notices after the money is gone is not a guard.

The cap here ($12) is deliberately below the account cap ($20). OpenRouter is locked
until chunk 7 regardless; this is the second line of defence.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from storygit.providers.base import BudgetExceeded
from storygit.providers.pricing import price_for

SCHEMA = """
CREATE TABLE IF NOT EXISTS spend (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    provider TEXT NOT NULL,
    model    TEXT NOT NULL,
    purpose  TEXT NOT NULL,
    cost_usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_spend_provider ON spend(provider);
"""


class BudgetGuard:
    """Tracks spend and refuses calls that would cross a cap.

    Attributes:
        cap_usd: The hard cap for metered providers.
    """

    METERED = ("openrouter",)
    """Providers whose spend counts against the cap. Gemini and Groq are free-tier."""

    def __init__(
        self,
        path: Path | str = ":memory:",
        *,
        cap_usd: float = 12.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Open (creating if needed) the spend database.

        Args:
            path: Database file, or ``":memory:"``.
            cap_usd: In-code cap for metered providers.
            clock: Injectable "now".
        """
        self.cap_usd = cap_usd
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._clock = clock or (lambda: datetime.now(UTC))

    def spent(self, provider: str | None = None) -> float:
        """Total recorded spend, optionally for one provider.

        Args:
            provider: Restrict to this provider; ``None`` sums the metered ones.

        Returns:
            Dollars spent.
        """
        if provider is None:
            placeholders = ",".join("?" for _ in self.METERED)
            row = self._conn.execute(
                f"SELECT COALESCE(SUM(cost_usd), 0) AS total FROM spend "
                f"WHERE provider IN ({placeholders})",
                self.METERED,
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM spend WHERE provider = ?",
                (provider,),
            ).fetchone()
        return float(row["total"])

    def remaining(self) -> float:
        """Dollars left under the cap."""
        return max(0.0, self.cap_usd - self.spent())

    def check(
        self,
        provider: str,
        model: str,
        *,
        prompt_tokens: int,
        max_output_tokens: int,
    ) -> None:
        """Refuse a call that could cross the cap.

        The estimate assumes the model emits its full output allowance, which is the
        conservative direction: the guard may refuse a call that would have fitted, and
        will never wave through one that would not.

        Args:
            provider: Backend name.
            model: Model id.
            prompt_tokens: Estimated input tokens.
            max_output_tokens: The request's output cap.

        Raises:
            BudgetExceeded: If the worst-case cost of this call exceeds what is left.
        """
        if provider not in self.METERED:
            return
        price = price_for(provider, model)
        worst_case = (
            prompt_tokens * price.prompt + max_output_tokens * price.completion
        ) / 1_000_000
        if worst_case > self.remaining():
            raise BudgetExceeded(
                f"{provider}/{model}: this call could cost ${worst_case:.4f}, "
                f"but only ${self.remaining():.4f} of the ${self.cap_usd:.2f} cap remains "
                f"(spent ${self.spent():.4f})"
            )

    def record(self, provider: str, model: str, purpose: str, cost_usd: float) -> None:
        """Record actual spend for a completed call."""
        self._conn.execute(
            "INSERT INTO spend(ts, provider, model, purpose, cost_usd) VALUES (?, ?, ?, ?, ?)",
            (self._clock().isoformat(), provider, model, purpose, cost_usd),
        )

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()
