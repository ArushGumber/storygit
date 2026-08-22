"""The open-thread ledger.

A serial drops threads; that is the failure mode that kills retention. Tracking
them explicitly means the system can say "you opened the sister subplot in episode
2 and have not touched it in nine episodes" without any model call.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from storygit.domain.ids import NodeId, ThreadId


class ThreadStatus(StrEnum):
    """Lifecycle of a narrative thread."""

    open = "open"
    paid_off = "paid_off"
    dropped = "dropped"


class Thread(BaseModel):
    """One open narrative question.

    Attributes:
        id: Stable identifier.
        description: What the thread is, in the writer's words.
        opened_at_beat: Beat that opened it.
        last_touched_beat: Most recent beat that advanced it.
        status: Whether it is still open, paid off, or abandoned.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: ThreadId
    description: str
    opened_at_beat: NodeId
    last_touched_beat: NodeId
    status: ThreadStatus = ThreadStatus.open
