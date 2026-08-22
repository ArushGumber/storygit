"""``Repository``: the only door into story state.

Every read goes through ``state()``; every write goes through ``commit_diff()``.
Nothing else in the system is allowed to mutate a story, which is what makes the
audit trail complete — if it happened, there is a snapshot for it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from storygit.domain.apply import apply
from storygit.domain.diff import Diff
from storygit.domain.errors import SnapshotNotFoundError
from storygit.domain.ids import SnapshotId
from storygit.domain.state import StoryState
from storygit.store.branches import (
    DEFAULT_BRANCH,
    BranchStore,
    MergeResult,
    merge,
    structural_diff,
)
from storygit.store.db import connect
from storygit.store.snapshots import SnapshotStore


class Repository:
    """A story database: snapshots, branches, and the apply-and-commit loop.

    Attributes:
        conn: The underlying SQLite connection.
        snapshots: Snapshot/object store.
        branches: Branch pointer store.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Wrap an open connection.

        Args:
            conn: Connection from :func:`storygit.store.db.connect`.
            clock: Injectable "now", so tests get deterministic snapshot ids.
        """
        self.conn = conn
        self._clock = clock or (lambda: datetime.now(UTC))
        self.snapshots = SnapshotStore(conn, clock=self._clock)
        self.branches = BranchStore(conn)

    @classmethod
    def open(cls, path: Path | str, *, clock: Callable[[], datetime] | None = None) -> Self:
        """Open or create a repository at a path.

        Args:
            path: Database file, or ``":memory:"``.
            clock: Injectable "now".

        Returns:
            An open repository.
        """
        return cls(connect(path), clock=clock)

    def close(self) -> None:
        """Close the underlying connection."""
        self.conn.close()

    def __enter__(self) -> Self:
        """Enter a context manager; returns self."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the connection on exit."""
        self.close()

    # --- lifecycle --------------------------------------------------------------

    def initialize(self, state: StoryState, *, branch: str = DEFAULT_BRANCH) -> SnapshotId:
        """Write the first snapshot and create the branch pointing at it.

        Args:
            state: The initial state (typically just a story root).
            branch: Branch name to create.

        Returns:
            The root snapshot's id.
        """
        snapshot_id = self.snapshots.write_snapshot(
            state, parent_id=None, author="system", intent="initialize"
        )
        self.branches.set(branch, snapshot_id)
        return snapshot_id

    def is_initialized(self, branch: str = DEFAULT_BRANCH) -> bool:
        """Whether the branch exists yet."""
        return self.branches.exists(branch)

    # --- reads ------------------------------------------------------------------

    def state(self, branch: str = DEFAULT_BRANCH) -> StoryState:
        """Current state of a branch."""
        return self.snapshots.load_state(self.branches.head(branch))

    def state_at(self, snapshot_id: SnapshotId) -> StoryState:
        """State as of a specific snapshot."""
        return self.snapshots.load_state(snapshot_id)

    def head(self, branch: str = DEFAULT_BRANCH) -> SnapshotId:
        """Snapshot a branch points at."""
        return self.branches.head(branch)

    def history(self, branch: str = DEFAULT_BRANCH, limit: int = 100) -> list[dict[str, Any]]:
        """Snapshot metadata for a branch, newest first."""
        return self.snapshots.history(self.branches.head(branch), limit=limit)

    # --- writes -----------------------------------------------------------------

    def commit_diff(
        self,
        diff: Diff,
        *,
        branch: str = DEFAULT_BRANCH,
        intent: str | None = None,
    ) -> SnapshotId:
        """Apply a diff to a branch and commit the result as a new snapshot.

        Args:
            diff: The diff to apply.
            branch: Branch to commit on.
            intent: Overrides the diff's own intent line on the snapshot row.

        Returns:
            The new snapshot's id.

        Raises:
            ApplyError: If the diff does not apply; nothing is written.
        """
        parent = self.branches.head(branch)
        new_state = apply(self.snapshots.load_state(parent), diff)
        snapshot_id = self.snapshots.write_snapshot(
            new_state,
            parent_id=parent,
            author=diff.author.value,
            intent=intent if intent is not None else diff.intent,
        )
        self.branches.set(branch, snapshot_id)
        return snapshot_id

    def preview_apply(self, diff: Diff, *, branch: str = DEFAULT_BRANCH) -> StoryState:
        """Apply a diff to a scratch copy without committing anything.

        This is what the UI uses to show a candidate's consequences before the
        writer decides.

        Args:
            diff: The diff to try.
            branch: Branch to preview against.

        Returns:
            The state that would result.
        """
        return apply(self.state(branch), diff)

    # --- branching --------------------------------------------------------------

    def create_branch(
        self, name: str, *, from_branch: str = DEFAULT_BRANCH, at: SnapshotId | None = None
    ) -> SnapshotId:
        """Create a branch at a snapshot.

        Args:
            name: New branch name.
            from_branch: Branch whose head to start from when ``at`` is omitted.
            at: Explicit snapshot to branch from.

        Returns:
            The snapshot the new branch points at.

        Raises:
            SnapshotNotFoundError: If the branch already exists.
        """
        if self.branches.exists(name):
            raise SnapshotNotFoundError(f"branch {name!r} already exists")
        snapshot_id = at or self.branches.head(from_branch)
        self.branches.set(name, snapshot_id)
        return snapshot_id

    def list_branches(self) -> dict[str, SnapshotId]:
        """All branches and their heads."""
        return self.branches.list()

    def diff(self, a: SnapshotId, b: SnapshotId) -> Diff:
        """Structural diff between two snapshots."""
        return structural_diff(self.snapshots.load_state(a), self.snapshots.load_state(b))

    def diff_branches(self, a: str, b: str) -> Diff:
        """Structural diff between two branch heads."""
        return self.diff(self.branches.head(a), self.branches.head(b))

    def merge_base(self, a: SnapshotId, b: SnapshotId) -> SnapshotId:
        """Most recent common ancestor of two snapshots.

        Args:
            a: One snapshot.
            b: The other.

        Returns:
            The common ancestor.

        Raises:
            SnapshotNotFoundError: If the two share no history.
        """
        ancestors_a = {info["id"] for info in self.snapshots.history(a, limit=10_000)}
        for info in self.snapshots.history(b, limit=10_000):
            if info["id"] in ancestors_a:
                return SnapshotId(info["id"])
        raise SnapshotNotFoundError(f"snapshots {a} and {b} share no common ancestor")

    def merge_branches(self, ours: str, theirs: str) -> MergeResult:
        """Three-way merge ``theirs`` into ``ours`` without committing.

        The caller inspects conflicts, then commits the result diff if it is happy.

        Args:
            ours: Branch being merged into.
            theirs: Branch being merged from.

        Returns:
            The merge result.
        """
        ours_head = self.branches.head(ours)
        theirs_head = self.branches.head(theirs)
        base = self.merge_base(ours_head, theirs_head)
        return merge(self.snapshots, base, ours_head, theirs_head)
