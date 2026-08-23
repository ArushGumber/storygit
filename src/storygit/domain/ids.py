"""Distinct identifier types and a deterministic identifier generator.

Every kind of object in the story graph gets its own :class:`typing.NewType` over
``str``. They are all strings at runtime, but mypy treats them as distinct, which
catches the easy mistake of passing a fact id where a node id belongs.
"""

from __future__ import annotations

import random
from typing import NewType

NodeId = NewType("NodeId", str)
EntityId = NewType("EntityId", str)
FactId = NewType("FactId", str)
ThreadId = NewType("ThreadId", str)
SnapshotId = NewType("SnapshotId", str)
ProposalId = NewType("ProposalId", str)

_ALPHABET = "0123456789abcdef"


class IdGenerator:
    """Generates short opaque ids, deterministically when seeded.

    Tests fix the seed so that whole snapshots — which are hashes of content that
    includes ids — come out byte-identical across runs.
    """

    def __init__(self, seed: int | None = None, *, stream: str = "", width: int = 10) -> None:
        """Create a generator.

        Args:
            seed: When given, ids are a deterministic function of the seed and the
                number of ids drawn so far. When ``None``, ids are unpredictable.
            stream: A label mixed into the seed. Two generators created with the same
                seed produce the *same* id sequence, so handing one to the test fixture
                and another to the proposer makes them collide on the first id. Give
                independent streams different labels, or better, share one generator.
            width: Number of hex characters after the type prefix.
        """
        self._rng = random.Random(f"{seed}:{stream}" if seed is not None else None)
        self._width = width

    def _raw(self) -> str:
        return "".join(self._rng.choice(_ALPHABET) for _ in range(self._width))

    def node(self) -> NodeId:
        """Return a fresh plan-node id."""
        return NodeId(f"n_{self._raw()}")

    def entity(self) -> EntityId:
        """Return a fresh entity id."""
        return EntityId(f"e_{self._raw()}")

    def fact(self) -> FactId:
        """Return a fresh fact id."""
        return FactId(f"f_{self._raw()}")

    def thread(self) -> ThreadId:
        """Return a fresh open-thread id."""
        return ThreadId(f"t_{self._raw()}")

    def proposal(self) -> ProposalId:
        """Return a fresh proposal id."""
        return ProposalId(f"p_{self._raw()}")
