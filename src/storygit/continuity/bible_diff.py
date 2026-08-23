"""The bible diff: what an accepted change did to the world, in the writer's terms.

Shown on every accept, because "what did I just agree to" deserves an answer that is
about the story rather than about JSON. Three kinds of line:

    + Kael keeps a secret: he can hear the ash.
    ~ Kael is at Ashfall. (no longer true from here)
    - Kael has the Warden's token.

The writer can **strike** any of them. Striking is not an undo — it produces a normal diff
that removes or ends the fact, which then propagates like any other change, and layer 1
re-runs on whatever depended on it. Even correcting the machine goes through the same
reviewable path as everything else.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from storygit.continuity.flags import Flag
from storygit.domain.diff import Diff, DiffAuthor, InvalidateFact, RemoveFact
from storygit.domain.ids import FactId, NodeId
from storygit.domain.state import StoryState
from storygit.domain.world import Fact


class BibleDiff(BaseModel):
    """Facts added, ended, and struck by a change.

    Attributes:
        added: Newly asserted facts.
        ended: Facts whose validity this change closed off.
        removed: Facts struck outright.
        lines: The same information as writer-readable sentences.
    """

    model_config = ConfigDict(frozen=True)

    added: tuple[Fact, ...] = ()
    ended: tuple[Fact, ...] = ()
    removed: tuple[Fact, ...] = ()
    lines: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Whether the change touched no facts at all."""
        return not (self.added or self.ended or self.removed)

    def counts(self) -> dict[str, int]:
        """``{added, ended, removed}`` counts, for the interface's summary line."""
        return {
            "added": len(self.added),
            "ended": len(self.ended),
            "removed": len(self.removed),
        }


def compute(before: StoryState, after: StoryState) -> BibleDiff:
    """Compare two states' world graphs.

    Args:
        before: State before the change.
        after: State after it.

    Returns:
        The bible diff.
    """
    names = after.entity_names() or before.entity_names()
    added = tuple(after.facts[f] for f in sorted(after.facts.keys() - before.facts.keys()))
    removed = tuple(before.facts[f] for f in sorted(before.facts.keys() - after.facts.keys()))
    ended = tuple(
        after.facts[f]
        for f in sorted(before.facts.keys() & after.facts.keys())
        if before.facts[f].valid_until_beat is None and after.facts[f].valid_until_beat is not None
    )
    lines = [
        *(f"+ {fact.sentence(names)}" for fact in added),
        *(f"~ {fact.sentence(names)} (no longer true from here)" for fact in ended),
        *(f"- {fact.sentence(names)}" for fact in removed),
    ]
    return BibleDiff(added=added, ended=ended, removed=removed, lines=tuple(lines))


def strike(
    state: StoryState,
    fact_id: FactId,
    *,
    at_beat: NodeId | None = None,
) -> Diff:
    """Build the diff for the writer striking a fact.

    Args:
        state: Current state.
        fact_id: The fact to strike.
        at_beat: When given, the fact is *ended* here rather than deleted — which is what
            the writer means by "that stopped being true", as opposed to "that was never
            true". Ending preserves the history that earlier beats depend on; deleting
            does not, and propagation will mark those beats.

    Returns:
        A human-authored diff.

    Raises:
        KeyError: If the fact does not exist.
    """
    fact = state.facts.get(fact_id)
    if fact is None:
        raise KeyError(f"no fact {fact_id}")
    names = state.entity_names()
    if at_beat is not None:
        return Diff(
            ops=(InvalidateFact(fact_id=fact_id, valid_until_beat=at_beat),),
            author=DiffAuthor.human,
            intent=f"end: {fact.sentence(names)}",
        )
    return Diff(
        ops=(RemoveFact(fact_id=fact_id),),
        author=DiffAuthor.human,
        intent=f"strike: {fact.sentence(names)}",
    )


def recheck_after_strike(state: StoryState, fact_id: FactId) -> list[Flag]:
    """Re-run layer 1 on the beats that depended on a struck fact.

    Striking a fact can *create* a contradiction as easily as remove one — a beat that
    relied on it may now rely on nothing, or on a different fact that conflicts. Checking
    only the affected beats keeps this instant.

    Args:
        state: The state after the strike.
        fact_id: The fact that was struck.

    Returns:
        Hard flags on the affected beats.
    """
    from storygit.continuity.layer1 import check_state

    affected = tuple(sorted(state.consumers_of.get(fact_id, ())))
    if not affected:
        return []
    return check_state(state, beats=affected)
