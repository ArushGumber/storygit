"""The dependency graph: which beats depend on which facts, and on each other.

Entirely mechanical. A beat declares the facts it establishes (``produces``) and
the facts it relies on (``consumes``); an edge from beat A to beat B exists exactly
when B consumes something A produced. No language model is asked whether two scenes
are related, because a model that is wrong 10% of the time about that would make
every propagation result untrustworthy.
"""

from __future__ import annotations

from typing import Protocol

from storygit.domain.ids import FactId, NodeId
from storygit.domain.state import StoryState


class EdgeProvider(Protocol):
    """Extra, non-declared dependency edges.

    The seam for the soft-edge ablation (chunk 3), where beat-to-beat embedding
    similarity above a threshold produces a weaker "maybe affected" signal. Keeping
    it behind a protocol means the deterministic core never grows a model
    dependency.
    """

    def extra_dependents(self, state: StoryState, node_id: NodeId) -> tuple[NodeId, ...]:
        """Nodes that may be affected by a change to ``node_id``."""
        ...


def producer_of(state: StoryState, fact_id: FactId) -> NodeId | None:
    """The beat that established a fact, if it is still in the tree."""
    return state.producer_of.get(fact_id)


def consumers_of(state: StoryState, fact_id: FactId) -> tuple[NodeId, ...]:
    """Beats that declare they rely on a fact."""
    return state.consumers_of.get(fact_id, ())


def dependents_of_facts(
    state: StoryState,
    fact_ids: set[FactId],
    *,
    stop_at: frozenset[NodeId] = frozenset(),
) -> dict[NodeId, FactId]:
    """Transitive closure of beats affected by a set of facts.

    Walks fact → consuming beats → the facts those beats produce → their consumers,
    and so on. Traversal stops at ``stop_at`` nodes: a locked beat is not going to
    change, so nothing it produces can change either.

    Args:
        state: The state to walk.
        fact_ids: Facts that changed or were removed.
        stop_at: Nodes to treat as immovable (in practice, the writer's locks).

    Returns:
        Affected beat id to the fact that first implicated it, so the flag can cite
        a specific cause rather than "something upstream changed".
    """
    found: dict[NodeId, FactId] = {}
    frontier: list[FactId] = sorted(fact_ids)
    seen_facts: set[FactId] = set(frontier)
    while frontier:
        fact_id = frontier.pop(0)
        for beat_id in consumers_of(state, fact_id):
            if beat_id in stop_at or beat_id in found:
                continue
            found[beat_id] = fact_id
            beat = state.nodes.get(beat_id)
            produced = getattr(beat, "produces", ())
            for downstream in produced:
                if downstream not in seen_facts:
                    seen_facts.add(downstream)
                    frontier.append(downstream)
    return found


def hard_constraints(state: StoryState) -> list[str]:
    """Everything generation must not contradict, as prompt-ready lines.

    Combines the writer's explicit constraints with every fact established by a
    locked node — the operational meaning of a lock.

    Args:
        state: The state to read.

    Returns:
        Sentences for the prompt builder.
    """
    names = state.entity_names()
    lines = list(state.ledger.hard_constraints)
    for node_id in sorted(state.ledger.locks):
        node = state.nodes.get(node_id)
        if node is None:
            continue
        label = node.title or node.node_type.value
        if node.what_happens:
            lines.append(f"Locked — {label}: {node.what_happens}")
        for fact_id in getattr(node, "produces", ()):
            fact = state.facts.get(fact_id)
            if fact is not None:
                lines.append(f"Locked fact — {fact.sentence(names)}")
    return lines
