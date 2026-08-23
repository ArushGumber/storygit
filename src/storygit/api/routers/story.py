"""Reading the story: the plan tree, a node, the world-state slice, threads, flags.

Every route here is a pure read. Nothing in this module can change the story, which is
worth knowing when reasoning about what a GET can do.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from storygit.api import schemas
from storygit.api.deps import AppState, app_state
from storygit.continuity import layer1
from storygit.domain.ids import EntityId, NodeId
from storygit.graph.slices import entities_in_scope

router = APIRouter(prefix="/api", tags=["story"])

State = Annotated[AppState, Depends(app_state)]


@router.get("/tree", response_model=schemas.TreeResponse)
def tree(state: State, branch: str | None = None) -> schemas.TreeResponse:
    """The plan tree with statuses, locks, stale reasons, and flag counts.

    Args:
        state: Application state.
        branch: Branch to read; the current one when omitted.

    Returns:
        Every node, flat, in reading order.
    """
    name = branch or state.branch
    story = state.repo.state(name)
    flags = tuple(layer1.check_state(story))
    return schemas.tree_response(story, name, flags=flags)


@router.get("/node/{node_id}", response_model=schemas.NodeDetail)
def node(state: State, node_id: str, branch: str | None = None) -> schemas.NodeDetail:
    """One node with its prose, provenance, authorship ratio, and flags.

    Args:
        state: Application state.
        node_id: The node.
        branch: Branch to read.

    Returns:
        The node detail.

    Raises:
        HTTPException: 404 if the node does not exist.
    """
    story = state.repo.state(branch or state.branch)
    target = story.nodes.get(NodeId(node_id))
    if target is None:
        raise HTTPException(status_code=404, detail=f"no node {node_id}")
    return schemas.node_detail(story, target, flags=tuple(layer1.check_state(story)))


@router.get("/slice", response_model=schemas.SliceResponse)
def world_slice(
    state: State,
    entities: Annotated[str, Query()] = "",
    beat: Annotated[str, Query()] = "",
    node: Annotated[str, Query()] = "",
    branch: str | None = None,
) -> schemas.SliceResponse:
    """The world-state pane: entities, their facts valid here, who knows what, threads.

    Args:
        state: Application state.
        entities: Comma-separated entity ids. Empty means everything in scope.
        beat: The beat to evaluate fact validity at.
        node: Scope the entities to the ones this node is about. This is what the
            interface actually uses — the writer selects a beat, not a set of entities.
        branch: Branch to read.

    Returns:
        The slice.
    """
    story = state.repo.state(branch or state.branch)
    scoped: set[EntityId] = {EntityId(e) for e in entities.split(",") if e.strip()}
    if not scoped and node:
        scoped = entities_in_scope(story, NodeId(node))
    at_beat = NodeId(beat) if beat else (NodeId(node) if node in story.seq else None)
    return schemas.slice_response(story, scoped, at_beat)


@router.get("/threads")
def threads(state: State, branch: str | None = None) -> dict[str, object]:
    """The open-thread ledger, with how long each has gone untouched."""
    story = state.repo.state(branch or state.branch)
    beats = story.beats_in_order()
    latest = story.seq.get(beats[-1].id, 0) if beats else 0
    return {
        "threads": [
            {
                "thread": thread.model_dump(mode="json"),
                "beats_since_touched": latest - story.seq.get(thread.last_touched_beat, 0),
                "opened_in": (
                    (story.nodes[thread.opened_at_beat].title or "an earlier beat")
                    if thread.opened_at_beat in story.nodes
                    else "a removed beat"
                ),
            }
            for thread in sorted(story.threads.values(), key=lambda t: t.id)
        ]
    }


@router.get("/flags")
def flags(state: State, branch: str | None = None, audit: bool = False) -> dict[str, object]:
    """Continuity flags on the current state.

    Args:
        state: Application state.
        branch: Branch to read.
        audit: Run the full audit (layers 1 and 2 over the whole graph, plus dropped
            threads) rather than the fast deterministic pass. Slower; the interface offers
            it as an explicit action.

    Returns:
        Flags, hard first.
    """
    name = branch or state.branch
    story = state.repo.state(name)
    if audit:
        # Through the engine rather than calling the audit directly: the engine knows
        # whether layer 2 is available in this deployment, and an audit that hard-coded
        # it would raise on a machine with no NLI model rather than degrading to layer 1.
        report = state.engine(name).audit()
        return {
            "flags": [f.model_dump(mode="json") for f in report.flags],
            "summary": report.summary(),
            "by_layer": report.by_layer,
        }
    found = layer1.check_state(story)
    return {"flags": [f.model_dump(mode="json") for f in found], "summary": ""}


@router.get("/ledger", response_model=schemas.LedgerResponse)
def ledger(state: State, branch: str | None = None) -> schemas.LedgerResponse:
    """The writer's standing instructions, plus what the preference layer has learned."""
    name = branch or state.branch
    story = state.repo.state(name)
    return schemas.ledger_response(story, state.engine(name).preference.state.summary())


@router.get("/authorship", response_model=schemas.AuthorshipResponse)
def authorship(state: State, branch: str | None = None) -> schemas.AuthorshipResponse:
    """How much of the story each source wrote, overall and per beat."""
    return schemas.authorship_response(state.repo.state(branch or state.branch))


@router.get("/history")
def history(state: State, branch: str | None = None, limit: int = 50) -> dict[str, object]:
    """The snapshot chain: what changed, when, and who authored it."""
    return {"history": state.repo.history(branch or state.branch, limit=limit)}
