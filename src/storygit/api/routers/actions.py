"""Everything that changes the story.

Every route here goes through the engine, and the engine goes through
``Repository.commit_diff``. There is no other path — a POST cannot mutate state any other
way, which is what makes the snapshot history a complete record rather than a partial one.

Long calls (proposing) run in a thread pool rather than on the event loop. Generation takes
seconds and involves local embedding work that would otherwise block every other request.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from storygit.agents.schemas import Level
from storygit.api import schemas
from storygit.api.deps import AppState, app_state
from storygit.continuity import bible_diff
from storygit.domain.ids import EntityId, FactId, NodeId, ProposalId
from storygit.domain.nodes import NodeStatus

router = APIRouter(prefix="/api", tags=["actions"])

State = Annotated[AppState, Depends(app_state)]


@router.post("/propose", response_model=schemas.ProposeResponse)
async def propose(state: State, request: schemas.ProposeRequest) -> schemas.ProposeResponse:
    """Generate labelled, checked, ranked candidates at a node.

    The slow route: six samples plus six judge calls, concurrently, so it costs roughly one
    round trip rather than twelve. Still seconds, and the interface shows progress.

    Args:
        state: Application state.
        request: Node, level, and the writer's intent.

    Returns:
        Candidates, shortlisted first.

    Raises:
        HTTPException: 422 if the level is not a plan level.
    """
    try:
        level = Level(request.level)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"unknown level {request.level!r}") from exc

    engine = state.engine()
    candidates = await engine.propose_at(
        level,
        node_id=NodeId(request.node_id) if request.node_id else None,
        intent=request.intent,
    )
    return schemas.ProposeResponse(
        candidates=tuple(schemas.candidate_view(c) for c in candidates),
        shown=engine.shown(),
    )


@router.post("/action/accept", response_model=schemas.ActionResponse)
async def accept(state: State, request: schemas.ActionRequest) -> schemas.ActionResponse:
    """Commit a proposal, extract facts from any new prose, and propagate.

    Args:
        state: Application state.
        request: The proposal to accept.

    Returns:
        The bible diff, the stale marks, and the resulting flags.

    Raises:
        HTTPException: 404 if the proposal is not one this engine is holding.
    """
    engine = state.engine()
    try:
        result = await engine.accept(ProposalId(request.proposal_id), shown_with=engine.shown())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return schemas.action_response(result)


@router.post("/action/reject")
def reject(state: State, request: schemas.ActionRequest) -> dict[str, Any]:
    """Reject a proposal, recording the direction so it is not proposed again."""
    engine = state.engine()
    try:
        engine.reject(ProposalId(request.proposal_id), reason=request.reason)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "rejected": request.proposal_id}


@router.post("/action/edit", response_model=schemas.ActionResponse)
async def edit(state: State, request: schemas.EditRequest) -> schemas.ActionResponse:
    """Accept a proposal with the writer's own words, recording the before/after pair."""
    engine = state.engine()
    try:
        result = await engine.edit(ProposalId(request.proposal_id), request.text)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return schemas.action_response(result)


class WriteRequest(BaseModel):
    """The hand-written path: the writer writes, the system only reads."""

    node_id: NodeId
    text: str


@router.post("/action/write", response_model=schemas.ActionResponse)
async def write(state: State, request: WriteRequest) -> schemas.ActionResponse:
    """Write a beat by hand. No generation; extraction and checking run as normal."""
    result = await state.engine().write_beat(NodeId(request.node_id), request.text)
    return schemas.action_response(result)


@router.post("/node/{node_id}/lock")
def lock(state: State, node_id: str) -> dict[str, Any]:
    """Freeze a node. Propagation skips it and its facts become hard constraints."""
    snapshot = state.engine().lock(NodeId(node_id))
    return {"ok": True, "snapshot_id": str(snapshot)}


@router.post("/node/{node_id}/unlock")
def unlock(state: State, node_id: str) -> dict[str, Any]:
    """Unfreeze a node."""
    snapshot = state.engine().unlock(NodeId(node_id))
    return {"ok": True, "snapshot_id": str(snapshot)}


@router.post("/node/{node_id}/dismiss-stale")
def dismiss_stale(state: State, node_id: str) -> dict[str, Any]:
    """The writer's "it still works": clear a stale mark without regenerating."""
    snapshot = state.engine().dismiss_stale(NodeId(node_id))
    return {"ok": True, "snapshot_id": str(snapshot)}


@router.post("/node/{node_id}/regenerate", response_model=schemas.ProposeResponse)
async def regenerate(
    state: State, node_id: str, subtree: bool = False, intent: str = ""
) -> schemas.ProposeResponse:
    """Propose replacements for a stale node, or for a whole stale subtree.

    Returns candidates rather than applying anything, even for a subtree. Regenerating is
    the single most destructive action available, and it goes through exactly the same
    review as everything else: previewed as diffs, accepted one at a time.

    Args:
        state: Application state.
        node_id: The stale node.
        subtree: Include every stale node beneath it.
        intent: Optional extra instruction.

    Returns:
        Candidates for the node.

    Raises:
        HTTPException: 404 if the node does not exist.
    """
    engine = state.engine()
    story = engine.state()
    target = story.nodes.get(NodeId(node_id))
    if target is None:
        raise HTTPException(status_code=404, detail=f"no node {node_id}")

    reason = target.stale_reason or "this node's dependencies changed"
    stale_below = [
        n.id for n in story.descendants_of(NodeId(node_id)) if n.status is NodeStatus.stale
    ]
    prompt = f"This needs revisiting: {reason}"
    if subtree and stale_below:
        prompt += f" There are {len(stale_below)} stale nodes beneath it."
    if intent:
        prompt = f"{intent}\n\n{prompt}"

    candidates = await engine.propose_at(
        Level(target.node_type.value),
        node_id=target.parent_id,
        intent=prompt,
    )
    return schemas.ProposeResponse(
        candidates=tuple(schemas.candidate_view(c) for c in candidates),
        shown=engine.shown(),
    )


@router.post("/fact/{fact_id}/strike", response_model=schemas.ActionResponse)
def strike_fact(state: State, fact_id: str, at_beat: str = "") -> schemas.ActionResponse:
    """Strike a fact from the bible, propagating and re-checking what depended on it.

    Args:
        state: Application state.
        fact_id: The fact.
        at_beat: End the fact here rather than deleting it — "that stopped being true"
            rather than "that was never true". Ending preserves the history earlier beats
            depend on.

    Returns:
        The resulting marks and flags.

    Raises:
        HTTPException: 404 if the fact does not exist.
    """
    engine = state.engine()
    before = engine.state()
    try:
        snapshot, flags, marks = engine.strike_fact(
            FactId(fact_id), at_beat=NodeId(at_beat) if at_beat else None
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    diff = bible_diff.compute(before, engine.state())
    counts = diff.counts()
    return schemas.ActionResponse(
        snapshot_id=str(snapshot),
        bible_diff=diff.lines,
        added=counts["added"],
        ended=counts["ended"],
        removed=counts["removed"],
        marks=tuple(schemas.mark_dict(m) for m in marks),
        flags=tuple(flags),
    )


class DialRequest(BaseModel):
    """Move the coherent-to-surprising dial."""

    value: float


class NoteRequest(BaseModel):
    """Name a style note by its text, to add or to remove it."""

    text: str


class CriterionRequest(BaseModel):
    """Add a writer-defined scoring axis."""

    name: str
    description: str
    # Bounded here as well as in the domain model, so an out-of-range weight is a 422
    # naming the field rather than a bare 500 with no body. A writer reaching for 1.5 to
    # mean "this matters more" got the latter, and only noticed because the criterion was
    # missing from the ledger afterwards.
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class NameRequest(BaseModel):
    """A request that names one thing in the ledger."""

    name: str


class MergeRequest(BaseModel):
    """Fold `source_id` into `target_id`, keeping the source's names as aliases."""

    source_id: str
    target_id: str


@router.post("/ledger/dial")
def set_dial(state: State, request: DialRequest) -> dict[str, Any]:
    """Move the dial."""
    snapshot = state.engine().set_dial(max(0.0, min(1.0, request.value)))
    return {"ok": True, "snapshot_id": str(snapshot)}


@router.post("/ledger/style-note")
def add_style_note(state: State, request: NoteRequest) -> dict[str, Any]:
    """Add a style note."""
    snapshot = state.engine().add_style_note(request.text)
    return {"ok": True, "snapshot_id": str(snapshot)}


@router.post("/ledger/criterion")
def add_criterion(state: State, request: CriterionRequest) -> dict[str, Any]:
    """Add a writer-defined scoring axis, which the judge then scores every candidate on."""
    snapshot = state.engine().add_criterion(
        request.name, request.description, weight=request.weight
    )
    return {"ok": True, "snapshot_id": str(snapshot)}


@router.post("/ledger/hard-constraint")
def add_hard_constraint(state: State, request: NoteRequest) -> dict[str, Any]:
    """Add a rule generation must never break, as opposed to a style preference."""
    snapshot = state.engine().add_hard_constraint(request.text)
    return {"ok": True, "snapshot_id": str(snapshot)}


@router.post("/entity/merge")
def merge_entities(state: State, request: MergeRequest) -> dict[str, Any]:
    """Fold one entity into another after the resolver kept a duplicate apart.

    Conservative alias resolution will sometimes create a second entity for a name it
    could not match exactly. This is the one-click correction the resolver's caution
    assumes exists.
    """
    snapshot = state.engine().merge_entities(
        EntityId(request.source_id), EntityId(request.target_id)
    )
    return {"ok": True, "snapshot_id": str(snapshot)}


@router.post("/ledger/style-note/remove")
def remove_style_note(state: State, request: NoteRequest) -> dict[str, Any]:
    """Delete a style rule, including one mined from the writer's own edits.

    A `remove` rather than a `DELETE` to match the rest of this router, which is a list of
    writer actions rather than a resource collection.
    """
    snapshot = state.engine().remove_style_note(request.text)
    return {"ok": True, "snapshot_id": str(snapshot)}


@router.post("/ledger/criterion/remove")
def remove_criterion(state: State, request: NameRequest) -> dict[str, Any]:
    """Delete a writer-defined scoring axis; the judge stops scoring candidates on it."""
    snapshot = state.engine().remove_criterion(request.name)
    return {"ok": True, "snapshot_id": str(snapshot)}


@router.post("/ledger/mine-edits")
async def mine_edits(state: State) -> dict[str, Any]:
    """Read the writer's edits into style rules and put them in the ledger.

    An explicit action rather than something that happens on every accept, because it costs
    a model call per edit pair and the writer should not wait on a network round trip to see
    the result of clicking accept.
    """
    snapshot = await state.engine().mine_edits()
    ledger = state.engine().state().ledger
    return {
        "ok": True,
        "snapshot_id": str(snapshot) if snapshot else None,
        "style_notes": [n.model_dump(mode="json") for n in ledger.style_notes],
    }
