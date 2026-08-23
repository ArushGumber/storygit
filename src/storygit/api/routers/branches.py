"""Branching: explore, compare, and merge — with conflicts surfaced, never guessed."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from storygit.api import schemas
from storygit.api.deps import AppState, app_state
from storygit.domain.diff import delta_summary
from storygit.domain.errors import SnapshotNotFoundError

router = APIRouter(prefix="/api", tags=["branches"])

State = Annotated[AppState, Depends(app_state)]


class BranchRequest(BaseModel):
    """Create a branch."""

    name: str
    from_branch: str = ""


class SwitchRequest(BaseModel):
    """Switch which branch is being edited."""

    name: str


class MergeRequest(BaseModel):
    """Merge one branch into another."""

    ours: str
    theirs: str
    commit: bool = False


@router.get("/branches", response_model=schemas.BranchesResponse)
def list_branches(state: State) -> schemas.BranchesResponse:
    """Every branch, its head, and which one is current."""
    return schemas.BranchesResponse(
        current=state.branch,
        branches={name: str(head) for name, head in state.repo.list_branches().items()},
    )


@router.post("/branch")
def create_branch(state: State, request: BranchRequest) -> dict[str, Any]:
    """Branch at a snapshot. A branch is a pointer, so exploring costs nothing.

    Raises:
        HTTPException: 409 if the name is taken.
    """
    try:
        snapshot = state.repo.create_branch(
            request.name, from_branch=request.from_branch or state.branch
        )
    except SnapshotNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True, "branch": request.name, "snapshot_id": str(snapshot)}


@router.post("/branch/switch")
def switch_branch(state: State, request: SwitchRequest) -> dict[str, Any]:
    """Switch the branch being edited.

    Raises:
        HTTPException: 404 if the branch does not exist.
    """
    if not state.repo.branches.exists(request.name):
        raise HTTPException(status_code=404, detail=f"no branch {request.name!r}")
    state.branch = request.name
    return {"ok": True, "current": state.branch}


@router.get("/branch/diff")
def diff_branches(state: State, a: str, b: str) -> dict[str, Any]:
    """What differs between two branches, in English.

    Raises:
        HTTPException: 404 if either branch does not exist.
    """
    try:
        difference = state.repo.diff_branches(a, b)
    except SnapshotNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "a": a,
        "b": b,
        "op_count": len(difference),
        "summary": delta_summary(difference, state.repo.state(a)),
    }


@router.post("/branch/merge", response_model=schemas.MergeResponse)
def merge_branches(state: State, request: MergeRequest) -> schemas.MergeResponse:
    """Three-way merge, returning conflicts rather than resolving them.

    Args:
        state: Application state.
        request: Which branches, and whether to commit a clean result.

    Returns:
        The preview, and the snapshot id when a clean merge was committed.

    Raises:
        HTTPException: 404 if a branch does not exist or they share no history.
    """
    try:
        result = state.repo.merge_branches(request.ours, request.theirs)
    except SnapshotNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    committed: str | None = None
    if result.clean and request.commit and len(result.diff):
        committed = str(state.repo.commit_diff(result.diff, branch=request.ours))

    return schemas.MergeResponse(
        clean=result.clean,
        conflicts=tuple(c.model_dump(mode="json") for c in result.conflicts),
        summary=tuple(delta_summary(result.diff, state.repo.state(request.ours))),
        committed=committed,
    )
