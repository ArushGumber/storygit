"""Serving what the evaluation produced: the Gallery sessions and the Eval figures.

Both tabs are pure reads off disk. Nothing here regenerates anything — the Gallery replays
recorded snapshots and the Eval tab shows committed SVGs, so opening the interface never
spends quota and never depends on a model being reachable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from storygit.api.deps import AppState, app_state
from storygit.gallery import Session, replay
from storygit.store.repository import Repository

router = APIRouter(prefix="/api", tags=["artifacts"])

State = Annotated[AppState, Depends(app_state)]


def _gallery_dir(state: AppState) -> Path:
    return state.results_dir / "gallery"


@router.get("/gallery")
def list_sessions(state: State) -> dict[str, Any]:
    """Every recorded session, with its title and what it demonstrates."""
    index = _gallery_dir(state) / "index.json"
    if index.exists():
        return {"sessions": json.loads(index.read_text())}
    sessions = []
    for path in sorted(_gallery_dir(state).glob("*.json")):
        if path.name == "index.json":
            continue
        data = json.loads(path.read_text())
        sessions.append(
            {
                "name": data.get("name", path.stem),
                "title": data.get("title", ""),
                "summary": data.get("summary", ""),
                "steps": len(data.get("steps", [])),
            }
        )
    return {"sessions": sessions}


@router.get("/gallery/{name}")
def session(state: State, name: str, resolve: bool = True) -> dict[str, Any]:
    """One recorded session, optionally with each step's state reconstructed.

    Args:
        state: Application state.
        name: Session name.
        resolve: Replay each step against the snapshots it names, so the client gets the
            plan tree and world state as they stood. This is what makes the Gallery a
            record rather than a slideshow.

    Returns:
        The session, and the replayed states when resolved.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    path = _gallery_dir(state) / f"{Path(name).name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no gallery session {name!r}")

    loaded = Session.load(path)
    body: dict[str, Any] = {"session": loaded.model_dump(mode="json")}
    if resolve and loaded.db_path and Path(loaded.db_path).exists():
        repo = Repository.open(loaded.db_path)
        try:
            body["replay"] = [r.model_dump(mode="json") for r in replay(loaded, repo)]
        finally:
            repo.close()
    return body


@router.get("/eval/summary")
def eval_summary(state: State) -> dict[str, Any]:
    """The evaluation summary, or a clear statement that it has not been run.

    Raises:
        HTTPException: never — an unrun evaluation is a state the interface must render,
            not an error.
    """
    path = state.results_dir / "summary.json"
    if not path.exists():
        return {"available": False, "detail": "run `python -m eval.run` to generate this"}
    return {"available": True, **json.loads(path.read_text())}


@router.get("/eval/plots")
def eval_plots(state: State) -> dict[str, Any]:
    """The figures the evaluation produced, with a caption for each.

    Captions live here rather than in the frontend because they describe what a curve
    *demonstrates*, which is a property of the experiment rather than of the page.
    """
    captions = {
        "checker_recall.svg": (
            "What each checker layer contributes. Layer 1 is deterministic and free; "
            "layer 2 handles only the free-text residue it cannot decide."
        ),
        "stale_precision_recall.svg": (
            "Staleness prediction. Declared dependency edges are exact but blind to what "
            "nobody declared; embedding edges trade precision for recall, swept by "
            "threshold."
        ),
        "diversity_quality.svg": (
            "Diversity against quality on six candidates whose three best are "
            "near-paraphrases. The temperature-only baseline takes all three."
        ),
        "mmr_lambda_sweep.svg": (
            "Where MMR's diversity term stops binding. At the conventional 0.7 it "
            "reproduces the baseline exactly; the DPP reaches the diverse answer with no "
            "parameter at all."
        ),
        "bandit_regret.svg": (
            "Cumulative pseudo-regret. Thompson sampling pays its exploration cost early "
            "and then flattens; epsilon-greedy keeps paying forever."
        ),
        "pretraining_lift.svg": (
            "Whether the population prior earns its place on a writer it never saw. It "
            "helps while data is scarce and converges to irrelevance, which is the shape a "
            "prior should have."
        ),
        "acceptance.svg": (
            "Rolling acceptance rate per simulated writer. Read beside the edit-distance "
            "curve: accepting more while rewriting just as much is not improvement."
        ),
        "edit_distance.svg": ("How much the writer rewrote what they took, over the run."),
        "bandit_selftest.svg": (
            "The sampler in isolation, against arms whose true rates are known. This is "
            "the only setting where \u201cis the regret sublinear\u201d has an unambiguous "
            "answer; the numbers that matter come from the simulated-writer runs."
        ),
    }
    # A figure with no caption is a figure whose filename gets shown to a reader, which
    # is why test_every_figure_has_a_caption exists rather than a fallback string.
    available = []
    for path in sorted(state.results_dir.glob("*.svg")):
        available.append(
            {
                "name": path.name,
                "url": f"/api/eval/plot/{path.name}",
                "caption": captions.get(path.name, ""),
            }
        )
    return {"plots": available}


@router.get("/eval/plot/{name}")
def eval_plot(state: State, name: str) -> FileResponse:
    """Serve one figure.

    Raises:
        HTTPException: 404 if the figure does not exist.
    """
    safe = Path(name).name
    path = state.results_dir / safe
    if path.suffix != ".svg" or not path.exists():
        raise HTTPException(status_code=404, detail=f"no plot {name!r}")
    return FileResponse(path, media_type="image/svg+xml")
