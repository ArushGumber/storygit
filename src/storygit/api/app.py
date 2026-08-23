"""The FastAPI application: a thin adapter, and nothing else.

No business logic lives in this layer. Routes translate JSON into engine calls and engine
results back into JSON, and every failure becomes the same problem shape. That constraint
is what makes chunks 1-5 testable without any web machinery at all, and it is why the whole
HTTP layer is a few hundred lines.

Run it:

    uvicorn storygit.api.app:app --reload            # dev, with a Vite proxy in front
    STORYGIT_DB=story.db python -m storygit.api.app  # prod, serving the built frontend
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from storygit.api.deps import AppState, build_state, seed_if_empty
from storygit.api.errors import problem, status_for
from storygit.api.routers import actions, artifacts, branches, story
from storygit.domain.errors import StoryGitError
from storygit.providers.base import ProviderError

DEV_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)
"""Vite's dev server. In production the frontend is served from this same origin, so CORS
does not apply at all — it exists only for the two-process development setup."""


def create_app(
    *,
    state: AppState | None = None,
    db_path: str | Path | None = None,
    frontend_dir: str | Path | None = None,
    results_dir: str | Path = "eval/results",
    seed: int | None = None,
) -> FastAPI:
    """Build the application.

    Args:
        state: Pre-built state. Tests inject one carrying a mock provider router.
        db_path: Story database, when building state here.
        frontend_dir: Built frontend to serve. When present, unknown paths fall through to
            its ``index.html`` so client-side routing works on a hard refresh.
        results_dir: Where the evaluation results live.
        seed: Id-generator seed, for reproducible tests.

    Returns:
        The application.
    """
    resolved_state = state or build_state(
        db_path or os.environ.get("STORYGIT_DB", "story.db"),
        results_dir=results_dir,
        seed=seed,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Open the repository and provider clients once, and close them once."""
        application.state.storygit = resolved_state
        seed_if_empty(resolved_state)
        try:
            yield
        finally:
            await resolved_state.router.aclose()
            resolved_state.close()

    application = FastAPI(
        title="storygit",
        version="0.1.0",
        summary="Version control for story state.",
        lifespan=lifespan,
    )

    from fastapi.middleware.cors import CORSMiddleware

    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEV_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(StoryGitError)
    async def _domain_error(request: Request, exc: StoryGitError) -> JSONResponse:
        """Domain failures become the shared problem shape."""
        redact = getattr(request.app.state.storygit.settings, "redact", None)
        return JSONResponse(status_code=status_for(exc), content=problem(exc, redact=redact))

    @application.exception_handler(ProviderError)
    async def _provider_error(request: Request, exc: ProviderError) -> JSONResponse:
        """Provider failures too, carrying ``retry_after`` when there is one."""
        redact = getattr(request.app.state.storygit.settings, "redact", None)
        return JSONResponse(status_code=status_for(exc), content=problem(exc, redact=redact))

    application.include_router(story.router)
    application.include_router(actions.router)
    application.include_router(branches.router)
    application.include_router(artifacts.router)

    @application.get("/api/health")
    def health() -> dict[str, Any]:
        """Liveness, and enough state for the interface to render its header."""
        current = resolved_state.repo.state(resolved_state.branch)
        return {
            "ok": True,
            "branch": resolved_state.branch,
            "nodes": len(current.nodes),
            "facts": len(current.facts),
            "providers": sorted(resolved_state.router.providers),
            "openrouter_enabled": resolved_state.settings.openrouter_is_enabled,
        }

    _mount_frontend(application, frontend_dir)
    return application


def _mount_frontend(application: FastAPI, frontend_dir: str | Path | None) -> None:
    """Serve the built frontend, if there is one.

    One process, one port, no separate static server. Unknown paths fall through to
    ``index.html`` because the frontend routes client-side, and a hard refresh on
    ``/gallery`` must not 404.
    """
    if frontend_dir is None:
        frontend_dir = os.environ.get("STORYGIT_FRONTEND", "frontend/dist")
    root = Path(frontend_dir)
    if not (root / "index.html").exists():
        return

    application.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")

    @application.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        """Serve a static file from the build directory, otherwise the app shell.

        The resolved path is checked against the build root before anything is read.
        ``root / path`` on an unsanitised URL segment resolves ``../`` happily, so this
        route would otherwise serve any file the process can read to anyone who can reach
        the port --- including the workspace ``.env`` two directories up. The sibling route
        that serves figures got this right; this one, the only hand-rolled static handler,
        was also the only route no test mounted.
        """
        base = root.resolve()
        candidate = (base / path).resolve()
        if path and candidate.is_file() and candidate.is_relative_to(base):
            return FileResponse(candidate)
        return FileResponse(base / "index.html")


app = create_app()
"""The module-level application, for ``uvicorn storygit.api.app:app``."""


def main() -> None:
    """Run the server directly, serving the built frontend from one port."""
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("STORYGIT_HOST", "127.0.0.1"),
        port=int(os.environ.get("STORYGIT_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
