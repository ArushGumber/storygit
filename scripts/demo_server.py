#!/usr/bin/env python
"""Boot the API against a scratch copy of the demo story, for the writer's session.

The writer subagent drives the HTTP API — the same actions the interface emits — so this
is the production app with a real provider behind it, not a harness. It works on a *copy*
so a session can be replayed or abandoned without touching the committed demo state.

    STORYGIT_DEMO_SRC=demo/story.db scripts/demo_server.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import uvicorn

from storygit.api.app import create_app
from storygit.api.deps import build_state
from storygit.domain.ids import IdGenerator
from storygit.providers.router import build_router
from storygit.seed import seed_story
from storygit.store.repository import Repository


def main() -> None:
    """Serve the demo database, copying from a source if one is named."""
    target = Path(os.environ.get("STORYGIT_DB", "demo/story.db"))
    source = os.environ.get("STORYGIT_DEMO_SRC", "")
    if source and Path(source).exists() and Path(source) != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        repo = Repository.open(target)
        seed_story(repo, IdGenerator(seed=7, stream="demo"))
        repo.close()

    state = build_state(
        str(target),
        router=build_router(),
        results_dir=Path("eval/results"),
        seed=7,
    )
    uvicorn.run(
        create_app(state=state, frontend_dir="frontend/dist"),
        host=os.environ.get("STORYGIT_HOST", "127.0.0.1"),
        port=int(os.environ.get("STORYGIT_PORT", "8000")),
        log_level="warning",
    )


if __name__ == "__main__":
    main()
