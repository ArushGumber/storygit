"""Shared state for the HTTP layer, constructed once and injected.

The repository, the provider router, and the engine are expensive to build — a SQLite
connection, an httpx client, and eventually a few hundred megabytes of local models — so
they are built once at startup and handed to routes by dependency injection rather than
constructed per request.

**Single-writer assumption**, stated here because it is load-bearing and easy to miss: one
story database, one writer, one process. There is no session, no auth, and no locking
between concurrent writers, because there are none. That is the right shape for a tool a
novelist runs on their own machine, and `learning_systems.tex` describes exactly what would
change at platform scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from fastapi import Request

from storygit.config import Settings, get_settings
from storygit.domain.ids import IdGenerator
from storygit.engine import Engine
from storygit.providers.router import Router, build_router
from storygit.store.branches import DEFAULT_BRANCH
from storygit.store.repository import Repository


@dataclass
class AppState:
    """Everything the routes need, built once.

    Attributes:
        repo: The story repository.
        router: Provider access.
        settings: Loaded configuration.
        branch: The branch currently being edited.
        engines: One engine per branch, cached — each carries its own preference state and
            its own set of pending proposals, so they must not be shared.
        results_dir: Where the evaluation wrote its results, for the Eval and Gallery tabs.
    """

    repo: Repository
    router: Router
    settings: Settings
    branch: str = DEFAULT_BRANCH
    engines: dict[str, Engine] = field(default_factory=dict)
    results_dir: Path = Path("eval/results")
    seed: int | None = None

    def engine(self, branch: str | None = None) -> Engine:
        """The engine for a branch, created on first use.

        Args:
            branch: Branch name; the current branch when omitted.

        Returns:
            The engine, which holds the pending proposals for that branch.
        """
        name = branch or self.branch
        if name not in self.engines:
            self.engines[name] = Engine(
                self.repo,
                self.router,
                ids=IdGenerator(seed=self.seed, stream=f"api:{name}"),
                branch=name,
            )
        return self.engines[name]

    def close(self) -> None:
        """Close the repository. The router is closed asynchronously by the lifespan."""
        self.repo.close()


def build_state(
    db_path: str | Path,
    *,
    settings: Settings | None = None,
    router: Router | None = None,
    results_dir: str | Path = "eval/results",
    seed: int | None = None,
) -> AppState:
    """Build the application state.

    Args:
        db_path: Story database.
        settings: Settings; loaded from the environment when omitted.
        router: Provider router; built from settings when omitted. Tests inject a mock.
        results_dir: Where the evaluation results live.
        seed: Id-generator seed, for reproducible tests.

    Returns:
        The state.
    """
    resolved = settings or get_settings()
    return AppState(
        repo=Repository.open(db_path),
        router=router or build_router(resolved),
        settings=resolved,
        results_dir=Path(results_dir),
        seed=seed,
    )


def seed_if_empty(state: AppState) -> None:
    """Put the orphan seed in place if the database is new.

    A brand new story with no root would make every route return an empty tree, which is
    indistinguishable from a bug. Seeding on first open means the tool opens onto something.
    """
    if state.repo.is_initialized(state.branch):
        return
    from storygit.seed import seed_story

    seed_story(state.repo, IdGenerator(seed=state.seed, stream="seed"), branch=state.branch)


def app_state(request: Request) -> AppState:
    """FastAPI dependency: pull the shared state off the app.

    The ``Request`` annotation is load-bearing. Without it FastAPI cannot tell this is an
    injected object and treats the parameter as a required query string, which turns every
    route in the application into a 422.
    """
    state: AppState = request.app.state.storygit
    return state
