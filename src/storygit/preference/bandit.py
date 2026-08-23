r"""Thompson sampling over safe and exploratory candidate slots.

A ranker that always shows its top three converges: the writer only ever sees what the
model already believes they want, so the model never learns it was wrong. That is the
classic exploration problem, and in a creative tool it has a second cost — the writer stops
being surprised, which is most of what they came for.

Two arms, over the *composition* of the three candidates shown:

* **safe** — three top-ranked candidates.
* **explore** — two top-ranked plus one high-surprise, low-rank candidate.

Thompson sampling picks between them: keep a Beta posterior over each arm's success rate,
sample one value from each posterior, and play the arm with the higher sample. Reward is
any positive writer action on the exploratory slot.

Why Thompson rather than epsilon-greedy: epsilon-greedy explores at a fixed rate forever,
wasting a slot long after the answer is known, and explores *uniformly*, treating a clearly
bad arm the same as a marginal one. Thompson's exploration rate falls out of the posterior
— wide posterior, more exploration; narrow posterior, less — with no schedule to tune. It
has strong regret guarantees and, at two arms, is about fifteen lines of code.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Arm(StrEnum):
    """How the shown set is composed."""

    safe = "safe"
    explore = "explore"


class BanditState(BaseModel):
    """Beta posteriors over each arm's success rate.

    A Beta(alpha, beta) posterior after a Beta(1,1) uniform prior is just
    "alpha-1 successes and beta-1 failures so far", which is why the update is an
    increment and why the whole thing survives being written to JSON.

    Attributes:
        alpha: Successes plus one, per arm.
        beta: Failures plus one, per arm.
        pulls: How many times each arm was played.
        rewards: Total reward per arm.
    """

    model_config = ConfigDict(frozen=True)

    alpha: dict[str, float] = Field(default_factory=lambda: {"safe": 1.0, "explore": 1.0})
    beta: dict[str, float] = Field(default_factory=lambda: {"safe": 1.0, "explore": 1.0})
    pulls: dict[str, int] = Field(default_factory=lambda: {"safe": 0, "explore": 0})
    rewards: dict[str, float] = Field(default_factory=lambda: {"safe": 0.0, "explore": 0.0})

    def mean(self, arm: Arm) -> float:
        """Posterior mean success rate for an arm."""
        a, b = self.alpha[arm.value], self.beta[arm.value]
        return a / (a + b)

    def mean_safe(self) -> float:
        """Posterior mean for the safe arm."""
        return self.mean(Arm.safe)

    def mean_explore(self) -> float:
        """Posterior mean for the exploratory arm."""
        return self.mean(Arm.explore)

    def update(self, arm: Arm, reward: float) -> BanditState:
        """Record the outcome of one pull.

        Args:
            arm: The arm that was played.
            reward: 1.0 for a positive writer action, 0.0 otherwise. Fractional rewards
                are allowed (an edit-then-accept is a partial success) and the Beta
                update handles them, which is standard for Bernoulli-like bandits with
                soft feedback.

        Returns:
            The updated state.
        """
        reward = max(0.0, min(1.0, reward))
        return self.model_copy(
            update={
                "alpha": {**self.alpha, arm.value: self.alpha[arm.value] + reward},
                "beta": {**self.beta, arm.value: self.beta[arm.value] + (1.0 - reward)},
                "pulls": {**self.pulls, arm.value: self.pulls[arm.value] + 1},
                "rewards": {**self.rewards, arm.value: self.rewards[arm.value] + reward},
            }
        )

    def save(self, path: Path | str) -> None:
        """Persist to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2))

    @staticmethod
    def load(path: Path | str) -> BanditState:
        """Load from JSON, returning a fresh uniform state if the file is absent."""
        file = Path(path)
        if not file.exists():
            return BanditState()
        return BanditState.model_validate(json.loads(file.read_text()))


class ThompsonBandit:
    """Two-armed Thompson sampler over shown-set composition.

    Attributes:
        state: The current posteriors.
    """

    def __init__(self, state: BanditState | None = None, *, seed: int | None = None) -> None:
        """Create the sampler.

        Args:
            state: Starting posteriors; uniform if omitted.
            seed: Fixes the sampling, so an evaluation run reproduces exactly.
        """
        self.state = state or BanditState()
        self._rng = random.Random(seed)

    def choose(self) -> Arm:
        """Sample one value from each posterior and play the better draw.

        Returns:
            The arm to play.
        """
        draws = {
            arm: self._rng.betavariate(self.state.alpha[arm.value], self.state.beta[arm.value])
            for arm in Arm
        }
        return max(draws, key=lambda arm: draws[arm])

    def update(self, arm: Arm, reward: float) -> None:
        """Record an outcome."""
        self.state = self.state.update(arm, reward)

    def compose(
        self, ranked: list[int], surprising: list[int], k: int = 3
    ) -> tuple[list[int], Arm]:
        """Choose the shown set.

        Args:
            ranked: Candidate indices, best-ranked first.
            surprising: Candidate indices ordered by surprise, most surprising first.
            k: How many to show.

        Returns:
            ``(indices to show, arm played)``. On the explore arm the last slot is the
            most surprising candidate that is not already in the safe set; if there is no
            such candidate the composition falls back to safe, and the arm reported is
            ``safe``, so the bandit is never credited for exploration that did not happen.
        """
        if not ranked:
            return [], Arm.safe
        arm = self.choose()
        safe = ranked[:k]
        # k < 2 leaves no slot for a surprising candidate, so no exploration can happen --
        # and reporting the explore arm here credited the bandit for it anyway, against the
        # promise three lines above. The old ternary was also tautological: `Arm.safe if arm
        # is Arm.safe else arm` is `arm`.
        if arm is Arm.safe or k < 2:
            return safe, Arm.safe

        keep = ranked[: k - 1]
        for index in surprising:
            if index not in keep:
                return [*keep, index], Arm.explore
        return safe, Arm.safe


def pseudo_regret_curve(arms: Sequence[Arm], rates: Mapping[Arm, float]) -> list[float]:
    """Cumulative pseudo-regret: what the *choices* cost, not what the coins did.

    Pseudo-regret charges each pull the gap between the best arm's expected rate and the
    played arm's expected rate. Realized regret (see :func:`realized_regret_curve`) charges
    the gap to the observed reward, which makes the curve a random walk dominated by
    Bernoulli noise and hides the only thing worth seeing.

    A pseudo-regret curve is non-decreasing by construction, so "sublinear" is visible
    directly: the curve flattens as the sampler stops paying to try the worse arm.

    Args:
        arms: The arm played at each step, in order.
        rates: Each arm's true expected reward. Known in a self-test and in the simulated
            evaluation, where the persona's acceptance rate per arm is a property of the
            persona.

    Returns:
        Cumulative pseudo-regret after each pull.
    """
    if not rates:
        return [0.0] * len(arms)
    best = max(rates.values())
    total = 0.0
    out: list[float] = []
    for arm in arms:
        total += best - rates.get(arm, best)
        out.append(total)
    return out


def realized_regret_curve(rewards: Sequence[tuple[Arm, float]], best_rate: float) -> list[float]:
    """Cumulative regret against the observed rewards.

    Noisy, and the honest choice when the arms' true rates are unknown — with real writers
    they always are. Reported alongside pseudo-regret rather than instead of it.

    Args:
        rewards: ``(arm played, reward)`` in order.
        best_rate: The best arm's assumed rate.

    Returns:
        Cumulative realized regret after each pull.
    """
    total = 0.0
    out: list[float] = []
    for _, reward in rewards:
        total += best_rate - reward
        out.append(total)
    return out
