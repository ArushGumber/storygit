"""Regret self-test for the Thompson sampler, and the figure it produces.

Not part of the evaluation proper — the real bandit numbers come from the simulated-writer
runs. This exists so the sampler's behaviour can be checked in isolation against known
arm rates, which is the only setting where "is the regret sublinear" has an unambiguous
answer.

    python -m eval.bandit_selftest
"""

from __future__ import annotations

import random
from pathlib import Path

from storygit.preference.bandit import (
    Arm,
    ThompsonBandit,
    pseudo_regret_curve,
    realized_regret_curve,
)

RESULTS = Path(__file__).parent / "results"


def run(
    *,
    rates: dict[Arm, float] | None = None,
    rounds: int = 400,
    seed: int = 1,
) -> dict[str, object]:
    """Play a two-armed Bernoulli bandit and collect the regret curve.

    Args:
        rates: True success rate per arm.
        rounds: How many pulls.
        seed: RNG seed for both the sampler and the environment.

    Returns:
        The history, the regret curve, an epsilon-greedy comparison, and the posteriors.
    """
    rates = rates or {Arm.safe: 0.4, Arm.explore: 0.7}
    best = max(rates.values())

    rng = random.Random(seed)
    bandit = ThompsonBandit(seed=seed)
    history: list[tuple[Arm, float]] = []
    for _ in range(rounds):
        arm = bandit.choose()
        reward = 1.0 if rng.random() < rates[arm] else 0.0
        bandit.update(arm, reward)
        history.append((arm, reward))

    # Epsilon-greedy at a fixed rate, for contrast: it keeps paying the exploration cost
    # long after the answer is known, which is the whole argument for posterior sampling.
    eps_rng = random.Random(seed)
    counts = dict.fromkeys(Arm, 0)
    totals = dict.fromkeys(Arm, 0.0)
    eps_history: list[tuple[Arm, float]] = []
    for _ in range(rounds):
        if eps_rng.random() < 0.1 or min(counts.values()) == 0:
            arm = eps_rng.choice(list(Arm))
        else:
            arm = max(Arm, key=lambda a: totals[a] / max(1, counts[a]))
        reward = 1.0 if eps_rng.random() < rates[arm] else 0.0
        counts[arm] += 1
        totals[arm] += reward
        eps_history.append((arm, reward))

    return {
        "regret": pseudo_regret_curve([arm for arm, _ in history], rates),
        "epsilon_regret": pseudo_regret_curve([arm for arm, _ in eps_history], rates),
        "regret_realized": realized_regret_curve(history, best),
        "pulls": dict(bandit.state.pulls),
        "posterior_safe": bandit.state.mean_safe(),
        "posterior_explore": bandit.state.mean_explore(),
        "rates": {arm.value: rate for arm, rate in rates.items()},
    }


def main() -> None:
    """Run the self-test and write ``eval/results/bandit_selftest.svg``."""
    from eval.plots import line_plot

    result = run()
    regret = result["regret"]
    assert isinstance(regret, list)
    epsilon = result["epsilon_regret"]
    assert isinstance(epsilon, list)

    path = line_plot(
        {
            "Thompson sampling": regret,
            "epsilon-greedy (0.1)": epsilon,
        },
        title="Cumulative regret, two-armed bandit (p = 0.40 vs 0.70)",
        xlabel="candidate sets shown",
        ylabel="cumulative regret",
        path=RESULTS / "bandit_selftest.svg",
    )

    print(f"pulls: {result['pulls']}")
    print(
        f"posterior means: safe={result['posterior_safe']:.3f} "
        f"explore={result['posterior_explore']:.3f}  (true 0.40 / 0.70)"
    )
    print(f"final regret: thompson={regret[-1]:.1f}  epsilon-greedy={epsilon[-1]:.1f}")
    print(
        f"first half {regret[len(regret) // 2]:.1f} -> second half "
        f"{regret[-1] - regret[len(regret) // 2]:.1f}  (sublinear if the second is smaller)"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
