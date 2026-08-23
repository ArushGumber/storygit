"""Regret self-test for the Thompson sampler, run in isolation.

Not part of the evaluation proper — the real bandit numbers come from the simulated-writer
runs. This exists so the sampler's behaviour can be checked against *known* arm rates, which
is the only setting where "is the regret sublinear" has an unambiguous answer.

It is a thin command over :func:`eval.offline.bandit_regret` rather than its own simulation.
It used to be a second implementation of the same 400-round experiment with the same seed
and the same arms, which meant two figures in the interface showing byte-identical curves
under different titles, and two loops to keep in step if either ever changed.

    python -m eval.bandit_selftest
"""

from __future__ import annotations

from typing import Any


def run(*, rounds: int = 400, seed: int = 1) -> dict[str, Any]:
    """Play the two-armed Bernoulli bandit and collect the regret curves.

    Args:
        rounds: How many pulls.
        seed: RNG seed for both the sampler and the environment.

    Returns:
        The regret curves, the pull counts, the posteriors, and the true rates.
    """
    from eval.offline import bandit_regret

    return bandit_regret(rounds=rounds, seed=seed)


def main() -> None:
    """Print the isolation check.

    No figure: the offline tier already writes ``bandit_regret.svg`` from exactly this
    computation, and a second copy of it under another name is not a second result.
    """
    result = run()
    thompson = result["final"]["thompson"]
    epsilon = result["final"]["epsilon_greedy"]
    posterior = result["posterior"]
    rates = result["true_rates"]

    print(f"pulls: {result['pulls']}")
    print(
        f"posterior means: safe={posterior['safe']:.3f} explore={posterior['explore']:.3f}"
        f"  (true {rates['safe']:.2f} / {rates['explore']:.2f})"
    )
    print(f"pseudo-regret after {len(result['thompson'])} rounds:")
    print(f"  thompson       {thompson:.1f}")
    print(f"  epsilon-greedy {epsilon:.1f}")
    # Sublinear regret means the curve flattens: the second half costs less than the first.
    curve = result["thompson"]
    half = len(curve) // 2
    first, second = curve[half - 1], curve[-1] - curve[half - 1]
    shape = "sub" if second < first else "super"
    print(f"first half {first:.1f}, second half {second:.1f} -- {shape}linear")


if __name__ == "__main__":
    main()
