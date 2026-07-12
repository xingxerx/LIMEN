# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Problem-structure signal for substrate-aware routing.

:func:`limen.router.budget_router.route` picks a backend purely on cost
tier, qubit count, and validation status -- it has no notion of which
*substrate* actually suits a problem's structure (dense/frustrated Ising
vs. a sparse, easily-satisfiable coupling graph). This module supplies
that missing signal as a :class:`ProblemProfile`, computed once per QUBO,
that :func:`route` can use as a tiebreaker via
``BackendProfile.substrate_affinity`` (see budget_router.py).

Scope limit: :func:`frustration_index` is a heuristic, not the exact
(NP-hard, balance-theoretic) frustration index from spin-glass theory. It
estimates how hard the QUBO's couplings are to jointly satisfy by running
a few random-restart greedy single-bit-flip descents to a local minimum
and averaging the fraction of couplings left unsatisfied there. It is a
deterministic function of (qubo, seed) -- not a certified bound.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ProblemProfile:
    """Structural signal for one QUBO, used only as a routing tiebreaker.

    Attributes:
        n_vars: Number of QUBO variables.
        edge_density: Fraction of possible variable pairs that carry a
            nonzero off-diagonal coupling (0.0 = no couplings, 1.0 =
            fully connected).
        frustration_index: Heuristic estimate in [0.0, 1.0] of how hard
            the couplings are to jointly satisfy at a local optimum (see
            module docstring). 0.0 = fully satisfiable (e.g. bipartite,
            unfrustrated), higher = more couplings fight each other.
        max_coupling_magnitude: Largest absolute off-diagonal weight.
    """

    n_vars: int
    edge_density: float
    frustration_index: float
    max_coupling_magnitude: float


def _off_diagonal_terms(
    qubo: list[tuple[tuple[int, int], float]]
) -> list[tuple[tuple[int, int], float]]:
    return [((i, j), w) for (i, j), w in qubo if i != j and w != 0.0]


def frustration_index(
    qubo: list[tuple[tuple[int, int], float]],
    n_vars: int,
    *,
    restarts: int = 4,
    seed: int = 0,
) -> float:
    """Heuristic frustration estimate for a QUBO (see module docstring).

    For each off-diagonal coupling ``(i, j, w)``, "satisfied" means the
    assignment agrees with what minimizes that term in isolation: both
    variables on when ``w < 0`` (the term rewards it), not both on when
    ``w > 0`` (the term penalizes it). Runs *restarts* independent
    random-restart greedy descents to a local minimum of the full QUBO
    energy (deterministic given *seed*) and returns the average fraction
    of couplings left unsatisfied at those local optima.

    Returns 0.0 for a QUBO with no off-diagonal couplings (nothing to
    frustrate) or zero variables.
    """
    off_diag = _off_diagonal_terms(qubo)
    if n_vars == 0 or not off_diag:
        return 0.0

    diag: dict[int, float] = {}
    for (i, j), w in qubo:
        if i == j:
            diag[i] = diag.get(i, 0.0) + w

    def energy(x: list[int]) -> float:
        e = sum(w * x[i] for i, w in diag.items())
        e += sum(w * x[i] * x[j] for (i, j), w in off_diag)
        return e

    rng = random.Random(seed)
    fractions: list[float] = []
    for _ in range(restarts):
        x = [rng.randint(0, 1) for _ in range(n_vars)]
        improved = True
        while improved:
            improved = False
            for i in range(n_vars):
                current = energy(x)
                x[i] ^= 1
                if energy(x) < current:
                    improved = True
                else:
                    x[i] ^= 1

        unsatisfied = 0
        for (i, j), w in off_diag:
            both_on = x[i] == 1 and x[j] == 1
            satisfied = both_on if w < 0 else not both_on
            if not satisfied:
                unsatisfied += 1
        fractions.append(unsatisfied / len(off_diag))

    return sum(fractions) / len(fractions)


def compute_problem_profile(
    qubo: list[tuple[tuple[int, int], float]],
    n_vars: int,
    *,
    restarts: int = 4,
    seed: int = 0,
) -> ProblemProfile:
    """Compute a :class:`ProblemProfile` for one QUBO."""
    off_diag = _off_diagonal_terms(qubo)
    max_possible_pairs = n_vars * (n_vars - 1) / 2 if n_vars > 1 else 0
    distinct_pairs = {frozenset(pair) for pair, _ in off_diag}
    edge_density = (
        len(distinct_pairs) / max_possible_pairs if max_possible_pairs > 0 else 0.0
    )
    max_coupling = max((abs(w) for _, w in off_diag), default=0.0)
    return ProblemProfile(
        n_vars=n_vars,
        edge_density=edge_density,
        frustration_index=frustration_index(qubo, n_vars, restarts=restarts, seed=seed),
        max_coupling_magnitude=max_coupling,
    )
