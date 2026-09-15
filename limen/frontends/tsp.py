# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""TSP frontend for LIMEN Discovery Loop P1.

Encodes a TSP Problem into the same one-hot QUBO used by the eil51
benchmark and the VRP frontend (``x_{city}_{position}``), then exposes
tour decode / length / approx_ratio helpers so a certificate solution can
be scored against ``Problem.known_optimum``.

Full TSPLIB eil51 is 51 cities (2601 binary variables) — far beyond the
statevector smoke path. Fixtures therefore use an eil51 *prefix*
(first N cities) with a classically verified optimum; see
``eil51_prefix_problem``.
"""

from __future__ import annotations

import math
from itertools import permutations
from typing import Any

from limen.formulation.problem import Problem
from limen.router.budget_router import RouteRequest

Coordinate = tuple[int, int] | tuple[float, float]

# TSPLIB eil51 EUC_2D coordinates; classical optimal tour length = 426.
EIL51_COORDS: list[tuple[int, int]] = [
    (37, 52), (49, 49), (52, 64), (20, 26), (40, 30),
    (21, 47), (17, 63), (31, 62), (52, 33), (51, 21),
    (42, 41), (31, 32), (5, 25), (12, 42), (36, 16),
    (52, 41), (27, 23), (17, 33), (13, 13), (57, 58),
    (62, 42), (42, 57), (16, 57), (8, 52), (7, 38),
    (27, 68), (30, 48), (43, 67), (58, 48), (58, 27),
    (37, 69), (38, 46), (46, 10), (61, 33), (62, 63),
    (63, 69), (32, 22), (45, 35), (59, 15), (5, 6),
    (10, 17), (21, 10), (5, 64), (30, 15), (39, 10),
    (32, 39), (25, 32), (25, 55), (48, 28), (56, 37),
    (30, 40),
]
EIL51_OPTIMAL_TOUR_LENGTH = 426
EIL51_N_CITIES = 51


def euc_2d(a: Coordinate, b: Coordinate) -> int:
    """Rounded Euclidean distance (TSPLIB EUC_2D)."""
    return int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + 0.5)


def distance_matrix(coords: list[Coordinate]) -> list[list[int]]:
    """Build a TSPLIB-style EUC_2D distance matrix."""
    n = len(coords)
    return [[euc_2d(coords[i], coords[j]) for j in range(n)] for i in range(n)]


def tsp_qubo(
    dist: list[list[int]],
    penalty_a: float | None = None,
    penalty_b: float = 1.0,
) -> dict[tuple[str, str], float]:
    """Build a QUBO for an n-city TSP from a distance matrix.

    Variables ``x_{i}_{t}`` in {0,1}: city i visited at position t.

    Same expansion as ``benchmarks/tsp_eil51_benchmark.tsp_qubo`` and the
    VRP frontend one-hot tour — deliberately shared so Discovery Loop
    certificates stay comparable to existing eil51 results.

        H = A * Σ_i (1 - Σ_t x_it)^2        [each city once]
          + A * Σ_t (1 - Σ_i x_it)^2        [each position once]
          + B * Σ_{u,v,t} d_uv x_ut x_v,t+1 [tour length]
    """
    n = len(dist)
    if n < 2:
        raise ValueError("TSP requires at least 2 cities")
    max_dist = max(dist[i][j] for i in range(n) for j in range(n) if i != j)
    if penalty_a is None:
        penalty_a = penalty_b * max_dist * n * 5

    qubo: dict[tuple[str, str], float] = {}

    def var(i: int, t: int) -> str:
        return f"x_{i}_{t}"

    def add(u: str, v: str, w: float) -> None:
        key = (u, v) if u <= v else (v, u)
        qubo[key] = qubo.get(key, 0.0) + w

    for i in range(n):
        for t in range(n):
            add(var(i, t), var(i, t), -penalty_a)
            for s in range(t + 1, n):
                add(var(i, t), var(i, s), 2.0 * penalty_a)

    for t in range(n):
        for i in range(n):
            add(var(i, t), var(i, t), -penalty_a)
            for j in range(i + 1, n):
                add(var(i, t), var(j, t), 2.0 * penalty_a)

    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            for t in range(n):
                s = (t + 1) % n
                add(var(u, t), var(v, s), penalty_b * dist[u][v])

    return qubo


def tour_length(tour: list[int], dist: list[list[int]]) -> int:
    """Sum of edge lengths around the closed tour."""
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def classical_optimal_tour(dist: list[list[int]]) -> tuple[int, list[int]]:
    """Brute-force optimal TSP tour (practical for n <= ~10)."""
    n = len(dist)
    best_len = math.inf
    best_tour: list[int] = []
    for perm in permutations(range(1, n)):
        tour = [0] + list(perm)
        length = tour_length(tour, dist)
        if length < best_len:
            best_len = length
            best_tour = tour
    return int(best_len), best_tour


def tour_to_assignment(tour: list[int]) -> dict[str, int]:
    """Encode a city-order tour as the one-hot ``x_{city}_{pos}`` assignment."""
    n = len(tour)
    assignment: dict[str, int] = {}
    for t, city in enumerate(tour):
        for i in range(n):
            assignment[f"x_{i}_{t}"] = 1 if i == city else 0
    return assignment


def decode_tour(assignment: dict[str, int], n: int) -> list[int] | None:
    """Decode a logical assignment into a tour, or None if infeasible."""
    pos_to_city: dict[int, int] = {}
    city_to_pos: dict[int, int] = {}
    for i in range(n):
        for t in range(n):
            if assignment.get(f"x_{i}_{t}", 0) == 1:
                if t in pos_to_city or i in city_to_pos:
                    return None
                pos_to_city[t] = i
                city_to_pos[i] = t
    if len(pos_to_city) != n:
        return None
    return [pos_to_city[t] for t in range(n)]


def approx_ratio(
    assignment: dict[str, int],
    dist: list[list[int]],
    known_optimum: float,
) -> float | None:
    """Tour length / known_optimum, or None if the assignment is infeasible.

    A feasible optimal tour yields exactly 1.0 when ``known_optimum`` is
    the classical optimum tour length.
    """
    if known_optimum <= 0:
        raise ValueError("known_optimum must be positive")
    tour = decode_tour(assignment, len(dist))
    if tour is None:
        return None
    return tour_length(tour, dist) / float(known_optimum)


def eil51_prefix_problem(
    n_cities: int = 2,
    *,
    penalty_a: float | None = None,
    penalty_b: float = 1.0,
) -> Problem:
    """Build a TSP Problem from the first ``n_cities`` of TSPLIB eil51.

    Full eil51 (51 cities, optimum 426) needs 51^2 = 2601 QUBO variables
    and is not a simulator smoke target. The default ``n_cities=2`` keeps
    4 binary variables so ``run_route_request`` on the statevector
    simulator recovers approx_ratio 1.000. Larger prefixes remain valid
    encodings (same ``tsp_qubo``) for classical ground-state checks.
    """
    if n_cities < 2 or n_cities > EIL51_N_CITIES:
        raise ValueError(f"n_cities must be in [2, {EIL51_N_CITIES}], got {n_cities}")
    coords: list[Coordinate] = list(EIL51_COORDS[:n_cities])
    dist = distance_matrix(coords)
    opt_len, opt_tour = classical_optimal_tour(dist)
    return Problem(
        id=f"tsp/eil51/n={n_cities}",
        problem_class="tsp",
        known_optimum=float(opt_len),
        metadata={
            "source": "tsplib_eil51_prefix",
            "n_cities": n_cities,
            "full_eil51_n_cities": EIL51_N_CITIES,
            "full_eil51_optimal_tour_length": EIL51_OPTIMAL_TOUR_LENGTH,
            "coords": coords,
            "distance_matrix": dist,
            "classical_optimal_tour": opt_tour,
            "penalty_a": penalty_a,
            "penalty_b": penalty_b,
            "why_not_full_eil51": (
                "Full eil51 is 51 cities / 2601 binary vars; statevector "
                "QAOA smoke uses an eil51 coordinate prefix with a "
                "brute-force-verified known_optimum instead."
            ),
        },
    )


def encode_tsp(problem: Problem) -> Problem:
    """Encode a TSP Problem into QUBO variables / objective in place-style copy.

    Returns a new Problem whose ``variables`` and ``objective`` are the
    one-hot tour QUBO accepted by ``RouteRequest``. Requires
    ``problem_class == "tsp"`` and ``metadata`` carrying either
    ``distance_matrix`` or ``coords``.
    """
    if problem.problem_class != "tsp":
        raise ValueError(
            f"encode_tsp expects problem_class='tsp', got {problem.problem_class!r}"
        )
    meta = dict(problem.metadata)
    if "distance_matrix" in meta:
        dist = meta["distance_matrix"]
    elif "coords" in meta:
        dist = distance_matrix(list(meta["coords"]))
        meta["distance_matrix"] = dist
    else:
        raise ValueError("TSP Problem metadata must include distance_matrix or coords")

    penalty_a = meta.get("penalty_a")
    penalty_b = float(meta.get("penalty_b", 1.0))
    qubo = tsp_qubo(dist, penalty_a=penalty_a, penalty_b=penalty_b)
    names = sorted({v for pair in qubo for v in pair})
    return Problem(
        id=problem.id,
        problem_class=problem.problem_class,
        variables=names,
        constraints=list(problem.constraints),
        objective=qubo,
        known_optimum=problem.known_optimum,
        metadata=meta,
    )


def to_route_request(
    problem: Problem,
    *,
    fidelity_target: float = 0.9,
    credit_budget: float = 0.0,
    offline: bool = False,
) -> RouteRequest:
    """Wrap an encoded Problem's objective as a ``RouteRequest``.

    ``credit_budget=0`` forces the free simulator tier (no QPU spend).
    """
    if not problem.objective:
        raise ValueError("Problem.objective is empty; call encode_tsp first")
    return RouteRequest(
        qubo=dict(problem.objective),
        fidelity_target=fidelity_target,
        credit_budget=credit_budget,
        offline=offline,
    )


def score_assignment(problem: Problem, assignment: dict[str, int]) -> dict[str, Any]:
    """Score a bit assignment against a TSP Problem's known_optimum.

    Returns a dict with tour, tour_length, and approx_ratio (None when
    infeasible or known_optimum is missing).
    """
    dist = problem.metadata.get("distance_matrix")
    if dist is None and "coords" in problem.metadata:
        dist = distance_matrix(list(problem.metadata["coords"]))
    if dist is None:
        raise ValueError("cannot score TSP without distance_matrix or coords")
    tour = decode_tour(assignment, len(dist))
    length = tour_length(tour, dist) if tour is not None else None
    ratio = None
    if length is not None and problem.known_optimum is not None:
        ratio = length / float(problem.known_optimum)
    return {
        "tour": tour,
        "tour_length": length,
        "approx_ratio": ratio,
        "known_optimum": problem.known_optimum,
    }
