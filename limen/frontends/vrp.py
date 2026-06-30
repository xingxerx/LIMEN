# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Vehicle Routing Problem (VRP) frontend for LIMEN.

Encodes a multi-vehicle routing instance as a QUBO and converts it into a
LogicalGraph, reusing the TSP one-hot tour encoding that is already proven
on real QPU hardware (see benchmarks/tsp_eil51_benchmark.py).

The fleet constraint is added via the standard depot-duplication trick:
the depot is replicated once per vehicle, with zero distance between any
two depot copies. A single TSP tour over (customers + depot copies) then
splits at the depot copies into ``num_vehicles`` independent routes, with
no change to the underlying one-hot QUBO structure.

No vehicle capacity constraint is modeled (pure multi-depot-split routing).
Coordinates are treated as planar (Euclidean), which is an adequate
approximation for small-scale local coordinates but not geodesic distance
between distant (lat, lon) points.
"""

import math

from limen.core.ir import Interaction, LogicalGraph, Variable

Coordinate = tuple[float, float]


def _euclidean(a: Coordinate, b: Coordinate) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def distance_matrix(coords: list[Coordinate]) -> list[list[float]]:
    """Build a planar Euclidean distance matrix from a list of coordinates."""
    n = len(coords)
    return [[_euclidean(coords[i], coords[j]) for j in range(n)] for i in range(n)]


def _augmented_distance_matrix(
    coords: list[Coordinate],
    depot: int,
    num_vehicles: int,
) -> tuple[list[list[float]], list[int]]:
    """Replicate the depot ``num_vehicles`` times with zero inter-depot distance.

    Returns the augmented distance matrix together with ``customer_ids``,
    the mapping from augmented customer node index back to the original
    coordinate index (depot excluded).
    """
    customer_ids = [i for i in range(len(coords)) if i != depot]
    n_customers = len(customer_ids)
    n_total = n_customers + num_vehicles

    dist = [[0.0] * n_total for _ in range(n_total)]
    for a in range(n_customers):
        for b in range(n_customers):
            dist[a][b] = _euclidean(coords[customer_ids[a]], coords[customer_ids[b]])
        depot_dist = _euclidean(coords[customer_ids[a]], coords[depot])
        for v in range(num_vehicles):
            dist[a][n_customers + v] = depot_dist
            dist[n_customers + v][a] = depot_dist
    # Depot copies are distance 0 from one another: cutting the single tour
    # at any depot-copy pair costs nothing, so the optimizer is free to
    # partition customers across vehicles however the route lengths dictate.

    return dist, customer_ids


def vrp_qubo(
    coords: list[Coordinate],
    num_vehicles: int,
    depot: int = 0,
    penalty_a: float | None = None,
    penalty_b: float = 1.0,
) -> tuple[dict[tuple[str, str], float], list[int]]:
    """Build a QUBO for a multi-vehicle routing problem.

    Variables x_{i}_{t} ∈ {0,1}: augmented node i occupies tour position t,
    where augmented nodes are the customers (all coords except ``depot``)
    followed by ``num_vehicles`` zero-distance copies of the depot.

    Same one-hot structure as ``tsp_qubo`` in the TSP benchmark:

        H = A * Σ_i (1 - Σ_t x_it)^2        [each node once]
          + A * Σ_t (1 - Σ_i x_it)^2        [each position once]
          + B * Σ_{u,v,t} d_uv x_ut x_v,t+1  [tour length]

    Splitting the resulting closed tour at the depot copies (see
    ``decode_routes``) yields ``num_vehicles`` vehicle routes.

    Returns:
        A tuple of (qubo dict, customer_ids) where customer_ids maps
        augmented customer node index back to the original coordinate
        index, for use with ``decode_routes``.
    """
    if num_vehicles < 1:
        raise ValueError("num_vehicles must be at least 1")

    dist, customer_ids = _augmented_distance_matrix(coords, depot, num_vehicles)
    n = len(dist)
    max_dist = max(
        (dist[i][j] for i in range(n) for j in range(n) if i != j),
        default=0.0,
    )
    if penalty_a is None:
        penalty_a = penalty_b * max_dist * n * 5

    qubo: dict[tuple[str, str], float] = {}

    def var(i: int, t: int) -> str:
        return f"x_{i}_{t}"

    def add(u: str, v: str, w: float) -> None:
        key = (u, v) if u <= v else (v, u)
        qubo[key] = qubo.get(key, 0.0) + w

    # Each node visited exactly once: A*(1 - Σ_t x_it)^2
    for i in range(n):
        for t in range(n):
            add(var(i, t), var(i, t), -penalty_a)
            for s in range(t + 1, n):
                add(var(i, t), var(i, s), 2.0 * penalty_a)

    # Each position filled exactly once: A*(1 - Σ_i x_it)^2
    for t in range(n):
        for i in range(n):
            add(var(i, t), var(i, t), -penalty_a)
            for j in range(i + 1, n):
                add(var(i, t), var(j, t), 2.0 * penalty_a)

    # Tour length objective: B * Σ_{u≠v,t} d_uv * x_ut * x_v,t+1
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            for t in range(n):
                s = (t + 1) % n
                add(var(u, t), var(v, s), penalty_b * dist[u][v])

    return qubo, customer_ids


def decode_routes(
    assignment: dict[str, int],
    num_customers: int,
    num_vehicles: int,
    customer_ids: list[int],
) -> list[list[int]] | None:
    """Decode a logical-variable assignment into per-vehicle routes.

    Expects keys of the form ``x_{node}_{position}`` as produced by
    ``vrp_qubo``, where node indices < num_customers are customers and
    node indices >= num_customers are depot copies.

    Returns a list of ``num_vehicles`` routes, each a list of original
    coordinate indices in visiting order (depot excluded; empty for an
    unused vehicle). Returns None if the assignment is infeasible
    (collisions or unfilled positions).
    """
    n_total = num_customers + num_vehicles
    pos_to_node: dict[int, int] = {}
    node_to_pos: dict[int, int] = {}
    for i in range(n_total):
        for t in range(n_total):
            if assignment.get(f"x_{i}_{t}", 0) == 1:
                if t in pos_to_node or i in node_to_pos:
                    return None  # collision -> infeasible
                pos_to_node[t] = i
                node_to_pos[i] = t
    if len(pos_to_node) != n_total:
        return None  # not all positions filled

    tour = [pos_to_node[t] for t in range(n_total)]
    depot_positions = sorted(t for t, node in enumerate(tour) if node >= num_customers)
    if not depot_positions:
        return None  # no depot copies found -> infeasible

    routes: list[list[int]] = []
    for k, pos in enumerate(depot_positions):
        next_pos = depot_positions[(k + 1) % len(depot_positions)]
        route: list[int] = []
        i = (pos + 1) % n_total
        while i != next_pos:
            node = tour[i]
            if node < num_customers:
                route.append(customer_ids[node])
            i = (i + 1) % n_total
        routes.append(route)
    return routes


def from_vrp(
    coords: list[Coordinate],
    num_vehicles: int,
    depot: int = 0,
    penalty_a: float | None = None,
    penalty_b: float = 1.0,
    metadata: dict | None = None,
) -> tuple[LogicalGraph, list[int]]:
    """Convert a VRP instance into a LogicalGraph.

    Args:
        coords: List of (x, y) coordinates; ``coords[depot]`` is the depot.
        num_vehicles: Size of the fleet.
        depot: Index of the depot within ``coords``.
        penalty_a: Constraint penalty weight (auto-selected when None).
        penalty_b: Objective (distance) weight.
        metadata: Optional extra metadata merged into the graph's metadata.

    Returns:
        A tuple of (LogicalGraph, customer_ids), where customer_ids maps
        augmented customer node index back to the original coordinate
        index, for use with ``decode_routes``.

    Raises:
        ValueError: If the resulting graph fails validation.
    """
    qubo, customer_ids = vrp_qubo(coords, num_vehicles, depot, penalty_a, penalty_b)

    names: set[str] = set()
    for i, j in qubo:
        names.add(i)
        names.add(j)

    variables = [Variable(name=n, domain="binary") for n in sorted(names)]
    interactions = [
        Interaction(i=min(i, j), j=max(i, j), weight=float(w))
        for (i, j), w in sorted(
            (((min(k), max(k)), v) for k, v in qubo.items())
        )
    ]

    combined: dict = {
        "source": "vrp",
        "num_vehicles": num_vehicles,
        "depot": depot,
        "num_customers": len(customer_ids),
    }
    if metadata:
        combined.update(metadata)

    graph = LogicalGraph(variables=variables, interactions=interactions, metadata=combined)

    errors = graph.validate()
    if errors:
        raise ValueError(f"LogicalGraph validation failed: {errors}")

    return graph, customer_ids
