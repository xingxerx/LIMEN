# Copyright (C) 2026 xingxerx
#
# Licensed under the Elastic License 2.0. See the LICENSE file in the
# repository root for the full terms.

"""Max-Cut frontend: QUBO encoding plus small instance generators.

Lifted out of examples/max_cut.py (which now imports from here) so the
study runner, tests and examples share one encoding. For an edge (u, v)
the term ``x_u + x_v - 2 x_u x_v`` is 1 when the edge is cut; QUBO
minimisation negates it: ``-x_u - x_v + 2 x_u x_v``, i.e. diagonal weight
``-deg(node)`` per node and ``+2.0`` per edge. (examples/max_cut.py used
``+1.0`` per edge, which is the Ising-style half-weight; both have the
same minimiser. The study uses the exact cut-count form so the QUBO
energy equals ``-cut_edges``.)
"""

from __future__ import annotations

import random

Edge = tuple[str, str]


def build_max_cut_qubo(edges: list[Edge]) -> dict[tuple[str, str], float]:
    """QUBO whose energy is exactly ``-(number of cut edges)``."""
    qubo: dict[tuple[str, str], float] = {}
    for u, v in edges:
        a, b = sorted((u, v))
        qubo[(a, b)] = qubo.get((a, b), 0.0) + 2.0
        qubo[(a, a)] = qubo.get((a, a), 0.0) - 1.0
        qubo[(b, b)] = qubo.get((b, b), 0.0) - 1.0
    return qubo


def count_cut_edges(assignment: dict[str, int], edges: list[Edge]) -> int:
    """Edges crossing the partition defined by *assignment*."""
    return sum(1 for u, v in edges if assignment.get(u, 0) != assignment.get(v, 0))


def ring_edges(n: int) -> list[Edge]:
    """Cycle graph on n nodes (optimum: n cut edges if n even, n-1 if odd)."""
    return [(f"v{i}", f"v{(i + 1) % n}") for i in range(n)]


def random_regular_edges(n: int, degree: int = 3, seed: int = 0) -> list[Edge]:
    """Approximately *degree*-regular random graph via the configuration model.

    Self-loops and duplicate edges are dropped, so a few nodes may end up
    with lower degree; good enough as a benchmark family and deterministic
    for a given (n, degree, seed).
    """
    if n * degree % 2:
        raise ValueError("n * degree must be even")
    rng = random.Random(seed)
    stubs = [i for i in range(n) for _ in range(degree)]
    rng.shuffle(stubs)
    seen: set[tuple[int, int]] = set()
    edges: list[Edge] = []
    for i in range(0, len(stubs), 2):
        a, b = stubs[i], stubs[i + 1]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in seen:
            continue
        seen.add(key)
        edges.append((f"v{key[0]}", f"v{key[1]}"))
    return edges


def ring_optimum(n: int) -> int:
    return n if n % 2 == 0 else n - 1
