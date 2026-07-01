# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Decode and score raw QPU measurement counts from five real-hardware TSP
eil51 (4-city, 16-variable) runs, using LIMEN's actual QUBO encoding and
tour-decoding logic (ported from benchmarks/tsp_eil51_benchmark.py).

Verifies whether LIMEN's QAOA-on-real-hardware pipeline is finding correct
or near-optimal feasible tours at a non-trivial rate, across:

  1. ibm_kingston   job d8ll3ojnn5bs738s8j3g   (documented milestone, ratio 1.000)
  2. ibm_kingston   job d8viq20pknjs73a14nh0   (tsp_eil51_20260626_215323)
  3. ibm_fez        job d8vl8j1ropqc738cjvug   (tsp_eil51_20260627_004217)
  4. Rigetti        job c281a62f-52c1-43b5-9c25-93fc2e6258a2
  5. Rigetti        job 2100264a-70b9-4cd4-a55e-a718da15d3d8

Usage::

    python examples/analyze_tsp_qpu_results.py
"""

from __future__ import annotations

import json
import math
import pathlib
from itertools import permutations

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# ---------------------------------------------------------------------------
# eil51 sub-problem (first 4 cities) — identical to benchmarks/tsp_eil51_benchmark.py
# ---------------------------------------------------------------------------

EIL51_COORDS: list[tuple[int, int]] = [
    (37, 52), (49, 49), (52, 64), (20, 26),
]
N_CITIES = 4
CLASSICAL_OPT_LEN = 102


def _euc_2d(a: tuple[int, int], b: tuple[int, int]) -> int:
    return int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + 0.5)


def _distance_matrix(coords: list[tuple[int, int]]) -> list[list[int]]:
    n = len(coords)
    return [[_euc_2d(coords[i], coords[j]) for j in range(n)] for i in range(n)]


def _tour_length(tour: list[int], dist: list[list[int]]) -> int:
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def _classical_optimal_tour(dist: list[list[int]]) -> tuple[int, list[int]]:
    n = len(dist)
    best_len = math.inf
    best_tour: list[int] = []
    for perm in permutations(range(1, n)):
        tour = [0] + list(perm)
        length = _tour_length(tour, dist)
        if length < best_len:
            best_len = length
            best_tour = tour
    return int(best_len), best_tour


DIST = _distance_matrix(EIL51_COORDS)
_opt_len, _opt_tour = _classical_optimal_tour(DIST)
assert _opt_len == CLASSICAL_OPT_LEN, f"expected {CLASSICAL_OPT_LEN}, got {_opt_len}"


# ---------------------------------------------------------------------------
# LIMEN variable naming / embedding replication
#
# The QAOA circuit is built over physical QUBO variables named q0..q{n-1}
# (from limen.core.compiler.default_hardware_graph), with a 1-to-1
# lexicographic embedding: sorted(logical x_i_t names) <-> sorted(q-node
# names). Qiskit's SamplerV2 measures all qubits in circuit order and
# get_counts() bitstrings are big-endian (q0 = rightmost char). Both the
# qiskit_backend (_counts_to_samples/_bits_to_assignment) and the run_qiskit_qpu
# call use `variables = sorted({name for pair in qubo for name in pair})`,
# i.e. the alphabetically-sorted physical q-node names, as the bit-index
# order. We replicate that exactly, then invert the embedding back to
# logical x_i_t names before decoding a tour.
# ---------------------------------------------------------------------------

N_VARS = N_CITIES * N_CITIES  # 16

_logical_vars = sorted(f"x_{i}_{t}" for i in range(N_CITIES) for t in range(N_CITIES))
_physical_nodes = sorted(f"q{i}" for i in range(N_VARS))
# lexicographic 1-to-1 embedding, exactly as compile_lexicographic() does it
EMBEDDING: dict[str, str] = dict(zip(_logical_vars, _physical_nodes))
PHYS_TO_LOGICAL: dict[str, str] = {p: l for l, p in EMBEDDING.items()}

# Bit-index order used when building/measuring the circuit: sorted physical
# variable names (q0, q1, q10, q11, ..., q15, q2, q3, ..., q9) — NOT numeric.
BIT_ORDER_PHYSICAL: list[str] = sorted(_physical_nodes)


def _bits_to_logical_assignment(bitstring: str) -> dict[str, int]:
    """Qiskit bitstring (big-endian, q0=rightmost) -> logical x_i_t assignment."""
    bits = bitstring[::-1]
    assignment: dict[str, int] = {}
    for idx, phys in enumerate(BIT_ORDER_PHYSICAL):
        logical = PHYS_TO_LOGICAL[phys]
        assignment[logical] = int(bits[idx])
    return assignment


def _decode_tour(assignment: dict[str, int], n: int) -> list[int] | None:
    """Decode a logical x_i_t assignment into a tour, or None if infeasible."""
    pos_to_city: dict[int, int] = {}
    city_to_pos: dict[int, int] = {}
    for i in range(n):
        for t in range(n):
            key = f"x_{i}_{t}"
            if assignment.get(key, 0) == 1:
                if t in pos_to_city or i in city_to_pos:
                    return None
                pos_to_city[t] = i
                city_to_pos[i] = t
    if len(pos_to_city) != n:
        return None
    return [pos_to_city[t] for t in range(n)]


# ---------------------------------------------------------------------------
# Loading raw counts from each results file
# ---------------------------------------------------------------------------

def _load_ibm_counts(path: pathlib.Path, job_id: str) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        if item.get("job_id") == job_id:
            return item["pubs"][0]["counts"]
    raise KeyError(f"job_id {job_id} not found in {path}")


def _load_oq_counts(path: pathlib.Path, job_id: str) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for item in data:
        if item.get("job_id") == job_id:
            return item["counts"]
    raise KeyError(f"job_id {job_id} not found in {path}")


RUNS = [
    {
        "label": "ibm_kingston (milestone)",
        "job_id": "d8ll3ojnn5bs738s8j3g",
        "loader": lambda: _load_ibm_counts(
            RESULTS / "fetched_jobs_d8ll3ojnn5bs738s8j3g.json",
            "d8ll3ojnn5bs738s8j3g",
        ),
    },
    {
        "label": "ibm_kingston (20260626_215323)",
        "job_id": "d8viq20pknjs73a14nh0",
        "loader": lambda: _load_ibm_counts(
            RESULTS / "fetched_jobs_d8vl8j1ropqc738cjvug_d91euieu9n7c73amjlcg_d91epe6u9n7c73amjc.json",
            "d8viq20pknjs73a14nh0",
        ),
    },
    {
        "label": "ibm_fez (20260627_004217)",
        "job_id": "d8vl8j1ropqc738cjvug",
        "loader": lambda: _load_ibm_counts(
            RESULTS / "fetched_jobs_d8vl8j1ropqc738cjvug_d91euieu9n7c73amjlcg_d91epe6u9n7c73amjc.json",
            "d8vl8j1ropqc738cjvug",
        ),
    },
    {
        "label": "rigetti:cepheus-1-108q (c281a62f, official)",
        "job_id": "c281a62f-52c1-43b5-9c25-93fc2e6258a2",
        "loader": lambda: _load_oq_counts(
            RESULTS / "fetched_openquantum_jobs_c281a62f-52c1-43b5-9c25-93fc2e6258a2_2100264a-70b9-4cd4-a55e.json",
            "c281a62f-52c1-43b5-9c25-93fc2e6258a2",
        ),
    },
    {
        "label": "rigetti:cepheus-1-108q (2100264a, retry)",
        "job_id": "2100264a-70b9-4cd4-a55e-a718da15d3d8",
        "loader": lambda: _load_oq_counts(
            RESULTS / "fetched_openquantum_jobs_c281a62f-52c1-43b5-9c25-93fc2e6258a2_2100264a-70b9-4cd4-a55e.json",
            "2100264a-70b9-4cd4-a55e-a718da15d3d8",
        ),
    },
]


def analyze(counts: dict[str, int]) -> dict:
    total_shots = sum(counts.values())
    feasible_shots = 0
    best_len: int | None = None
    best_bitstring: str | None = None
    best_tour: list[int] | None = None
    optimal_found = False
    distinct_feasible_tours: set[tuple[int, ...]] = set()

    for bitstring, count in counts.items():
        assignment = _bits_to_logical_assignment(bitstring)
        tour = _decode_tour(assignment, N_CITIES)
        if tour is None:
            continue
        feasible_shots += count
        length = _tour_length(tour, DIST)
        distinct_feasible_tours.add(tuple(tour))
        if length == CLASSICAL_OPT_LEN:
            optimal_found = True
        if best_len is None or length < best_len:
            best_len = length
            best_bitstring = bitstring
            best_tour = tour

    return {
        "total_shots": total_shots,
        "distinct_bitstrings": len(counts),
        "feasible_shots": feasible_shots,
        "feasible_rate": feasible_shots / total_shots if total_shots else 0.0,
        "distinct_feasible_tours": len(distinct_feasible_tours),
        "best_len": best_len,
        "best_bitstring": best_bitstring,
        "best_tour": best_tour,
        "approx_ratio": (best_len / CLASSICAL_OPT_LEN) if best_len is not None else None,
        "optimal_found": optimal_found,
    }


def main() -> None:
    print(f"Classical optimal tour length: {CLASSICAL_OPT_LEN} (tour {_opt_tour})")
    print()

    rows = []
    for run in RUNS:
        counts = run["loader"]()
        result = analyze(counts)
        rows.append((run["label"], run["job_id"], result))

    header = (
        f"{'Run':40s} {'Shots':>7s} {'Feasible':>10s} {'#FeasTours':>10s} "
        f"{'BestLen':>8s} {'ApproxRatio':>11s} {'Optimal?':>8s}"
    )
    print(header)
    print("-" * len(header))
    for label, job_id, r in rows:
        best_len_s = str(r["best_len"]) if r["best_len"] is not None else "N/A"
        ratio_s = f"{r['approx_ratio']:.3f}" if r["approx_ratio"] is not None else "N/A"
        print(
            f"{label:40s} {r['total_shots']:7d} "
            f"{r['feasible_rate']*100:9.2f}% {r['distinct_feasible_tours']:10d} "
            f"{best_len_s:>8s} {ratio_s:>11s} {str(r['optimal_found']):>8s}"
        )

    print()
    print("Details:")
    for label, job_id, r in rows:
        print(f"\n  {label}  (job {job_id})")
        print(f"    total shots            : {r['total_shots']}")
        print(f"    distinct bitstrings    : {r['distinct_bitstrings']}")
        print(f"    feasible shots         : {r['feasible_shots']} ({r['feasible_rate']*100:.2f}%)")
        print(f"    distinct feasible tours: {r['distinct_feasible_tours']}")
        print(f"    best tour length       : {r['best_len']}")
        print(f"    best bitstring         : {r['best_bitstring']}")
        print(f"    best tour (city order) : {r['best_tour']}")
        print(f"    approx ratio           : {r['approx_ratio']}")
        print(f"    optimal (102) found    : {r['optimal_found']}")


if __name__ == "__main__":
    main()
