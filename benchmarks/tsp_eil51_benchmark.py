# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""TSP eil51 benchmark: LIMEN pipeline on IBM ibm_kingston.

Loads the TSPLIB eil51 instance (51 cities, classical optimal tour = 426),
converts a tractable sub-problem to QUBO, runs the full LIMEN pipeline
(lexicographic compiler + Stackelberg co-design), executes QAOA on
ibm_kingston via SamplerV2, emits a CompilationCertificate, and compares
the QPU result against the classical sub-problem optimum.

Full eil51 requires 51^2 = 2601 qubits, far beyond any current QPU. The
benchmark selects the first --cities cities (default 4, giving 16 variables)
so the circuit fits on ibm_kingston (156 qubits). The classical optimal for
the N-city sub-problem is found by brute force, and the full-problem
reference of 426 is noted in the output.

Usage::

    python benchmarks/tsp_eil51_benchmark.py [--cities N] [--shots N] [--sim-only]

IBM credentials are read from .env at the project root
(IBM_QUANTUM_TOKEN, IBM_QUANTUM_CRN).
"""

import argparse
import json
import math
import os
import pathlib
import sys
import time
from itertools import permutations

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

_reconfigure = getattr(sys.stdout, "reconfigure", None)
if _reconfigure and (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
    _reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.analog.certificate import certify_ising
from limen.backends.qiskit_backend import (
    _qubo_to_ising,
    run_qiskit,
    run_qiskit_qpu,
)
from limen.codesign.solver import run_codesign
from limen.validator.validator import validate

_BACKEND_NAME = "ibm_kingston"
_REPS = 1
_CODESIGN_ITERATIONS = 15
_CODESIGN_RUNS = 500
_VALIDATOR_RUNS = 1000

# ---------------------------------------------------------------------------
# eil51 city coordinates — TSPLIB EUC_2D, 51 cities, optimal tour = 426
# ---------------------------------------------------------------------------

EIL51_COORDS: list[tuple[int, int]] = [
    (37, 52), (49, 49), (52, 64), (20, 26), (40, 30),
    (21, 47), (17, 63), (31, 62), (52, 33), (51, 21),
    (42, 41), (31, 32), ( 5, 25), (12, 42), (36, 16),
    (52, 41), (27, 23), (17, 33), (13, 13), (57, 58),
    (62, 42), (42, 57), (16, 57), ( 8, 52), ( 7, 38),
    (27, 68), (30, 48), (43, 67), (58, 48), (58, 27),
    (37, 69), (38, 46), (46, 10), (61, 33), (62, 63),
    (63, 69), (32, 22), (45, 35), (59, 15), ( 5,  6),
    (10, 17), (21, 10), ( 5, 64), (30, 15), (39, 10),
    (32, 39), (25, 32), (25, 55), (48, 28), (56, 37),
    (30, 40),
]
EIL51_OPTIMAL_TOUR_LENGTH = 426
EIL51_N_CITIES = 51

# ---------------------------------------------------------------------------
# Distance and QUBO helpers
# ---------------------------------------------------------------------------

def _euc_2d(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Rounded Euclidean distance (TSPLIB EUC_2D)."""
    return int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + 0.5)


def _distance_matrix(coords: list[tuple[int, int]]) -> list[list[int]]:
    n = len(coords)
    return [[_euc_2d(coords[i], coords[j]) for j in range(n)] for i in range(n)]


def tsp_qubo(
    dist: list[list[int]],
    penalty_a: float | None = None,
    penalty_b: float = 1.0,
) -> dict[tuple[str, str], float]:
    """Build a QUBO for an n-city TSP from a distance matrix.

    Variables x_{i}_{t} ∈ {0,1}: city i visited at position t.

    H = A * Σ_i (1 - Σ_t x_it)^2        [each city once]
      + A * Σ_t (1 - Σ_i x_it)^2        [each position once]
      + B * Σ_{u,v,t} d_uv x_ut x_v,t+1 [tour length]

    Auto-selects penalty_a = B * max_distance * n * 5 when None, which
    ensures feasible states are strictly preferred over infeasible ones.
    """
    n = len(dist)
    max_dist = max(dist[i][j] for i in range(n) for j in range(n) if i != j)
    if penalty_a is None:
        penalty_a = penalty_b * max_dist * n * 5

    qubo: dict[tuple[str, str], float] = {}

    def var(i: int, t: int) -> str:
        return f"x_{i}_{t}"

    def add(u: str, v: str, w: float) -> None:
        key = (u, v) if u <= v else (v, u)
        qubo[key] = qubo.get(key, 0.0) + w

    # Each city visited exactly once: A*(1 - Σ_t x_it)^2
    # Expanding: A*(-Σ_t x_it + 2*Σ_{t<s} x_it*x_is) + const
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

    return qubo


# ---------------------------------------------------------------------------
# TSP tour helpers
# ---------------------------------------------------------------------------

def _tour_length(tour: list[int], dist: list[list[int]]) -> int:
    """Sum of edge lengths around the tour (returns to start)."""
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))


def _classical_optimal_tour(dist: list[list[int]]) -> tuple[int, list[int]]:
    """Brute-force optimal TSP tour (n ≤ ~10)."""
    n = len(dist)
    best_len = math.inf
    best_tour: list[int] = []
    # Fix city 0 as start to eliminate rotational symmetry.
    for perm in permutations(range(1, n)):
        tour = [0] + list(perm)
        length = _tour_length(tour, dist)
        if length < best_len:
            best_len = length
            best_tour = tour
    return int(best_len), best_tour


def _remap_to_logical(
    assignment: dict[str, int],
    embedding: dict[str, list[str]],
) -> dict[str, int]:
    """Translate a physical-qubit assignment back to logical variable names."""
    phys_to_log = {phys: log for log, [phys] in embedding.items()}
    return {phys_to_log.get(k, k): v for k, v in assignment.items()}


def _decode_tour(
    assignment: dict[str, int],
    n: int,
) -> list[int] | None:
    """Decode a logical-variable assignment into a TSP tour, or None if infeasible.

    Expects keys of the form ``x_{city}_{position}`` with integer city and
    position indices in range(n).
    """
    pos_to_city: dict[int, int] = {}
    city_to_pos: dict[int, int] = {}
    for i in range(n):
        for t in range(n):
            key = f"x_{i}_{t}"
            if assignment.get(key, 0) == 1:
                if t in pos_to_city or i in city_to_pos:
                    return None  # collision → infeasible
                pos_to_city[t] = i
                city_to_pos[i] = t
    if len(pos_to_city) != n:
        return None  # not all positions filled
    return [pos_to_city[t] for t in range(n)]


def _qubo_to_int_ising(
    qubo: dict[tuple[str, str], float],
) -> tuple[dict[int, float], dict[tuple[int, int], float], dict[str, int]]:
    """Convert a string-keyed QUBO to integer-indexed Ising (h, J, var_idx)."""
    variables = sorted({name for pair in qubo for name in pair})
    var_idx = {v: i for i, v in enumerate(variables)}
    h_str, J_str = _qubo_to_ising(qubo)
    h = {var_idx[v]: w for v, w in h_str.items()}
    J = {
        (var_idx[min(u, v)], var_idx[max(u, v)]): w
        for (u, v), w in J_str.items()
    }
    return h, J, var_idx


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="TSP eil51 LIMEN/IBM QPU benchmark")
    parser.add_argument(
        "--cities", type=int, default=4,
        help="Number of cities for the sub-problem (default 4 → 16 qubits)",
    )
    parser.add_argument(
        "--shots", type=int, default=1000,
        help="Number of circuit shots (default 1000)",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0,
        help="Seconds to wait for the QPU job before giving up (default 600)",
    )
    parser.add_argument(
        "--sim-only", action="store_true",
        help="Skip QPU, run simulator only even if IBM credentials are present",
    )
    parser.add_argument(
        "--backend", default=_BACKEND_NAME,
        help=f"IBM backend name to target (default {_BACKEND_NAME})",
    )
    parser.add_argument(
        "--algorithm", default="auto",
        choices=["auto", "qaoa", "exact"],
        help="Simulator algorithm: auto tries qaoa then falls back to exact (default auto)",
    )
    parser.add_argument(
        "--target-kappa", type=float, default=0.62,
        help=(
            "Stackelberg co-design target κ (default 0.62). "
            "For 1-to-1 gate-model embeddings the loop adjusts "
            "cost-Hamiltonian scale only (no chain-break terms), so κ "
            "plateaus near 0.62 regardless of iterations — this default "
            "matches that equilibrium. Use 0.85 to match the annealer convention "
            "(will not converge on gate-model problems)."
        ),
    )
    args = parser.parse_args()

    n_cities = args.cities
    shots = args.shots
    n_vars = n_cities * n_cities

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")
    qpu_enabled = bool(token and crn) and not args.sim_only

    print(f"=== TSP eil51 LIMEN Benchmark ===")
    print(f"Sub-problem : first {n_cities} cities of eil51 ({n_vars} QUBO variables)")
    print(f"Full problem: {EIL51_N_CITIES} cities, classical optimal = {EIL51_OPTIMAL_TOUR_LENGTH}")
    print(f"Backend     : {args.backend if qpu_enabled else 'AerSimulator (no IBM credentials)'}")
    print()

    # ── 1. Build sub-problem ─────────────────────────────────────────────
    coords = EIL51_COORDS[:n_cities]
    dist = _distance_matrix(coords)

    print(f"[1/6] Building TSP QUBO for {n_cities} cities ...")
    qubo = tsp_qubo(dist)
    print(f"      QUBO size: {len(qubo)} terms, {n_vars} variables")

    # ── 2. Classical brute-force optimum ─────────────────────────────────
    print(f"[2/6] Computing classical optimal tour by brute force ...")
    classical_opt_len, classical_opt_tour = _classical_optimal_tour(dist)
    print(f"      Classical optimal tour length: {classical_opt_len}")
    print(f"      Classical optimal tour       : {classical_opt_tour}")

    # ── 3. LIMEN compilation ─────────────────────────────────────────────
    print(f"[3/6] Compiling through LIMEN (lexicographic + Stackelberg) ...")
    graph = from_qubo_dict(qubo)
    encoding = compile_lexicographic(graph, default_hardware_graph(n_vars))
    print(f"      Logical vars  : {len(graph.variables)}")
    print(f"      Hardware nodes: {encoding.metadata['hardware_nodes']}")
    print(f"      Chain strength: {encoding.chain_strength:.4f}")

    vr = validate(encoding, runs=_VALIDATOR_RUNS, seed=42)
    print(f"      Validator confidence: {vr.confidence * 100:.1f}%")

    cd = run_codesign(
        encoding,
        target_kappa=args.target_kappa,
        max_iterations=_CODESIGN_ITERATIONS,
        runs_per_iteration=_CODESIGN_RUNS,
        seed=42,
    )
    print(f"      Stackelberg kappa: {cd.kappa:.4f} (converged={cd.converged})")
    encoding = cd.encoding  # use the co-design-optimised encoding

    # ── 4. CompilationCertificate ────────────────────────────────────────
    print(f"[4/6] Generating CompilationCertificate ...")
    target_h, target_J, var_idx = _qubo_to_int_ising(qubo)

    # The lexicographic compiler does a 1-to-1 embedding: logical var x_{i}_{t}
    # maps to physical qubit q_k where k = sorted position. Re-derive the
    # compiled Ising coefficients using the same integer index as target so
    # certify_ising can compute a meaningful diff.
    phys_to_logical: dict[str, str] = {
        phys: log for log, [phys] in encoding.embedding.items()
    }
    compiled_h: dict[int, float] = {}
    compiled_J: dict[tuple[int, int], float] = {}
    for (pi, pj), w in encoding.qubo.items():
        li = phys_to_logical.get(pi, pi)
        lj = phys_to_logical.get(pj, pj)
        ii = var_idx.get(li, 0)
        ij = var_idx.get(lj, 0)
        if pi == pj:
            compiled_h[ii] = compiled_h.get(ii, 0.0) + w / 2.0
        else:
            # QUBO off-diagonal → Ising J (w/4) and h contributions (w/4 each)
            compiled_h[ii] = compiled_h.get(ii, 0.0) + w / 4.0
            compiled_h[ij] = compiled_h.get(ij, 0.0) + w / 4.0
            key = (min(ii, ij), max(ii, ij))
            compiled_J[key] = compiled_J.get(key, 0.0) + w / 4.0

    cert = certify_ising(
        target_h=target_h,
        target_J=target_J,
        compiled_h=compiled_h,
        compiled_J=compiled_J,
        n_sites=n_vars,
        natively_realizable=True,
        notes=[
            f"eil51 sub-problem: {n_cities} cities, {n_vars} QUBO variables",
            f"Gate model (IBM ibm_kingston): all Ising terms natively realizable",
            f"Lexicographic 1-to-1 embedding: compiled == target up to index relabelling",
        ],
    )
    print(f"      L1 bound (op-norm error): {cert.l1_bound:.6f}")
    if cert.operator_norm is not None:
        print(f"      Exact operator norm     : {cert.operator_norm:.6f}")
    else:
        print(f"      Exact operator norm     : N/A (n_sites={n_vars} > 20)")

    # ── 5. QPU / simulator run ───────────────────────────────────────────
    sim_label = "AerSimulator (qaoa)" if args.algorithm != "exact" else "exact enumeration"
    run_label = args.backend if qpu_enabled else sim_label
    print(f"[5/6] Running QAOA on {run_label} ...")
    t0 = time.time()
    if qpu_enabled:
        assert token is not None and crn is not None  # narrowed above by qpu_enabled check
        qr = run_qiskit_qpu(
            encoding=encoding,
            token=token,
            crn=crn,
            backend_name=args.backend,
            shots=shots,
            reps=_REPS,
            cost_scale=cd.kappa,
            timeout=args.timeout,
        )
        job_id = qr.metadata.get("job_id")
        print(f"      Job id: {job_id}")
    else:
        # Try qaoa (needs qiskit_aer); fall back to exact enumeration.
        sim_algorithm = args.algorithm
        if sim_algorithm == "auto":
            try:
                from qiskit_aer.primitives import StatevectorSampler  # type: ignore[import]  # noqa: F401
                sim_algorithm = "qaoa"
            except (ModuleNotFoundError, ImportError):
                sim_algorithm = "exact"
                print(f"      StatevectorSampler unavailable — falling back to exact enumeration")
        qr = run_qiskit(
            encoding=encoding,
            num_shots=shots,
            algorithm=sim_algorithm,
            reps=_REPS,
            seed=42,
        )
        job_id = None
    elapsed = time.time() - t0
    print(f"      Elapsed: {elapsed:.1f}s")
    depth_str = (
        str(qr.circuit_depth)
        if qr.circuit_depth is not None
        else "N/A (exact enumeration — install qiskit-aer for QAOA circuit depth)"
    )
    print(f"      Circuit depth: {depth_str}")
    print(f"      Best QUBO energy: {qr.best_energy:.4f}")

    # ── 6. Tour interpretation and comparison ────────────────────────────
    print(f"[6/6] Interpreting results ...")
    # Remap physical qubit labels back to logical variable names (x_{i}_{t})
    # before decoding, since the lexicographic compiler renamed them.
    best_logical = _remap_to_logical(qr.best_assignment, encoding.embedding)
    qpu_tour = _decode_tour(best_logical, n_cities)
    if qpu_tour is not None:
        qpu_tour_len = _tour_length(qpu_tour, dist)
        approx_ratio = qpu_tour_len / classical_opt_len
        feasible = True
        print(f"      QPU best tour  : {qpu_tour}")
        print(f"      QPU tour length: {qpu_tour_len}")
        print(f"      Classical opt  : {classical_opt_len} (approx ratio {approx_ratio:.3f})")
    else:
        qpu_tour_len = None
        approx_ratio = None
        feasible = False
        print(f"      QPU best assignment is infeasible (constraint violation).")
        print(f"      Classical opt  : {classical_opt_len}")

    # Count feasible solutions across all samples.
    feasible_count = sum(
        1 for s in qr.samples
        if _decode_tour(_remap_to_logical(s, encoding.embedding), n_cities) is not None
    )
    feasible_rate = feasible_count / len(qr.samples) if qr.samples else 0.0
    print(f"      Feasible sample rate: {feasible_rate * 100:.1f}% ({feasible_count}/{len(qr.samples)})")

    # ── Output ───────────────────────────────────────────────────────────
    result = {
        "benchmark": "tsp_eil51",
        "date": time.strftime("%Y-%m-%d"),
        "full_problem": {
            "name": "eil51",
            "n_cities": EIL51_N_CITIES,
            "classical_optimal_tour_length": EIL51_OPTIMAL_TOUR_LENGTH,
        },
        "sub_problem": {
            "n_cities": n_cities,
            "n_qubo_variables": n_vars,
            "classical_optimal_tour_length": classical_opt_len,
            "classical_optimal_tour": classical_opt_tour,
        },
        "compilation": {
            "compiler": "lexicographic",
            "chain_strength": encoding.chain_strength,
            "validator_confidence": vr.confidence,
            "target_kappa": args.target_kappa,
            "kappa": cd.kappa,
            "kappa_converged": cd.converged,
            "kappa_iterations": cd.iterations,
        },
        "certificate": cert.to_dict(),
        "qpu_run": {
            "backend": args.backend if qpu_enabled else "aer_simulator",
            "shots": shots,
            "reps": _REPS,
            "job_id": job_id,
            "circuit_depth": qr.circuit_depth,
            "best_qubo_energy": qr.best_energy,
            "best_tour": qpu_tour,
            "best_tour_length": qpu_tour_len,
            "feasible": feasible,
            "approximation_ratio": approx_ratio,
            "feasible_sample_rate": feasible_rate,
            "elapsed_seconds": elapsed,
        },
    }

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"tsp_eil51_{stamp}.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # ── Markdown summary ──────────────────────────────────────────────────
    qpu_or_sim = args.backend if qpu_enabled else "AerSimulator"
    cert_norm = (
        f"{cert.operator_norm:.2e}" if cert.operator_norm is not None else f"≤ {cert.l1_bound:.2e} (L1)"
    )
    approx_str = f"{approx_ratio:.3f}" if approx_ratio is not None else "N/A (infeasible)"

    md = "\n".join([
        "# TSP eil51 LIMEN Benchmark",
        "",
        f"- **Date**: {time.strftime('%Y-%m-%d')}",
        f"- **Backend**: {qpu_or_sim}",
        f"- **Sub-problem**: first {n_cities} cities of eil51 ({n_vars} QUBO variables)",
        f"- **Full eil51 reference**: {EIL51_N_CITIES} cities, classical optimal = {EIL51_OPTIMAL_TOUR_LENGTH}",
        f"- **QAOA**: p={_REPS}, β=γ=0.1, {shots} shots",
        "",
        "## Compilation",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Compiler | lexicographic (1-to-1 embedding) |",
        f"| QUBO variables | {n_vars} |",
        f"| Chain strength | {encoding.chain_strength:.4f} |",
        f"| Validator confidence | {vr.confidence * 100:.1f}% |",
        f"| Target κ | {args.target_kappa:.2f} |",
        f"| Stackelberg κ | {cd.kappa:.4f} (converged={cd.converged}, iters={cd.iterations}) |",
        "",
        "## Compilation Certificate",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| n_sites | {cert.n_sites} |",
        f"| L1 bound (||H_target - H_compiled||_op ≤) | {cert.l1_bound:.2e} |",
        f"| Exact operator norm | {cert_norm} |",
        f"| Max linear error | {cert.max_linear_error:.2e} |",
        f"| Max quadratic error | {cert.max_quadratic_error:.2e} |",
        f"| Natively realizable | {cert.natively_realizable} |",
        "",
        "## QPU Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Circuit depth | {depth_str} |",
        f"| Best QUBO energy | {qr.best_energy:.4f} |",
        f"| Best tour | {qpu_tour} |",
        f"| Best tour length | {qpu_tour_len if qpu_tour_len is not None else 'N/A (infeasible)'} |",
        f"| Classical sub-problem optimal | {classical_opt_len} |",
        f"| Approximation ratio | {approx_str} |",
        f"| Feasible sample rate | {feasible_rate * 100:.1f}% |",
        "",
        "## Notes",
        "",
    ] + [f"- {n}" for n in cert.notes] + [
        "",
        f"*Raw JSON: `{out_json.name}`*",
    ]) + "\n"

    out_md = pathlib.Path(__file__).resolve().parent / "TSP_EIL51_RESULTS.md"
    out_md.write_text(md, encoding="utf-8")

    print()
    print(md)
    print(f"JSON   : {out_json}")
    print(f"Markdown: {out_md}")


if __name__ == "__main__":
    main()
