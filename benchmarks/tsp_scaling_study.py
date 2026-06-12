# Copyright 2026 LIMEN Contributors — Apache 2.0
"""TSP scaling study: find where LIMEN hits its limits.

Sweeps n=4..12 cities (16..144 qubits) on Aer simulator, measuring:
  - Compilation time
  - Feasible sample rate
  - Approximation ratio (vs brute-force classical optimal)
  - Circuit depth

ibm_kingston hard ceiling: 156 qubits → max 12 cities (144 qubits).
Brute-force classical optimal is only tractable up to ~10 cities (n! search).
Above that, nearest-neighbour heuristic is used as the reference.

Usage:
    python benchmarks/tsp_scaling_study.py
    python benchmarks/tsp_scaling_study.py --max-cities 10 --shots 2000
    python benchmarks/tsp_scaling_study.py --qpu   # hit real ibm_kingston at each size

Results → results/tsp_scaling_<timestamp>.json + benchmarks/TSP_SCALING_RESULTS.md
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

# ── EIL51 coordinates ────────────────────────────────────────────────────────

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

IBM_KINGSTON_QUBITS = 156
BRUTE_FORCE_LIMIT   = 10   # n! search only feasible up to here

# ── Geometry helpers ─────────────────────────────────────────────────────────

def _euc_2d(a: tuple[int, int], b: tuple[int, int]) -> int:
    return int(math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + 0.5)

def _distance_matrix(coords: list[tuple[int, int]]) -> list[list[int]]:
    n = len(coords)
    return [[_euc_2d(coords[i], coords[j]) for j in range(n)] for i in range(n)]

def _tour_length(tour: list[int], dist: list[list[int]]) -> int:
    n = len(tour)
    return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))

def _classical_optimal(dist: list[list[int]]) -> tuple[int, list[int]]:
    """Brute-force for n <= BRUTE_FORCE_LIMIT, nearest-neighbour heuristic above."""
    n = len(dist)
    if n <= BRUTE_FORCE_LIMIT:
        best, best_tour = math.inf, []
        for perm in permutations(range(1, n)):
            tour = [0] + list(perm)
            tl = _tour_length(tour, dist)
            if tl < best:
                best, best_tour = tl, tour
        return int(best), best_tour
    else:
        # Nearest-neighbour heuristic (upper bound, not optimal)
        visited = [False] * n
        tour = [0]
        visited[0] = True
        for _ in range(n - 1):
            cur = tour[-1]
            nxt = min((j for j in range(n) if not visited[j]),
                      key=lambda j: dist[cur][j])
            tour.append(nxt)
            visited[nxt] = True
        return _tour_length(tour, dist), tour

# ── QUBO builder ─────────────────────────────────────────────────────────────

def _tsp_qubo(dist: list[list[int]]) -> dict[tuple[str, str], float]:
    n = len(dist)
    max_dist = max(dist[i][j] for i in range(n) for j in range(n) if i != j)
    penalty_a = max_dist * n * 5.0

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
                add(var(u, t), var(v, (t + 1) % n), dist[u][v])

    return qubo

# ── Tour decoder ─────────────────────────────────────────────────────────────

def _remap_to_logical(assignment: dict[str, int],
                      embedding: dict[str, list[str]]) -> dict[str, int]:
    phys_to_log = {phys: log for log, [phys] in embedding.items()}
    return {phys_to_log.get(k, k): v for k, v in assignment.items()}

def _decode_tour(assignment: dict[str, int], n: int) -> list[int] | None:
    pos_to_city: dict[int, int] = {}
    city_to_pos: dict[int, int] = {}
    for i in range(n):
        for t in range(n):
            if assignment.get(f"x_{i}_{t}", 0) == 1:
                if t in pos_to_city or i in city_to_pos:
                    return None
                pos_to_city[t] = i
                city_to_pos[i] = t
    return [pos_to_city[t] for t in range(n)] if len(pos_to_city) == n else None

# ── Single size benchmark ─────────────────────────────────────────────────────

def _run_one(n_cities: int, shots: int, use_qpu: bool,
             token: str | None, crn: str | None,
             sim_limit: int = 20) -> dict:
    from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
    from limen.analog.certificate import certify_ising
    from limen.backends.qiskit_backend import _qubo_to_ising, run_qiskit, run_qiskit_qpu
    from limen.codesign.solver import run_codesign
    from limen.validator.validator import validate

    n_vars = n_cities * n_cities
    coords = EIL51_COORDS[:n_cities]
    dist   = _distance_matrix(coords)

    # Classical reference
    classical_len, classical_tour = _classical_optimal(dist)
    ref_type = "brute_force" if n_cities <= BRUTE_FORCE_LIMIT else "nearest_neighbour"

    # QUBO + compile
    t_compile = time.time()
    qubo     = _tsp_qubo(dist)
    graph    = from_qubo_dict(qubo)
    encoding = compile_lexicographic(graph, default_hardware_graph(n_vars))
    vr       = validate(encoding, runs=500, seed=42)
    cd       = run_codesign(encoding, target_kappa=0.85,
                            max_iterations=10, runs_per_iteration=300, seed=42)
    encoding = cd.encoding
    t_compile = time.time() - t_compile

    # Certificate
    variables  = sorted({name for pair in qubo for name in pair})
    var_idx    = {v: i for i, v in enumerate(variables)}
    h_str, J_str = _qubo_to_ising(qubo)
    target_h   = {var_idx[v]: w for v, w in h_str.items()}
    target_J   = {(var_idx[min(u,v)], var_idx[max(u,v)]): w for (u,v),w in J_str.items()}
    phys_map   = {phys: var_idx.get(log, 0)
                  for log, [phys] in encoding.embedding.items()}
    h_str2, J_str2 = _qubo_to_ising(encoding.qubo)
    comp_h     = {phys_map.get(v, v): w for v, w in
                  {var_idx.get(v, 0): w for v, w in h_str2.items()}.items()}
    comp_J     = {(phys_map.get(u,u), phys_map.get(v,v)): w
                  for (u,v),w in J_str2.items()}
    cert = certify_ising(
        target_h=target_h, target_J=target_J,
        compiled_h=comp_h, compiled_J=comp_J,
        n_sites=n_vars, natively_realizable=True,
    )

    # QPU or simulator run — skipped when n_vars exceeds sim_limit
    sim_skipped = not use_qpu and n_vars > sim_limit
    t_run = 0.0
    qr = None
    backend_used = "skipped"

    if sim_skipped:
        pass
    elif use_qpu and token and crn:
        t_run = time.time()
        qr = run_qiskit_qpu(encoding=encoding, token=token, crn=crn,
                             backend_name="ibm_kingston", shots=shots,
                             reps=1, cost_scale=cd.kappa)
        backend_used = "ibm_kingston"
        t_run = time.time() - t_run
    else:
        t_run = time.time()
        try:
            qr = run_qiskit(encoding=encoding, num_shots=shots,
                            algorithm="qaoa", reps=1, seed=42)
            backend_used = "aer_simulator"
        except ImportError:
            qr = run_qiskit(encoding=encoding, num_shots=shots,
                            algorithm="exact", reps=1, seed=42)
            backend_used = "exact_enumeration"
        t_run = time.time() - t_run

    # Decode
    feasible_tours = []
    if qr is not None:
        for s in qr.samples:
            logical = _remap_to_logical(s, encoding.embedding)
            tour    = _decode_tour(logical, n_cities)
            if tour:
                feasible_tours.append((_tour_length(tour, dist), tour))

    feasible_count = len(feasible_tours)
    feasible_rate  = feasible_count / len(qr.samples) if qr is not None and qr.samples else None
    best_qpu_len   = min(feasible_tours, key=lambda x: x[0])[0] if feasible_tours else None
    approx_ratio   = best_qpu_len / classical_len if best_qpu_len is not None else None

    return {
        "n_cities":         n_cities,
        "n_qubits":         n_vars,
        "qpu_ceiling":      n_vars <= IBM_KINGSTON_QUBITS,
        "classical_len":    classical_len,
        "classical_tour":   classical_tour,
        "classical_ref":    ref_type,
        "compile_seconds":  round(t_compile, 2),
        "run_seconds":      round(t_run, 2),
        "backend":          backend_used,
        "circuit_depth":    qr.circuit_depth if qr is not None else None,
        "chain_strength":   encoding.chain_strength,
        "validator_conf":   round(vr.confidence, 3),
        "kappa":            round(cd.kappa, 4),
        "kappa_converged":  cd.converged,
        "op_norm_error":    cert.operator_norm if cert.operator_norm is not None else cert.l1_bound,
        "total_shots":      len(qr.samples) if qr is not None else 0,
        "feasible_shots":   feasible_count,
        "feasible_rate":    round(feasible_rate, 4) if feasible_rate is not None else None,
        "best_qpu_length":  best_qpu_len,
        "approximation_ratio": round(approx_ratio, 4) if approx_ratio else None,
        "found_optimal":    best_qpu_len == classical_len if best_qpu_len else False,
        "sim_skipped":      sim_skipped,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TSP scaling study for LIMEN")
    parser.add_argument("--min-cities", type=int, default=4)
    parser.add_argument("--max-cities", type=int, default=8,
                        help="Max cities to test (QPU ceiling = 12, default 8)")
    parser.add_argument("--shots",      type=int, default=1000)
    parser.add_argument("--qpu",        action="store_true",
                        help="Run on real ibm_kingston (costs QPU credits)")
    parser.add_argument("--sim-limit",  type=int, default=20,
                        help="Max qubits for Aer simulation; larger problems "
                             "compile but skip the quantum run (default 20)")
    args = parser.parse_args()

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn   = os.environ.get("IBM_QUANTUM_CRN")

    qpu_ceiling = int(math.floor(math.sqrt(IBM_KINGSTON_QUBITS)))  # 12

    print("=" * 60)
    print("  TSP SCALING STUDY — LIMEN × ibm_kingston")
    print("=" * 60)
    print(f"  Cities range : {args.min_cities} → {args.max_cities}")
    print(f"  Qubits range : {args.min_cities**2} → {args.max_cities**2}")
    print(f"  QPU ceiling  : {qpu_ceiling} cities ({qpu_ceiling**2} qubits)")
    print(f"  Shots        : {args.shots}")
    print(f"  Sim limit    : {args.sim_limit} qubits (compile-only above this)")
    print(f"  Backend      : {'ibm_kingston (QPU)' if args.qpu else 'Aer / exact fallback'}")
    print()

    results = []
    for n in range(args.min_cities, args.max_cities + 1):
        n_vars = n * n
        within_ceiling = n_vars <= IBM_KINGSTON_QUBITS
        ceiling_str    = "OK" if within_ceiling else "OVER QPU LIMIT"
        print(f"[{n} cities / {n_vars} qubits] {ceiling_str}")

        if not within_ceiling and args.qpu:
            print(f"  Skipping QPU run — {n_vars} qubits exceeds ibm_kingston ({IBM_KINGSTON_QUBITS})")
            results.append({"n_cities": n, "n_qubits": n_vars,
                            "qpu_ceiling": False, "skipped": True})
            continue

        try:
            r = _run_one(n, args.shots, args.qpu, token, crn, sim_limit=args.sim_limit)
            results.append(r)
            if r.get("sim_skipped"):
                print(f"  compile={r['compile_seconds']}s  run=skipped (>{args.sim_limit}q sim limit)")
            else:
                status = "✓ OPTIMAL" if r["found_optimal"] else (
                    f"ratio={r['approximation_ratio']}" if r["approximation_ratio"] else "no feasible"
                )
                feasible_pct = r['feasible_rate'] * 100 if r['feasible_rate'] is not None else 0.0
                print(f"  compile={r['compile_seconds']}s  run={r['run_seconds']}s  "
                      f"feasible={feasible_pct:.1f}%  {status}")
        except MemoryError as e:
            print(f"  OOM — statevector too large at {n_vars} qubits: {e}")
            results.append({"n_cities": n, "n_qubits": n_vars,
                            "qpu_ceiling": within_ceiling, "error": "OOM"})
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"n_cities": n, "n_qubits": n_vars,
                            "qpu_ceiling": within_ceiling, "error": str(e)})
        print()

    # ── Save JSON ────────────────────────────────────────────────────────────
    out_dir  = pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp    = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"tsp_scaling_{stamp}.json"
    out_json.write_text(json.dumps({
        "study":    "tsp_scaling",
        "date":     time.strftime("%Y-%m-%d"),
        "shots":    args.shots,
        "backend":  "ibm_kingston" if args.qpu else "aer_simulator",
        "results":  results,
    }, indent=2, sort_keys=True), encoding="utf-8")

    # ── Markdown table ───────────────────────────────────────────────────────
    header = (
        "# TSP Scaling Study — LIMEN × ibm_kingston\n\n"
        f"- **Date**: {time.strftime('%Y-%m-%d')}\n"
        f"- **Shots per run**: {args.shots}\n"
        f"- **Backend**: {'ibm_kingston (QPU)' if args.qpu else 'Aer simulator'}\n"
        f"- **QPU ceiling**: {qpu_ceiling} cities ({qpu_ceiling**2} qubits / 156 available)\n\n"
        "| Cities | Qubits | Within QPU | Classical Opt | Feasible% | "
        "Best QPU | Approx Ratio | Found Optimal | Compile(s) |\n"
        "|--------|--------|------------|---------------|-----------|"
        "----------|--------------|---------------|------------|\n"
    )
    rows = []
    for r in results:
        if r.get("skipped") or r.get("error"):
            tag = r.get("error", "skipped")
            rows.append(f"| {r['n_cities']} | {r['n_qubits']} | "
                        f"{'✓' if r.get('qpu_ceiling') else '✗'} | — | — | — | — | {tag} | — |")
        elif r.get("sim_skipped"):
            rows.append(
                f"| {r['n_cities']} | {r['n_qubits']} | "
                f"{'✓' if r['qpu_ceiling'] else '✗'} | "
                f"{r['classical_len']} ({r['classical_ref'][:2]}) | "
                f"— | — | — | sim_skipped | {r['compile_seconds']} |"
            )
        else:
            feasible_pct = r['feasible_rate'] * 100 if r['feasible_rate'] is not None else 0.0
            rows.append(
                f"| {r['n_cities']} | {r['n_qubits']} | "
                f"{'✓' if r['qpu_ceiling'] else '✗'} | "
                f"{r['classical_len']} ({r['classical_ref'][:2]}) | "
                f"{feasible_pct:.1f}% | "
                f"{r['best_qpu_length'] or 'none'} | "
                f"{r['approximation_ratio'] or 'N/A'} | "
                f"{'✓' if r['found_optimal'] else '✗'} | "
                f"{r['compile_seconds']} |"
            )

    md = header + "\n".join(rows) + f"\n\n*Raw JSON: `{out_json.name}`*\n"
    out_md = pathlib.Path(__file__).resolve().parent / "TSP_SCALING_RESULTS.md"
    out_md.write_text(md, encoding="utf-8")

    # ── Print summary ────────────────────────────────────────────────────────
    print()
    print(md)
    print(f"JSON    : {out_json}")
    print(f"Markdown: {out_md}")

    # ── Key findings ─────────────────────────────────────────────────────────
    good   = [r for r in results if r.get("found_optimal")]
    failed = [r for r in results if not r.get("found_optimal") and not r.get("error") and not r.get("skipped")]
    errors = [r for r in results if r.get("error")]

    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if good:
        print(f"  Optimal found   : {[r['n_cities'] for r in good]} cities")
    if failed:
        print(f"  Sub-optimal     : {[r['n_cities'] for r in failed]} cities")
    if errors:
        print(f"  Errors/OOM      : {[r['n_cities'] for r in errors]} cities")
    print(f"  QPU hard ceiling: {qpu_ceiling} cities ({qpu_ceiling**2} qubits)")
    print()

if __name__ == "__main__":
    main()
