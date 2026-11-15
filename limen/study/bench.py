# Copyright (C) 2026 xingxerx
#
# Licensed under the Elastic License 2.0. See the LICENSE file in the
# repository root for the full terms.

"""Recognisable benchmark instances through run_route_request on tier 0.

TSP comes from the Discovery Loop's Problem schema and eil51 encoder
(limen.frontends.tsp), Max-Cut from limen.frontends.maxcut. Each result
records approximation ratio against a brute-force or known optimum where
one exists, wall time, and which simulator actually ran.
"""

from __future__ import annotations

from typing import Any

from limen.frontends.maxcut import build_max_cut_qubo, count_cut_edges, random_regular_edges, ring_edges, ring_optimum
from limen.frontends.pyqubo import from_qubo_dict
from limen.frontends.tsp import eil51_prefix_problem, encode_tsp, score_assignment, to_route_request
from limen.router import RouteRequest
from limen.study._common import StudyContext, Timer, write_record


def _run(req: RouteRequest, ctx: StudyContext) -> tuple[Any, float]:
    from limen.pipeline import run_route_request

    with Timer() as t:
        cert = run_route_request(req, memory=None, emit_report=True, local_gpu_first=ctx.local_gpu)
    return cert, t.seconds


def bench_tsp(n_cities: int, *, ctx: StudyContext) -> dict[str, Any]:
    problem = encode_tsp(eil51_prefix_problem(n_cities))
    req = to_route_request(problem, credit_budget=0.0)
    cert, seconds = _run(req, ctx)
    score = score_assignment(problem, cert.solution)
    report = cert.metadata.get("route_report", {})
    return {
        "problem": problem.id,
        "n_vars": len(problem.variables),
        "backend": report.get("backend"),
        "tier": report.get("tier"),
        "device": cert.metadata.get("device"),
        "seconds": seconds,
        "energy": cert.energy,
        "is_optimal": cert.is_optimal,
        "tour": score["tour"],
        "tour_length": score["tour_length"],
        "known_optimum": score["known_optimum"],
        "approx_ratio": score["approx_ratio"],
        "feasible": score["tour"] is not None,
        "note": (
            None if score["tour"] is not None else
            "argmax bitstring violates the TSP one-hot constraints (p=1 QAOA, "
            "penalty QUBO); ratio undefined. Known weakness: feasible-sample "
            "rate was 0.15% on ibm_kingston at 16 vars (benchmarks/TSP_EIL51_RESULTS.md)"
        ),
    }


def bench_maxcut(n_nodes: int, *, ctx: StudyContext, kind: str = "regular3", seed: int = 0) -> dict[str, Any]:
    from limen.qubo_spectrum import qubo_energy_spectrum

    edges = ring_edges(n_nodes) if kind == "ring" else random_regular_edges(n_nodes, 3, seed)
    qubo = build_max_cut_qubo(edges)
    req = RouteRequest(qubo, fidelity_target=0.9, credit_budget=0.0)
    cert, seconds = _run(req, ctx)
    cut = count_cut_edges(cert.solution, edges)
    reference = "exact"
    if kind == "ring":
        optimum: int | None = ring_optimum(n_nodes)
    else:
        spectrum = qubo_energy_spectrum(qubo)  # None past 20 vars
        optimum = int(-spectrum.best_energy) if spectrum is not None else None
    if optimum is None:
        # No exact optimum past 20 vars. Best-of-restarts local search is a
        # lower bound on the true max cut, so the ratio against it is an
        # upper bound on the true approximation ratio; the record says so.
        from limen.gates.qaoa import variable_order
        from limen.study._common import local_search_energy

        order = variable_order(from_qubo_dict(qubo))
        optimum = int(round(-local_search_energy(qubo, order, seed=seed, restarts=50)))
        reference = "local_search_lower_bound"
    report = cert.metadata.get("route_report", {})
    return {
        "problem": f"maxcut/{kind}/n={n_nodes}/seed={seed}",
        "n_vars": n_nodes,
        "edges": len(edges),
        "backend": report.get("backend"),
        "tier": report.get("tier"),
        "device": cert.metadata.get("device"),
        "seconds": seconds,
        "energy": cert.energy,
        "cut_edges": cut,
        "known_optimum": optimum,
        "reference": reference,
        "approx_ratio": (cut / optimum) if optimum else None,
    }


def run_bench(
    *,
    ctx: StudyContext,
    tsp_cities: tuple[int, ...] = (3, 4),
    maxcut_nodes: tuple[int, ...] = (12, 20),
    maxcut_kind: str = "regular3",
    on_row: Any = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    def _guard(label: str, n_vars: int, fn: Any) -> None:
        # One instance failing (typically: wider than any simulator can hold
        # right now, e.g. 30 vars with the GPU cap at 29 because another
        # process holds VRAM) must not lose the rows already measured.
        try:
            rows.append(fn())
        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
            rows.append({
                "problem": label, "n_vars": n_vars, "backend": None, "tier": None, "device": None,
                "seconds": 0.0, "approx_ratio": None, "error": f"{type(exc).__name__}: {exc}",
            })
        if on_row is not None:
            on_row(rows[-1])

    for n in tsp_cities:
        _guard(f"tsp/eil51/n={n}", n * n, lambda n=n: bench_tsp(n, ctx=ctx))
    for n in maxcut_nodes:
        _guard(f"maxcut/{maxcut_kind}/n={n}", n, lambda n=n: bench_maxcut(n, ctx=ctx, kind=maxcut_kind))
    record = {"study": "bench", "context": ctx.summary(), "rows": rows}
    record["path"] = write_record(ctx, "bench", "bench", record)
    return record
