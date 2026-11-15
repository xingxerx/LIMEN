# Copyright (C) 2026 xingxerx
#
# Licensed under the Elastic License 2.0. See the LICENSE file in the
# repository root for the full terms.

"""Circuit cutting at tier-0 partition sizes.

Two measurements, both built on limen.cutting as it already exists:

* validity: cut a problem that still fits the exact simulator into pieces
  of ``max_sub`` qubits, reconstruct every <Z_i> through the cutting path,
  and compare against the exact marginals from the uncut circuit. This is
  the only way to put a number on cutting error, and it is only possible
  while the uncut circuit fits, i.e. up to the tier-0 cap.
* scaling: run limen.pipeline.run_cut_route_request offline on a problem
  wider than the cap, with pieces sized to the cap, and record partitions,
  cuts, wall time and the decoded energy against a classical local-search
  baseline. No exact answer exists here; the point is cost and sanity.
"""

from __future__ import annotations

import random
from typing import Any

from limen.frontends.maxcut import build_max_cut_qubo, count_cut_edges, random_regular_edges, ring_edges
from limen.frontends.pyqubo import from_qubo_dict
from limen.gates.qaoa import compile_qaoa, variable_order
from limen.router import LOCAL_GPU_BACKEND_NAME, BackendProfile, RouteRequest, Tier, route
from limen.study._common import StudyContext, Timer, exact_prob_fn, write_record, z_marginals


def _dispatch_backend(ctx: StudyContext) -> str:
    return LOCAL_GPU_BACKEND_NAME if ctx.local_gpu else "aer_simulator"


def random_field_ring(n: int, seed: int = 0) -> dict[tuple[str, str], float]:
    """Ring of +1 couplings with random on-site fields in +-[0.5, 1.5].

    Max-Cut QUBOs are symmetric under a global bit flip, so every exact
    <Z_i> is 0 and a marginal comparison says nothing (and threshold
    decoding of near-zero marginals is a coin toss, which is a separate
    weakness of run_cut_route_request on symmetric problems). The fields
    here break that symmetry so the marginals are O(1) and a cutting error
    is measurable.
    """
    rng = random.Random(seed)
    qubo: dict[tuple[str, str], float] = {}
    for i in range(n):
        a, b = f"v{i}", f"v{(i + 1) % n}"
        a, b = sorted((a, b))
        qubo[(a, b)] = qubo.get((a, b), 0.0) + 1.0
        qubo[(f"v{i}", f"v{i}")] = rng.choice((-1.0, 1.0)) * rng.uniform(0.5, 1.5)
    return qubo


def cutting_validity(
    *,
    ctx: StudyContext,
    n_vars: int = 12,
    max_sub: int = 6,
    shots: int = 8000,
    seed: int = 0,
    problem: str = "random_field_ring",
) -> dict[str, Any]:
    """Cutting-reconstructed <Z_i> vs exact <Z_i> on an uncut circuit that fits."""
    from limen.cutting import reconstruct_from_results, reconstruct_z_marginals_via_cutting
    from limen.cutting.local_dispatch import run_cut_circuit_locally
    from limen.pipeline import _qaoa_params_for_oversized

    if n_vars > ctx.exact_max_qubits:
        raise ValueError(f"validity needs an exact baseline: {n_vars}q > {ctx.exact_max_qubits}q")
    if problem == "maxcut":
        edges = random_regular_edges(n_vars, 3, seed)
        qubo = build_max_cut_qubo(edges)
    else:
        edges = ring_edges(n_vars)
        qubo = random_field_ring(n_vars, seed)
    graph = from_qubo_dict(qubo)
    order = variable_order(graph)
    n = len(order)
    params, note = _qaoa_params_for_oversized(qubo, graph, n)
    circuit = compile_qaoa(graph, [params["gamma"]], [params["beta"]])

    with Timer() as t_exact:
        exact = z_marginals(exact_prob_fn(n, ctx)(circuit), n)

    backend_name = _dispatch_backend(ctx)
    dispatch_meta: dict[str, Any] = {}

    def dispatch_fn(cut_plan: Any) -> Any:
        res = run_cut_circuit_locally(cut_plan, shots=shots, backend_name=backend_name)
        dispatch_meta.update(res.metadata)
        return res

    with Timer() as t_cut:
        rec = reconstruct_z_marginals_via_cutting(
            circuit, list(range(n)), max_sub, dispatch_fn, reconstruct_from_results
        )

    errors = {q: abs(rec.marginals[q] - exact[q]) for q in range(n)}
    sign_agree = sum(1 for q in range(n) if (rec.marginals[q] >= 0) == (exact[q] >= 0))
    degenerate = all(abs(v) < 1e-9 for v in exact.values())
    record = {
        "study": "cutting_validity",
        "context": ctx.summary(),
        "problem": {"class": problem, "n_vars": n, "edges": len(edges), "seed": seed},
        "qaoa_params": params,
        "param_note": note,
        "degenerate_symmetric": degenerate,
        "note": (
            "every exact <Z_i> is 0 (global bit-flip symmetry): the comparison "
            "only measures sampling noise and threshold decoding is a coin toss here"
            if degenerate else "exact marginals are nonzero; errors measure cutting + sampling"
        ),
        "max_subcircuit_qubits": max_sub,
        "num_partitions": rec.num_partitions,
        "num_cuts": rec.num_cuts,
        "shots_per_subexperiment": shots,
        "dispatch": dispatch_meta,
        "exact_seconds": t_exact.seconds,
        "cutting_seconds": t_cut.seconds,
        "max_abs_marginal_error": max(errors.values()),
        "mean_abs_marginal_error": sum(errors.values()) / n,
        "marginal_sign_agreement": f"{sign_agree}/{n}",
        "marginals_exact": exact,
        "marginals_cut": rec.marginals,
    }
    record["path"] = write_record(ctx, "cutting", f"validity-n{n}-sub{max_sub}", record)
    return record


# local search reference lives in limen.study._common (shared with bench).
from limen.study._common import local_search_energy as _local_search_energy


def cutting_scaling(
    *,
    ctx: StudyContext,
    n_vars: int = 40,
    max_sub: int | None = None,
    shots: int = 4000,
    seed: int = 0,
) -> dict[str, Any]:
    """run_cut_route_request offline on a problem wider than the local cap."""
    from limen.pipeline import run_cut_route_request

    max_sub = max_sub or (ctx.gpu_cap if ctx.local_gpu else 20)
    edges = random_regular_edges(n_vars, 3, seed)
    qubo = build_max_cut_qubo(edges)
    order = variable_order(from_qubo_dict(qubo))
    # A plan whose backend is exactly max_sub wide forces cutting into
    # tier-0-sized pieces; the sim profile stands in for "the local GPU".
    fleet = (
        BackendProfile("cut_target", "sim", max_sub, 0.0, validated=True),
        BackendProfile("ibm_fez", "ibm", 156, 0.002, validated=True),
    )
    req = RouteRequest(qubo, fidelity_target=0.9, credit_budget=0.0, force_tier=Tier.SIM)
    plan = route(req, fleet)
    if not plan.use_cutting:
        raise ValueError(f"{len(order)} vars fit {max_sub}q; nothing to cut")

    with Timer() as t:
        cert = run_cut_route_request(qubo, plan, offline=True, shots=shots)
    baseline_e = _local_search_energy(qubo, order, seed)
    record = {
        "study": "cutting_scaling",
        "context": ctx.summary(),
        "problem": {"class": "maxcut", "n_vars": len(order), "edges": len(edges), "seed": seed},
        "max_subcircuit_qubits": max_sub,
        "num_partitions": cert.num_partitions,
        "num_cuts": cert.num_cuts,
        "shots_per_subexperiment": shots,
        "seconds": t.seconds,
        "decoded_classical_energy": cert.decoded_classical_energy,
        "decoded_cut_edges": count_cut_edges(cert.solution, edges),
        "local_search_energy": baseline_e,
        "local_search_cut_edges": int(-baseline_e),
        "reconstructed_expected_energy": cert.reconstructed_expected_energy,
        "notes": list(cert.notes),
    }
    record["path"] = write_record(ctx, "cutting", f"scaling-n{len(order)}-sub{max_sub}", record)
    return record
