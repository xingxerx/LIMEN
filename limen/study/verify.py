# Copyright (C) 2026 xingxerx
#
# Licensed under the Elastic License 2.0. See the LICENSE file in the
# repository root for the full terms.

"""Tier 2 certificate check against the exact answer.

A Tier 2 plan carries surface-code ECC settings and a predicted logical
error bound. Executing that plan offline (RouteRequest.offline=True keeps
every planning decision but runs on the local simulator) and comparing
with the tier-0 ideal answer tells you whether the certified solution is
the true optimum and how the cert's own success probability relates to
the ideal one. The exact side is only available while the problem fits
the tier-0 cap, which is why this check is bounded at 30 qubits.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from limen.frontends.maxcut import build_max_cut_qubo, random_regular_edges
from limen.router import RouteRequest, Tier, route
from limen.study._common import StudyContext, Timer, baseline_backend, write_record


def verify_tier2(
    qubo: dict[tuple[str, str], float],
    *,
    ctx: StudyContext,
    fidelity_target: float = 0.99,
    credit_budget: float = 50.0,
) -> dict[str, Any]:
    from limen.pipeline import run_pipeline

    # Tier 2 plan with ECC, executed on the exact local baseline.
    req = RouteRequest(
        qubo,
        fidelity_target=fidelity_target,
        credit_budget=credit_budget,
        force_tier=Tier.HW_CERTIFIED,
        offline=True,
    )
    plan = route(req)
    n = plan.n_vars
    base_backend, base_name = baseline_backend(n, ctx)
    tier2_kwargs = dict(plan.pipeline_kwargs)
    tier2_kwargs.update(backend=base_backend, qpu_backend_name=base_name)
    with Timer() as t2:
        tier2 = run_pipeline(qubo, **tier2_kwargs)

    # Tier 0 ideal: same simulator, no ECC term.
    ideal_kwargs = dict(tier2_kwargs)
    ideal_kwargs["encode_logical"] = False
    for k in ("distance", "physical_error_rate", "measured_logical_error"):
        ideal_kwargs.pop(k, None)
    with Timer() as t0:
        ideal = run_pipeline(qubo, **ideal_kwargs)

    record = {
        "study": "verify_tier2",
        "context": ctx.summary(),
        "n_vars": n,
        "plan": plan.to_dict(),
        "baseline": {"backend": base_backend, "qpu_backend_name": base_name, "device": tier2.metadata.get("device")},
        "tier2_seconds": t2.seconds,
        "ideal_seconds": t0.seconds,
        "tier2_solution": tier2.solution,
        "ideal_solution": ideal.solution,
        "solutions_agree": tier2.solution == ideal.solution,
        "tier2_energy": tier2.energy,
        "ideal_energy": ideal.energy,
        "classical_energy": ideal.classical_energy,
        "tier2_is_optimal": tier2.is_optimal,
        "ideal_is_optimal": ideal.is_optimal,
        "tier2_success_probability": tier2.success_probability,
        "ideal_success_probability": ideal.success_probability,
        "ecc": {
            "distance": tier2.distance,
            "physical_error_rate": tier2.physical_error_rate,
            "logical_error_rate": tier2.logical_error_rate,
            "aggregate_logical_error_rate": tier2.aggregate_logical_error_rate,
            "predicted_logical_error_bound": tier2.predicted_logical_error_bound,
            "n_logical_qubits": tier2.n_logical_qubits,
        },
        "notes": list(tier2.notes),
    }
    record["path"] = write_record(ctx, "verify", f"tier2-n{n}", record)
    return record


def verify_demo(n_nodes: int = 12, *, ctx: StudyContext, seed: int = 0) -> dict[str, Any]:
    qubo = build_max_cut_qubo(random_regular_edges(n_nodes, 3, seed))
    return verify_tier2(qubo, ctx=ctx)
