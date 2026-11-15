# Copyright (C) 2026 xingxerx
#
# Licensed under the Elastic License 2.0. See the LICENSE file in the
# repository root for the full terms.

"""Hardware gap: ideal (tier 0) vs. measured (QPU) success probability.

Generalises the ``_certify`` step of examples/router_tier2_kingston_fetch.py
so the baseline can be the GPU statevector instead of the pure-Python one,
which is what lets the gap be measured at 20-30 qubits, where a Heron's
noise actually registers. The record written for a real job has exactly
the shape limen.router.history._scan_router_tier2 already reads (top-level
``plan``, ``measured_success_deficit``, ``timestamps``), so the gap flows
into informed_fleet as ``measured_logical_error`` with no new scanner,
and into RouterMemory via record_route_outcome.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from limen.router import DEFAULT_FLEET, RoutePlan, RouteRequest, RouterMemory, Tier, route
from limen.study._common import StudyContext, baseline_backend, timestamp, write_record


def hardware_gap(
    qubo: dict[tuple[str, str], float],
    plan: RoutePlan,
    hw_counts: dict[str, int],
    *,
    ctx: StudyContext,
    job_id: str | None = None,
    memory: RouterMemory | None = None,
    simulated: bool = False,
) -> dict[str, Any]:
    """Certify *hw_counts* against an exact baseline and record the deficit.

    Args:
        qubo: The QUBO *plan* was routed for.
        plan: The hardware RoutePlan (pipeline_kwargs carry the ECC settings).
        hw_counts: Measurement counts from the hardware job (Qiskit order).
        ctx: Study context; decides statevector vs aer_gpu for the baseline.
        job_id: IBM job id, for provenance.
        memory: Ledger that receives the measured logical error (real runs).
        simulated: True when *hw_counts* came from a noise model. Simulated
            records go under results/study/gap/ and never reach history or
            the ledger; real ones go to the results root so the router
            learns from them.
    """
    from limen.pipeline import run_pipeline

    n = plan.n_vars
    base_backend, base_name = baseline_backend(n, ctx)
    offline_kwargs = dict(plan.pipeline_kwargs)
    offline_kwargs.update(backend=base_backend, qpu_backend_name=base_name)
    t0 = time.perf_counter()
    baseline = run_pipeline(qubo, **offline_kwargs)
    t_base = time.perf_counter() - t0

    hw_cert = run_pipeline(qubo, **plan.pipeline_kwargs, qpu_counts=hw_counts)

    model_predicted = hw_cert.aggregate_logical_error_rate or 0.0
    predicted = (
        hw_cert.predicted_logical_error_bound
        if hw_cert.predicted_logical_error_bound is not None
        else model_predicted
    )
    measured = max(0.0, baseline.success_probability - hw_cert.success_probability)
    shots = max(1, sum(hw_counts.values()))

    # Whole-distribution gap. success_probability is ~2**-n for p=1 QAOA,
    # so at 20+ qubits the deficit above is sampling noise; fidelity / TVD /
    # expected-energy on the final circuit is the signal that survives.
    from limen.frontends.pyqubo import from_qubo_dict
    from limen.gates.qaoa import bitstring_to_assignment, compile_qaoa, variable_order
    from limen.pipeline import _counts_to_probabilities, _energy
    from limen.study._common import distribution_gap, exact_prob_fn

    graph = from_qubo_dict(qubo)
    order = variable_order(graph)
    final_circuit = compile_qaoa(
        graph,
        [baseline.qaoa_params["gamma"]] * baseline.qaoa_layers,
        [baseline.qaoa_params["beta"]] * baseline.qaoa_layers,
    )
    ideal_dist = exact_prob_fn(n, ctx, shots=max(shots, 4000))(final_circuit)
    hw_dist = _counts_to_probabilities(hw_counts)
    dist_gap = distribution_gap(
        ideal_dist, hw_dist, energy_of=lambda bits: _energy(qubo, bitstring_to_assignment(bits, order))
    )
    from limen.pipeline import _AER_EXACT_MAX_QUBITS

    if n > _AER_EXACT_MAX_QUBITS:
        # Past the exact-probabilities cap the "ideal" side is itself a
        # finite sample; two sparse samples over 2**n outcomes barely
        # overlap (seen: fidelity 0.0005, TVD 0.98 at 20 qubits with 4000
        # shots each), so those two numbers measure shot count, not noise.
        # expected_energy_gap is a mean over the samples and stays valid.
        dist_gap["distribution_fidelity"] = None
        dist_gap["total_variation_distance"] = None
        dist_gap["distribution_note"] = (
            f"ideal distribution is sampled above {_AER_EXACT_MAX_QUBITS} qubits; "
            "fidelity/TVD between two sparse samples are not reported, read "
            "expected_energy_gap"
        )
    noise = 2.0 * math.sqrt(
        max(hw_cert.success_probability * (1.0 - hw_cert.success_probability), 1e-12) / shots
    )
    now = timestamp()
    record: dict[str, Any] = {
        "generated_at": now,
        "study": "gap",
        "simulated_hardware": simulated,
        "job_id": job_id,
        "plan": plan.to_dict(),
        "baseline_backend": {
            "backend": base_backend,
            "qpu_backend_name": base_name,
            "device": baseline.metadata.get("device"),
            "seconds": round(t_base, 3),
        },
        "baseline_certificate": baseline.to_dict(),
        "hardware_certificate": hw_cert.to_dict(),
        "predicted_aggregate_logical_error_rate": model_predicted,
        "predicted_logical_error_bound": predicted,
        "measured_success_deficit": measured,
        "two_sigma_sampling_noise": noise,
        "measured_within_prediction": measured <= predicted + noise,
        **dist_gap,
        "success_probability_note": (
            "success_probability is the mass on one optimal bitstring and scales "
            "like 2**-n for p=1 QAOA; above ~16 qubits read distribution_fidelity, "
            "total_variation_distance and expected_energy_gap instead"
            if n > 16 else "success_probability is informative at this width"
        ),
        "solutions_agree": baseline.solution == hw_cert.solution,
        "energy_gap": hw_cert.energy - baseline.energy,
        # history._queue_seconds_from_timestamps reads these; a fetched job
        # exposes no queue time here, so submitted == completed.
        "timestamps": {"submitted": now, "completed": now},
    }

    if simulated:
        record["path"] = write_record(ctx, "gap", f"gap-sim-{plan.backend.name}", record)
        return record

    if ctx.results_dir is not None:
        ctx.results_dir.mkdir(parents=True, exist_ok=True)
        path = ctx.results_dir / f"router_gap_{plan.backend.name}_{now}.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        record["path"] = path
    if memory is not None:
        memory.record_route_outcome(
            plan.backend.name,
            logical_error=measured,
            source=f"study.gap:{job_id or 'fetched'}",
        )
    return record


def hardware_plan(
    qubo: dict[tuple[str, str], float],
    backend_name: str,
    *,
    fidelity_target: float = 0.95,
    credit_budget: float = 4.0,
    certified: bool = False,
    fleet: Any = None,
) -> RoutePlan:
    """A Tier 1 (or Tier 2) plan pinned to *backend_name*, for a gap run."""
    fleet = fleet or DEFAULT_FLEET
    target = tuple(p for p in fleet if p.name == backend_name)
    if not target:
        raise ValueError(f"{backend_name} is not in the fleet")
    sims = tuple(p for p in fleet if p.kind == "sim")
    req = RouteRequest(
        qubo,
        fidelity_target=fidelity_target,
        credit_budget=credit_budget,
        force_tier=Tier.HW_CERTIFIED if certified else Tier.HW_STANDARD,
    )
    return route(req, (*sims, *target))


def simulated_hardware_counts(
    qubo: dict[tuple[str, str], float],
    plan: RoutePlan,
    *,
    ctx: StudyContext,
    depolarizing_1q: float = 1e-3,
    depolarizing_2q: float = 1e-2,
    readout: float = 1e-2,
    shots: int | None = None,
    seed: int = 7,
) -> dict[str, int]:
    """Stand-in for a QPU job: the plan's final QAOA circuit under a simple
    depolarizing + readout noise model on Aer. Lets the gap loop run end to
    end with no credentials; the result is labelled simulated and kept out
    of the router's history."""
    from qiskit import transpile
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    from limen.frontends.pyqubo import from_qubo_dict
    from limen.gates.qaoa import compile_qaoa
    from limen.gates.qiskit_exec import make_aer_simulator, to_qiskit_circuit
    from limen.pipeline import run_pipeline

    base_backend, base_name = baseline_backend(plan.n_vars, ctx)
    kw = dict(plan.pipeline_kwargs)
    kw.update(backend=base_backend, qpu_backend_name=base_name)
    ideal = run_pipeline(qubo, **kw)
    layers = ideal.qaoa_layers
    circuit = compile_qaoa(
        from_qubo_dict(qubo),
        [ideal.qaoa_params["gamma"]] * layers,
        [ideal.qaoa_params["beta"]] * layers,
    )
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(
        depolarizing_error(depolarizing_1q, 1), ["rx", "ry", "rz", "h", "x", "sx", "u"]
    )
    noise.add_all_qubit_quantum_error(depolarizing_error(depolarizing_2q, 2), ["cx", "rzz", "cz", "ecr"])
    noise.add_all_qubit_readout_error(ReadoutError([[1 - readout, readout], [readout, 1 - readout]]))
    qc = to_qiskit_circuit(circuit)
    n_shots = shots or plan.shots
    aer_name = base_name if base_backend == "aer" else "aer_simulator"
    try:
        sim = make_aer_simulator(aer_name, seed, noise_model=noise)
        counts = sim.run(transpile(qc, sim), shots=n_shots).result().get_counts()
    except Exception:
        sim = make_aer_simulator("aer_simulator", seed, noise_model=noise)
        counts = sim.run(transpile(qc, sim), shots=n_shots).result().get_counts()
    return dict(counts)


def gap_offline_demo(
    qubo: dict[tuple[str, str], float],
    backend_name: str = "ibm_fez",
    *,
    ctx: StudyContext,
    **noise: float,
) -> dict[str, Any]:
    """Whole gap loop with simulated hardware: plan -> noisy counts -> record."""
    plan = hardware_plan(qubo, backend_name)
    counts = simulated_hardware_counts(qubo, plan, ctx=ctx, **noise)
    return hardware_gap(qubo, plan, counts, ctx=ctx, simulated=True)


def gap_live(
    qubo: dict[tuple[str, str], float],
    backend_name: str,
    *,
    ctx: StudyContext,
    token: str,
    crn: str,
    memory: RouterMemory | None = None,
    certified: bool = False,
    poll_initial_seconds: float = 30.0,
    poll_backoff_cap_seconds: float = 300.0,
    poll_ceiling_seconds: float = 6 * 3600.0,
) -> dict[str, Any]:
    """Submit the plan's circuit to *backend_name*, wait, record the real gap.

    Reuses limen.pipeline.submit_qpu_job and the crash-safe poller
    (_poll_qpu_counts persists the job id to results_dir at every poll).
    Spends credits: one job of plan.shots shots.
    """
    from limen.pipeline import _poll_qpu_counts, submit_qpu_job

    plan = hardware_plan(qubo, backend_name, certified=certified)
    job_id = submit_qpu_job(
        qubo,
        qpu_backend_name=plan.pipeline_kwargs["qpu_backend_name"],
        qpu_shots=plan.shots,
        qpu_token=token,
        qpu_instance=crn,
    )
    counts = _poll_qpu_counts(
        job_id,
        plan,
        token,
        crn,
        results_dir=ctx.results_dir,
        poll_initial_seconds=poll_initial_seconds,
        poll_backoff_cap_seconds=poll_backoff_cap_seconds,
        poll_ceiling_seconds=poll_ceiling_seconds,
    )
    return hardware_gap(qubo, plan, counts, ctx=ctx, job_id=job_id, memory=memory, simulated=False)
