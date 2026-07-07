# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Fetch a router_tier2_kingston.py job by ID and certify it once it lands.

router_tier2_kingston.py blocks on job.result(), so if the terminal that
submitted it is closed (or the device is in maintenance and the job sits
in queue for a long time), nothing certifies the result when it finally
completes. This script re-attaches to that job by id, polls until it is
DONE, and feeds the returned counts through the exact same certification
path run_pipeline uses for a live QPU run (grid-search energy check +
surface-code ECC certificate) via run_pipeline's qpu_counts override —
so the output is a real EndToEndCertificate, not just raw counts.

Requires IBM credentials (never written to the plan, certificate, or the
results file):

    export IBM_QUANTUM_TOKEN=...
    export IBM_QUANTUM_CRN=...
    python examples/router_tier2_kingston_fetch.py <job_id>
"""

from __future__ import annotations

import datetime
import json
import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass

from limen.pipeline import run_pipeline
from limen.router import DEFAULT_FLEET, RouteRequest, Tier, route
from examples.router_tier2_kingston import star_maxcut

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
POLL_SECONDS = 30


def _rebuild_plan():
    """Recompute the identical RoutePlan router_tier2_kingston.py used.

    route() is a pure function of (request, fleet), so this reproduces
    the same tier/backend/ECC distance/patch assignments without needing
    to have persisted the original plan.
    """
    qubo = star_maxcut(4)
    request = RouteRequest(
        qubo,
        fidelity_target=0.9,
        credit_budget=1000 * 0.002,
    )
    fleet = tuple(
        p for p in DEFAULT_FLEET if p.kind == "sim" or p.name == "ibm_kingston"
    )
    plan = route(request, fleet=fleet)
    assert plan.tier == Tier.HW_CERTIFIED, plan.tier
    return qubo, plan


def _poll_until_done(service, job_id: str):
    job = service.job(job_id)
    while True:
        status = str(job.status())
        print(f"[fetch] job {job_id} status: {status}")
        if status == "DONE":
            return job
        if status in ("ERROR", "CANCELLED"):
            raise RuntimeError(f"job {job_id} ended with status {status}")
        time.sleep(POLL_SECONDS)


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python examples/router_tier2_kingston_fetch.py <job_id>",
            file=sys.stderr,
        )
        return 2

    job_id = sys.argv[1]
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")
    if not (token and crn):
        print("Set IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN first.", file=sys.stderr)
        return 2

    from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore[import]

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token, instance=crn
    )
    job = _poll_until_done(service, job_id)
    pub_result = job.result()[0]
    counts: dict[str, int] = pub_result.data.c.get_counts()
    metrics = job.metrics()
    timestamps = metrics.get("timestamps") if metrics else None

    qubo, plan = _rebuild_plan()

    # Offline baseline: same plan executed on the exact simulator.
    offline_kwargs = dict(plan.pipeline_kwargs)
    offline_kwargs.update(backend="statevector", qpu_backend_name="aer_simulator")
    baseline = run_pipeline(qubo, **offline_kwargs)
    assert baseline.is_optimal, "statevector baseline must certify optimal"

    # Certify the already-fetched counts through the same path a live
    # QPU run would take — no new job is submitted.
    hw_cert = run_pipeline(qubo, **plan.pipeline_kwargs, qpu_counts=counts)

    predicted = hw_cert.aggregate_logical_error_rate or 0.0
    physical_error_rate = plan.pipeline_kwargs["physical_error_rate"]
    measured = max(0.0, baseline.success_probability - hw_cert.success_probability)
    noise = 2.0 * math.sqrt(
        max(hw_cert.success_probability * (1.0 - hw_cert.success_probability), 1e-12)
        / plan.shots
    )
    within = measured <= predicted + noise
    delta_vs_physical_error_rate = measured - physical_error_rate
    delta_vs_predicted = measured - predicted

    record = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "job_id": job_id,
        "timestamps": timestamps,
        "plan": plan.to_dict(),
        "baseline_certificate": baseline.to_dict(),
        "hardware_certificate": hw_cert.to_dict(),
        "predicted_aggregate_logical_error_rate": predicted,
        "physical_error_rate": physical_error_rate,
        "measured_success_deficit": measured,
        "two_sigma_sampling_noise": noise,
        "measured_within_prediction": within,
        "delta_measured_vs_physical_error_rate": delta_vs_physical_error_rate,
        "delta_measured_vs_predicted_aggregate": delta_vs_predicted,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"router_tier2_kingston_{job_id}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nLogged to {out}")

    print(f"optimal on hardware:                {hw_cert.is_optimal}")
    print(f"physical error rate (baked in plan): {physical_error_rate:.3e}")
    print(f"predicted aggregate logical error:   {predicted:.3e}")
    print(f"measured success deficit:            {measured:.3e} (±{noise:.3e})")
    print(f"delta vs physical_error_rate:        {delta_vs_physical_error_rate:+.3e}")
    print(f"delta vs predicted aggregate:        {delta_vs_predicted:+.3e}")
    print(f"within prediction: {within}")
    ok = bool(hw_cert.is_optimal) and within
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
