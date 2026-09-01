# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Poll a router_tier2_kingston.py job by ID and certify it once it lands.

This is the only place in the router_tier2_kingston workflow that waits.
router_tier2_kingston.py submits and exits immediately; this script
re-attaches to that job by id (which never expires — IBM Runtime jobs
run independently of any local process) and:

  - Polls with exponential backoff (30s -> 5min cap) instead of blocking
    indefinitely. After a 24h ceiling it gives up and marks the local
    job state TIMED_OUT rather than hanging the terminal forever — you
    can always re-run this script later to keep waiting.
  - Persists lifecycle state (SUBMITTED -> QUEUED -> RUNNING ->
    DONE/ERROR/CANCELLED/TIMED_OUT) to results/ at every poll, so a
    crash mid-poll loses nothing: restarting just resumes polling the
    same job id, and the 24h ceiling is measured from the original
    submission time, not from this process's start time.
  - Is idempotent: if a certificate already exists for this job id, it
    prints a summary and exits instead of re-certifying or erroring.
  - Never auto-resubmits. A job that errors or is cancelled on the QPU
    side (calibration fault, cancelled by IBM) is surfaced as ERROR/
    CANCELLED and left there — silently resubmitting would burn credits
    on a problem this pipeline can't fix (see module docstring in
    router_tier2_kingston.py). Re-submitting is a deliberate, separate
    decision: run router_tier2_kingston.py again.

Feeds the fetched counts through the exact certification path
run_pipeline uses for a live QPU run (grid-search energy check +
surface-code ECC certificate) via run_pipeline's qpu_counts override —
so the output is a real EndToEndCertificate, not just raw counts.

Requires IBM credentials:

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
from limen.router import (
    DEFAULT_FLEET,
    JobState,
    JobStatus,
    RoutePlan,
    RouteRequest,
    Tier,
    informed_fleet,
    route,
)
from limen.router.job_state import cert_path, load_state, now_iso, save_state, state_path
from examples.router_tier2_kingston import star_maxcut

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"

POLL_INITIAL_SECONDS = 30.0
POLL_BACKOFF_CAP_SECONDS = 300.0
POLL_CEILING_SECONDS = 24 * 3600.0

# Maps qiskit-ibm-runtime's job.status() strings onto our own JobStatus.
_IBM_STATUS_MAP = {
    "INITIALIZING": JobStatus.QUEUED,
    "QUEUED": JobStatus.QUEUED,
    "RUNNING": JobStatus.RUNNING,
    "DONE": JobStatus.DONE,
    "ERROR": JobStatus.ERROR,
    "CANCELLED": JobStatus.CANCELLED,
}


def _rebuild_plan() -> tuple[dict, RoutePlan]:
    """Recompute the identical RoutePlan router_tier2_kingston.py used.

    route() is a pure function of (request, fleet), so this reproduces
    the same tier/backend/ECC distance/patch assignments without needing
    to have persisted the original plan. Only used as a fallback when no
    local state file exists (e.g. a job id from before this state-file
    tracking existed).
    """
    qubo = star_maxcut(4)
    # Mirror examples/router_tier2_kingston.py: that script pins Tier 2
    # (see its request comment re: accepted proposal
    # 2026-07-24-raise-criticality-threshold), so the rebuild must too or
    # route() would no longer reproduce the persisted plan.
    request = RouteRequest(
        qubo,
        fidelity_target=0.9,
        credit_budget=1000 * 0.002,
        force_tier=Tier.HW_CERTIFIED,
    )
    fleet = tuple(
        p
        for p in informed_fleet(RESULTS_DIR, DEFAULT_FLEET)
        if p.kind == "sim" or p.name == "ibm_kingston"
    )
    plan = route(request, fleet=fleet)
    assert plan.tier == Tier.HW_CERTIFIED, plan.tier
    return qubo, plan


def _poll_until_terminal(service, state: JobState) -> JobState:
    """Poll job status with exponential backoff, persisting state at every
    step, until a terminal status is reached or the 24h ceiling expires."""
    job = service.job(state.job_id)
    try:
        submitted_at = datetime.datetime.fromisoformat(state.submitted_at)
    except ValueError:
        submitted_at = datetime.datetime.strptime(state.submitted_at, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=datetime.timezone.utc)

    backoff = POLL_INITIAL_SECONDS

    while True:
        raw_status = str(job.status())
        mapped = _IBM_STATUS_MAP.get(raw_status, JobStatus.QUEUED)
        state.status = mapped
        state.last_polled_at = now_iso()
        save_state(RESULTS_DIR, state)
        print(f"[fetch] job {state.job_id} status: {raw_status}")

        if mapped in (JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED):
            return state

        elapsed = (
            datetime.datetime.now(datetime.timezone.utc) - submitted_at
        ).total_seconds()
        if elapsed >= POLL_CEILING_SECONDS:
            state.status = JobStatus.TIMED_OUT
            state.error = (
                f"Polling ceiling of {POLL_CEILING_SECONDS / 3600:.0f}h exceeded; "
                "job may still complete on IBM's side. Re-run this script later "
                "to keep waiting — the job id does not expire."
            )
            save_state(RESULTS_DIR, state)
            return state

        time.sleep(backoff)
        backoff = min(backoff * 2, POLL_BACKOFF_CAP_SECONDS)


def _certify(job_id: str, plan_dict: dict, counts: dict[str, int]) -> dict:
    """Rebuild the request's qubo and certify already-fetched counts
    through the same grid-search/energy/ECC path a live QPU run uses."""
    qubo = star_maxcut(4)

    offline_kwargs = dict(plan_dict["pipeline_kwargs"])
    offline_kwargs.update(backend="statevector", qpu_backend_name="aer_simulator")
    baseline = run_pipeline(qubo, **offline_kwargs)
    assert baseline.is_optimal, "statevector baseline must certify optimal"

    hw_cert = run_pipeline(qubo, **plan_dict["pipeline_kwargs"], qpu_counts=counts)

    model_predicted = hw_cert.aggregate_logical_error_rate or 0.0
    # The bound is max(model, measured prior from run history) — see
    # EndToEndCertificate. Plans routed before measured_logical_error
    # forwarding existed carry no prior; the bound then equals the model.
    predicted = (
        hw_cert.predicted_logical_error_bound
        if hw_cert.predicted_logical_error_bound is not None
        else model_predicted
    )
    physical_error_rate = plan_dict["pipeline_kwargs"]["physical_error_rate"]
    measured = max(0.0, baseline.success_probability - hw_cert.success_probability)
    noise = 2.0 * math.sqrt(
        max(hw_cert.success_probability * (1.0 - hw_cert.success_probability), 1e-12)
        / plan_dict["shots"]
    )
    within = measured <= predicted + noise

    return {
        "generated_at": now_iso(),
        "job_id": job_id,
        "plan": plan_dict,
        "baseline_certificate": baseline.to_dict(),
        "hardware_certificate": hw_cert.to_dict(),
        "predicted_aggregate_logical_error_rate": model_predicted,
        "predicted_logical_error_bound": predicted,
        "measured_logical_error_prior": hw_cert.measured_logical_error_prior,
        "physical_error_rate": physical_error_rate,
        "measured_success_deficit": measured,
        "two_sigma_sampling_noise": noise,
        "measured_within_prediction": within,
        "delta_measured_vs_physical_error_rate": measured - physical_error_rate,
        "delta_measured_vs_predicted_aggregate": measured - model_predicted,
        "delta_measured_vs_predicted_bound": measured - predicted,
    }


def _print_summary(record: dict) -> bool:
    hw = record["hardware_certificate"]
    print(f"optimal on hardware:                {hw['is_optimal']}")
    print(f"physical error rate (baked in plan): {record['physical_error_rate']:.3e}")
    print(
        f"predicted aggregate logical error:   "
        f"{record['predicted_aggregate_logical_error_rate']:.3e}"
    )
    # Certs written before the measured-prior bound existed lack the key.
    if "predicted_logical_error_bound" in record:
        print(
            f"predicted bound (max(model, prior)): "
            f"{record['predicted_logical_error_bound']:.3e}"
        )
    print(
        f"measured success deficit:            {record['measured_success_deficit']:.3e} "
        f"(±{record['two_sigma_sampling_noise']:.3e})"
    )
    print(
        f"delta vs physical_error_rate:        "
        f"{record['delta_measured_vs_physical_error_rate']:+.3e}"
    )
    print(
        f"delta vs predicted aggregate:        "
        f"{record['delta_measured_vs_predicted_aggregate']:+.3e}"
    )
    print(f"within prediction: {record['measured_within_prediction']}")
    ok = bool(hw["is_optimal"]) and record["measured_within_prediction"]
    print("PASS" if ok else "FAIL")
    return ok


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python examples/router_tier2_kingston_fetch.py <job_id>",
            file=sys.stderr,
        )
        return 2
    job_id = sys.argv[1]

    # Idempotent: a cert already on disk means this job was already
    # certified — print it and stop instead of re-certifying.
    existing_cert = cert_path(RESULTS_DIR, job_id)
    if existing_cert.exists():
        print(f"[fetch] {existing_cert} already exists — skipping re-certification.")
        record = json.loads(existing_cert.read_text())
        return 0 if _print_summary(record) else 1

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")
    if not (token and crn):
        print("Set IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN first.", file=sys.stderr)
        return 2

    state = load_state(RESULTS_DIR, job_id)
    if state is None:
        # No local state (e.g. a job id predating this tracking, or the
        # state file was lost) — rebuild the plan deterministically and
        # start tracking from now.
        _, plan = _rebuild_plan()
        state = JobState(
            job_id=job_id,
            status=JobStatus.SUBMITTED,
            plan=plan.to_dict(),
            submitted_at=now_iso(),
        )
        save_state(RESULTS_DIR, state)

    if state.status in (JobStatus.ERROR, JobStatus.CANCELLED):
        print(
            f"[fetch] job {job_id} previously ended with status "
            f"{state.status.value} — not polling again. Re-run "
            "router_tier2_kingston.py to submit a fresh job if you want "
            "to retry.",
            file=sys.stderr,
        )
        return 1
    if state.status == JobStatus.TIMED_OUT:
        print(
            f"[fetch] job {job_id} previously hit the local polling "
            "ceiling; resuming polling now.",
        )
        state.status = JobStatus.QUEUED

    from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore[import]

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token, instance=crn
    )
    state = _poll_until_terminal(service, state)

    if state.status == JobStatus.TIMED_OUT:
        print(f"[fetch] {state.error}", file=sys.stderr)
        return 1
    if state.status in (JobStatus.ERROR, JobStatus.CANCELLED):
        print(
            f"[fetch] job {job_id} ended with status {state.status.value} — "
            "not auto-resubmitting. Re-run router_tier2_kingston.py "
            "manually if you want to retry.",
            file=sys.stderr,
        )
        return 1

    job = service.job(job_id)
    pub_result = job.result()[0]
    # Register name varies (c, meas, ...): take the first BitArray in the DataBin.
    _regs = [
        k for k in dir(pub_result.data) if not k.startswith("_")
        and hasattr(getattr(pub_result.data, k), "get_counts")
    ]
    if not _regs:
        raise RuntimeError(f"no classical registers in result: {pub_result.data}")
    counts: dict[str, int] = getattr(pub_result.data, _regs[0]).get_counts()
    metrics = job.metrics()
    timestamps = metrics.get("timestamps") if metrics else None

    record = _certify(job_id, state.plan, counts)
    record["timestamps"] = timestamps

    RESULTS_DIR.mkdir(exist_ok=True)
    out = cert_path(RESULTS_DIR, job_id)
    out.write_text(json.dumps(record, indent=2))
    print(f"\nLogged to {out}")

    # The state file's only purpose was tracking this job until a
    # certificate existed; now that it does, it's redundant bookkeeping.
    state_path(RESULTS_DIR, job_id).unlink(missing_ok=True)

    ok = _print_summary(record)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
