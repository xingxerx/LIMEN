# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Tier 2 hardware validation: submit a small instance to ibm_kingston.

Routes a d=3 ECC-budgeted Max-Cut instance (5 logical vars -> 45 physical
qubits of patch budget, well inside kingston's 156) through the budget
router at Tier 2, then submits the routed plan to real hardware and
exits immediately — it does not wait for the job to finish.

Submitting and waiting are deliberately separate processes: an IBM
Runtime job runs on IBM's servers regardless of what this script does
afterwards, and kingston's queue can sit in "maintenance" for a long
time. Blocking this process on job.result() would mean a closed
terminal (or a hardware queue outlasting your patience) loses the
ability to certify a completed job. Instead this script just persists
{job_id, plan, submitted_at} to a local state file and exits; run
examples/router_tier2_kingston_fetch.py <job_id> whenever you're ready
to poll for and certify the result — it re-attaches by job id, which
never expires.

Requires IBM credentials (never written to the plan, state file, or
certificate):

    export IBM_QUANTUM_TOKEN=...
    export IBM_QUANTUM_CRN=...
    python examples/router_tier2_kingston.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

# Allow running directly from the project root without a full package install.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Load .env from the project root if present (never required — real env vars
# take precedence via dotenv's override=False default).
try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass

from limen.pipeline import submit_qpu_job
from limen.router import (
    DEFAULT_FLEET,
    JobState,
    JobStatus,
    RouteRequest,
    Tier,
    informed_fleet,
    route,
)
from limen.router.job_state import now_iso, retry_transient, save_state

SHOTS = 1000
RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def star_maxcut(n_leaves: int) -> dict[tuple[str, str], float]:
    """Star-graph Max-Cut: the hub's heavy-tailed criticality is exactly
    the spectrum the router sends to Tier 2 on its own."""
    qubo: dict[tuple[str, str], float] = {}
    for i in range(n_leaves):
        leaf = f"leaf{i}"
        qubo[("hub", leaf)] = 2.0
        qubo[("hub", "hub")] = qubo.get(("hub", "hub"), 0.0) - 1.0
        qubo[(leaf, leaf)] = -1.0
    return qubo


def main() -> int:
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")
    if not (token and crn):
        print("Set IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN first.", file=sys.stderr)
        return 2

    qubo = star_maxcut(4)  # 5 logical vars -> 45 physical qubits of d=3 patches

    request = RouteRequest(
        qubo,
        fidelity_target=0.9,  # -> distance 3
        credit_budget=SHOTS * 0.002,  # kingston cost estimate -> 1000 shots
    )
    # Pin the fleet to kingston (plus the sims) so the router can't pick a
    # sibling 156q device on the name tiebreak. Fold in run history and
    # live calibration so physical_error_rate reflects measured hardware,
    # not the 1e-3 hardcoded default (see limen.router.informed_fleet).
    fleet = tuple(
        p
        for p in informed_fleet(RESULTS_DIR, DEFAULT_FLEET)
        if p.kind == "sim" or p.name == "ibm_kingston"
    )
    plan = route(request, fleet=fleet)
    assert plan.tier == Tier.HW_CERTIFIED, plan.tier
    assert plan.backend.name == "ibm_kingston"
    assert plan.ecc_distance == 3
    assert plan.patch_assignments, "Tier 2 plan must carry patch assignments"
    print("RoutePlan:")
    print(json.dumps(plan.to_dict(), indent=2))

    # Submission is safe to retry on a transient network error (nothing
    # has been accepted by IBM yet); a real submission failure (bad
    # credentials, unknown backend) is not retried and surfaces directly.
    try:
        job_id = retry_transient(
            lambda: submit_qpu_job(
                qubo,
                qpu_backend_name=plan.backend.name,
                qpu_shots=plan.shots,
                qpu_token=token,
                qpu_instance=crn,
            )
        )
    except Exception as exc:
        print(f"Submission failed: {exc!r}", file=sys.stderr)
        return 1

    state = JobState(
        job_id=job_id,
        status=JobStatus.SUBMITTED,
        plan=plan.to_dict(),
        submitted_at=now_iso(),
    )
    save_state(RESULTS_DIR, state)

    print(f"\n[limen] Submitted job {job_id} on {plan.backend.name} ({plan.shots} shots)")
    print(f"[limen] State saved to results/router_tier2_kingston_{job_id}.state.json")
    print(
        "[limen] This process does not wait for the result. When you're "
        "ready to check on it or certify a completed run:\n"
        f"    python examples/router_tier2_kingston_fetch.py {job_id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
