# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Tier 2 hardware validation: route a small instance to ibm_kingston.

Routes a d=3 ECC-budgeted Max-Cut instance (5 logical vars -> 45 physical
qubits of patch budget, well inside kingston's 156) through the budget
router at Tier 2, executes the plan on real hardware, and compares the
measured logical error against the offline surface-code prediction from
the certificate.

The "measured logical error" here is the hardware success-probability
deficit relative to the exact statevector baseline for the same circuit:
everything the device loses versus the noiseless simulation. The success
bar is that this deficit stays within sampling noise of the certificate's
predicted aggregate logical error rate — i.e. the offline prediction is
not *underestimating* the hardware.

Requires IBM credentials (never written to the plan, certificate, or the
results file):

    export IBM_QUANTUM_TOKEN=...
    export IBM_QUANTUM_CRN=...
    python examples/router_tier2_kingston.py

Results are logged to results/router_tier2_kingston_<timestamp>.json,
following the eil51 run convention.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import pathlib
import sys

from limen.pipeline import run_pipeline
from limen.router import DEFAULT_FLEET, RouteRequest, Tier, route

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
    # sibling 156q device on the name tiebreak.
    fleet = tuple(
        p for p in DEFAULT_FLEET if p.kind == "sim" or p.name == "ibm_kingston"
    )
    plan = route(request, fleet=fleet)
    assert plan.tier == Tier.HW_CERTIFIED, plan.tier
    assert plan.backend.name == "ibm_kingston"
    assert plan.ecc_distance == 3
    assert plan.patch_assignments, "Tier 2 plan must carry patch assignments"
    print("RoutePlan:")
    print(json.dumps(plan.to_dict(), indent=2))

    # Offline baseline: same plan executed on the exact simulator.
    offline_kwargs = dict(plan.pipeline_kwargs)
    offline_kwargs.update(backend="statevector", qpu_backend_name="aer_simulator")
    baseline = run_pipeline(qubo, **offline_kwargs)
    assert baseline.is_optimal, "statevector baseline must certify optimal"

    # Hardware run: the routed plan, plus credentials supplied at call time.
    hw_cert = run_pipeline(
        qubo, **plan.pipeline_kwargs, qpu_token=token, qpu_instance=crn
    )

    predicted = hw_cert.aggregate_logical_error_rate
    measured = max(0.0, baseline.success_probability - hw_cert.success_probability)
    # Two-sigma binomial sampling noise on the measured success probability.
    noise = 2.0 * math.sqrt(
        max(hw_cert.success_probability * (1.0 - hw_cert.success_probability), 1e-12)
        / plan.shots
    )
    within = measured <= (predicted or 0.0) + noise

    record = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan": plan.to_dict(),
        "baseline_certificate": baseline.to_dict(),
        "hardware_certificate": hw_cert.to_dict(),
        "predicted_aggregate_logical_error_rate": predicted,
        "measured_success_deficit": measured,
        "two_sigma_sampling_noise": noise,
        "measured_within_prediction": within,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"router_tier2_kingston_{stamp}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nLogged to {out}")

    print(f"optimal on hardware: {hw_cert.is_optimal}")
    print(f"predicted aggregate logical error: {predicted:.3e}")
    print(f"measured success deficit:          {measured:.3e} (±{noise:.3e})")
    print(f"within prediction: {within}")
    ok = bool(hw_cert.is_optimal) and within
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
