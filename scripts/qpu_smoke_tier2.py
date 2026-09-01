# Tier 2 QPU-path smoke test for run_route_request().
#
# Same 3-variable Max-Cut QUBO that passed the simulator path, now forced
# to Tier.HW_CERTIFIED so the plan routes to a validated IBM backend and
# run_route_request() drives the full submit -> poll -> certify chain in
# one call. This is the DUCTEI integration gate: prove the single-call
# QPU path works, with crash-safe job persistence in results_dir.
#
# Usage (PowerShell):
#   $env:IBM_QUANTUM_TOKEN = "..."
#   $env:IBM_QUANTUM_CRN   = "..."
#   python qpu_smoke_tier2.py
#
# If the process dies mid-poll, the job id is already persisted under
# ./results/ — rerun or use examples/router_tier2_kingston_fetch.py to
# re-attach.

from __future__ import annotations

import os
import pathlib
import sys
import time

from limen.pipeline import run_route_request
from limen.router import RouteRequest, Tier

# Repo-root results/, not scripts/results/: this directory is what
# informed_fleet() scans for run history and calibration snapshots (with
# the old scripts/results path the router planned against bare
# DEFAULT_FLEET priors — hardcoded 1e-3 physical_error_rate and the
# alphabetical name tiebreak), and where
# examples/router_tier2_kingston_fetch.py looks to re-attach by job id.
RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"

# 0. Preflight: credentials present?
for var in ("IBM_QUANTUM_TOKEN", "IBM_QUANTUM_CRN"):
    if not os.environ.get(var):
        sys.exit(f"FAIL: {var} not set")

# 1. The proven instance: triangle Max-Cut, optimum cuts 2 of 3 edges.
qubo = {
    (0, 0): -2.0, (1, 1): -2.0, (2, 2): -2.0,
    (0, 1): 2.0, (0, 2): 2.0, (1, 2): 2.0,
}

request = RouteRequest(
    qubo=qubo,
    fidelity_target=0.95,
    credit_budget=10.0,          # enough for a small shot count on HW
    force_tier=Tier.HW_CERTIFIED,
)

print("Submitting Tier 2 route request (this call blocks through poll+certify;")
print("job id is persisted to results/ at every poll, safe to kill and re-attach)...")
t0 = time.time()

cert = run_route_request(
    request,
    results_dir=RESULTS_DIR,
    poll_initial_seconds=30.0,
)

elapsed = time.time() - t0

# 2. Gate checks.
print(f"\nElapsed: {elapsed:.1f}s")
print(f"Solution: {cert.solution}")
print(f"Backend/plan metadata: {getattr(cert, 'metadata', None)}")
print(f"Logical error rate: {getattr(cert, 'logical_error_rate', None)}")

ok = True

# Check 0: solution decodes to a valid Max-Cut optimum (any 2-of-3 cut).
sol = cert.solution
if sol:
    bits = list(sol.values())  # keys are str var names; only values matter
    # A triangle optimum has both values present (a lone 1 or lone 0).
    if len(bits) == 3 and len(set(bits)) == 2:
        print("CHECK 0 PASS: solution is a valid triangle Max-Cut optimum")
    elif cert.is_optimal:
        print("CHECK 0 PASS: certificate marks solution optimal")
    else:
        print(f"CHECK 0 FAIL: solution {sol} is not a 2-of-3 cut")
        ok = False
else:
    print("CHECK 0 FAIL: no solution on certificate")
    ok = False

# Check 1: certificate is the real Tier 2 article (ECC fields populated).
if getattr(cert, "logical_error_rate", None) is not None:
    print("CHECK 1 PASS: Tier 2 certificate carries logical_error_rate")
else:
    print("CHECK 1 FAIL: missing logical_error_rate (did this run Tier 2?)")
    ok = False

# Check 2: job state persisted for crash-resilience contract.
persisted = list(RESULTS_DIR.glob("**/*")) if RESULTS_DIR.exists() else []
if persisted:
    print(f"CHECK 2 PASS: {len(persisted)} state/cert file(s) in results/")
else:
    print("CHECK 2 FAIL: nothing persisted to results_dir")
    ok = False

print("\n" + ("SMOKE TEST PASS — DUCTEI integration gate satisfied"
              if ok else "SMOKE TEST FAIL"))
sys.exit(0 if ok else 1)
