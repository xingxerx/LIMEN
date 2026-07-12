# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Fetch live IBM calibration data for one backend and cache it to results/.

router_tier2_kingston_d965qgotcv6s73djc1l0.json showed the router's
hardcoded physical_error_rate (1e-3) is ~2 orders of magnitude below the
measured deficit on real ibm_kingston hardware. This script replaces
that guess with a measured one: it queries the backend's calibration
properties (two-qubit gate error, readout error), derives a scalar
physical_error_rate estimate, and writes it to
results/calibration_<backend>_<timestamp>.json.

limen.router.calibration.scan_calibration + apply_calibration then fold
the cached snapshot into a fleet, so route()'s Tier 2 planning uses the
measured value instead of RouteRequest's hardcoded default — no network
call required at plan time.

Requires IBM credentials:

    export IBM_QUANTUM_TOKEN=...
    export IBM_QUANTUM_CRN=...
    python examples/fetch_backend_calibration.py ibm_kingston
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass

from limen.router.calibration import fetch_backend_calibration

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "Usage: python examples/fetch_backend_calibration.py <backend_name>",
            file=sys.stderr,
        )
        return 2
    backend_name = sys.argv[1]

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")
    if not (token and crn):
        print("Set IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN first.", file=sys.stderr)
        return 2

    from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore[import]

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token, instance=crn
    )
    record = fetch_backend_calibration(service, backend_name)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = record["generated_at"].replace("+00:00", "Z").replace(":", "")
    out = RESULTS_DIR / f"calibration_{backend_name}_{timestamp}.json"
    out.write_text(json.dumps(record, indent=2))

    print(f"avg two-qubit gate error: {record['avg_two_qubit_gate_error']:.3e}")
    print(f"avg readout error:        {record['avg_readout_error']:.3e}")
    print(f"physical_error_rate:      {record['physical_error_rate']:.3e}")
    print(f"\nLogged to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
