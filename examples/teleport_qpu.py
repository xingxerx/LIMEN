# Copyright (C) 2026 Jemone McCubbin / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""IBM QPU demo: real-hardware quantum teleportation for LIMEN.

Submits the standard 3-qubit teleportation circuit
(limen.quantum_channel.teleport.teleport_circuit) to a real IBM quantum
processor (ibm_kingston) via QiskitRuntimeService and SamplerV2, then
estimates the teleportation fidelity from the measured counts.

Required environment variables:
    IBM_QUANTUM_TOKEN  — IBM Quantum Platform API token
    IBM_QUANTUM_CRN    — IBM Quantum instance CRN (service instance identifier)

Usage::

    IBM_QUANTUM_TOKEN=<token> IBM_QUANTUM_CRN=<crn> python examples/teleport_qpu.py
"""

import json
import os
import pathlib
import sys
import time

# Allow running directly from the project root without a full package install.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Load .env from the project root if present (never required — real env vars
# take precedence via dotenv's override=False default).
try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass

from limen.quantum_channel.teleport import run_teleport_qpu, teleport_circuit

_BACKEND_NAME = "ibm_kingston"
_NUM_SHOTS = 1000


def main() -> None:
    """Submit the teleportation circuit to a real IBM QPU and save the result."""
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")

    if not token:
        print(
            "ERROR: IBM_QUANTUM_TOKEN environment variable is not set.\n"
            "Export your IBM Quantum Platform API token before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not crn:
        print(
            "ERROR: IBM_QUANTUM_CRN environment variable is not set.\n"
            "Export your IBM Quantum instance CRN before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    qc = teleport_circuit()
    print(
        f"Submitting teleportation circuit ({qc.num_qubits} qubits, "
        f"depth={qc.depth()}) to IBM QPU ({_BACKEND_NAME}) ..."
    )

    result = run_teleport_qpu(
        token=token,
        crn=crn,
        backend_name=_BACKEND_NAME,
        shots=_NUM_SHOTS,
    )

    print(f"Job id            : {result.job_id}")
    print(f"Fidelity estimate : {result.fidelity_estimate:.4f}")
    print(f"Success           : {result.success}")
    if result.channel_delta:
        print(f"T2 (median, us)   : {result.channel_delta.t2_us:.2f}")
        print(f"Fidelity penalty  : {result.channel_delta.fidelity_penalty():.6f}")

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"teleport_qpu_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_json.write_text(
        json.dumps(
            {
                "backend": _BACKEND_NAME,
                "shots": _NUM_SHOTS,
                "circuit_depth": qc.depth(),
                **result.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Raw JSON          : {out_json}")


if __name__ == "__main__":
    main()
