# limen/quantum_channel/teleport_analysis.py
from __future__ import annotations
from typing import Optional

from limen.quantum_channel.teleport import TeleportResult


def estimate_fidelity(
    counts: dict[str, int],
    backend: str,
    job_id: Optional[str] = None,
    expected_bit: str = "0",
) -> TeleportResult:
    """Estimate teleportation fidelity from raw IBM job measurement counts.

    teleport_circuit() always prepares q0 in |0>, so a faithful teleportation
    measures Bob's qubit (q2, classical bit 2) as ``expected_bit``. Qiskit
    formats counts keys as "c2c1c0" (bit 2 leftmost), so fidelity_estimate
    is the fraction of shots whose leftmost character matches expected_bit.
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return TeleportResult(
            fidelity_estimate=0.0, success=False, backend=backend, job_id=job_id
        )

    correct_outcomes = sum(
        count
        for bitstring, count in counts.items()
        if bitstring.replace(" ", "")[0] == expected_bit
    )

    fidelity_estimate = correct_outcomes / total_shots
    return TeleportResult(
        fidelity_estimate=fidelity_estimate,
        success=fidelity_estimate > 0.5,
        backend=backend,
        job_id=job_id,
    )
