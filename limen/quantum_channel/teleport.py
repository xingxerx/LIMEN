# limen/quantum_channel/teleport.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class TeleportResult:
    fidelity_estimate: float
    success: bool
    backend: str
    job_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "fidelity_estimate": self.fidelity_estimate,
            "success": self.success,
            "backend": self.backend,
            "job_id": self.job_id,
        }


def teleport_circuit():
    """
    Standard 3-qubit teleportation circuit.
    q0 = state to send
    q1/q2 = Bell pair (logical node A / node B)
    Requires qiskit. Gated — raises ImportError if not installed.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3, 3)

    # Bell pair between q1 and q2
    qc.h(1)
    qc.cx(1, 2)

    # Alice's operations
    qc.cx(0, 1)
    qc.h(0)

    # Measure Alice's qubits
    qc.measure([0, 1], [0, 1])

    # Classical feedforward corrections on Bob's qubit
    qc.cx(1, 2)
    qc.cz(0, 2)

    qc.measure(2, 2)
    return qc
