# limen/quantum_channel/qkd.py
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class QKDResult:
    raw_key: list[int]
    sifted_key: list[int]
    qber: float                   # Quantum Bit Error Rate
    secure: bool                  # True if qber < 0.11
    backend: str
    job_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sifted_key_length": len(self.sifted_key),
            "qber": self.qber,
            "secure": self.secure,
            "backend": self.backend,
            "job_id": self.job_id,
        }


def bb84_circuit(n_bits: int):
    """
    Build a BB84 QKD circuit for n_bits.
    Requires qiskit. Gated — will raise ImportError if not installed.
    Returns (circuit, alice_bases, alice_bits, bob_bases).
    """
    from qiskit import QuantumCircuit

    alice_bits  = [random.randint(0, 1) for _ in range(n_bits)]
    alice_bases = [random.randint(0, 1) for _ in range(n_bits)]
    bob_bases   = [random.randint(0, 1) for _ in range(n_bits)]

    qc = QuantumCircuit(n_bits, n_bits)
    for i in range(n_bits):
        if alice_bits[i] == 1:
            qc.x(i)
        if alice_bases[i] == 1:
            qc.h(i)
    for i in range(n_bits):
        if bob_bases[i] == 1:
            qc.h(i)
    qc.measure(range(n_bits), range(n_bits))

    return qc, alice_bases, alice_bits, bob_bases


def sift_and_evaluate(
    alice_bits: list[int],
    alice_bases: list[int],
    bob_bases: list[int],
    bob_results: list[int],
    backend: str = "classical",
    job_id: Optional[str] = None,
) -> QKDResult:
    """Sift keys on matching bases, compute QBER, return QKDResult."""
    sifted_alice, sifted_bob = [], []
    for i in range(len(alice_bases)):
        if alice_bases[i] == bob_bases[i]:
            sifted_alice.append(alice_bits[i])
            sifted_bob.append(bob_results[i])

    if not sifted_alice:
        return QKDResult([], [], 1.0, False, backend, job_id)

    errors = sum(a != b for a, b in zip(sifted_alice, sifted_bob))
    qber   = errors / len(sifted_alice)

    return QKDResult(
        raw_key    = alice_bits,
        sifted_key = sifted_alice,
        qber       = qber,
        secure     = qber < 0.11,
        backend    = backend,
        job_id     = job_id,
    )
