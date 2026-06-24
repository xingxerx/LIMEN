# limen/quantum_channel/teleport.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from limen.quantum_channel.channel_delta import ChannelDeltaModel

# IBM dynamic-circuit classical feedforward (qc.measure -> conditional gate)
# completes on the order of ~1us on real hardware.
_DEFAULT_FEEDFORWARD_LATENCY_MS = 0.001


@dataclass
class TeleportResult:
    fidelity_estimate: float
    success: bool
    backend: str
    job_id: Optional[str] = None
    channel_delta: Optional[ChannelDeltaModel] = None

    def to_dict(self) -> dict:
        return {
            "fidelity_estimate": self.fidelity_estimate,
            "success": self.success,
            "backend": self.backend,
            "job_id": self.job_id,
            "channel_delta": self.channel_delta.to_dict() if self.channel_delta else None,
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


def run_teleport_qpu(
    token: str,
    crn: str,
    backend_name: str = "ibm_kingston",
    shots: int = 1000,
) -> TeleportResult:
    """Submit the teleportation circuit to a real IBM QPU and estimate fidelity.

    Requires qiskit and qiskit-ibm-runtime. Gated — raises ImportError if not
    installed.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2  # type: ignore[import]
    from qiskit.transpiler.preset_passmanagers import (  # type: ignore[import]
        generate_preset_pass_manager,
    )

    from limen.quantum_channel.teleport_analysis import estimate_fidelity

    qc = teleport_circuit()

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=crn,
    )
    backend = service.backend(backend_name)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled = pm.run(qc)

    sampler = SamplerV2(mode=backend)
    job = sampler.run([transpiled], shots=shots)
    job_id = job.job_id()
    pub_result = job.result()[0]
    counts: dict[str, int] = pub_result.data.c.get_counts()

    result = estimate_fidelity(counts, backend=backend_name, job_id=job_id)
    result.channel_delta = _channel_delta_from_backend(backend)
    return result


def _channel_delta_from_backend(backend) -> Optional[ChannelDeltaModel]:
    """Build a ChannelDeltaModel from the backend's live T2 calibration."""
    t2_values = [
        props.t2
        for q in range(backend.num_qubits)
        if (props := backend.qubit_properties(q)) is not None and props.t2
    ]
    if not t2_values:
        return None

    median_t2_us = sorted(t2_values)[len(t2_values) // 2] * 1e6
    return ChannelDeltaModel(
        latency_ms=_DEFAULT_FEEDFORWARD_LATENCY_MS, t2_us=median_t2_us
    )
