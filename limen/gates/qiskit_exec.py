"""Qiskit execution backend for CircuitIR.

Maps CircuitIR gate instructions onto qiskit.QuantumCircuit method calls
and executes via a local AerSimulator or IBM Runtime, reusing the same
backend-selection pattern already established in
limen.communication.channel.QuantumChannel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from limen.gates.ir import CircuitIR

_INSTALL_MSG = (
    "The Qiskit SDK is required to execute CircuitIR. "
    "Install it with: pip install limen[ibm]  "
    "(or: pip install qiskit qiskit-aer)"
)


def _check_qiskit() -> None:
    try:
        import qiskit  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc


@dataclass
class CircuitResult:
    """The result of executing a CircuitIR.

    Attributes:
        counts: Measurement counts, keyed by bitstring.
        circuit_depth: Transpiled circuit depth.
        metadata: Backend options and execution telemetry.
    """

    counts: dict[str, int]
    circuit_depth: int
    metadata: dict[str, Any] = field(default_factory=dict)


def to_qiskit_circuit(circuit: CircuitIR) -> Any:
    """Build a qiskit.QuantumCircuit from a CircuitIR, with measurement on all qubits.

    Args:
        circuit: The CircuitIR to translate. Must pass circuit.validate().

    Returns:
        A qiskit QuantumCircuit with one classical bit per qubit, measured
        at the end in qubit order.

    Raises:
        ImportError: If qiskit is not installed.
        ValueError: If circuit.validate() reports any errors.
    """
    _check_qiskit()
    from qiskit import QuantumCircuit

    errors = circuit.validate()
    if errors:
        raise ValueError(f"invalid CircuitIR: {errors}")

    qc = QuantumCircuit(circuit.n_qubits, circuit.n_qubits)
    gate_methods = {
        "h": qc.h,
        "x": qc.x,
        "y": qc.y,
        "z": qc.z,
        "s": qc.s,
        "t": qc.t,
        "rx": qc.rx,
        "ry": qc.ry,
        "rz": qc.rz,
        "u": qc.u,
        "cx": qc.cx,
        "cz": qc.cz,
        "swap": qc.swap,
    }
    for ins in circuit.instructions:
        gate_methods[ins.name](*ins.params, *ins.qubits)

    qc.measure(range(circuit.n_qubits), range(circuit.n_qubits))
    return qc


def run_circuit(
    circuit: CircuitIR,
    backend_name: str = "aer_simulator",
    shots: int = 1000,
    seed: int = 42,
) -> CircuitResult:
    """Execute a CircuitIR on a local Qiskit simulator and return measurement counts.

    Args:
        circuit: The CircuitIR to execute.
        backend_name: "aer_simulator" or "statevector" for ideal simulation.
        shots: Number of measurement shots.
        seed: RNG seed for the simulator.

    Returns:
        A CircuitResult with measurement counts and transpiled circuit depth.
    """
    _check_qiskit()
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    qc = to_qiskit_circuit(circuit)
    method = "statevector" if backend_name == "statevector" else "automatic"
    backend = AerSimulator(method=method, seed_simulator=seed)
    transpiled = transpile(qc, backend)
    job = backend.run(transpiled, shots=shots)
    counts = job.result().get_counts()

    return CircuitResult(
        counts=counts,
        circuit_depth=transpiled.depth(),
        metadata={"backend": backend_name, "shots": shots, "seed": seed},
    )
