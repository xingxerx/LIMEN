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
    "Install it with: pip install limen-compiler[ibm]  "
    "(or: pip install qiskit qiskit-aer)"
)


# Backend name for the local GPU statevector tier 0. Kept as a literal here
# (not imported from limen.router) so limen.gates has no dependency on the
# router package; limen.router.budget_router.LOCAL_GPU_BACKEND_NAME must match.
GPU_BACKEND_NAME = "aer_gpu"


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
    metadata: dict[str, Any] = {"backend": backend_name, "shots": shots, "seed": seed}
    if backend_name == GPU_BACKEND_NAME:
        # Local GPU statevector tier 0 (limen.router.route_local_gpu_first).
        # Single precision halves VRAM per amplitude; the router's qubit cap
        # (gpu_qubit_cap, 8 bytes/amplitude) assumes exactly this setting.
        # A CPU-only qiskit-aer build accepts device="GPU" at construction
        # and only fails inside run(), so the whole attempt is guarded and
        # falls back to the CPU simulator, recording why in the metadata
        # rather than failing the job.
        try:
            backend = AerSimulator(
                method="statevector", device="GPU", precision="single", seed_simulator=seed
            )
            transpiled = transpile(qc, backend)
            counts = backend.run(transpiled, shots=shots).result().get_counts()
            metadata["device"] = "GPU"
        except Exception as exc:  # AerError / RuntimeError from the C++ controller
            backend = AerSimulator(method="automatic", seed_simulator=seed)
            transpiled = transpile(qc, backend)
            counts = backend.run(transpiled, shots=shots).result().get_counts()
            metadata["device"] = "CPU"
            metadata["gpu_fallback_reason"] = f"{type(exc).__name__}: {exc}"
    else:
        method = "statevector" if backend_name == "statevector" else "automatic"
        backend = AerSimulator(method=method, seed_simulator=seed)
        transpiled = transpile(qc, backend)
        counts = backend.run(transpiled, shots=shots).result().get_counts()

    return CircuitResult(
        counts=counts,
        circuit_depth=transpiled.depth(),
        metadata=metadata,
    )
