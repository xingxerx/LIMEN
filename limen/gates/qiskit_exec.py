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
# Aer stabilizer (Clifford) simulator: used by limen.ecc for surface-code
# round trips too wide for any statevector (distance 5 is 49 qubits).
STABILIZER_BACKEND_NAME = "aer_stabilizer"


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

    qc = to_qiskit_circuit(circuit)
    metadata: dict[str, Any] = {"backend": backend_name, "shots": shots, "seed": seed}

    def _run(backend: Any) -> tuple[dict[str, int], int]:
        transpiled = transpile(qc, backend)
        return backend.run(transpiled, shots=shots).result().get_counts(), transpiled.depth()

    counts, depth = run_on_aer(backend_name, _run, seed=seed, metadata=metadata)
    return CircuitResult(counts=counts, circuit_depth=depth, metadata=metadata)


def make_aer_simulator(backend_name: str = "aer_simulator", seed: int = 42, **options: Any) -> Any:
    """Build the AerSimulator a LIMEN backend name stands for.

    ``"statevector"`` -> exact CPU statevector; ``"aer_gpu"`` ->
    statevector on ``device="GPU"`` in single precision (the setting the
    router's VRAM cap assumes); anything else -> ``method="automatic"``.
    Shared by run_circuit and limen.cutting.local_dispatch so the GPU
    tier is configured in exactly one place.
    """
    _check_qiskit()
    from qiskit_aer import AerSimulator

    if backend_name == GPU_BACKEND_NAME:
        return AerSimulator(
            method="statevector", device="GPU", precision="single", seed_simulator=seed, **options
        )
    if backend_name == STABILIZER_BACKEND_NAME:
        # Clifford-only circuits (surface-code syndrome extraction) at any
        # width; thousands of qubits are fine. Not a statevector.
        return AerSimulator(method="stabilizer", seed_simulator=seed, **options)
    method = "statevector" if backend_name == "statevector" else "automatic"
    return AerSimulator(method=method, seed_simulator=seed, **options)


def run_on_aer(
    backend_name: str,
    fn: Any,
    *,
    seed: int = 42,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Call ``fn(backend)`` on the simulator *backend_name* names, with GPU fallback.

    For ``"aer_gpu"`` the attempt is wrapped: a CPU-only qiskit-aer build
    accepts ``device="GPU"`` at construction and only fails inside
    ``run()``, so on any exception the same *fn* is re-run on the CPU
    ``automatic`` simulator and *metadata* records ``device="CPU"`` plus
    ``gpu_fallback_reason`` instead of the job failing. Other backend
    names run once with no fallback (their errors are real errors).
    """
    meta = metadata if metadata is not None else {}
    if backend_name != GPU_BACKEND_NAME:
        meta.setdefault("device", "CPU")
        return fn(make_aer_simulator(backend_name, seed))
    try:
        result = fn(make_aer_simulator(backend_name, seed))
        meta["device"] = "GPU"
        return result
    except Exception as exc:  # AerError / RuntimeError from the C++ controller
        meta["device"] = "CPU"
        meta["gpu_fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return fn(make_aer_simulator("aer_simulator", seed))
