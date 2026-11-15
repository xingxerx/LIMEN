"""Circuit-level logical round-trip for a surface-code patch.

Where limen.ecc.certificate computes the logical error rate analytically
by enumerating error patterns, this module runs the correction loop as an
actual gate circuit on limen.gates.simulator: it injects X errors, builds
and executes a Z-syndrome-extraction circuit, reads the syndrome off the
ancilla register of the resulting statevector, decodes, applies the
correction, and reports whether a logical bit-flip survived.

Scope matches limen.ecc.decoder / limen.ecc.certificate exactly:
independent per-qubit bit-flip (X-error) noise only. The computational
basis state |0...0> on the data qubits is the logical-|0> codeword for
this noise model (a +1 eigenstate of every Z-stabilizer and of logical
Z), so no separate state-preparation unitary is needed; preparing a
general superposition codeword (the X-stabilizer-projected encoder) is
out of scope for this milestone, consistent with the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from limen.ecc.decoder import LookupDecoder
from limen.ecc.surface_code import SurfaceCodePatch
from limen.gates.ir import CircuitIR, GateInstruction
from limen.gates.simulator import probabilities


@dataclass
class RoundTripResult:
    """Outcome of one executed encode -> error -> syndrome -> decode -> correct loop.

    Attributes:
        error_qubits: Data-qubit indices that were bit-flipped.
        syndrome: Z-stabilizer syndrome read from the executed circuit,
            in patch.z_stabilizers order.
        correction: Data-qubit indices the decoder chose to flip.
        logical_error: True if (error XOR correction) anticommutes with
            logical Z (an undetected logical bit-flip survived).
        residual_weight: Number of data qubits still flipped after correction.
        circuit: The executed syndrome-extraction CircuitIR.
    """

    error_qubits: list[int]
    syndrome: tuple[int, ...]
    correction: list[int]
    logical_error: bool
    residual_weight: int
    circuit: CircuitIR
    metadata: dict[str, Any] = field(default_factory=dict)


def build_z_syndrome_circuit(
    patch: SurfaceCodePatch, x_errors: list[int] | None = None
) -> CircuitIR:
    """Build a Z-syndrome-extraction circuit, optionally pre-injecting X errors.

    Data qubits keep their indices 0..n_data-1; one ancilla per
    Z-stabilizer is appended in patch.z_stabilizers order, so the
    ancilla for z_stabilizers[j] is qubit n_data + j. Each X error is an
    x gate on a data qubit; each Z-stabilizer is read by cx(data, ancilla)
    over its support.

    Args:
        patch: The SurfaceCodePatch to build the circuit for.
        x_errors: Data-qubit indices to bit-flip before extraction.

    Returns:
        A CircuitIR with n_qubits = n_data + len(patch.z_stabilizers).
    """
    n_data = len(patch.data_qubits)
    errors = list(x_errors or [])
    instructions: list[GateInstruction] = [GateInstruction("x", [q]) for q in errors]

    for j, support in enumerate(patch.z_stabilizers):
        ancilla = n_data + j
        for q in support:
            instructions.append(GateInstruction("cx", [q, ancilla]))

    return CircuitIR(
        n_qubits=n_data + len(patch.z_stabilizers),
        instructions=instructions,
        metadata={
            "distance": patch.distance,
            "n_data_qubits": n_data,
            "n_z_stabilizers": len(patch.z_stabilizers),
            "x_errors": errors,
        },
    )


def run_logical_roundtrip(
    patch: SurfaceCodePatch, decoder: LookupDecoder, x_errors: list[int]
) -> RoundTripResult:
    """Execute one bit-flip correction loop on the statevector simulator.

    The data qubits start in the |0...0> logical-|0> codeword. The
    syndrome is not computed analytically: it is read from the ancilla
    register of the statevector produced by executing the extraction
    circuit, which is what makes this a circuit-level (gate-executed)
    verification of the same X-error model certify_logical_qubit scores.

    Args:
        patch: The SurfaceCodePatch under test.
        decoder: A LookupDecoder built for the same patch.
        x_errors: Data-qubit indices to bit-flip.

    Returns:
        A RoundTripResult describing the syndrome, correction, and whether
        a logical error survived.
    """
    n_data = len(patch.data_qubits)
    n_z = len(patch.z_stabilizers)
    circuit = build_z_syndrome_circuit(patch, x_errors)

    outcome = _execute_syndrome_circuit(circuit)
    syndrome = tuple(int(outcome[n_data + j]) for j in range(n_z))

    correction = decoder.decode(syndrome)
    residual = set(x_errors) ^ set(correction)
    logical_error = len(residual & set(patch.logical_z)) % 2 == 1

    return RoundTripResult(
        error_qubits=list(x_errors),
        syndrome=syndrome,
        correction=correction,
        logical_error=logical_error,
        residual_weight=len(residual),
        circuit=circuit,
        metadata={"executed_outcome": outcome},
    )


# The pure-Python statevector (and its Rust kernel) stops at 24 qubits. A
# distance-3 patch is 9 data + 8 ancilla = 17 qubits and fits; distance 5
# is 25 + 24 = 49 and never did, so Tier 2 at fidelity >= 0.99 (d=5) could
# not be verified at all. The syndrome circuit is pure Clifford (x, h, cx),
# so above the limit it runs on Aer's stabilizer simulator instead, which
# handles any distance. The state is a definite computational basis state
# either way, so one shot reads the syndrome exactly.
_PY_STATEVECTOR_MAX_QUBITS = 24


def _execute_syndrome_circuit(circuit: CircuitIR) -> str:
    """Qubit-0-first outcome bitstring of a Clifford syndrome circuit."""
    if circuit.n_qubits <= _PY_STATEVECTOR_MAX_QUBITS:
        dist = probabilities(circuit)
        # All registers are in a definite computational state, so the
        # distribution has a single outcome to read the syndrome from.
        return max(dist, key=dist.get)
    try:
        from limen.gates.qiskit_exec import STABILIZER_BACKEND_NAME, run_circuit
    except ImportError as exc:  # pragma: no cover - qiskit missing
        raise ImportError(
            f"surface-code round trip on {circuit.n_qubits} qubits needs qiskit-aer's "
            "stabilizer simulator: pip install limen-compiler[ibm]"
        ) from exc
    result = run_circuit(circuit, backend_name=STABILIZER_BACKEND_NAME, shots=1)
    (qiskit_bits,) = result.counts  # exactly one outcome
    return qiskit_bits[::-1]  # Qiskit is qubit-0 rightmost; CircuitIR convention is leftmost


def verify_corrects_all_weight_one(
    patch: SurfaceCodePatch, decoder: LookupDecoder
) -> bool:
    """Return True iff every single-qubit X error is corrected with no logical error.

    Runs an executed round-trip for each data qubit; a distance-3 code
    must survive all of them.
    """
    return all(
        not run_logical_roundtrip(patch, decoder, [q]).logical_error
        for q in range(len(patch.data_qubits))
    )
