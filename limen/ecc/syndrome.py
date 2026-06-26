"""Syndrome-extraction circuit construction for a SurfaceCodePatch.

Builds one round of stabilizer measurement on top of limen.gates.ir,
making CircuitIR's first real consumer. Each stabilizer gets one
dedicated ancilla qubit, appended after the data qubits. Multi-round
extraction with ancilla reset is out of scope for this milestone.
"""

from __future__ import annotations

from limen.ecc.surface_code import SurfaceCodePatch
from limen.gates.ir import CircuitIR, GateInstruction


def build_syndrome_circuit(patch: SurfaceCodePatch) -> CircuitIR:
    """Build a single round of stabilizer-measurement circuit for `patch`.

    Z-stabilizers: cx(data_i, ancilla) for each data qubit in the
    support, measuring the ancilla in the Z basis to read off the
    stabilizer eigenvalue.
    X-stabilizers: h(ancilla), cx(ancilla, data_i) for each data qubit,
    h(ancilla), then measure.

    Ancilla qubits are appended after the data qubits, in stabilizer
    order: all Z-stabilizer ancillas first, then all X-stabilizer
    ancillas (matching the order patch.z_stabilizers + patch.x_stabilizers).

    Args:
        patch: The SurfaceCodePatch to build a syndrome circuit for.

    Returns:
        A CircuitIR with n_qubits = len(data_qubits) + n_stabilizers.
    """
    n_data = len(patch.data_qubits)
    instructions: list[GateInstruction] = []

    ancilla = n_data
    for support in patch.z_stabilizers:
        for q in support:
            instructions.append(GateInstruction("cx", [q, ancilla]))
        ancilla += 1

    for support in patch.x_stabilizers:
        instructions.append(GateInstruction("h", [ancilla]))
        for q in support:
            instructions.append(GateInstruction("cx", [ancilla, q]))
        instructions.append(GateInstruction("h", [ancilla]))
        ancilla += 1

    n_qubits = n_data + len(patch.z_stabilizers) + len(patch.x_stabilizers)
    return CircuitIR(n_qubits=n_qubits, instructions=instructions, metadata={
        "distance": patch.distance,
        "n_data_qubits": n_data,
        "n_z_stabilizers": len(patch.z_stabilizers),
        "n_x_stabilizers": len(patch.x_stabilizers),
    })
