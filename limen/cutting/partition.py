# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Find cuts in a CircuitIR and partition it into sub-circuits.

Thin wrapper around qiskit_addon_cutting.find_cuts + partition_problem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from limen.gates.ir import CircuitIR
from limen.gates.qiskit_exec import to_qiskit_circuit

_INSTALL_MSG = (
    "qiskit-addon-cutting is required for circuit cutting. "
    "Install it with: pip install limen[cutting]"
)


def _check_qiskit_addon_cutting() -> None:
    try:
        import qiskit_addon_cutting  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc


def _unmeasured_copy(qc: Any) -> Any:
    """Return a copy of qc with no measurements and no classical bits.

    partition_problem and find_cuts both require circuits with no
    classical registers/bits; to_qiskit_circuit always appends measurements,
    so this strips them back off rather than duplicating circuit-building
    logic in two places.
    """
    from qiskit import QuantumCircuit

    stripped = QuantumCircuit(qc.num_qubits)
    for instruction in qc.data:
        if instruction.operation.name == "measure":
            continue
        qubit_indices = [qc.find_bit(q).index for q in instruction.qubits]
        stripped.append(instruction.operation, qubit_indices)
    return stripped


@dataclass
class CutPlan:
    """The result of finding cuts in and partitioning a circuit.

    Attributes:
        subcircuits: Partition label -> uncut sub-circuit (qiskit QuantumCircuit).
        subobservables: Partition label -> PauliList of sub-observables.
        bases: One QPDBasis per cut gate/wire, in cut order.
        num_cuts: Number of cuts inserted.
        original_num_qubits: Qubit count of the circuit before cutting.
        metadata: find_cuts' own metadata dict (cuts, sampling_overhead, minimum_reached).
    """

    subcircuits: dict[Any, Any]
    subobservables: dict[Any, Any]
    bases: list[Any]
    num_cuts: int
    original_num_qubits: int
    metadata: dict[str, Any] = field(default_factory=dict)


def find_cuts_and_partition(
    circuit: CircuitIR,
    observable: str,
    max_subcircuit_qubits: int,
) -> CutPlan:
    """Find cut locations in circuit and partition it into sub-circuits.

    Args:
        circuit: The CircuitIR to cut. Gates acting on more than 2 qubits
            are not supported by find_cuts and will raise ValueError.
        observable: A Pauli string (e.g. "ZZZZ") of length circuit.n_qubits,
            the observable whose expectation value will be reconstructed.
        max_subcircuit_qubits: Maximum qubits per sub-circuit -- the qubit
            budget of the smallest validated backend you intend to target.

    Returns:
        A CutPlan with one sub-circuit/sub-observable per partition.

    Raises:
        ImportError: If qiskit-addon-cutting is not installed.
        ValueError: If circuit.validate() fails, or if find_cuts/partition_problem reject the input.
    """
    _check_qiskit_addon_cutting()
    from qiskit.quantum_info import PauliList
    from qiskit_addon_cutting import (
        DeviceConstraints,
        OptimizationParameters,
        find_cuts,
        partition_problem,
    )

    if len(observable) != circuit.n_qubits:
        raise ValueError(
            f"observable length ({len(observable)}) must equal circuit.n_qubits "
            f"({circuit.n_qubits})"
        )

    qc = _unmeasured_copy(to_qiskit_circuit(circuit))

    constraints = DeviceConstraints(qubits_per_subcircuit=max_subcircuit_qubits)
    optimization = OptimizationParameters()
    cut_circuit, find_cuts_metadata = find_cuts(qc, optimization, constraints)

    partitioned = partition_problem(cut_circuit, observables=PauliList([observable]))
    if partitioned.subobservables is None:
        raise ValueError("partition_problem returned no subobservables")

    return CutPlan(
        subcircuits=dict(partitioned.subcircuits),
        subobservables=dict(partitioned.subobservables),
        bases=list(partitioned.bases),
        num_cuts=len(find_cuts_metadata["cuts"]),
        original_num_qubits=circuit.n_qubits,
        metadata=find_cuts_metadata,
    )
