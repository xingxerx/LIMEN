"""Gate-model circuit IR: a parallel track to the diagonal Ising/QUBO IR in limen.core.

Exports CircuitIR/GateInstruction unconditionally (no extra dependencies).
Qiskit-dependent execution (limen.gates.qiskit_exec) is imported lazily by
callers, following the same optional-SDK convention as the rest of LIMEN.
"""

from limen.gates.ir import CircuitIR, GateInstruction, KNOWN_GATES
from limen.gates.synthesis import decompose_unitary_1q, u_matrix

__all__ = [
    "CircuitIR",
    "GateInstruction",
    "KNOWN_GATES",
    "decompose_unitary_1q",
    "u_matrix",
]
