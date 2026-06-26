"""Gate-model circuit IR for LIMEN.

A parallel track to limen.core.ir.LogicalGraph: where LogicalGraph
represents diagonal Ising/QUBO Hamiltonians (Z and ZZ terms only),
CircuitIR represents an arbitrary sequence of quantum gates, including
off-diagonal operators (X, Y, H, and arbitrary single-qubit unitaries
via limen.gates.synthesis). The two IRs are not unified in this
milestone; the existing analog/Ising compiler path is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# name -> (arity, expected number of params)
KNOWN_GATES: dict[str, tuple[int, int]] = {
    "h": (1, 0),
    "x": (1, 0),
    "y": (1, 0),
    "z": (1, 0),
    "s": (1, 0),
    "t": (1, 0),
    "rx": (1, 1),
    "ry": (1, 1),
    "rz": (1, 1),
    "u": (1, 3),
    "cx": (2, 0),
    "cz": (2, 0),
    "swap": (2, 0),
}


@dataclass
class GateInstruction:
    """A single gate application.

    Attributes:
        name: Gate name; must be a key in KNOWN_GATES.
        qubits: Qubit indices the gate acts on, in the order KNOWN_GATES
            expects (e.g. cx is [control, target]).
        params: Gate parameters (e.g. [theta] for rx/ry/rz, [theta, phi,
            lambda] for u). Empty for parameter-free gates.
    """

    name: str
    qubits: list[int]
    params: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "qubits": list(self.qubits), "params": list(self.params)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GateInstruction":
        return cls(
            name=d["name"],
            qubits=[int(q) for q in d["qubits"]],
            params=[float(p) for p in d.get("params", [])],
        )


@dataclass
class CircuitIR:
    """A gate-model quantum circuit.

    Attributes:
        n_qubits: Number of qubits in the circuit.
        instructions: Ordered sequence of gate applications.
        metadata: Arbitrary annotations.
    """

    n_qubits: int
    instructions: list[GateInstruction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "instructions": [ins.to_dict() for ins in self.instructions],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CircuitIR":
        return cls(
            n_qubits=int(d["n_qubits"]),
            instructions=[GateInstruction.from_dict(i) for i in d.get("instructions", [])],
            metadata=dict(d.get("metadata", {})),
        )

    def validate(self) -> list[str]:
        """Check the circuit for structural errors.

        Checks performed:
            - Every instruction's gate name is in KNOWN_GATES.
            - Every instruction's qubit count matches the gate's arity.
            - Every instruction's param count matches the gate's signature.
            - Every qubit index is within range(n_qubits).

        Returns:
            A list of error strings. An empty list means the circuit is valid.
        """
        errors: list[str] = []
        for idx, ins in enumerate(self.instructions):
            spec = KNOWN_GATES.get(ins.name)
            if spec is None:
                errors.append(f"Instruction[{idx}]: unknown gate '{ins.name}'")
                continue
            arity, n_params = spec
            if len(ins.qubits) != arity:
                errors.append(
                    f"Instruction[{idx}]: gate '{ins.name}' expects {arity} qubit(s), "
                    f"got {len(ins.qubits)}"
                )
            if len(ins.params) != n_params:
                errors.append(
                    f"Instruction[{idx}]: gate '{ins.name}' expects {n_params} param(s), "
                    f"got {len(ins.params)}"
                )
            for q in ins.qubits:
                if not (0 <= q < self.n_qubits):
                    errors.append(
                        f"Instruction[{idx}]: qubit index {q} out of range "
                        f"for n_qubits={self.n_qubits}"
                    )
        return errors
