"""Tests for the gate-model CircuitIR."""

import unittest

from limen.gates.ir import CircuitIR, GateInstruction


class TestCircuitIR(unittest.TestCase):

    def test_roundtrip(self):
        circuit = CircuitIR(
            n_qubits=2,
            instructions=[
                GateInstruction("h", [0]),
                GateInstruction("cx", [0, 1]),
                GateInstruction("rz", [1], [0.5]),
            ],
            metadata={"name": "bell-ish"},
        )
        circuit2 = CircuitIR.from_dict(circuit.to_dict())
        self.assertEqual(circuit2.n_qubits, circuit.n_qubits)
        self.assertEqual(len(circuit2.instructions), 3)
        self.assertEqual(circuit2.instructions[2].params, [0.5])
        self.assertEqual(circuit2.metadata, circuit.metadata)

    def test_valid_circuit_has_no_errors(self):
        circuit = CircuitIR(
            n_qubits=2, instructions=[GateInstruction("h", [0]), GateInstruction("cx", [0, 1])]
        )
        self.assertEqual(circuit.validate(), [])

    def test_rejects_unknown_gate(self):
        circuit = CircuitIR(n_qubits=1, instructions=[GateInstruction("frobnicate", [0])])
        errors = circuit.validate()
        self.assertTrue(any("unknown gate" in e for e in errors))

    def test_rejects_out_of_range_qubit(self):
        circuit = CircuitIR(n_qubits=1, instructions=[GateInstruction("cx", [0, 1])])
        errors = circuit.validate()
        self.assertTrue(any("out of range" in e for e in errors))

    def test_rejects_wrong_arity(self):
        circuit = CircuitIR(n_qubits=2, instructions=[GateInstruction("h", [0, 1])])
        errors = circuit.validate()
        self.assertTrue(any("expects 1 qubit" in e for e in errors))

    def test_rejects_wrong_param_count(self):
        circuit = CircuitIR(n_qubits=1, instructions=[GateInstruction("rz", [0], [])])
        errors = circuit.validate()
        self.assertTrue(any("expects 1 param" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
