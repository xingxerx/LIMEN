"""Tests for syndrome-extraction circuit construction."""

import unittest

from limen.ecc.surface_code import build_surface_code
from limen.ecc.syndrome import build_syndrome_circuit
from limen.gates.ir import GateInstruction


class TestSyndromeCircuit(unittest.TestCase):

    def setUp(self):
        self.patch = build_surface_code(3)
        self.circuit = build_syndrome_circuit(self.patch)

    def test_qubit_count_is_data_plus_ancillas(self):
        n_data = len(self.patch.data_qubits)
        n_stabilizers = len(self.patch.z_stabilizers) + len(self.patch.x_stabilizers)
        self.assertEqual(self.circuit.n_qubits, n_data + n_stabilizers)

    def test_circuit_is_valid(self):
        self.assertEqual(self.circuit.validate(), [])

    def test_z_stabilizer_uses_cx_into_ancilla(self):
        n_data = len(self.patch.data_qubits)
        first_z_support = self.patch.z_stabilizers[0]
        ancilla = n_data
        expected = [GateInstruction("cx", [q, ancilla]) for q in first_z_support]
        self.assertEqual(self.circuit.instructions[: len(expected)], expected)

    def test_x_stabilizer_uses_hadamard_sandwich(self):
        n_data = len(self.patch.data_qubits)
        n_z = len(self.patch.z_stabilizers)
        ancilla = n_data + n_z
        first_x_support = self.patch.x_stabilizers[0]

        # Skip past all Z-stabilizer instructions to the first X-stabilizer block.
        offset = sum(len(s) for s in self.patch.z_stabilizers)
        block = self.circuit.instructions[offset : offset + len(first_x_support) + 2]

        self.assertEqual(block[0], GateInstruction("h", [ancilla]))
        self.assertEqual(block[-1], GateInstruction("h", [ancilla]))
        for q, ins in zip(first_x_support, block[1:-1]):
            self.assertEqual(ins, GateInstruction("cx", [ancilla, q]))


if __name__ == "__main__":
    unittest.main()
