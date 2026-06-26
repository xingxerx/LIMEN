"""Tests for the pure-Python CircuitIR statevector simulator (offline, no Qiskit)."""

import math
import unittest

from limen.gates.ir import CircuitIR, GateInstruction
from limen.gates.simulator import probabilities, sample_counts, statevector


class TestStatevectorSimulator(unittest.TestCase):

    def _bell(self) -> CircuitIR:
        return CircuitIR(
            n_qubits=2,
            instructions=[GateInstruction("h", [0]), GateInstruction("cx", [0, 1])],
        )

    def test_bell_statevector(self):
        state = statevector(self._bell())
        self.assertAlmostEqual(abs(state[0]), 1 / math.sqrt(2), places=9)
        self.assertAlmostEqual(abs(state[3]), 1 / math.sqrt(2), places=9)
        self.assertAlmostEqual(abs(state[1]), 0.0, places=9)
        self.assertAlmostEqual(abs(state[2]), 0.0, places=9)

    def test_bell_probabilities(self):
        dist = probabilities(self._bell())
        self.assertEqual(set(dist), {"00", "11"})
        self.assertAlmostEqual(dist["00"], 0.5, places=9)
        self.assertAlmostEqual(dist["11"], 0.5, places=9)

    def test_norm_is_preserved(self):
        circuit = CircuitIR(
            n_qubits=3,
            instructions=[
                GateInstruction("h", [0]),
                GateInstruction("rx", [1], [0.7]),
                GateInstruction("ry", [2], [1.3]),
                GateInstruction("cx", [0, 2]),
                GateInstruction("cz", [1, 2]),
            ],
        )
        norm = sum((a.conjugate() * a).real for a in statevector(circuit))
        self.assertAlmostEqual(norm, 1.0, places=9)

    def test_x_flips_qubit_zero_first_ordering(self):
        circuit = CircuitIR(n_qubits=2, instructions=[GateInstruction("x", [0])])
        dist = probabilities(circuit)
        # qubit 0 is the leftmost character: only qubit 0 is set.
        self.assertEqual(set(dist), {"10"})

    def test_swap_exchanges_qubits(self):
        circuit = CircuitIR(
            n_qubits=2,
            instructions=[GateInstruction("x", [0]), GateInstruction("swap", [0, 1])],
        )
        dist = probabilities(circuit)
        self.assertEqual(set(dist), {"01"})

    def test_u_gate_matches_hadamard(self):
        # U(pi/2, 0, pi) == H up to global phase.
        circuit = CircuitIR(
            n_qubits=1,
            instructions=[GateInstruction("u", [0], [math.pi / 2, 0.0, math.pi])],
        )
        dist = probabilities(circuit)
        self.assertAlmostEqual(dist["0"], 0.5, places=9)
        self.assertAlmostEqual(dist["1"], 0.5, places=9)

    def test_sample_counts_deterministic_and_supported(self):
        a = sample_counts(self._bell(), shots=500, seed=7)
        b = sample_counts(self._bell(), shots=500, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(sum(a.values()), 500)
        self.assertTrue(set(a).issubset({"00", "11"}))

    def test_invalid_circuit_raises(self):
        bad = CircuitIR(n_qubits=1, instructions=[GateInstruction("cx", [0, 1])])
        with self.assertRaises(ValueError):
            statevector(bad)


if __name__ == "__main__":
    unittest.main()
