"""Tests for the QUBO/Ising -> QAOA CircuitIR bridge (offline, no Qiskit)."""

import unittest

from limen.frontends.pyqubo import from_qubo_dict
from limen.gates.qaoa import (
    bitstring_to_assignment,
    compile_qaoa,
    qubo_to_ising,
    variable_order,
)


class TestQuboToIsing(unittest.TestCase):

    def test_linear_term_mapping(self):
        graph = from_qubo_dict({("a", "a"): -1.0})
        h, j, order = qubo_to_ising(graph)
        # w * x = w/2 - (w/2) Z  ->  h = -0.5 * w
        self.assertAlmostEqual(h[0], 0.5)
        self.assertEqual(j, {})
        self.assertEqual(order, ["a"])

    def test_quadratic_term_mapping(self):
        graph = from_qubo_dict({("a", "b"): 4.0})
        h, j, order = qubo_to_ising(graph)
        self.assertEqual(order, ["a", "b"])
        self.assertAlmostEqual(h[0], -1.0)
        self.assertAlmostEqual(h[1], -1.0)
        self.assertAlmostEqual(j[(0, 1)], 1.0)


class TestCompileQaoa(unittest.TestCase):

    def setUp(self):
        self.graph = from_qubo_dict(
            {("x0", "x0"): -1.0, ("x1", "x1"): -1.0, ("x0", "x1"): 2.0}
        )

    def test_circuit_is_valid(self):
        circuit = compile_qaoa(self.graph, [0.5], [0.5])
        self.assertEqual(circuit.validate(), [])
        self.assertEqual(circuit.n_qubits, 2)

    def test_starts_with_hadamard_layer(self):
        circuit = compile_qaoa(self.graph, [0.5], [0.5])
        self.assertEqual(circuit.instructions[0].name, "h")
        self.assertEqual(circuit.instructions[1].name, "h")
        self.assertEqual(circuit.metadata["ansatz"], "qaoa")
        self.assertEqual(circuit.metadata["layers"], 1)

    def test_layer_count_scales_instructions(self):
        one = compile_qaoa(self.graph, [0.5], [0.5])
        two = compile_qaoa(self.graph, [0.5, 0.5], [0.5, 0.5])
        # Two layers add another cost+mixer block beyond the shared H init.
        self.assertGreater(len(two.instructions), len(one.instructions))

    def test_mismatched_params_raises(self):
        with self.assertRaises(ValueError):
            compile_qaoa(self.graph, [0.5, 0.5], [0.5])

    def test_bitstring_to_assignment_uses_variable_order(self):
        order = variable_order(self.graph)
        self.assertEqual(order, ["x0", "x1"])
        self.assertEqual(bitstring_to_assignment("10", order), {"x0": 1, "x1": 0})


if __name__ == "__main__":
    unittest.main()
