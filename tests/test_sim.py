import unittest

import pytest

limen_core = pytest.importorskip("limen_core", reason="limen_core Rust extension not installed")

from limen.exceptions import SizeViolation

class TestIsingSimulator(unittest.TestCase):
    def test_solve_exact_size_violation(self):
        sim = limen_core.sim.IsingSimulator()
        variables = [limen_core.sim.Variable(name=f"v{i}", domain="binary") for i in range(21)]
        graph = limen_core.sim.LogicalGraph(variables=variables, interactions=[])
        with self.assertRaises(SizeViolation):
            sim.solve_exact(graph)

    def test_solve_exact_simple(self):
        sim = limen_core.sim.IsingSimulator()
        # H = -1 * s0 * s1 - 0.5 * s0
        # States (s0, s1):
        # (1, 1): -1 - 0.5 = -1.5
        # (1, -1): 1 - 0.5 = 0.5
        # (-1, 1): 1 + 0.5 = 1.5
        # (-1, -1): -1 + 0.5 = -0.5
        # Min energy at (1, 1)
        v0 = limen_core.sim.Variable(name="v0", domain="binary")
        v1 = limen_core.sim.Variable(name="v1", domain="binary")
        i0 = limen_core.sim.Interaction(i="v0", j="v1", weight=-1.0)
        i1 = limen_core.sim.Interaction(i="v0", j="v0", weight=-0.5)
        graph = limen_core.sim.LogicalGraph(variables=[v0, v1], interactions=[i0, i1])
        result = sim.solve_exact(graph)
        self.assertEqual(result, [1, 1])

if __name__ == "__main__":
    unittest.main()
