"""End-to-end smoke tests for the full LIMEN pipeline."""

import unittest

from limen import (
    Interaction,
    LogicalGraph,
    PhysicalEncoding,
    Variable,
    compile_lexicographic,
    default_hardware_graph,
    from_qubo_dict,
    validate,
)

_SAMPLE_QUBO = {("x0", "x0"): -1.0, ("x1", "x1"): -1.0, ("x0", "x1"): 2.0}


def _sample_graph() -> LogicalGraph:
    return from_qubo_dict(_SAMPLE_QUBO)


def _sample_encoding():
    return compile_lexicographic(_sample_graph(), default_hardware_graph(4))


class TestCorePipeline(unittest.TestCase):

    def test_logical_graph_roundtrip(self):
        g = LogicalGraph(
            variables=[Variable("x0"), Variable("x1")],
            interactions=[Interaction("x0", "x1", 2.0)],
        )
        g2 = LogicalGraph.from_dict(g.to_dict())
        self.assertEqual(len(g2.variables), len(g.variables))
        self.assertEqual(len(g2.interactions), len(g.interactions))
        self.assertEqual(g2.interactions[0].weight, g.interactions[0].weight)

    def test_validate_rejects_unknown_variable(self):
        g = LogicalGraph(
            variables=[Variable("x0")],
            interactions=[Interaction("x0", "x1", 1.0)],
        )
        errors = g.validate()
        self.assertTrue(len(errors) > 0)

    def test_duplicate_interaction_detected(self):
        g = LogicalGraph(
            variables=[Variable("x0"), Variable("x1")],
            interactions=[Interaction("x0", "x1", 1.0), Interaction("x1", "x0", 2.0)],
        )
        errors = g.validate()
        self.assertTrue(len(errors) > 0)

    def test_from_qubo_dict_builds_valid_graph(self):
        g = from_qubo_dict(_SAMPLE_QUBO)
        self.assertIsInstance(g, LogicalGraph)
        self.assertEqual(len(g.variables), 2)
        self.assertEqual(len(g.interactions), 3)
        self.assertEqual(g.validate(), [])

    def test_compile_lexicographic_basic(self):
        enc = _sample_encoding()
        self.assertEqual(len(enc.embedding), 2)
        self.assertEqual(len(enc.qubo), 3)
        self.assertGreaterEqual(enc.chain_strength, 1.0)

    def test_physical_encoding_roundtrip(self):
        enc = _sample_encoding()
        enc2 = PhysicalEncoding.from_dict(enc.to_dict())
        self.assertTrue(all(isinstance(k, tuple) for k in enc2.qubo))
        self.assertEqual(enc2.chain_strength, enc.chain_strength)

    def test_validate_high_confidence(self):
        enc = _sample_encoding()
        result = validate(enc, runs=500)
        self.assertGreater(result.confidence, 0.5)
        self.assertEqual(result.total_runs, 500)
        self.assertIsNotNone(result.classical_energy)

    def test_default_hardware_graph_structure(self):
        hw = default_hardware_graph(3)
        self.assertEqual(len(hw), 3)
        for neighbours in hw.values():
            self.assertEqual(len(neighbours), 2)


if __name__ == "__main__":
    unittest.main()
