import unittest
import limen_core
from limen.analog.neutral_atom import NeutralAtomCompiler
from limen.analog.photonic import PhotonicCompiler

class TestHardwareCompilers(unittest.TestCase):
    def test_neutral_atom_compiler(self):
        compiler = NeutralAtomCompiler()
        v0 = limen_core.sim.Variable(name="v0", domain="binary")
        v1 = limen_core.sim.Variable(name="v1", domain="binary")
        graph = limen_core.sim.LogicalGraph(variables=[v0, v1], interactions=[])
        layout = compiler.generate_layout(graph)
        self.assertEqual(len(layout.atom_positions), 2)
        self.assertTrue(len(layout.omega) > 0)

    def test_photonic_compiler(self):
        compiler = PhotonicCompiler()
        v0 = limen_core.sim.Variable(name="v0", domain="binary")
        v1 = limen_core.sim.Variable(name="v1", domain="binary")
        i0 = limen_core.sim.Interaction(i="v0", j="v1", weight=0.5)
        graph = limen_core.sim.LogicalGraph(variables=[v0, v1], interactions=[i0])
        encoding = compiler.build_gbs_encoding(graph)
        self.assertEqual(len(encoding.adjacency_matrix), 2)
        self.assertEqual(encoding.adjacency_matrix[0][1], 0.5)

if __name__ == "__main__":
    unittest.main()
