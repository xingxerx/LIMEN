"""Tests for the D-Wave backend adapter (limen/backends/dwave.py)."""

import sys
import unittest
from unittest.mock import MagicMock, patch

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.backends.dwave import DWaveResult, run_dwave

_DWAVE_AVAILABLE = False
try:
    import dwave  # noqa: F401
    _DWAVE_AVAILABLE = True
except ModuleNotFoundError:
    pass


def _make_encoding(qubo: dict):
    graph = from_qubo_dict(qubo)
    return compile_lexicographic(graph, default_hardware_graph(8))


_TRIVIAL_QUBO = {("q0", "q0"): -1.0, ("q1", "q1"): -1.0, ("q0", "q1"): 2.0}

_MAXCUT_QUBO = {
    ("A", "B"): 1.0, ("A", "C"): 1.0, ("B", "C"): 1.0,
    ("B", "D"): 1.0,
    ("A", "A"): -2.0, ("B", "B"): -3.0, ("C", "C"): -2.0, ("D", "D"): -1.0,
}


class TestDWaveImportError(unittest.TestCase):
    """Test that a helpful ImportError is raised when dwave is absent."""

    def test_import_error_raised_when_dwave_missing(self):
        """run_dwave must raise ImportError with an install hint if SDK absent."""
        encoding = _make_encoding(_TRIVIAL_QUBO)

        # Temporarily hide dwave and dimod from sys.modules.
        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith(("dwave", "dimod", "neal"))}
        try:
            with patch.dict("sys.modules", {"dimod": None, "dwave": None, "neal": None}):
                with self.assertRaises(ImportError) as ctx:
                    run_dwave(encoding, num_reads=10)
        finally:
            sys.modules.update(saved)

        self.assertIn("pip install", str(ctx.exception))


@unittest.skipUnless(_DWAVE_AVAILABLE, "dwave Ocean SDK not installed")
class TestDWaveSimulator(unittest.TestCase):
    """Tests that require the dwave Ocean SDK to be installed."""

    def test_returns_correct_types(self):
        """run_dwave returns a DWaveResult with correct field types."""
        encoding = _make_encoding(_TRIVIAL_QUBO)
        result = run_dwave(encoding, num_reads=50, seed=0)

        self.assertIsInstance(result, DWaveResult)
        self.assertEqual(len(result.samples), 50)
        self.assertEqual(len(result.energies), 50)
        self.assertIsInstance(result.best_assignment, dict)
        self.assertIsInstance(result.best_energy, float)
        self.assertIsInstance(result.timing, dict)
        for val in result.best_assignment.values():
            self.assertIn(val, (0, 1))

    def test_deterministic_with_same_seed(self):
        """Same seed and encoding must produce identical best_assignment."""
        encoding = _make_encoding(_TRIVIAL_QUBO)
        r1 = run_dwave(encoding, num_reads=100, seed=7)
        r2 = run_dwave(encoding, num_reads=100, seed=7)
        self.assertEqual(r1.best_assignment, r2.best_assignment)
        self.assertEqual(r1.best_energy, r2.best_energy)

    def test_maxcut_best_energy_non_positive(self):
        """Sampler must find a valid cut (energy <= 0) on a 4-node Max-Cut."""
        encoding = _make_encoding(_MAXCUT_QUBO)
        result = run_dwave(encoding, num_reads=200, seed=42)

        self.assertLessEqual(result.best_energy, 0.0)
        for val in result.best_assignment.values():
            self.assertIn(val, (0, 1))


if __name__ == "__main__":
    unittest.main()
