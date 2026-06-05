"""Tests for the Qiskit backend adapter (limen/backends/qiskit_backend.py)."""

import sys
import unittest

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.backends.qiskit_backend import QiskitResult, _qubo_to_ising, run_qiskit
from limen.validator.validator import brute_force_solve

_QISKIT_AVAILABLE = False
try:
    import qiskit  # noqa: F401
    import qiskit_aer  # noqa: F401
    _QISKIT_AVAILABLE = True
except ModuleNotFoundError:
    pass

# Small 2-variable QUBO: minimum at x0=1, x1=0 (energy -1)
_TRIVIAL_QUBO = {("q0", "q0"): -1.0, ("q1", "q1"): -1.0, ("q0", "q1"): 2.0}

# 4-node Max-Cut QUBO
_MAXCUT_QUBO = {
    ("A", "B"): 1.0, ("A", "C"): 1.0, ("B", "C"): 1.0, ("B", "D"): 1.0,
    ("A", "A"): -2.0, ("B", "B"): -3.0, ("C", "C"): -2.0, ("D", "D"): -1.0,
}


def _make_encoding(qubo: dict):
    return compile_lexicographic(from_qubo_dict(qubo), default_hardware_graph(8))


class TestQuboToIsing(unittest.TestCase):
    """Test _qubo_to_ising conversion against direct energy evaluation."""

    def test_energy_equivalence_all_assignments(self):
        """Ising energy + constant must equal QUBO energy at every assignment."""
        from itertools import product

        qubo = _TRIVIAL_QUBO
        h, J = _qubo_to_ising(qubo)
        variables = sorted({v for pair in qubo for v in pair})

        for bits in product((0, 1), repeat=len(variables)):
            assignment = dict(zip(variables, bits))
            qubo_e = sum(w * assignment[i] * assignment[j] for (i, j), w in qubo.items())

            spin = {v: 2 * assignment[v] - 1 for v in variables}
            ising_e = (
                sum(h[v] * spin[v] for v in variables)
                + sum(w * spin[i] * spin[j] for (i, j), w in J.items())
            )
            # The two energies may differ by a constant offset; verify the
            # *relative* ordering is preserved (and they differ by same const).
            self._check_const = getattr(self, "_check_const", qubo_e - ising_e)
            self.assertAlmostEqual(qubo_e - ising_e, self._check_const, places=10)


class TestQiskitImportError(unittest.TestCase):
    """Verify a helpful ImportError is raised when qiskit is absent."""

    def test_import_error_raised_when_qiskit_missing(self):
        """run_qiskit must raise ImportError with install hint if SDK absent."""
        encoding = _make_encoding(_TRIVIAL_QUBO)

        saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("qiskit")}
        try:
            with unittest.mock.patch.dict("sys.modules", {"qiskit": None}):
                with self.assertRaises(ImportError) as ctx:
                    run_qiskit(encoding, num_shots=10, algorithm="exact")
        finally:
            sys.modules.update(saved)

        self.assertIn("pip install", str(ctx.exception))


@unittest.skipUnless(_QISKIT_AVAILABLE, "qiskit / qiskit-aer not installed")
class TestQiskitSimulator(unittest.TestCase):
    """Tests that require qiskit and qiskit-aer to be installed."""

    def test_exact_matches_brute_force(self):
        """exact algorithm must return best_energy matching brute_force_solve."""
        encoding = _make_encoding(_TRIVIAL_QUBO)
        result = run_qiskit(encoding, num_shots=16, algorithm="exact", seed=0)

        bf = brute_force_solve(encoding.qubo)
        self.assertIsNotNone(bf)
        _, bf_energy = bf

        self.assertIsInstance(result, QiskitResult)
        self.assertAlmostEqual(result.best_energy, bf_energy, places=9)
        for val in result.best_assignment.values():
            self.assertIn(val, (0, 1))

    def test_qaoa_returns_valid_result(self):
        """qaoa algorithm returns a QiskitResult with valid binary assignment."""
        encoding = _make_encoding(_TRIVIAL_QUBO)
        result = run_qiskit(encoding, num_shots=50, algorithm="qaoa", reps=1, seed=42)

        self.assertIsInstance(result, QiskitResult)
        self.assertIsInstance(result.best_assignment, dict)
        for val in result.best_assignment.values():
            self.assertIn(val, (0, 1))
        self.assertEqual(result.metadata["algorithm"], "qaoa")

    def test_deterministic_same_seed(self):
        """Same seed must produce identical best_assignment on simulator."""
        encoding = _make_encoding(_TRIVIAL_QUBO)
        r1 = run_qiskit(encoding, num_shots=50, algorithm="exact", seed=7)
        r2 = run_qiskit(encoding, num_shots=50, algorithm="exact", seed=7)
        self.assertEqual(r1.best_assignment, r2.best_assignment)
        self.assertEqual(r1.best_energy, r2.best_energy)

    def test_maxcut_best_energy_non_positive(self):
        """Best energy on 4-node Max-Cut must be <= 0 with a valid binary result."""
        encoding = _make_encoding(_MAXCUT_QUBO)
        result = run_qiskit(encoding, num_shots=64, algorithm="exact", seed=42)

        self.assertLessEqual(result.best_energy, 0.0)
        for val in result.best_assignment.values():
            self.assertIn(val, (0, 1))


# Import mock is needed for TestQiskitImportError regardless of environment.
import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
