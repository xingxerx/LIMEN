"""SDK-dependent tests for the Qiskit backend adapter.

Skipped cleanly when qiskit / qiskit-aer are not installed.
See tests/test_backends_offline.py for ImportError guard and Ising tests.
"""

import pytest

pytest.importorskip("qiskit", reason="qiskit not installed")
pytest.importorskip("qiskit_aer", reason="qiskit-aer not installed")

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.backends.qiskit_backend import QiskitResult, run_qiskit
from limen.validator.validator import brute_force_solve

_TRIVIAL_QUBO = {("q0", "q0"): -1.0, ("q1", "q1"): -1.0, ("q0", "q1"): 2.0}
_MAXCUT_QUBO = {
    ("A", "B"): 1.0, ("A", "C"): 1.0, ("B", "C"): 1.0, ("B", "D"): 1.0,
    ("A", "A"): -2.0, ("B", "B"): -3.0, ("C", "C"): -2.0, ("D", "D"): -1.0,
}


def _make_encoding(qubo: dict):
    return compile_lexicographic(from_qubo_dict(qubo), default_hardware_graph(8))


def test_exact_matches_brute_force():
    """exact algorithm best_energy must match brute_force_solve."""
    encoding = _make_encoding(_TRIVIAL_QUBO)
    result = run_qiskit(encoding, num_shots=16, algorithm="exact", seed=0)
    _, bf_energy = brute_force_solve(encoding.qubo)

    assert isinstance(result, QiskitResult)
    assert abs(result.best_energy - bf_energy) < 1e-9
    assert all(v in (0, 1) for v in result.best_assignment.values())


def test_qaoa_returns_valid_result():
    """qaoa algorithm returns a QiskitResult with a valid binary assignment."""
    encoding = _make_encoding(_TRIVIAL_QUBO)
    result = run_qiskit(encoding, num_shots=50, algorithm="qaoa", reps=1, seed=42)

    assert isinstance(result, QiskitResult)
    assert all(v in (0, 1) for v in result.best_assignment.values())
    assert result.metadata["algorithm"] == "qaoa"


def test_deterministic_same_seed():
    """Same seed must produce identical best_assignment on the exact simulator."""
    encoding = _make_encoding(_TRIVIAL_QUBO)
    r1 = run_qiskit(encoding, num_shots=50, algorithm="exact", seed=7)
    r2 = run_qiskit(encoding, num_shots=50, algorithm="exact", seed=7)
    assert r1.best_assignment == r2.best_assignment
    assert r1.best_energy == r2.best_energy


def test_maxcut_best_energy_non_positive():
    """Best energy on 4-node Max-Cut must be <= 0 with a valid binary result."""
    encoding = _make_encoding(_MAXCUT_QUBO)
    result = run_qiskit(encoding, num_shots=64, algorithm="exact", seed=42)

    assert result.best_energy <= 0.0
    assert all(v in (0, 1) for v in result.best_assignment.values())
