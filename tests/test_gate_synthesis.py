"""Tests for single-qubit unitary synthesis (ZYZ decomposition)."""

import cmath
import math
import unittest

from limen.exceptions import GateSynthesisError
from limen.gates.synthesis import decompose_unitary_1q, u_matrix

_SQRT2 = math.sqrt(2.0)

PAULI_X = [[0, 1], [1, 0]]
HADAMARD = [[1 / _SQRT2, 1 / _SQRT2], [1 / _SQRT2, -1 / _SQRT2]]
IDENTITY = [[1, 0], [0, 1]]


def _assert_unitary_equal_up_to_phase(m1, m2, tol=1e-6):
    flat1 = [x for row in m1 for x in row]
    flat2 = [x for row in m2 for x in row]
    idx = max(range(4), key=lambda i: abs(flat1[i]))
    ratio = flat2[idx] / flat1[idx]
    assert abs(abs(ratio) - 1.0) < tol, f"ratio {ratio} is not a pure phase"
    for a, b in zip(flat1, flat2):
        assert abs(b - ratio * a) < tol, f"{b} != {ratio} * {a}"


class TestDecomposeUnitary1q(unittest.TestCase):

    def test_pauli_x(self):
        gate = decompose_unitary_1q(PAULI_X)
        reconstructed = u_matrix(*gate.params)
        _assert_unitary_equal_up_to_phase(PAULI_X, reconstructed)

    def test_hadamard(self):
        gate = decompose_unitary_1q(HADAMARD)
        reconstructed = u_matrix(*gate.params)
        _assert_unitary_equal_up_to_phase(HADAMARD, reconstructed)

    def test_identity(self):
        gate = decompose_unitary_1q(IDENTITY)
        reconstructed = u_matrix(*gate.params)
        _assert_unitary_equal_up_to_phase(IDENTITY, reconstructed)

    def test_arbitrary_angles_round_trip(self):
        for theta, phi, lam in [
            (0.3, 1.1, -0.7),
            (math.pi / 2, 0.0, math.pi),
            (2.5, -1.2, 0.4),
        ]:
            original = u_matrix(theta, phi, lam)
            gate = decompose_unitary_1q(original)
            reconstructed = u_matrix(*gate.params)
            _assert_unitary_equal_up_to_phase(original, reconstructed)

    def test_gate_name_and_qubit(self):
        gate = decompose_unitary_1q(HADAMARD)
        self.assertEqual(gate.name, "u")
        self.assertEqual(gate.qubits, [0])
        self.assertEqual(len(gate.params), 3)

    def test_rejects_multi_qubit_matrix(self):
        identity_3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        with self.assertRaises(GateSynthesisError):
            decompose_unitary_1q(identity_3)

    def test_rejects_non_unitary_matrix(self):
        with self.assertRaises(GateSynthesisError):
            decompose_unitary_1q([[1, 1], [0, 1]])


if __name__ == "__main__":
    unittest.main()
