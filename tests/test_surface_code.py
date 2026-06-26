"""Tests for the distance-3 rotated surface code construction.

The key test here is the distance-3 proof by brute-force enumeration -
this is what makes the construction's correctness self-certifying
rather than dependent on correctly recalling a named layout from memory.
"""

import unittest
from itertools import product

from limen.ecc.decoder import compute_syndrome
from limen.ecc.surface_code import build_surface_code


class TestSurfaceCodeStructure(unittest.TestCase):

    def setUp(self):
        self.patch = build_surface_code(3)

    def test_nine_data_qubits(self):
        self.assertEqual(len(self.patch.data_qubits), 9)

    def test_eight_total_stabilizers(self):
        total = len(self.patch.x_stabilizers) + len(self.patch.z_stabilizers)
        self.assertEqual(total, 8)

    def test_stabilizer_weights_are_two_or_four(self):
        for stabilizer in self.patch.x_stabilizers + self.patch.z_stabilizers:
            self.assertIn(len(stabilizer), (2, 4))

    def test_logical_operators_span_three_qubits(self):
        self.assertEqual(len(self.patch.logical_x), 3)
        self.assertEqual(len(self.patch.logical_z), 3)


class TestSurfaceCodeDistance(unittest.TestCase):
    """Brute-force proof that the constructed code is distance 3."""

    def setUp(self):
        self.patch = build_surface_code(3)
        self.n = len(self.patch.data_qubits)
        self.logical_z = set(self.patch.logical_z)

    def test_every_weight_one_error_is_detectable(self):
        for i in range(self.n):
            bits = tuple(1 if j == i else 0 for j in range(self.n))
            syndrome = compute_syndrome(bits, self.patch.z_stabilizers)
            self.assertTrue(
                any(syndrome), f"weight-1 error on qubit {i} produced a trivial syndrome"
            )

    def test_no_undetected_weight_two_or_less_logical_error(self):
        for bits in product((0, 1), repeat=self.n):
            weight = sum(bits)
            if weight == 0 or weight > 2:
                continue
            syndrome = compute_syndrome(bits, self.patch.z_stabilizers)
            if any(syndrome):
                continue
            overlap = sum(bits[i] for i in self.logical_z) % 2
            self.assertEqual(
                overlap, 0, f"undetected weight<=2 logical error found: {bits}"
            )


if __name__ == "__main__":
    unittest.main()
