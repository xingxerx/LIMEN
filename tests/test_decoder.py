"""Tests for the exact minimum-weight lookup decoder."""

import unittest

from limen.ecc.decoder import LookupDecoder
from limen.ecc.surface_code import build_surface_code


class TestLookupDecoder(unittest.TestCase):

    def setUp(self):
        self.patch = build_surface_code(3)
        self.decoder = LookupDecoder(self.patch)
        self.logical_z = set(self.patch.logical_z)

    def test_corrects_every_single_qubit_error(self):
        n = len(self.patch.data_qubits)
        for i in range(n):
            syndrome = self.decoder.syndrome_for([i])
            correction = self.decoder.decode(syndrome)
            residual = {i} ^ set(correction)
            overlap = len(residual & self.logical_z) % 2
            self.assertEqual(
                overlap, 0, f"weight-1 error on qubit {i} left a residual logical error"
            )

    def test_no_error_decodes_to_no_correction(self):
        syndrome = self.decoder.syndrome_for([])
        self.assertEqual(self.decoder.decode(syndrome), [])

    def test_unknown_syndrome_falls_back_to_no_correction(self):
        # A syndrome with a different shape can't occur in practice, but
        # decode() must still be total for any syndrome it's given.
        bogus = tuple(0 for _ in self.patch.z_stabilizers)
        self.assertEqual(self.decoder.decode(bogus), [])


if __name__ == "__main__":
    unittest.main()
