"""Tests for exact logical error rate certification."""

import unittest

from limen.ecc.certificate import certify_logical_qubit
from limen.ecc.decoder import LookupDecoder
from limen.ecc.surface_code import build_surface_code

# ibm_kingston's measured two-qubit error rate, per benchmarks/RESULTS.md.
IBM_KINGSTON_TWO_QUBIT_ERROR_RATE = 1.91e-3


class TestLogicalErrorCertificate(unittest.TestCase):

    def setUp(self):
        self.patch = build_surface_code(3)
        self.decoder = LookupDecoder(self.patch)

    def test_suppresses_logical_error_at_ibm_kingston_rate(self):
        cert = certify_logical_qubit(
            self.patch, self.decoder, IBM_KINGSTON_TWO_QUBIT_ERROR_RATE
        )
        self.assertLess(cert.logical_error_rate, IBM_KINGSTON_TWO_QUBIT_ERROR_RATE)
        # Net benefit should be substantial, not marginal, in this regime.
        self.assertLess(cert.logical_error_rate / IBM_KINGSTON_TWO_QUBIT_ERROR_RATE, 0.1)

    def test_scales_roughly_quadratically_for_small_p(self):
        p1, p2 = 1e-3, 2e-3
        cert1 = certify_logical_qubit(self.patch, self.decoder, p1)
        cert2 = certify_logical_qubit(self.patch, self.decoder, p2)
        # Doubling p should roughly quadruple logical_error_rate (d=3
        # corrects all weight-1 errors, fails starting at weight-2).
        ratio = cert2.logical_error_rate / cert1.logical_error_rate
        self.assertGreater(ratio, 3.0)
        self.assertLess(ratio, 5.0)

    def test_zero_physical_error_rate_gives_zero_logical_error(self):
        cert = certify_logical_qubit(self.patch, self.decoder, 0.0)
        self.assertEqual(cert.logical_error_rate, 0.0)

    def test_certificate_fields(self):
        cert = certify_logical_qubit(self.patch, self.decoder, 1e-3)
        self.assertEqual(cert.distance, 3)
        self.assertEqual(cert.n_physical_qubits, 9)
        self.assertEqual(cert.n_logical_qubits, 1)
        self.assertEqual(cert.decoder, "LookupDecoder")
        self.assertTrue(cert.notes)


if __name__ == "__main__":
    unittest.main()
