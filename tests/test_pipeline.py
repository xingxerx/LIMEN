"""End-to-end pipeline tests: QUBO -> QAOA -> simulate -> certify (offline)."""

import unittest

from limen.pipeline import EndToEndCertificate, run_pipeline


class TestEndToEndPipeline(unittest.TestCase):

    def test_single_variable_optimum(self):
        cert = run_pipeline({("a", "a"): -1.0}, encode_logical=False)
        self.assertEqual(cert.solution, {"a": 1})
        self.assertAlmostEqual(cert.energy, -1.0)
        self.assertTrue(cert.is_optimal)

    def test_unique_two_variable_optimum(self):
        # Minimize x0 - x1  ->  x0=0, x1=1, energy -1.
        cert = run_pipeline(
            {("x0", "x0"): 1.0, ("x1", "x1"): -1.0}, encode_logical=False
        )
        self.assertEqual(cert.solution, {"x0": 0, "x1": 1})
        self.assertTrue(cert.is_optimal)
        self.assertGreater(cert.success_probability, 0.0)

    def test_finds_classical_optimum_energy(self):
        cert = run_pipeline(
            {("x0", "x0"): -1.0, ("x1", "x1"): -1.0, ("x0", "x1"): 2.0},
            encode_logical=False,
        )
        self.assertEqual(cert.energy, cert.classical_energy)
        self.assertTrue(cert.is_optimal)

    def test_logical_certificate_composed_when_error_rate_given(self):
        cert = run_pipeline(
            {("a", "a"): -1.0, ("b", "b"): -1.0},
            physical_error_rate=0.01,
            distance=3,
        )
        self.assertIsNotNone(cert.logical_error_rate)
        self.assertIsNotNone(cert.aggregate_logical_error_rate)
        self.assertEqual(cert.distance, 3)
        self.assertEqual(cert.n_logical_qubits, 2)
        # Aggregate over 2 independent logical qubits >= single-qubit rate.
        self.assertGreaterEqual(
            cert.aggregate_logical_error_rate, cert.logical_error_rate
        )

    def test_logical_certificate_skipped_without_error_rate(self):
        cert = run_pipeline({("a", "a"): -1.0})
        self.assertIsNone(cert.logical_error_rate)
        self.assertIsNone(cert.distance)

    def test_certificate_is_serializable(self):
        cert = run_pipeline({("a", "a"): -1.0}, physical_error_rate=0.02)
        self.assertIsInstance(cert, EndToEndCertificate)
        d = cert.to_dict()
        self.assertEqual(d["solution"], {"a": 1})
        self.assertIn("logical_error_rate", d)
        self.assertIn("qaoa_params", d)


if __name__ == "__main__":
    unittest.main()
