"""Tests for the circuit-level surface-code logical round-trip (offline, no Qiskit).

These assert that the syndrome is genuinely read from a gate circuit
executed on the statevector simulator, and that the executed correction
loop matches the analytic X-error model in decoder.py / certificate.py.
"""

import unittest

from limen.ecc.decoder import LookupDecoder, compute_syndrome
from limen.ecc.encoder import (
    build_z_syndrome_circuit,
    run_logical_roundtrip,
    verify_corrects_all_weight_one,
)
from limen.ecc.surface_code import build_surface_code


class TestLogicalRoundTrip(unittest.TestCase):

    def setUp(self):
        self.patch = build_surface_code(3)
        self.decoder = LookupDecoder(self.patch)
        self.n_data = len(self.patch.data_qubits)

    def test_syndrome_circuit_is_valid(self):
        circuit = build_z_syndrome_circuit(self.patch, [0])
        self.assertEqual(circuit.validate(), [])
        self.assertEqual(
            circuit.n_qubits, self.n_data + len(self.patch.z_stabilizers)
        )

    def test_executed_syndrome_matches_analytic(self):
        # The syndrome read off the executed circuit must equal the one
        # compute_syndrome derives algebraically, for every weight-1 error.
        for q in range(self.n_data):
            result = run_logical_roundtrip(self.patch, self.decoder, [q])
            bits = tuple(1 if i == q else 0 for i in range(self.n_data))
            expected = compute_syndrome(bits, self.patch.z_stabilizers)
            self.assertEqual(result.syndrome, expected, f"qubit {q}")

    def test_no_error_is_noop(self):
        result = run_logical_roundtrip(self.patch, self.decoder, [])
        self.assertEqual(result.syndrome, tuple([0] * len(self.patch.z_stabilizers)))
        self.assertEqual(result.correction, [])
        self.assertFalse(result.logical_error)
        self.assertEqual(result.residual_weight, 0)

    def test_all_weight_one_errors_corrected(self):
        for q in range(self.n_data):
            result = run_logical_roundtrip(self.patch, self.decoder, [q])
            self.assertFalse(
                result.logical_error, f"weight-1 error on qubit {q} caused a logical error"
            )

    def test_verify_helper_passes_for_distance_three(self):
        self.assertTrue(verify_corrects_all_weight_one(self.patch, self.decoder))


if __name__ == "__main__":
    unittest.main()
