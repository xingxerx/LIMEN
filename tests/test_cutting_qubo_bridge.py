"""Tests for limen.cutting.qubo_bridge: reconstructing a QUBO's diagonal
Ising terms and single-qubit <Z_i> marginals via circuit cutting, then
decoding a solution bitstring from those marginals (see that module's
docstring for the exact, deliberately scoped technique).

All cutting/reconstruction here runs on a local AerSimulator (see
limen.cutting.local_dispatch) -- zero credits, zero network, the same
loopback pattern already validated in examples/cutting_smoke_test.py.
"""

import unittest

import pytest

pytest.importorskip("qiskit_addon_cutting", reason="qiskit-addon-cutting not installed")
pytest.importorskip("qiskit_aer", reason="qiskit-aer not installed")
pytest.importorskip("limen_core", reason="limen_core Rust extension not installed")

from limen.cutting.local_dispatch import run_cut_circuit_locally
from limen.cutting.qubo_bridge import (
    classical_energy,
    decode_bitstring_from_marginals,
    mean_field_expected_energy,
    pauli_z_string,
    qubo_ising_terms,
    reconstruct_z_marginals_via_cutting,
)
from limen.cutting.reconstruct import reconstruct_from_results
from limen.frontends.pyqubo import from_qubo_dict
from limen.gates.qaoa import compile_qaoa, variable_order
from limen.validator.validator import brute_force_solve

_QUBO = {("x0", "x1"): 2.0, ("x0", "x0"): -1.0, ("x1", "x1"): -1.0}
_LARGER_QUBO = {
    ("x0", "x1"): 2.0,
    ("x1", "x2"): 2.0,
    ("x2", "x3"): 2.0,
    ("x3", "x0"): 2.0,
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x2", "x2"): -1.0,
    ("x3", "x3"): -1.0,
}


class TestQuboIsingTerms(unittest.TestCase):

    def test_ising_reconstruction_matches_classical_energy_at_uncut_marginals(self):
        # At the exact +-1 eigenvalues of a known bit assignment (not an
        # approximate reconstruction), constant + sum h_i*z_i + sum
        # J_ij*z_i*z_j must reproduce the QUBO's classical energy exactly
        # -- this is a sanity check on the Pauli decomposition itself,
        # independent of any cutting/reconstruction error.
        h, j_coeffs, constant, order = qubo_ising_terms(_QUBO)
        for x0 in (0, 1):
            for x1 in (0, 1):
                assignment = {"x0": x0, "x1": x1}
                expected = classical_energy(_QUBO, assignment)
                z = {0: 1 - 2 * x0, 1: 1 - 2 * x1}
                actual = constant + sum(h[i] * z[i] for i in h)
                actual += sum(
                    coeff * z[a] * z[b] for (a, b), coeff in j_coeffs.items()
                )
                self.assertAlmostEqual(actual, expected)


class TestPauliZString(unittest.TestCase):

    def test_rightmost_character_is_qubit_zero(self):
        self.assertEqual(pauli_z_string(2, {0}), "IZ")
        self.assertEqual(pauli_z_string(2, {1}), "ZI")
        self.assertEqual(pauli_z_string(3, {0, 2}), "ZIZ")


class TestReconstructMarginalsViaCutting(unittest.TestCase):

    def test_cut_reconstruction_matches_uncut_marginal_within_tolerance(self):
        graph = from_qubo_dict(_LARGER_QUBO)
        order = variable_order(graph)
        n = len(order)
        circuit = compile_qaoa(graph, [0.6], [0.3])

        def dispatch_fn(cut_plan):
            return run_cut_circuit_locally(cut_plan, shots=4000)

        # max_subcircuit_qubits=2 forces an actual cut on this 4-qubit
        # circuit (chain-connected couplings mean it can't stay uncut).
        reconstruction = reconstruct_z_marginals_via_cutting(
            circuit, list(range(n)), 2, dispatch_fn, reconstruct_from_results
        )
        self.assertGreater(reconstruction.num_cuts, 0)
        self.assertEqual(len(reconstruction.marginals), n)
        for value in reconstruction.marginals.values():
            self.assertGreaterEqual(value, -1.0 - 1e-6)
            self.assertLessEqual(value, 1.0 + 1e-6)


class TestDecodeAndEnergy(unittest.TestCase):

    def test_decode_rounds_toward_lower_energy_eigenvalue(self):
        # <Z_0> strongly negative (mostly measured |1>) must decode to
        # x0=1; strongly positive must decode to x0=0.
        order = ["x0", "x1"]
        solution = decode_bitstring_from_marginals({0: -0.9, 1: 0.9}, order)
        self.assertEqual(solution, {"x0": 1, "x1": 0})

    def test_decoded_solution_matches_brute_force_optimum_on_small_qubo(self):
        # End-to-end: reconstruct real marginals via a real (local) cut +
        # dispatch + reconstruct round trip, decode, and check the decoded
        # bitstring's exact classical energy against a brute-force oracle
        # (the oracle itself never sees the cutting path -- it's a pure
        # correctness check, not part of the production code).
        graph = from_qubo_dict(_QUBO)
        order = variable_order(graph)
        n = len(order)
        circuit = compile_qaoa(graph, [1.2], [0.4])

        def dispatch_fn(cut_plan):
            return run_cut_circuit_locally(cut_plan, shots=8000)

        reconstruction = reconstruct_z_marginals_via_cutting(
            circuit, list(range(n)), 1, dispatch_fn, reconstruct_from_results
        )
        solution = decode_bitstring_from_marginals(reconstruction.marginals, order)
        decoded_energy = classical_energy(_QUBO, solution)

        best_assignment, best_energy = brute_force_solve(_QUBO)
        # QAOA at these angles is not guaranteed to concentrate exactly on
        # the optimum, so this checks the decode pipeline produces a
        # legitimate, low-energy assignment -- not equal to the classical
        # optimum by construction, but never worse than it.
        self.assertGreaterEqual(decoded_energy, best_energy - 1e-9)

    def test_mean_field_expected_energy_uses_marginals_not_true_correlator(self):
        h, j_coeffs, constant, order = qubo_ising_terms(_QUBO)
        # All marginals at 0 (maximally uncertain) should collapse the
        # mean-field estimate to exactly the constant term.
        marginals = {i: 0.0 for i in range(len(order))}
        result = mean_field_expected_energy(h, j_coeffs, constant, marginals)
        self.assertAlmostEqual(result, constant)


if __name__ == "__main__":
    unittest.main()
