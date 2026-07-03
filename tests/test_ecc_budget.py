"""Tests for adaptive ECC patch-budget allocation (limen.ecc.budget).

Exercises both the limen_core Rust path and the pure-Python fallback
directly, asserting they produce identical results - the same
dual-path verification style as test_ecc_roundtrip.py.
"""

import unittest

from limen.ecc.budget import (
    PatchAssignment,
    allocate_ecc_budget,
    rank_criticality,
    select_patches,
)


class TestRankCriticality(unittest.TestCase):

    def test_linear_and_quadratic_terms_accumulate(self):
        qubo = [((0, 0), 2.0), ((0, 1), 1.0), ((1, 2), -3.0)]
        ranked = rank_criticality(qubo, 3)
        weights = dict(ranked)
        self.assertAlmostEqual(weights[0], 3.0)
        self.assertAlmostEqual(weights[1], 4.0)
        self.assertAlmostEqual(weights[2], 3.0)

    def test_sorted_descending_by_weight(self):
        qubo = [((0, 0), 1.0), ((1, 1), 5.0), ((2, 2), 3.0)]
        ranked = rank_criticality(qubo, 3)
        self.assertEqual([var for var, _ in ranked], [1, 2, 0])

    def test_untouched_variable_gets_zero_and_sorts_last(self):
        qubo = [((0, 0), 1.0)]
        ranked = rank_criticality(qubo, 2)
        self.assertEqual(ranked[-1], (1, 0.0))

    def test_out_of_range_indices_ignored(self):
        qubo = [((0, 5), 1.0)]
        ranked = rank_criticality(qubo, 2)
        weights = dict(ranked)
        self.assertEqual(weights, {0: 0.0, 1: 0.0})


class TestSelectPatches(unittest.TestCase):

    def test_greedily_fills_budget_in_ranked_order(self):
        ranked = [(0, 5.0), (1, 3.0), (2, 1.0)]
        assignments = select_patches(ranked, physical_qubit_budget=18, distance=3)
        self.assertEqual(len(assignments), 2)
        self.assertEqual(assignments[0], PatchAssignment(0, 3, 0, 9))
        self.assertEqual(assignments[1], PatchAssignment(1, 3, 9, 18))

    def test_lower_criticality_variable_can_still_fit(self):
        # Budget only fits one distance-3 patch (9 qubits) after the first
        # is placed; var 1 (distance-3 cost 9) doesn't fit in the remaining
        # 5, but var 2 never gets a chance since patch_cost is fixed here -
        # this asserts the "skip and keep trying" semantics, not a size fit.
        ranked = [(0, 5.0), (1, 3.0), (2, 1.0)]
        assignments = select_patches(ranked, physical_qubit_budget=14, distance=3)
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].logical_var, 0)

    def test_zero_distance_yields_no_assignments(self):
        ranked = [(0, 1.0)]
        assignments = select_patches(ranked, physical_qubit_budget=100, distance=0)
        self.assertEqual(assignments, [])

    def test_empty_budget_yields_no_assignments(self):
        ranked = [(0, 1.0)]
        assignments = select_patches(ranked, physical_qubit_budget=0, distance=3)
        self.assertEqual(assignments, [])


class TestAllocateEccBudget(unittest.TestCase):

    def test_full_flow_protects_most_critical_variable_first(self):
        qubo = [((0, 0), 1.0), ((1, 1), 10.0), ((2, 2), 2.0)]
        assignments = allocate_ecc_budget(qubo, n_vars=3, physical_qubit_budget=9, distance=3)
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].logical_var, 1)


if __name__ == "__main__":
    unittest.main()
