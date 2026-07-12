"""Tests for limen.router.problem_profile: the substrate-aware routing
signal. frustration_index is a heuristic (see module docstring), so these
tests check directional sanity (frustrated > unfrustrated) and determinism,
not an exact ground-truth value."""

import unittest

from limen.router.problem_profile import compute_problem_profile, frustration_index


def _to_int_qubo(qubo: dict[tuple[int, int], float]) -> list[tuple[tuple[int, int], float]]:
    return list(qubo.items())


class TestFrustrationIndex(unittest.TestCase):

    def test_no_couplings_is_unfrustrated(self):
        qubo = _to_int_qubo({(0, 0): -1.0, (1, 1): -1.0})
        self.assertEqual(frustration_index(qubo, 2), 0.0)

    def test_bipartite_ferromagnetic_is_easily_satisfiable(self):
        # A single attractive (w<0) coupling between two free variables is
        # trivially satisfiable: set both to 1. Frustration should be 0.
        qubo = _to_int_qubo({(0, 1): -1.0})
        self.assertEqual(frustration_index(qubo, 2, restarts=4, seed=0), 0.0)

    def test_frustrated_triangle_scores_higher_than_unfrustrated_pair(self):
        # Three variables, all-antiferromagnetic (w>0) couplings on every
        # edge of a triangle -- with binary (not spin) variables this is
        # jointly satisfiable (all three off), so pick an assignment that
        # actually frustrates: one attractive (w<0) edge competing against
        # two repulsive (w>0) edges sharing a vertex forces a trade-off no
        # single assignment fully satisfies.
        frustrated = _to_int_qubo({(0, 1): -1.0, (0, 2): 1.0, (1, 2): 1.0})
        unfrustrated = _to_int_qubo({(0, 1): -1.0})
        f_score = frustration_index(frustrated, 3, restarts=8, seed=0)
        u_score = frustration_index(unfrustrated, 2, restarts=8, seed=0)
        self.assertGreater(f_score, u_score)

    def test_deterministic_given_seed(self):
        qubo = _to_int_qubo({(0, 1): -1.0, (0, 2): 1.0, (1, 2): 1.0})
        a = frustration_index(qubo, 3, restarts=6, seed=7)
        b = frustration_index(qubo, 3, restarts=6, seed=7)
        self.assertEqual(a, b)

    def test_bounded_in_unit_interval(self):
        qubo = _to_int_qubo({(0, 1): -1.0, (0, 2): 1.0, (1, 2): 1.0, (0, 0): 2.0})
        score = frustration_index(qubo, 3, restarts=6, seed=3)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestProblemProfile(unittest.TestCase):

    def test_edge_density_full_graph(self):
        qubo = _to_int_qubo({(0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0})
        profile = compute_problem_profile(qubo, 3)
        self.assertEqual(profile.edge_density, 1.0)

    def test_edge_density_sparse_graph(self):
        qubo = _to_int_qubo({(0, 1): 1.0})
        profile = compute_problem_profile(qubo, 4)
        # 1 edge out of C(4,2)=6 possible pairs.
        self.assertAlmostEqual(profile.edge_density, 1 / 6)

    def test_max_coupling_magnitude(self):
        qubo = _to_int_qubo({(0, 1): -3.5, (0, 2): 1.0})
        profile = compute_problem_profile(qubo, 3)
        self.assertEqual(profile.max_coupling_magnitude, 3.5)

    def test_single_variable_no_couplings(self):
        profile = compute_problem_profile([], 1)
        self.assertEqual(profile.edge_density, 0.0)
        self.assertEqual(profile.frustration_index, 0.0)
        self.assertEqual(profile.max_coupling_magnitude, 0.0)


if __name__ == "__main__":
    unittest.main()
