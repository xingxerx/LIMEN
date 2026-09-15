"""Discovery Loop P1: Problem schema + TSP encoder -> run_route_request
-> approx_ratio 1.000 on an eil51 prefix (simulator only, no QPU)."""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from limen.formulation import Problem
from limen.frontends.tsp import (
    approx_ratio,
    classical_optimal_tour,
    eil51_prefix_problem,
    encode_tsp,
    score_assignment,
    to_route_request,
    tour_to_assignment,
)
from limen.pipeline import run_route_request
from limen.router import DEFAULT_FLEET
from limen.validator.validator import brute_force_solve


class TestProblemSchema(unittest.TestCase):
    def test_requires_id_and_class(self):
        with self.assertRaises(ValueError):
            Problem(id="", problem_class="tsp")
        with self.assertRaises(ValueError):
            Problem(id="x", problem_class="")


class TestTspEncoder(unittest.TestCase):
    def test_eil51_prefix_known_optimum_matches_brute_force_tour(self):
        problem = eil51_prefix_problem(4)
        dist = problem.metadata["distance_matrix"]
        opt_len, opt_tour = classical_optimal_tour(dist)
        self.assertEqual(problem.known_optimum, float(opt_len))
        self.assertEqual(problem.metadata["classical_optimal_tour"], opt_tour)

    def test_encode_emits_route_request_qubo(self):
        problem = encode_tsp(eil51_prefix_problem(2))
        self.assertEqual(len(problem.variables), 4)
        self.assertTrue(problem.objective)
        request = to_route_request(problem)
        self.assertEqual(request.qubo, problem.objective)
        self.assertEqual(request.credit_budget, 0.0)

    def test_optimal_tour_assignment_has_approx_ratio_one(self):
        problem = encode_tsp(eil51_prefix_problem(4))
        tour = problem.metadata["classical_optimal_tour"]
        dist = problem.metadata["distance_matrix"]
        assignment = tour_to_assignment(tour)
        ratio = approx_ratio(assignment, dist, problem.known_optimum)
        self.assertAlmostEqual(ratio, 1.0, places=9)

    def test_qubo_ground_state_recovers_known_optimum_n4(self):
        """Encoding fidelity at the historical 4-city eil51 sub-problem size.

        Default QAOA does not reliably return feasible 16-var tours; the
        classical ground state of the same QUBO must still decode to a
        tour with approx_ratio 1.0.
        """
        problem = encode_tsp(eil51_prefix_problem(4))
        bf = brute_force_solve(problem.objective)
        self.assertIsNotNone(bf)
        assignment, _energy = bf
        scored = score_assignment(problem, assignment)
        self.assertIsNotNone(scored["tour"])
        self.assertAlmostEqual(scored["approx_ratio"], 1.0, places=9)


class TestDiscoveryLoopP1Acceptance(unittest.TestCase):
    def test_encode_route_approx_ratio_one_eil51_n2(self):
        """P1 acceptance: encode -> run_route_request (sim) -> ratio 1.000.

        Uses the first 2 cities of eil51 (4 QUBO vars). Full eil51 is
        documented in docs/DISCOVERY_LOOP_P1.md / problem metadata.
        """
        problem = encode_tsp(eil51_prefix_problem(2))
        request = to_route_request(problem, fidelity_target=0.9, credit_budget=0.0)

        with tempfile.TemporaryDirectory() as tmp:
            results_dir = pathlib.Path(tmp)
            cert = run_route_request(
                request,
                fleet=DEFAULT_FLEET,
                results_dir=results_dir,
                # MEMORY_AUTO (default) writes the P0 ledger under results_dir
            )

        self.assertTrue(cert.is_optimal)
        scored = score_assignment(problem, cert.solution)
        self.assertIsNotNone(scored["tour"], msg=f"infeasible solution {cert.solution}")
        self.assertAlmostEqual(scored["approx_ratio"], 1.0, places=9)
        self.assertEqual(scored["tour_length"], int(problem.known_optimum))


if __name__ == "__main__":
    unittest.main()
