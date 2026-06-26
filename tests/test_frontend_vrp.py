"""Tests for the VRP frontend: QUBO encoding, route decoding, and LogicalGraph
conversion, including an end-to-end run through limen.pipeline.run_pipeline."""

import itertools
import math
import unittest

from limen.core.ir import LogicalGraph
from limen.frontends.vrp import (
    _augmented_distance_matrix,
    decode_routes,
    distance_matrix,
    from_vrp,
    vrp_qubo,
)
from limen.pipeline import run_pipeline

# Depot at index 0, four customers roughly arranged in two clusters so the
# optimal split is unambiguous.
COORDS = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (0.0, 2.0)]


class TestDistanceMatrix(unittest.TestCase):
    def test_symmetric_and_zero_diagonal(self):
        dist = distance_matrix(COORDS)
        n = len(COORDS)
        for i in range(n):
            self.assertEqual(dist[i][i], 0.0)
            for j in range(n):
                self.assertAlmostEqual(dist[i][j], dist[j][i])

    def test_known_distance(self):
        dist = distance_matrix(COORDS)
        self.assertAlmostEqual(dist[0][1], 1.0)
        self.assertAlmostEqual(dist[1][2], 1.0)


class TestAugmentedDistanceMatrix(unittest.TestCase):
    def test_depot_copies_are_zero_distance_from_each_other(self):
        num_vehicles = 3
        dist, customer_ids = _augmented_distance_matrix(COORDS, depot=0, num_vehicles=num_vehicles)
        n_customers = len(customer_ids)
        for a in range(n_customers, n_customers + num_vehicles):
            for b in range(n_customers, n_customers + num_vehicles):
                self.assertEqual(dist[a][b], 0.0)

    def test_customer_ids_exclude_depot(self):
        _, customer_ids = _augmented_distance_matrix(COORDS, depot=0, num_vehicles=2)
        self.assertEqual(customer_ids, [1, 2, 3, 4])
        self.assertNotIn(0, customer_ids)


class TestVrpQubo(unittest.TestCase):
    def test_rejects_zero_vehicles(self):
        with self.assertRaises(ValueError):
            vrp_qubo(COORDS, num_vehicles=0)

    def test_qubo_variable_count(self):
        num_vehicles = 2
        qubo, customer_ids = vrp_qubo(COORDS, num_vehicles=num_vehicles)
        n_total = len(customer_ids) + num_vehicles
        names = {name for pair in qubo for name in pair}
        self.assertEqual(len(names), n_total * n_total)

    def test_qubo_is_symmetric_keyed(self):
        qubo, _ = vrp_qubo(COORDS, num_vehicles=2)
        for (u, v) in qubo:
            self.assertLessEqual(u, v)


def _brute_force_best_tour(dist):
    n = len(dist)

    def cost(tour):
        return sum(dist[tour[i]][tour[(i + 1) % n]] for i in range(n))

    best_cost = math.inf
    best_tour: list[int] = []
    for perm in itertools.permutations(range(1, n)):
        tour = [0] + list(perm)
        c = cost(tour)
        if c < best_cost:
            best_cost = c
            best_tour = tour
    return best_cost, best_tour


class TestDecodeRoutes(unittest.TestCase):
    def test_decodes_optimal_tour_into_feasible_routes(self):
        num_vehicles = 2
        dist, customer_ids = _augmented_distance_matrix(COORDS, depot=0, num_vehicles=num_vehicles)
        n_customers = len(customer_ids)
        _, best_tour = _brute_force_best_tour(dist)

        assignment = {f"x_{node}_{t}": 1 for t, node in enumerate(best_tour)}
        routes = decode_routes(assignment, n_customers, num_vehicles, customer_ids)

        self.assertIsNotNone(routes)
        self.assertEqual(len(routes), num_vehicles)
        visited = sorted(c for route in routes for c in route)
        self.assertEqual(visited, sorted(customer_ids))

    def test_infeasible_assignment_returns_none(self):
        # Missing positions -> infeasible.
        routes = decode_routes({}, num_customers=4, num_vehicles=2, customer_ids=[1, 2, 3, 4])
        self.assertIsNone(routes)

    def test_collision_returns_none(self):
        # Two nodes claiming the same tour position.
        assignment = {"x_0_0": 1, "x_1_0": 1}
        routes = decode_routes(assignment, num_customers=4, num_vehicles=2, customer_ids=[1, 2, 3, 4])
        self.assertIsNone(routes)


class TestFromVrp(unittest.TestCase):
    def test_returns_validated_logical_graph(self):
        graph, customer_ids = from_vrp(COORDS, num_vehicles=2)
        self.assertIsInstance(graph, LogicalGraph)
        self.assertEqual(graph.validate(), [])
        self.assertEqual(customer_ids, [1, 2, 3, 4])

    def test_metadata_records_instance_shape(self):
        graph, _ = from_vrp(COORDS, num_vehicles=3, depot=0)
        self.assertEqual(graph.metadata["source"], "vrp")
        self.assertEqual(graph.metadata["num_vehicles"], 3)
        self.assertEqual(graph.metadata["depot"], 0)
        self.assertEqual(graph.metadata["num_customers"], 4)

    def test_custom_metadata_is_merged(self):
        graph, _ = from_vrp(COORDS, num_vehicles=2, metadata={"label": "demo"})
        self.assertEqual(graph.metadata["label"], "demo")
        self.assertEqual(graph.metadata["source"], "vrp")


class TestVrpThroughPipeline(unittest.TestCase):
    """End-to-end: vrp_qubo -> run_pipeline -> decode_routes.

    run_pipeline's default backend statevector-simulates the full QAOA
    circuit, so the instance must stay tiny: 2 customers + 2 vehicles ->
    4 augmented nodes -> 16 binary variables (2^16 states).
    """

    PIPELINE_COORDS = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]

    def test_small_instance_certifies_and_decodes_feasibly(self):
        num_vehicles = 2
        qubo, customer_ids = vrp_qubo(self.PIPELINE_COORDS, num_vehicles=num_vehicles)
        n_customers = len(customer_ids)

        cert = run_pipeline(qubo, encode_logical=False)

        self.assertTrue(math.isfinite(cert.energy))
        routes = decode_routes(cert.solution, n_customers, num_vehicles, customer_ids)

        # The certified ground state of this small, well-penalized instance
        # must be feasible: every customer visited exactly once.
        self.assertIsNotNone(routes)
        self.assertEqual(len(routes), num_vehicles)
        visited = sorted(c for route in routes for c in route)
        self.assertEqual(visited, sorted(customer_ids))


if __name__ == "__main__":
    unittest.main()
