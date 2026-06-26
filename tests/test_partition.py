"""Tests for distributed QUBO partitioning, compilation, and merging."""

import unittest

from limen.core.compiler import compile_lexicographic, default_hardware_graph
from limen.core.ir import Interaction, LogicalGraph, Variable
from limen.distributed.partition import (
    merge_partition_results,
    namespaced_hardware_graph,
    partition_graph,
)
from limen.validator.validator import validate


def _six_var_graph() -> LogicalGraph:
    return LogicalGraph(
        variables=[Variable(f"x{i}") for i in range(6)],
        interactions=[
            Interaction("x0", "x1", 1.0),
            Interaction("x2", "x2", 0.5),
            Interaction("x4", "x5", 2.0),
            Interaction("x1", "x3", 3.0),
        ],
    )


def _cross_partition_graph() -> LogicalGraph:
    return LogicalGraph(
        variables=[Variable("x0"), Variable("x1"), Variable("x2"), Variable("x3")],
        interactions=[
            Interaction("x0", "x0", -1.0),
            Interaction("x1", "x1", -1.0),
            Interaction("x2", "x2", -1.0),
            Interaction("x3", "x3", -1.0),
            Interaction("x0", "x1", 2.0),
            Interaction("x2", "x3", 2.0),
            Interaction("x1", "x2", 1.5),
        ],
    )


class TestPartitionGraph(unittest.TestCase):

    def test_balanced_split(self):
        partitions = partition_graph(_six_var_graph(), num_partitions=2)
        self.assertEqual(partitions[0].local_vars, {"x0", "x1", "x2"})
        self.assertEqual(partitions[1].local_vars, {"x3", "x4", "x5"})

    def test_cross_edge_owned_by_lower_partition(self):
        partitions = partition_graph(_six_var_graph(), num_partitions=2)
        owned_pairs = {(ix.i, ix.j) for ix in partitions[0].graph.interactions}
        self.assertIn(("x1", "x3"), owned_pairs)
        self.assertEqual(partitions[0].boundary_refs, {"x3"})
        self.assertEqual(partitions[1].boundary_refs, set())
        # The cross edge must not appear in the partition that doesn't own it.
        other_pairs = {(ix.i, ix.j) for ix in partitions[1].graph.interactions}
        self.assertNotIn(("x1", "x3"), other_pairs)

    def test_each_sub_graph_is_locally_valid(self):
        partitions = partition_graph(_six_var_graph(), num_partitions=2)
        for p in partitions:
            self.assertEqual(p.graph.validate(), [])

    def test_no_interaction_dropped_or_duplicated(self):
        graph = _six_var_graph()
        partitions = partition_graph(graph, num_partitions=2)
        all_owned = [ix for p in partitions for ix in p.graph.interactions]
        self.assertEqual(len(all_owned), len(graph.interactions))

    def test_rejects_invalid_partition_count(self):
        with self.assertRaises(ValueError):
            partition_graph(_six_var_graph(), num_partitions=0)
        with self.assertRaises(ValueError):
            partition_graph(_six_var_graph(), num_partitions=7)


class TestNamespacedHardwareGraph(unittest.TestCase):

    def test_labels_are_namespaced(self):
        hw = namespaced_hardware_graph(3, "p0")
        self.assertEqual(sorted(hw), ["p0:q0", "p0:q1", "p0:q2"])

    def test_two_partitions_never_collide(self):
        a = namespaced_hardware_graph(3, "p0")
        b = namespaced_hardware_graph(3, "p1")
        self.assertEqual(set(a) & set(b), set())


class TestMergeCorrectness(unittest.TestCase):

    def test_merged_encoding_matches_single_shot_compile(self):
        graph = _cross_partition_graph()
        partitions = partition_graph(graph, num_partitions=2)

        encodings = [
            compile_lexicographic(
                p.graph, namespaced_hardware_graph(len(p.graph.variables), f"p{p.partition_id}")
            )
            for p in partitions
        ]
        merged = merge_partition_results(partitions, encodings, graph)

        # Same number of interaction terms, same total weight, just
        # re-labelled onto namespaced physical qubits.
        self.assertEqual(len(merged.qubo), len(graph.interactions))
        self.assertAlmostEqual(sum(merged.qubo.values()), sum(ix.weight for ix in graph.interactions))

        single_shot = compile_lexicographic(graph, default_hardware_graph(len(graph.variables)))

        merged_result = validate(merged, runs=500)
        single_result = validate(single_shot, runs=500)
        self.assertEqual(merged_result.classical_energy, single_result.classical_energy)

    def test_merged_embedding_covers_every_variable(self):
        graph = _cross_partition_graph()
        partitions = partition_graph(graph, num_partitions=2)
        encodings = [
            compile_lexicographic(
                p.graph, namespaced_hardware_graph(len(p.graph.variables), f"p{p.partition_id}")
            )
            for p in partitions
        ]
        merged = merge_partition_results(partitions, encodings, graph)
        self.assertEqual(set(merged.embedding), {v.name for v in graph.variables})


if __name__ == "__main__":
    unittest.main()
