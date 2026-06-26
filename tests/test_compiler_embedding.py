"""Tests for chain-based minor-embedding in the lexicographic compiler.

Covers: sparse hardware graphs that force multi-qubit chains, hardware
graphs too small/sparse for any embedding to exist (must raise
ValueError), and a regression check that complete hardware graphs still
produce single-qubit (1-to-1) embeddings exactly as before.
"""

import unittest

from limen.core.compiler import compile_lexicographic, default_hardware_graph
from limen.core.ir import Interaction, LogicalGraph, Variable


def _path_graph(n: int) -> dict[str, list[str]]:
    """Return a path graph p0 - p1 - ... - p(n-1)."""
    nodes = [f"p{i}" for i in range(n)]
    adj: dict[str, list[str]] = {node: [] for node in nodes}
    for idx in range(n - 1):
        adj[nodes[idx]].append(nodes[idx + 1])
        adj[nodes[idx + 1]].append(nodes[idx])
    return adj


def _cycle_graph(n: int) -> dict[str, list[str]]:
    """Return a cycle graph c0 - c1 - ... - c(n-1) - c0."""
    nodes = [f"c{i}" for i in range(n)]
    adj: dict[str, list[str]] = {node: [] for node in nodes}
    for idx in range(n):
        nxt = (idx + 1) % n
        adj[nodes[idx]].append(nodes[nxt])
        adj[nodes[nxt]].append(nodes[idx])
    return adj


def _grid_graph(width: int, height: int) -> dict[str, list[str]]:
    """Return a sparse 2D grid graph (4-neighbour lattice), a realistic
    sparse hardware topology -- each node connects only to its immediate
    horizontal/vertical neighbours, unlike a complete graph.
    """
    names = {(x, y): f"g{x}_{y}" for x in range(width) for y in range(height)}
    adj: dict[str, list[str]] = {name: [] for name in names.values()}
    for (x, y), name in names.items():
        for dx, dy in ((1, 0), (0, 1)):
            nx_, ny_ = x + dx, y + dy
            if (nx_, ny_) in names:
                other = names[(nx_, ny_)]
                adj[name].append(other)
                adj[other].append(name)
    return adj


def _triangle_graph() -> LogicalGraph:
    """A 3-variable logical graph where every pair interacts (a triangle)."""
    return LogicalGraph(
        variables=[Variable("x0"), Variable("x1"), Variable("x2")],
        interactions=[
            Interaction("x0", "x1", 1.0),
            Interaction("x1", "x2", 1.0),
            Interaction("x0", "x2", 1.0),
        ],
    )


def _is_adjacent(hardware_graph: dict[str, list[str]], a: str, b: str) -> bool:
    if a == b:
        return True
    return b in hardware_graph.get(a, ())


def _chains_touch(
    hardware_graph: dict[str, list[str]], chain_a: list[str], chain_b: list[str]
) -> bool:
    return any(
        _is_adjacent(hardware_graph, a, b) for a in chain_a for b in chain_b
    )


class TestChainEmbeddingOnSparseGraphs(unittest.TestCase):
    def test_triangle_on_cycle_forms_chains_and_is_adjacency_valid(self):
        """A triangle of interacting variables can't be placed 1-to-1 on a
        cycle (no 3-clique exists in a cycle of length > 3), so the
        embedder must grow at least one chain, and every interacting pair
        must end up adjacency-valid in the final embedding.
        """
        graph = _triangle_graph()
        hw = _cycle_graph(6)

        enc = compile_lexicographic(graph, hw)

        # Every logical variable has a non-empty chain.
        self.assertEqual(set(enc.embedding.keys()), {"x0", "x1", "x2"})
        for chain in enc.embedding.values():
            self.assertGreaterEqual(len(chain), 1)

        # Every interacting pair must be adjacency-valid (directly, or via
        # chains that touch).
        for ix in graph.interactions:
            chain_i = enc.embedding[ix.i]
            chain_j = enc.embedding[ix.j]
            self.assertTrue(
                _chains_touch(hw, chain_i, chain_j),
                f"chains for {ix.i!r} and {ix.j!r} are not adjacency-valid",
            )

        # At least one chain must have grown beyond a single qubit, since
        # a 3-clique cannot exist in a 6-cycle.
        max_chain_length = max(len(c) for c in enc.embedding.values())
        self.assertGreater(max_chain_length, 1)
        self.assertEqual(enc.metadata["max_chain_length"], max_chain_length)
        self.assertEqual(enc.metadata["embedding_quality"], "chained")

    def test_triangle_on_grid_forms_chains_and_is_adjacency_valid(self):
        """A 2D grid lattice is a realistic sparse hardware topology
        (each qubit connects only to its immediate neighbours, unlike a
        complete graph) that does contain triangle minors (unlike a
        tree), so this is a genuinely solvable sparse-embedding case.
        """
        graph = _triangle_graph()
        hw = _grid_graph(3, 3)

        enc = compile_lexicographic(graph, hw)

        for ix in graph.interactions:
            chain_i = enc.embedding[ix.i]
            chain_j = enc.embedding[ix.j]
            self.assertTrue(
                _chains_touch(hw, chain_i, chain_j),
                f"chains for {ix.i!r} and {ix.j!r} are not adjacency-valid",
            )

        max_chain_length = max(len(c) for c in enc.embedding.values())
        self.assertGreater(max_chain_length, 1)

    def test_deterministic_across_repeated_calls(self):
        """The same (graph, hardware_graph, seed) must always produce the
        identical embedding -- this is LIMEN's core determinism
        guarantee.
        """
        graph = _triangle_graph()
        hw = _cycle_graph(6)

        enc1 = compile_lexicographic(graph, hw, seed=42)
        enc2 = compile_lexicographic(graph, hw, seed=42)

        self.assertEqual(enc1.embedding, enc2.embedding)
        self.assertEqual(enc1.qubo, enc2.qubo)
        self.assertEqual(enc1.metadata, enc2.metadata)

    def test_chain_bias_couplers_present_for_grown_chains(self):
        """Any chain with more than one qubit must carry a ferromagnetic
        chain-strength coupler between every pair of qubits in it.
        """
        graph = _triangle_graph()
        hw = _cycle_graph(6)

        enc = compile_lexicographic(graph, hw, chain_strength=5.0)

        for chain in enc.embedding.values():
            if len(chain) < 2:
                continue
            sorted_chain = sorted(chain)
            for a_idx in range(len(sorted_chain)):
                for b_idx in range(a_idx + 1, len(sorted_chain)):
                    key = (sorted_chain[a_idx], sorted_chain[b_idx])
                    self.assertIn(key, enc.qubo)
                    self.assertLessEqual(enc.qubo[key], -5.0)


class TestEmbeddingInfeasibility(unittest.TestCase):
    def test_too_few_hardware_nodes_raises_value_error(self):
        """Hardware graph smaller than the logical graph must raise
        ValueError outright (pre-existing behaviour, still required).
        """
        graph = _triangle_graph()
        hw = _path_graph(2)

        with self.assertRaises(ValueError):
            compile_lexicographic(graph, hw)

    def test_triangle_on_path_graph_raises_value_error(self):
        """A path graph is a tree, and a tree can never contain a 3-cycle
        as a graph minor (no contraction of a cycle-free graph creates a
        cycle) -- so a triangle of mutually interacting variables can
        never be embedded on a path graph, regardless of heuristic
        cleverness. This must surface as ValueError, not a silently
        invalid (non-adjacent) coupler.
        """
        graph = _triangle_graph()
        hw = _path_graph(8)

        with self.assertRaises(ValueError):
            compile_lexicographic(graph, hw)

    def test_disconnected_hardware_graph_raises_value_error(self):
        """A hardware graph with enough nodes but no path at all between
        the components hosting two interacting variables can never
        satisfy that interaction's adjacency requirement -- the heuristic
        must detect this and raise ValueError rather than emit an
        invalid coupler.
        """
        graph = LogicalGraph(
            variables=[Variable("x0"), Variable("x1")],
            interactions=[Interaction("x0", "x1", 1.0)],
        )
        # Two isolated nodes -- enough count-wise, but zero edges anywhere,
        # so x0 and x1 (and any chain grown from them) can never touch.
        hw = {"q0": [], "q1": []}

        with self.assertRaises(ValueError):
            compile_lexicographic(graph, hw)


class TestCompleteGraphRegression(unittest.TestCase):
    def test_default_hardware_graph_keeps_single_qubit_chains(self):
        """On a complete hardware graph, every interacting pair is already
        adjacency-valid under the naive 1-to-1 assignment, so no chain
        should ever grow -- this preserves exact backward compatibility
        with every existing caller.
        """
        graph = _triangle_graph()
        hw = default_hardware_graph(4)

        enc = compile_lexicographic(graph, hw)

        for chain in enc.embedding.values():
            self.assertEqual(len(chain), 1)
        self.assertEqual(enc.metadata["max_chain_length"], 1)
        self.assertEqual(enc.metadata["embedding_quality"], "one_to_one")

    def test_complete_graph_qubo_shape_matches_previous_behaviour(self):
        """Physical QUBO must contain exactly one entry per logical
        interaction (no extra chain-bias entries) when the embedding is
        1-to-1.
        """
        graph = _triangle_graph()
        hw = default_hardware_graph(4)

        enc = compile_lexicographic(graph, hw)

        self.assertEqual(len(enc.qubo), len(graph.interactions))


if __name__ == "__main__":
    unittest.main()
