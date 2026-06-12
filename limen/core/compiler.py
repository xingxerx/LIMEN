"""Lexicographic compiler for LIMEN.

Converts a validated LogicalGraph into a PhysicalEncoding suitable for
a hardware backend, using a deterministic greedy minor-embedding strategy.
No D-Wave or minorminer dependencies are required.
"""

from dataclasses import dataclass, field
from typing import Any

from limen.core.ir import LogicalGraph


@dataclass
class PhysicalEncoding:
    """The result of compiling a LogicalGraph to a physical hardware encoding.

    Attributes:
        graph: The source LogicalGraph.
        qubo: Physical QUBO dict mapping (physical_qubit, physical_qubit)
            pairs to float weights.
        embedding: Mapping from logical variable name to the list of
            physical qubit labels it occupies (one per qubit for the
            naive 1-to-1 embedder).
        chain_strength: The chain coupling strength used during embedding.
        metadata: Arbitrary compiler annotations.
    """

    graph: LogicalGraph
    qubo: dict[tuple[str, str], float]
    embedding: dict[str, list[str]]
    chain_strength: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this encoding to a plain Python dict.

        QUBO tuple keys are stored as two-element lists so the result
        is JSON-safe.
        """
        return {
            "graph": self.graph.to_dict(),
            "qubo": [[list(k), v] for k, v in self.qubo.items()],
            "embedding": {k: list(v) for k, v in self.embedding.items()},
            "chain_strength": self.chain_strength,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PhysicalEncoding":
        """Deserialize a PhysicalEncoding from a plain Python dict.

        Converts QUBO key lists back to tuples and reconstructs the
        LogicalGraph via LogicalGraph.from_dict().
        """
        return cls(
            graph=LogicalGraph.from_dict(d["graph"]),
            qubo={tuple(k): float(v) for k, v in d["qubo"]},
            embedding={k: list(v) for k, v in d["embedding"].items()},
            chain_strength=float(d["chain_strength"]),
            metadata=dict(d.get("metadata", {})),
        )


def compile_lexicographic(
    graph: LogicalGraph,
    hardware_graph: dict[str, list[str]],
    chain_strength: float | None = None,
    seed: int = 42,
) -> PhysicalEncoding:
    """Compile a LogicalGraph to a PhysicalEncoding using a greedy lexicographic embedder.

    Args:
        graph: A validated LogicalGraph to compile.
        hardware_graph: Adjacency dict for the target hardware
            (node -> list of neighbour nodes).
        chain_strength: Chain coupling strength. If None, auto-calculated
            as ``1.5 * max(|weight|)`` across all interactions, with a
            floor of 1.0. Defaults to 1.0 when the graph has no interactions.
        seed: Random seed kept for API compatibility; the greedy algorithm
            is fully deterministic without it.

    Returns:
        A PhysicalEncoding containing the remapped QUBO, embedding, and
        compiler metadata.

    Raises:
        ValueError: If the hardware graph has fewer nodes than logical
            variables in the graph.
    """
    logical_vars = [v.name for v in graph.variables]
    physical_nodes = sorted(hardware_graph.keys())

    if len(physical_nodes) < len(logical_vars):
        raise ValueError(
            f"Hardware graph has {len(physical_nodes)} node(s) but the logical "
            f"graph requires {len(logical_vars)}. Use a larger hardware graph."
        )

    # Auto chain-strength — guard against empty interaction list.
    if chain_strength is None:
        weights = [abs(ix.weight) for ix in graph.interactions]
        chain_strength = max(1.0, 1.5 * max(weights)) if weights else 1.0

    # Logical QUBO.
    logical_qubo: dict[tuple[str, str], float] = {
        (ix.i, ix.j): ix.weight for ix in graph.interactions
    }

    # Greedy 1-to-1 embedding: assign each logical variable (alphabetical)
    # to the next available physical node (sorted order).
    embedding: dict[str, list[str]] = {
        lv: [physical_nodes[idx]] for idx, lv in enumerate(sorted(logical_vars))
    }

    # Remap QUBO keys through the embedding.
    physical_qubo: dict[tuple[str, str], float] = {
        (embedding[i][0], embedding[j][0]): w
        for (i, j), w in logical_qubo.items()
    }

    max_levels = max((v.levels for v in graph.variables), default=2)

    metadata: dict[str, Any] = {
        "compiler": "lexicographic",
        "seed": seed,
        "hardware_nodes": len(hardware_graph),
        "logical_variables": len(graph.variables),
        "max_levels": max_levels,
    }

    return PhysicalEncoding(
        graph=graph,
        qubo=physical_qubo,
        embedding=embedding,
        chain_strength=chain_strength,
        metadata=metadata,
    )


def default_hardware_graph(n: int) -> dict[str, list[str]]:
    """Return a complete graph over n nodes labelled 'q0'..'q{n-1}'.

    Useful for unit tests and examples that do not have access to real
    hardware topology data.

    Args:
        n: Number of physical qubits.

    Returns:
        Adjacency dict where every node is connected to every other node.
    """
    nodes = [f"q{i}" for i in range(n)]
    return {node: [other for other in nodes if other != node] for node in nodes}
