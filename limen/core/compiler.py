"""Lexicographic compiler for LIMEN.

Converts a validated LogicalGraph into a PhysicalEncoding suitable for
a hardware backend, using a deterministic greedy minor-embedding strategy.
No D-Wave or minorminer dependencies are required.
"""

from dataclasses import dataclass, field
from typing import Any

from limen.core.ir import LogicalGraph


def _adjacent(hardware_graph: dict[str, list[str]], a: str, b: str) -> bool:
    """Return True if nodes a and b are directly connected in hardware_graph."""
    if a == b:
        return True
    return b in hardware_graph.get(a, ())


def _chains_adjacent(
    hardware_graph: dict[str, list[str]],
    chain_a: list[str],
    chain_b: list[str],
) -> bool:
    """Return True if any qubit in chain_a is adjacent to any qubit in chain_b."""
    for a in chain_a:
        for b in chain_b:
            if _adjacent(hardware_graph, a, b):
                return True
    return False


def _grow_chains_until_adjacent(
    hardware_graph: dict[str, list[str]],
    chain_a: list[str],
    chain_b: list[str],
    used: set[str],
    max_growth_steps: int,
) -> bool:
    """Deterministically grow chain_a and chain_b via BFS until they touch.

    Alternates growth between the two chains (chain_a first, on ties),
    always picking the lexicographically smallest available neighbour of
    the chain's current node set, so the result depends only on
    (hardware_graph, the chains' starting contents) and not on iteration
    order elsewhere.

    Mutates chain_a/chain_b and `used` in place. Returns True if the
    chains became adjacent, False if growth was exhausted without success
    (every reachable neighbour is already claimed by some chain, or the
    growth-step budget was used up).
    """
    if _chains_adjacent(hardware_graph, chain_a, chain_b):
        return True

    chains = [chain_a, chain_b]
    turn = 0
    steps = 0
    while steps < max_growth_steps:
        grown = False
        # Try both chains this round (chain_a first) so a single exhausted
        # chain doesn't block the other from growing.
        for _ in range(2):
            chain = chains[turn % 2]
            turn += 1
            frontier = sorted(chain)
            candidates = sorted(
                {
                    nbr
                    for node in frontier
                    for nbr in hardware_graph.get(node, ())
                    if nbr not in used
                }
            )
            if candidates:
                new_node = candidates[0]
                chain.append(new_node)
                used.add(new_node)
                grown = True
                steps += 1
                if _chains_adjacent(hardware_graph, chain_a, chain_b):
                    return True
                break
        if not grown:
            # Neither chain could grow further: truly stuck.
            return False
    return _chains_adjacent(hardware_graph, chain_a, chain_b)


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

    The embedder starts from the simple, deterministic 1-to-1 assignment
    (logical variables in lexicographic order onto physical nodes in
    lexicographic order). For each pair of interacting logical variables
    whose assigned physical qubits are *not* adjacent in
    ``hardware_graph``, it grows one or both variables' chains — via
    deterministic breadth-first search over ``hardware_graph`` — until a
    pair of mutually-adjacent qubits (one per chain) is found. This is
    real, if minimal, minor-embedding: a logical variable may end up
    occupying a *chain* of several physical qubits rather than a single
    qubit, and every pair of qubits within one chain is coupled with
    ``chain_strength`` (the standard ferromagnetic "chain bias" used by
    D-Wave's ``embed_qubo``/``FixedEmbeddingComposite``) so the solver is
    encouraged to return the same value for every qubit in the chain.

    On a complete hardware graph (e.g. ``default_hardware_graph(n)``) the
    initial 1-to-1 assignment is already adjacency-valid for every
    interaction, so no chain ever grows and the result is identical to
    the previous single-qubit-per-variable behaviour.

    Important limitation: like minorminer, this is a *heuristic* for the
    (NP-hard, in general) minor-embedding problem. It is not guaranteed
    to find a valid embedding even when one exists, and when it does find
    one it is not guaranteed to be the smallest possible embedding (in
    qubit count or chain length). It processes interactions in a fixed
    order — sorted by ``(i, j)`` — and grows chains by always picking the
    lexicographically smallest available neighbour, so for a fixed
    ``(graph, hardware_graph)`` pair the result is always bit-for-bit
    identical, but a different hardware graph or variable naming can turn
    a solvable instance into one this heuristic fails on.

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
        compiler metadata. ``metadata`` additionally records
        ``"max_chain_length"`` (the size of the largest chain produced)
        and ``"embedding_quality"`` (``"one_to_one"`` if every chain has
        exactly one qubit, else ``"chained"``).

    Raises:
        ValueError: If the hardware graph has fewer nodes than logical
            variables in the graph, or if the deterministic chain-growth
            heuristic cannot find a valid embedding (e.g. the hardware
            graph is too sparse or too small for some interaction to be
            realized, even after exhausting its growth budget).
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
    # to the next available physical node (sorted order). Each chain
    # starts as a singleton; chains only grow if a later adjacency check
    # fails.
    sorted_logical_vars = sorted(logical_vars)
    embedding: dict[str, list[str]] = {
        lv: [physical_nodes[idx]] for idx, lv in enumerate(sorted_logical_vars)
    }
    used_nodes: set[str] = {chain[0] for chain in embedding.values()}

    # Total available "growth budget": every physical node not already
    # claimed by some chain can be claimed at most once across all
    # interactions. This bounds the BFS growth loop and lets us detect
    # genuine infeasibility deterministically rather than looping forever.
    growth_budget = max(0, len(physical_nodes) - len(sorted_logical_vars))

    # Process interactions in a fixed, deterministic order so chain
    # growth never depends on input ordering.
    interaction_pairs = sorted(
        {
            (ix.i, ix.j) if ix.i <= ix.j else (ix.j, ix.i)
            for ix in graph.interactions
            if ix.i != ix.j
        }
    )

    for i, j in interaction_pairs:
        chain_i = embedding[i]
        chain_j = embedding[j]
        if _chains_adjacent(hardware_graph, chain_i, chain_j):
            continue
        ok = _grow_chains_until_adjacent(
            hardware_graph, chain_i, chain_j, used_nodes, growth_budget
        )
        if not ok:
            raise ValueError(
                f"Could not find a valid minor-embedding for interaction "
                f"({i!r}, {j!r}): no adjacency-connecting chain growth was "
                f"found within the available hardware graph. The hardware "
                f"graph may be too sparse or too small for this logical "
                f"graph to be embedded by this heuristic."
            )

    # Build the physical QUBO. Linear (self-loop) terms map onto one
    # representative qubit per chain. Inter-variable terms are carried by
    # the adjacent pair of qubits (one from each chain) that satisfies the
    # adjacency requirement — falling back to the representative qubits
    # when the chains are already singletons (the common case).
    physical_qubo: dict[tuple[str, str], float] = {}
    for (i, j), w in logical_qubo.items():
        if i == j:
            rep = embedding[i][0]
            key = (rep, rep)
            physical_qubo[key] = physical_qubo.get(key, 0.0) + w
            continue
        chain_i = embedding[i]
        chain_j = embedding[j]
        pair = None
        for a in chain_i:
            for b in chain_j:
                if _adjacent(hardware_graph, a, b):
                    pair = (a, b)
                    break
            if pair is not None:
                break
        if pair is None:
            # Should be unreachable given the adjacency-validation loop
            # above, but guard defensively rather than emit an invalid
            # coupler.
            raise ValueError(
                f"Internal embedding error: chains for {i!r} and {j!r} are "
                f"not adjacent after embedding."
            )
        physical_qubo[pair] = physical_qubo.get(pair, 0.0) + w

    # Chain bias: couple every pair of qubits within a multi-qubit chain
    # ferromagnetically (negative weight favours equal values for a
    # standard QUBO minimization), matching D-Wave's embed_qubo
    # convention.
    for chain in embedding.values():
        if len(chain) < 2:
            continue
        sorted_chain = sorted(chain)
        for idx_a in range(len(sorted_chain)):
            for idx_b in range(idx_a + 1, len(sorted_chain)):
                key = (sorted_chain[idx_a], sorted_chain[idx_b])
                physical_qubo[key] = physical_qubo.get(key, 0.0) - chain_strength

    max_levels = max((v.levels for v in graph.variables), default=2)
    max_chain_length = max((len(chain) for chain in embedding.values()), default=1)

    metadata: dict[str, Any] = {
        "compiler": "lexicographic",
        "seed": seed,
        "hardware_nodes": len(hardware_graph),
        "logical_variables": len(graph.variables),
        "max_levels": max_levels,
        "max_chain_length": max_chain_length,
        "embedding_quality": "one_to_one" if max_chain_length == 1 else "chained",
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
