"""Max-Cut example for LIMEN.

The Max-Cut problem asks: given an undirected graph, partition its nodes
into two sets S and T such that the number of edges crossing the partition
(one endpoint in S, the other in T) is maximised.

It maps naturally to QUBO because each binary variable x_i encodes which
partition node i belongs to (0 or 1). For an edge (u, v), the term

    x_u + x_v - 2*x_u*x_v

equals 1 when u and v are in different partitions and 0 otherwise. To
*minimise* (as QUBO solvers do) we negate and minimise:

    sum_{(u,v) in E}  (x_u*x_v - x_u - x_v)

giving the linear diagonal weight of -deg(node) for each node and +1.0
for each edge's off-diagonal term.
"""

from limen import (
    compile_lexicographic,
    default_hardware_graph,
    from_qubo_dict,
    validate,
)
from limen.validator.validator import brute_force_solve

# ── Problem definition ────────────────────────────────────────────────

NODES = ["A", "B", "C", "D", "E"]
EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E")]


def build_max_cut_qubo(
    edges: list[tuple[str, str]],
) -> dict[tuple[str, str], float]:
    """Convert a Max-Cut instance to a QUBO dict."""
    qubo: dict[tuple[str, str], float] = {}
    for u, v in edges:
        qubo[(u, v)] = qubo.get((u, v), 0.0) + 1.0
        qubo[(u, u)] = qubo.get((u, u), 0.0) - 1.0
        qubo[(v, v)] = qubo.get((v, v), 0.0) - 1.0
    return qubo


def count_cut_edges(
    assignment: dict[str, int], edges: list[tuple[str, str]]
) -> int:
    """Count edges that cross the partition defined by assignment."""
    return sum(1 for u, v in edges if assignment.get(u, 0) != assignment.get(v, 0))


# ── Pipeline ──────────────────────────────────────────────────────────

def main() -> None:
    qubo = build_max_cut_qubo(EDGES)

    graph = from_qubo_dict(qubo)
    encoding = compile_lexicographic(graph, default_hardware_graph(8))
    result = validate(encoding, runs=2000)

    # Recover best assignment from classical solver for the report.
    bf = brute_force_solve(encoding.qubo)
    best_assignment_physical, _ = bf  # guaranteed: ≤20 vars

    # Map physical qubit labels back to logical variable names.
    phys_to_logical = {v[0]: k for k, v in encoding.embedding.items()}
    best_assignment = {
        phys_to_logical[p]: val for p, val in best_assignment_physical.items()
    }

    # ── Report ────────────────────────────────────────────────────────
    W = 55
    sep = "─" * W

    print(f"── LIMEN Max-Cut Example {'─' * (W - 24)}")
    print(f"  Graph : {len(NODES)} nodes, {len(EDGES)} edges")
    print()
    print("  Logical IR")
    var_names = ", ".join(v.name for v in graph.variables)
    print(f"    Variables   : {var_names}")
    print(f"    Interactions: {len(graph.interactions)}")
    print()
    print("  Physical Encoding")
    print(f"    Hardware nodes : {encoding.metadata['hardware_nodes']}")
    print(f"    Logical vars   : {encoding.metadata['logical_variables']}")
    print(f"    Chain strength : {encoding.chain_strength:.4f}")
    emb_str = ", ".join(
        f"{lv}→{pq[0]}" for lv, pq in sorted(encoding.embedding.items())
    )
    print(f"    Embedding      : {emb_str}")
    print()
    print(f"  Validation (2000 runs)")
    print(f"    Best energy     : {result.best_energy:.4f}")
    ce = f"{result.classical_energy:.4f}" if result.classical_energy is not None else "N/A"
    print(f"    Classical energy: {ce}")
    print(f"    Confidence      : {result.confidence * 100:.1f}%")
    notes_str = " | ".join(result.notes) if result.notes else "None"
    print(f"    Notes           : {notes_str}")
    print(sep)

    # ── Best assignment ───────────────────────────────────────────────
    print()
    print("  Best assignment:")
    for node in sorted(best_assignment):
        print(f"    {node} = {best_assignment[node]}")

    partition_s = sorted(n for n, v in best_assignment.items() if v == 1)
    partition_t = sorted(n for n, v in best_assignment.items() if v == 0)
    cut = count_cut_edges(best_assignment, EDGES)

    print()
    print(f"    Partition S (=1): {', '.join(partition_s) or '(empty)'}")
    print(f"    Partition T (=0): {', '.join(partition_t) or '(empty)'}")
    print(f"    Edges in cut    : {cut}")


if __name__ == "__main__":
    main()
