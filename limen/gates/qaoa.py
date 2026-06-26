"""QAOA bridge: compile a diagonal Ising/QUBO LogicalGraph into a CircuitIR.

This is the connective tissue between the two LIMEN IRs. limen.core.ir
represents an optimization problem as a diagonal cost function (Z and ZZ
terms); limen.gates.ir represents an executable gate-model circuit. The
Quantum Approximate Optimization Algorithm (Farhi-Goldstone-Gutmann)
encodes that cost function as a parameterized circuit whose measurement
distribution concentrates on low-energy bitstrings.

Mapping: a QUBO variable x in {0,1} relates to a Pauli-Z eigenvalue z in
{+1,-1} by x = (1 - z)/2. Substituting into the QUBO objective yields an
Ising form C = sum h_i Z_i + sum J_ij Z_i Z_j (plus a constant that does
not affect optimization). The cost unitary exp(-i gamma C) is realized
with RZ(2 gamma h_i) and, per coupling, CX-RZ(2 gamma J_ij)-CX; the mixer
exp(-i beta sum X_i) is RX(2 beta) on every qubit.
"""

from __future__ import annotations

from limen.core.ir import LogicalGraph
from limen.gates.ir import CircuitIR, GateInstruction


def variable_order(graph: LogicalGraph) -> list[str]:
    """Return the deterministic qubit ordering: variable names sorted lexically.

    Qubit index q corresponds to variable_order(graph)[q], matching the
    sorted-name convention used by limen.core.compiler.
    """
    return sorted(v.name for v in graph.variables)


def qubo_to_ising(
    graph: LogicalGraph,
) -> tuple[dict[int, float], dict[tuple[int, int], float], list[str]]:
    """Convert a QUBO LogicalGraph into Ising coefficients over qubit indices.

    Args:
        graph: A LogicalGraph whose interactions encode a QUBO objective
            (i == j interactions are linear terms).

    Returns:
        A (h, J, order) tuple where h maps qubit index -> single-Z
        coefficient, J maps (i, j) with i < j -> ZZ coefficient, and
        order is the variable_order(graph) list. The Ising constant is
        discarded (irrelevant to optimization).
    """
    order = variable_order(graph)
    index = {name: i for i, name in enumerate(order)}
    h: dict[int, float] = {i: 0.0 for i in range(len(order))}
    j_coeffs: dict[tuple[int, int], float] = {}

    for ix in graph.interactions:
        qi, qj = index[ix.i], index[ix.j]
        w = ix.weight
        if qi == qj:
            # Linear term w * x_i = w/2 - (w/2) Z_i.
            h[qi] += -0.5 * w
        else:
            # Quadratic w * x_i x_j = w/4 (1 - Z_i - Z_j + Z_i Z_j).
            h[qi] += -0.25 * w
            h[qj] += -0.25 * w
            a, b = (qi, qj) if qi < qj else (qj, qi)
            j_coeffs[(a, b)] = j_coeffs.get((a, b), 0.0) + 0.25 * w

    return h, j_coeffs, order


def compile_qaoa(
    graph: LogicalGraph, gammas: list[float], betas: list[float]
) -> CircuitIR:
    """Compile a QUBO LogicalGraph into a p-layer QAOA CircuitIR.

    Args:
        graph: The QUBO LogicalGraph to encode.
        gammas: Cost-layer angles, one per QAOA layer.
        betas: Mixer-layer angles, one per QAOA layer (len == len(gammas)).

    Returns:
        A CircuitIR with n_qubits == number of variables, beginning with a
        Hadamard on every qubit and applying len(gammas) cost+mixer layers.

    Raises:
        ValueError: If len(gammas) != len(betas).
    """
    if len(gammas) != len(betas):
        raise ValueError(
            f"gammas and betas must have equal length, got {len(gammas)} and {len(betas)}"
        )

    h, j_coeffs, order = qubo_to_ising(graph)
    n = len(order)
    instructions: list[GateInstruction] = [GateInstruction("h", [q]) for q in range(n)]

    coupling_order = sorted(j_coeffs)
    for gamma, beta in zip(gammas, betas):
        for q in range(n):
            if abs(h[q]) > 1e-15:
                instructions.append(GateInstruction("rz", [q], [2.0 * gamma * h[q]]))
        for (a, b) in coupling_order:
            weight = j_coeffs[(a, b)]
            if abs(weight) <= 1e-15:
                continue
            instructions.append(GateInstruction("cx", [a, b]))
            instructions.append(GateInstruction("rz", [b], [2.0 * gamma * weight]))
            instructions.append(GateInstruction("cx", [a, b]))
        for q in range(n):
            instructions.append(GateInstruction("rx", [q], [2.0 * beta]))

    return CircuitIR(
        n_qubits=n,
        instructions=instructions,
        metadata={
            "ansatz": "qaoa",
            "layers": len(gammas),
            "variable_order": order,
        },
    )


def bitstring_to_assignment(bitstring: str, order: list[str]) -> dict[str, int]:
    """Map a qubit-0-first bitstring to a {variable_name: 0/1} assignment.

    The bitstring convention matches limen.gates.simulator: character q is
    the measured value of qubit q, i.e. of variable order[q].
    """
    return {name: int(bitstring[q]) for q, name in enumerate(order)}
