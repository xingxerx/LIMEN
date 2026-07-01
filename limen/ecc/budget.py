"""Adaptive ECC patch-budget allocation.

Bridges the QUBO frontends (limen.frontends / limen.validator) to the
surface-code layer in limen.ecc: rank QUBO variables by criticality
(a bit-flip there is more likely to land the solver in the wrong basin)
and greedily assign surface-code patches to the most critical variables
until a physical-qubit budget runs out.

Mirrors src/scoring.rs::qubo_criticality and src/ecc/selector.rs::select_patches
exactly (same tie-breaking, same greedy-in-ranked-order semantics), using the
limen_core Rust extension when built and an identical pure-Python fallback
otherwise - the same pattern as limen.ecc.decoder.LookupDecoder.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatchAssignment:
    """A surface-code patch allocation for one logical (QUBO) variable.

    physical_start..physical_end is the contiguous block of physical
    qubits reserved for this variable's patch.
    """

    logical_var: int
    distance: int
    physical_start: int
    physical_end: int


def rank_criticality(
    qubo: list[tuple[tuple[int, int], float]], n_vars: int
) -> list[tuple[int, float]]:
    """Rank QUBO variables by criticality for ECC patch-budget allocation.

    Criticality of variable i is the sum of |weight| over every QUBO
    term touching i. Returns (var, weight) pairs covering every index in
    range(n_vars), sorted descending by weight; untouched variables get
    weight 0.0 and sort last.
    """
    try:
        import limen_core

        _rust_rank = limen_core.qubo_criticality
    except (ImportError, AttributeError):
        _rust_rank = None

    if _rust_rank is not None:
        return list(_rust_rank(qubo, n_vars))

    weights = [0.0] * n_vars
    for (i, j), w in qubo:
        if i >= n_vars or j >= n_vars:
            continue
        abs_w = abs(w)
        if i == j:
            weights[i] += abs_w
        else:
            weights[i] += abs_w
            weights[j] += abs_w
    ranked = list(enumerate(weights))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def select_patches(
    ranked: list[tuple[int, float]], physical_qubit_budget: int, distance: int
) -> list[PatchAssignment]:
    """Greedily assign surface-code patches to the most critical variables.

    `ranked` is expected sorted by descending criticality (the output of
    rank_criticality); ties or unsorted input are not re-sorted here.
    Each patch costs distance*distance physical qubits. Variables that
    don't fit in the remaining budget are skipped (lower-criticality
    variables later in the list may still fit and will be tried).
    """
    try:
        import limen_core

        _rust_select = limen_core.ecc.select_patches
    except (ImportError, AttributeError):
        _rust_select = None

    if _rust_select is not None:
        return [
            PatchAssignment(a.logical_var, a.distance, a.physical_start, a.physical_end)
            for a in _rust_select(ranked, physical_qubit_budget, distance)
        ]

    patch_cost = distance * distance
    assignments: list[PatchAssignment] = []
    if patch_cost == 0:
        return assignments

    used = 0
    for var, _criticality in ranked:
        if used + patch_cost > physical_qubit_budget:
            continue
        assignments.append(PatchAssignment(var, distance, used, used + patch_cost))
        used += patch_cost
    return assignments


def allocate_ecc_budget(
    qubo: list[tuple[tuple[int, int], float]],
    n_vars: int,
    physical_qubit_budget: int,
    distance: int = 3,
) -> list[PatchAssignment]:
    """Rank QUBO variables by criticality and assign surface-code patches
    within a physical-qubit budget - the full adaptive ECC budget flow.
    """
    ranked = rank_criticality(qubo, n_vars)
    return select_patches(ranked, physical_qubit_budget, distance)
