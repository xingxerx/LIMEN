# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Penalty-weight selection for limen.formulation.

Constraint penalties must outweigh whatever the objective could gain by
violating them, or the optimizer will happily violate a constraint to
lower the energy. This is a separate concern from LIMEN's Stackelberg
co-design loop (limen.codesign): co-design tunes embedding/chain-strength
*after* a QUBO already exists, against a fixed hardware graph; this module
picks the penalty coefficient *before* the QUBO exists, from the
objective's own coefficients, and has no hardware dependency. The two
don't compose into one search — co-design would just be re-tuning a
weight this module already fixed.
"""

from __future__ import annotations


def default_penalty_weight(
    objective_qubo: dict[tuple[str, str], float],
    constraint_variables: set[str],
    scale: float = 2.0,
) -> float:
    """Pick a penalty weight that dominates the objective's local influence.

    Heuristic: sum the absolute value of every objective coefficient that
    touches at least one variable in the constraint, then multiply by
    ``scale``. This bounds the maximum energy the objective could gain by
    flipping those variables away from a feasible assignment, so a
    penalty of this size (or larger) makes constraint violation strictly
    unprofitable regardless of the rest of the objective.

    Args:
        objective_qubo: The objective's QUBO terms, (var, var) -> weight.
        constraint_variables: The variables the constraint touches.
        scale: Safety margin multiplier. 2.0 is a common default in the
            QUBO-formulation literature; raise it if the solver is still
            finding infeasible optima, lower it if the penalty is
            drowning out the objective entirely.

    Returns:
        A positive float penalty weight. Never zero, even if the
        objective doesn't touch these variables at all, so unconstrained-
        objective cases still get a usable penalty magnitude.
    """
    touched_weight = sum(
        abs(w)
        for (i, j), w in objective_qubo.items()
        if i in constraint_variables or j in constraint_variables
    )
    return max(scale * touched_weight, 1.0)
