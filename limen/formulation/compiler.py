# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Constraint -> QUBO compiler for limen.formulation.

Translates the typed constraints in limen.formulation.constraints into
penalty terms and merges them with an objective QUBO into a LogicalGraph,
the same IR limen.frontends.pyqubo.from_qubo_dict produces. Two
primitives do all the work:

    - Linear equality (Sum c_i x_i == rhs) penalized as weight * (Sum c_i
      x_i - rhs)^2, expanded exactly using x_i^2 = x_i for binary x_i.
    - Linear inequality (<=, >=), reduced to an equality by adding a
      binary-encoded slack variable (Sum c_i x_i +/- Sum 2^b s_b = rhs),
      then handled by the same equality path.

Every higher-level constraint (OneHot, AtMostK, AtLeastK, AllDifferent)
compiles down to one or more of these two primitives. AtMostK/AtLeastK
with k=1 (and k=0) are special-cased to skip slack variables entirely,
since "at most one of these is 1" has an exact zero-auxiliary penalty
(pairwise product), unlike the general-k case.
"""

from __future__ import annotations

import math

from limen.core.ir import Interaction, LogicalGraph, Variable
from limen.formulation.constraints import (
    AllDifferent,
    AtLeastK,
    AtMostK,
    Constraint,
    Equality,
    Inequality,
    OneHot,
)
from limen.formulation.penalty import default_penalty_weight

_Terms = dict[tuple[str, str], float]


def _key(i: str, j: str) -> tuple[str, str]:
    return (i, j) if i <= j else (j, i)


def expand_equality_penalty(
    coeffs: dict[str, float], rhs: float, weight: float
) -> tuple[_Terms, float]:
    """Expand weight * (Sum coeffs[v]*v - rhs)^2 into QUBO terms.

    Uses x_i^2 = x_i (binary variables) to fold every squared linear term
    into a linear QUBO term.

    Args:
        coeffs: Mapping of variable name to coefficient.
        rhs: Target value of the linear sum.
        weight: Penalty weight multiplying the whole squared expression.

    Returns:
        A tuple (terms, constant): terms is a QUBO-shaped dict keyed by
        sorted (var, var) pairs (var, var) for linear terms; constant is
        the weight * rhs**2 energy offset the expansion drops (constant
        across all assignments, so it doesn't affect argmin, but callers
        that report absolute energies should add it back).
    """
    terms: _Terms = {}
    names = list(coeffs)
    for v in names:
        c = coeffs[v]
        linear = weight * (c * c - 2.0 * rhs * c)
        k = _key(v, v)
        terms[k] = terms.get(k, 0.0) + linear
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            vi, vj = names[a], names[b]
            quad = weight * 2.0 * coeffs[vi] * coeffs[vj]
            k = _key(vi, vj)
            terms[k] = terms.get(k, 0.0) + quad
    constant = weight * rhs * rhs
    return terms, constant


def _bounds(coeffs: dict[str, float]) -> tuple[float, float]:
    lo = sum(min(0.0, c) for c in coeffs.values())
    hi = sum(max(0.0, c) for c in coeffs.values())
    return lo, hi


def _compile_inequality(
    coeffs: dict[str, float],
    rhs: float,
    sense: str,
    weight: float | None,
    scale: float,
    aux_prefix: str,
    objective: dict[tuple[str, str], float],
) -> tuple[_Terms, float, set[str]]:
    """Reduce an inequality to an equality via a binary-encoded slack and expand it."""
    if sense == ">=":
        coeffs = {v: -c for v, c in coeffs.items()}
        rhs = -rhs

    lo, _hi = _bounds(coeffs)
    bound = rhs - lo
    if not math.isclose(bound, round(bound), abs_tol=1e-9):
        raise ValueError(
            "Inequality coefficients/rhs must be integers for exact slack "
            f"encoding; got a non-integer slack bound {bound} from "
            f"coeffs={coeffs}, rhs={rhs}."
        )
    bound_int = round(bound)
    if bound_int < 0:
        raise ValueError(
            f"Inequality is infeasible for every binary assignment: even "
            f"the minimum achievable sum ({lo}) already exceeds rhs "
            f"({rhs})."
        )

    combined = dict(coeffs)
    aux_vars: set[str] = set()
    if bound_int > 0:
        nbits = max(1, bound_int.bit_length())
        for b in range(nbits):
            slack_name = f"{aux_prefix}_slack{b}"
            aux_vars.add(slack_name)
            combined[slack_name] = combined.get(slack_name, 0.0) + float(2**b)

    touched = set(coeffs)
    w = weight if weight is not None else default_penalty_weight(objective, touched, scale)
    terms, constant = expand_equality_penalty(combined, rhs, w)
    return terms, constant, touched | aux_vars


def _compile_one(
    constraint: Constraint,
    weight: float | None,
    scale: float,
    aux_prefix: str,
    objective: dict[tuple[str, str], float],
) -> tuple[_Terms, float, set[str]]:
    """Compile a single typed constraint into (terms, constant, touched_variables)."""
    if isinstance(constraint, Equality):
        touched = set(constraint.coeffs)
        w = weight if weight is not None else default_penalty_weight(objective, touched, scale)
        terms, constant = expand_equality_penalty(constraint.coeffs, constraint.rhs, w)
        return terms, constant, touched

    if isinstance(constraint, OneHot):
        eq = Equality(coeffs={v: 1.0 for v in constraint.variables}, rhs=1.0)
        return _compile_one(eq, weight, scale, aux_prefix, objective)

    if isinstance(constraint, AtMostK):
        if constraint.k <= 0:
            eq = Equality(coeffs={v: 1.0 for v in constraint.variables}, rhs=0.0)
            return _compile_one(eq, weight, scale, aux_prefix, objective)
        if constraint.k == 1:
            touched = set(constraint.variables)
            w = (
                weight
                if weight is not None
                else default_penalty_weight(objective, touched, scale)
            )
            terms: _Terms = {}
            vs = constraint.variables
            for a in range(len(vs)):
                for b in range(a + 1, len(vs)):
                    k = _key(vs[a], vs[b])
                    terms[k] = terms.get(k, 0.0) + w
            return terms, 0.0, touched
        ineq = Inequality(
            coeffs={v: 1.0 for v in constraint.variables}, rhs=float(constraint.k), sense="<="
        )
        return _compile_inequality(
            ineq.coeffs, ineq.rhs, ineq.sense, weight, scale, aux_prefix, objective
        )

    if isinstance(constraint, AtLeastK):
        if constraint.k <= 0:
            return {}, 0.0, set()
        ineq = Inequality(
            coeffs={v: 1.0 for v in constraint.variables}, rhs=float(constraint.k), sense=">="
        )
        return _compile_inequality(
            ineq.coeffs, ineq.rhs, ineq.sense, weight, scale, aux_prefix, objective
        )

    if isinstance(constraint, Inequality):
        return _compile_inequality(
            constraint.coeffs, constraint.rhs, constraint.sense, weight, scale, aux_prefix, objective
        )

    if isinstance(constraint, AllDifferent):
        terms: _Terms = {}
        constant = 0.0
        touched: set[str] = set()
        for item in constraint.items:
            row_vars = [constraint.var_name(item, slot) for slot in constraint.slots]
            row = OneHot(variables=row_vars, name=f"{constraint.name or 'alldiff'}_{item}_row")
            t, c, tv = _compile_one(row, weight, scale, f"{aux_prefix}_{item}row", objective)
            _merge(terms, t)
            constant += c
            touched |= tv
        for slot in constraint.slots:
            col_vars = [constraint.var_name(item, slot) for item in constraint.items]
            col = AtMostK(variables=col_vars, k=1, name=f"{constraint.name or 'alldiff'}_{slot}_col")
            t, c, tv = _compile_one(col, weight, scale, f"{aux_prefix}_{slot}col", objective)
            _merge(terms, t)
            constant += c
            touched |= tv
        return terms, constant, touched

    raise TypeError(f"Unsupported constraint type: {type(constraint).__name__}")


def _merge(dst: _Terms, src: _Terms) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + v


class ConstraintCompiler:
    """Builds a LogicalGraph from an objective QUBO plus typed constraints.

    Usage::

        compiler = ConstraintCompiler()
        compiler.set_objective(qubo)
        compiler.add_constraint(OneHot(["x0", "x1", "x2"]))
        compiler.add_constraint(AtMostK(["x0", "x3"], k=1), weight=50.0)
        graph = compiler.compile()

    Each constraint gets its own auto-selected penalty weight (see
    limen.formulation.penalty.default_penalty_weight) unless a weight is
    passed explicitly to add_constraint. Weights are selected against the
    *original* objective only, not against other constraints — constraint
    interactions are additive, so this stays correct as long as each
    weight individually dominates the objective's local pull; it does not
    protect against many constraints collectively outweighing a very
    small per-constraint weight choice made by a caller who overrides the
    default.
    """

    def __init__(self) -> None:
        self._objective: dict[tuple[str, str], float] = {}
        self._pending: list[tuple[Constraint, float | None]] = []

    def set_objective(self, qubo: dict[tuple[str, str], float]) -> None:
        """Set the objective QUBO. Overwrites any previously set objective."""
        self._objective = dict(qubo)

    def add_constraint(self, constraint: Constraint, weight: float | None = None) -> None:
        """Queue a typed constraint for compilation, with an optional explicit weight."""
        self._pending.append((constraint, weight))

    def compile(self, scale: float = 2.0) -> LogicalGraph:
        """Compile the objective and all queued constraints into a LogicalGraph.

        Args:
            scale: Safety-margin multiplier passed to
                default_penalty_weight for every constraint that wasn't
                given an explicit weight.

        Returns:
            A validated LogicalGraph. Its metadata carries
            ``penalty_offset`` (the constant energy term dropped by the
            penalty expansions — add it back to recover the true
            objective-plus-penalty energy of any assignment) and
            ``aux_variables`` (slack/assignment variables introduced by
            constraint compilation, not part of the caller's original
            problem).

        Raises:
            ValueError: If the resulting graph fails validation, or if
                an Inequality has a non-integer slack bound or is
                infeasible for every binary assignment.
        """
        interactions: _Terms = dict(self._objective)
        constant = 0.0
        variables: set[str] = set()
        for i, j in interactions:
            variables.add(i)
            variables.add(j)

        for idx, (constraint, weight) in enumerate(self._pending):
            prefix = f"__c{idx}_{constraint.name or type(constraint).__name__}"
            terms, const, touched = _compile_one(
                constraint, weight, scale, prefix, self._objective
            )
            _merge(interactions, terms)
            constant += const
            variables |= touched

        # Slack variables introduced by inequality encodings are always
        # named f"{prefix}_slack{b}" with the "__c<idx>_" prefix above, so
        # they're identifiable without separately threading an aux-var set
        # through every _compile_one call.
        aux_vars = {v for v in variables if "_slack" in v}

        var_list = [Variable(name=v) for v in sorted(variables)]
        ix_list = [
            Interaction(i=k[0], j=k[1], weight=w)
            for k, w in sorted(interactions.items())
            if w != 0.0
        ]
        graph = LogicalGraph(
            variables=var_list,
            interactions=ix_list,
            metadata={
                "source": "formulation",
                "penalty_offset": constant,
                "aux_variables": sorted(aux_vars),
            },
        )
        errors = graph.validate()
        if errors:
            raise ValueError(f"LogicalGraph validation failed: {errors}")
        return graph
