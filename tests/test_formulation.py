"""Validation loop for limen.formulation: constraints in, QUBO out, brute-force
solve, check the solution actually satisfies the original constraints.

Follows the same physics-validation philosophy as tests/test_physics_validation.py
(discover properties over randomized inputs rather than asserting one baked-in
answer) and the brute-force pattern in tests/test_frontend_vrp.py.
"""

import itertools
import math
import random

import pytest

from limen.formulation import (
    AllDifferent,
    AtLeastK,
    AtMostK,
    ConstraintCompiler,
    Equality,
    Inequality,
    OneHot,
)


def _qubo_energy(interactions, assignment) -> float:
    total = 0.0
    for ix in interactions:
        if ix.i == ix.j:
            total += ix.weight * assignment[ix.i]
        else:
            total += ix.weight * assignment[ix.i] * assignment[ix.j]
    return total


def _brute_force(graph):
    names = sorted(v.name for v in graph.variables)
    best_energy = math.inf
    best_assignment = None
    for bits in itertools.product((0, 1), repeat=len(names)):
        assignment = dict(zip(names, bits))
        e = _qubo_energy(graph.interactions, assignment)
        if e < best_energy:
            best_energy = e
            best_assignment = assignment
    return best_energy, best_assignment


def _random_objective(rng, names):
    qubo = {(n, n): rng.uniform(-1.0, 1.0) for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if rng.random() < 0.5:
                qubo[(names[i], names[j])] = rng.uniform(-1.0, 1.0)
    return qubo


# ---------------------------------------------------------------------------
# Equality / OneHot
# ---------------------------------------------------------------------------


def test_equality_is_satisfied_at_the_optimum():
    """A penalized equality must dominate any objective pull away from it."""
    rng = random.Random(1)
    names = ["a", "b", "c", "d"]
    for _ in range(15):
        compiler = ConstraintCompiler()
        compiler.set_objective(_random_objective(rng, names))
        compiler.add_constraint(Equality(coeffs={"a": 1.0, "b": 1.0, "c": 1.0}, rhs=2.0))
        graph = compiler.compile()
        _, best = _brute_force(graph)
        assert best["a"] + best["b"] + best["c"] == 2


def test_onehot_group_has_exactly_one_selected():
    rng = random.Random(2)
    names = ["x0", "x1", "x2", "x3"]
    for _ in range(15):
        compiler = ConstraintCompiler()
        compiler.set_objective(_random_objective(rng, names))
        compiler.add_constraint(OneHot(names))
        graph = compiler.compile()
        _, best = _brute_force(graph)
        assert sum(best[n] for n in names) == 1


def test_penalty_offset_recovers_true_energy_when_feasible():
    """metadata['penalty_offset'] + QUBO energy == objective energy for a
    feasible (constraint-satisfying) assignment, since the penalty term is
    exactly zero there."""
    objective = {("a", "a"): 0.7, ("b", "b"): -0.3, ("a", "b"): 0.4}
    compiler = ConstraintCompiler()
    compiler.set_objective(objective)
    compiler.add_constraint(Equality(coeffs={"a": 1.0, "b": 1.0}, rhs=1.0))
    graph = compiler.compile()

    feasible = {"a": 1, "b": 0}
    qubo_energy = _qubo_energy(graph.interactions, feasible)
    recovered = qubo_energy + graph.metadata["penalty_offset"]
    expected_objective_energy = objective[("a", "a")] * 1 + objective[("a", "b")] * 1 * 0
    assert recovered == pytest.approx(expected_objective_energy)


# ---------------------------------------------------------------------------
# AtMostK / AtLeastK (k=1 pairwise path and general slack-encoded path)
# ---------------------------------------------------------------------------


def test_at_most_one_never_selects_two():
    rng = random.Random(3)
    names = ["p", "q", "r"]
    for _ in range(15):
        compiler = ConstraintCompiler()
        compiler.set_objective(_random_objective(rng, names))
        compiler.add_constraint(AtMostK(names, k=1))
        graph = compiler.compile()
        _, best = _brute_force(graph)
        assert sum(best[n] for n in names) <= 1


def test_at_most_k_general_respects_bound():
    rng = random.Random(4)
    names = ["v0", "v1", "v2", "v3"]
    for _ in range(10):
        compiler = ConstraintCompiler()
        compiler.set_objective(_random_objective(rng, names))
        compiler.add_constraint(AtMostK(names, k=2))
        graph = compiler.compile()
        _, best = _brute_force(graph)
        assert sum(best[n] for n in names) <= 2


def test_at_least_k_respects_bound():
    rng = random.Random(5)
    names = ["v0", "v1", "v2", "v3"]
    for _ in range(10):
        compiler = ConstraintCompiler()
        compiler.set_objective(_random_objective(rng, names))
        compiler.add_constraint(AtLeastK(names, k=3))
        graph = compiler.compile()
        _, best = _brute_force(graph)
        assert sum(best[n] for n in names) >= 3


def test_at_most_zero_forces_all_off():
    compiler = ConstraintCompiler()
    compiler.set_objective({("a", "a"): -5.0, ("b", "b"): -5.0})
    compiler.add_constraint(AtMostK(["a", "b"], k=0))
    graph = compiler.compile()
    _, best = _brute_force(graph)
    assert best["a"] == 0
    assert best["b"] == 0


# ---------------------------------------------------------------------------
# Inequality (direct, arbitrary coefficients)
# ---------------------------------------------------------------------------


def test_inequality_le_respects_bound():
    compiler = ConstraintCompiler()
    compiler.set_objective({("a", "a"): -3.0, ("b", "b"): -2.0, ("c", "c"): -1.0})
    compiler.add_constraint(Inequality(coeffs={"a": 2.0, "b": 1.0, "c": 1.0}, rhs=3.0, sense="<="))
    graph = compiler.compile()
    _, best = _brute_force(graph)
    assert 2 * best["a"] + best["b"] + best["c"] <= 3


def test_inequality_ge_respects_bound():
    compiler = ConstraintCompiler()
    compiler.set_objective({("a", "a"): 3.0, ("b", "b"): 2.0, ("c", "c"): 1.0})
    compiler.add_constraint(Inequality(coeffs={"a": 1.0, "b": 1.0, "c": 1.0}, rhs=2.0, sense=">="))
    graph = compiler.compile()
    _, best = _brute_force(graph)
    assert best["a"] + best["b"] + best["c"] >= 2


def test_inequality_infeasible_bound_raises():
    compiler = ConstraintCompiler()
    compiler.set_objective({})
    compiler.add_constraint(Inequality(coeffs={"a": -1.0}, rhs=-5.0, sense="<="))
    with pytest.raises(ValueError):
        compiler.compile()


def test_inequality_non_integer_bound_raises():
    compiler = ConstraintCompiler()
    compiler.set_objective({})
    compiler.add_constraint(Inequality(coeffs={"a": 0.5}, rhs=0.7, sense="<="))
    with pytest.raises(ValueError):
        compiler.compile()


# ---------------------------------------------------------------------------
# AllDifferent (row one-hot + column at-most-one assignment pattern)
# ---------------------------------------------------------------------------


def test_all_different_produces_a_valid_assignment():
    items = ["i0", "i1"]
    slots = ["s0", "s1"]
    compiler = ConstraintCompiler()
    compiler.set_objective({})
    compiler.add_constraint(AllDifferent(items=items, slots=slots))
    graph = compiler.compile()
    _, best = _brute_force(graph)

    for item in items:
        row = [best[f"{item}__{slot}"] for slot in slots]
        assert sum(row) == 1, f"{item} must be assigned exactly one slot"
    for slot in slots:
        col = [best[f"{item}__{slot}"] for item in items]
        assert sum(col) <= 1, f"{slot} must not be shared by two items"
