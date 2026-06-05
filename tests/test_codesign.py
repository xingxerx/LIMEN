"""Tests for the Stackelberg co-design layer (limen/codesign/).

Skipped cleanly if limen_core Rust extension is not installed.
"""

import pytest

limen_core = pytest.importorskip("limen_core", reason="limen_core Rust extension not installed")

from limen_core import StackelbergSolver, EquilibriumScore

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.codesign.solver import CoDesignResult, run_codesign
from limen.codesign.portfolio import PortfolioResult, compile_portfolio

_TRIVIAL_QUBO = {("q0", "q0"): -1.0, ("q1", "q1"): -1.0, ("q0", "q1"): 2.0}


def _make_encoding(qubo: dict = _TRIVIAL_QUBO):
    return compile_lexicographic(from_qubo_dict(qubo), default_hardware_graph(8))


def test_equilibrium_score_kappa_bounds():
    """EquilibriumScore kappa must lie in [0.0, 1.0]."""
    solver = StackelbergSolver(target_kappa=0.85, max_iterations=10, learning_rate=0.1)
    _, score = solver.solve(
        confidences=[0.8],
        best_energies=[-2.0],
        second_best_energies=[-1.0],
        chain_break_fractions=[0.0],
        current_chain_strength=2.0,
    )
    assert isinstance(score, EquilibriumScore)
    assert 0.0 <= score.kappa <= 1.0


def test_stackelberg_converges_at_high_confidence():
    """Solver returns kappa >= target when fed consistently high confidence."""
    solver = StackelbergSolver(target_kappa=0.5, max_iterations=10, learning_rate=0.1)
    cs, score = solver.solve(
        confidences=[0.9] * 5,
        best_energies=[-2.0] * 5,
        second_best_energies=[-1.0] * 5,
        chain_break_fractions=[0.0] * 5,
        current_chain_strength=2.0,
    )
    assert score.kappa >= 0.5
    assert cs > 0.0


def test_stackelberg_increases_chain_strength_when_not_converged():
    """Solver must recommend a higher chain strength when kappa is below target."""
    solver = StackelbergSolver(target_kappa=0.99, max_iterations=5, learning_rate=0.2)
    current_cs = 2.0
    cs, score = solver.solve(
        confidences=[0.3] * 5,
        best_energies=[-0.5] * 5,
        second_best_energies=[-0.4] * 5,
        chain_break_fractions=[0.0] * 5,
        current_chain_strength=current_cs,
    )
    assert cs > current_cs


def test_run_codesign_returns_result():
    """run_codesign returns a valid CoDesignResult with sane fields."""
    encoding = _make_encoding()
    result = run_codesign(
        encoding,
        target_kappa=0.5,
        max_iterations=5,
        runs_per_iteration=200,
        seed=0,
    )
    assert isinstance(result, CoDesignResult)
    assert result.kappa >= 0.0
    assert len(result.confidence_history) >= 1
    assert isinstance(result.converged, bool)


def test_compile_portfolio_ranks_candidates():
    """compile_portfolio produces sorted candidates with the best first."""
    encoding = _make_encoding()
    result = compile_portfolio(
        encoding,
        backends=["dwave_sim", "qiskit_exact"],
        target_kappa=0.5,
        max_iterations=3,
        runs_per_iteration=100,
        seed=0,
    )
    assert isinstance(result, PortfolioResult)
    assert len(result.candidates) == 2
    assert result.candidates[0].kappa >= result.candidates[1].kappa
    assert result.best_backend in ["dwave_sim", "qiskit_exact"]
