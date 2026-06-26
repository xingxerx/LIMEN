# Copyright 2026 LIMEN Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the pure-Python Stackelberg solver fallback.

These mirror the solver-level cases in test_codesign.py so the fallback's
semantics stay locked to the Rust implementation. When limen_core is built,
an extra parity test compares the two implementations directly.
"""

import pytest

from limen.codesign._pyfallback import EquilibriumScore, StackelbergSolver


def test_kappa_bounds():
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


def test_converges_at_high_confidence():
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


def test_increases_chain_strength_when_not_converged():
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
    assert score.kappa < 0.99
    assert cs > current_cs


def test_chain_break_fraction_lowers_kappa():
    """A higher chain-break fraction must lower kappa, all else equal."""
    solver = StackelbergSolver(target_kappa=0.85, max_iterations=10, learning_rate=0.1)
    _, clean = solver.solve([0.8], [-2.0], [-1.0], [0.0], 2.0)
    _, noisy = solver.solve([0.8], [-2.0], [-1.0], [0.5], 2.0)
    assert noisy.kappa < clean.kappa
    assert noisy.kappa == pytest.approx(clean.kappa - 0.2 * 0.5)


def test_kappa_std_zero_for_single_iteration():
    """kappa_std is 0.0 with fewer than 2 observations."""
    solver = StackelbergSolver(target_kappa=0.85, max_iterations=10, learning_rate=0.1)
    _, score = solver.solve([0.8], [-2.0], [-1.0], [0.0], 2.0)
    assert score.kappa_std == 0.0


def test_parity_with_rust_extension():
    """Pure-Python fallback must match limen_core exactly on the same inputs."""
    rust = pytest.importorskip("limen_core", reason="limen_core not built")

    args = (
        [0.3, 0.5, 0.7, 0.6],
        [-2.0, -2.0, -2.0, -2.0],
        [-1.0, -1.2, -0.8, -1.0],
        [0.1, 0.0, 0.2, 0.05],
        2.0,
    )
    py_cs, py_score = StackelbergSolver(0.85, 10, 0.1).solve(*args)
    rs_cs, rs_score = rust.StackelbergSolver(0.85, 10, 0.1).solve(*args)

    assert py_cs == pytest.approx(rs_cs)
    assert py_score.kappa == pytest.approx(rs_score.kappa)
    assert py_score.kappa_std == pytest.approx(rs_score.kappa_std)
    assert py_score.energy_gap == pytest.approx(rs_score.energy_gap)


def test_qubo_energy_spectrum_parity_with_rust_extension():
    """Pure-Python qubo_energy_spectrum fallback must match limen_core exactly."""
    rust = pytest.importorskip("limen_core", reason="limen_core not built")

    from limen._qubo_spectrum_pyfallback import qubo_energy_spectrum as py_spectrum

    # A small QUBO with linear and quadratic terms, indexed 0..n_vars-1.
    qubo_terms = [
        ((0, 0), 1.0),
        ((1, 1), -2.0),
        ((2, 2), 0.5),
        ((0, 1), -1.5),
        ((1, 2), 2.0),
        ((0, 2), 0.75),
    ]
    n_vars = 3

    py_bits, py_energy, py_distinct = py_spectrum(qubo_terms, n_vars)
    rs_bits, rs_energy, rs_distinct = rust.qubo_energy_spectrum(qubo_terms, n_vars)

    assert list(py_bits) == list(rs_bits)
    assert py_energy == pytest.approx(rs_energy)
    assert list(py_distinct) == pytest.approx(list(rs_distinct))


def test_qubo_energy_spectrum_pyfallback_size_guard():
    """Pure-Python fallback raises ValueError for n_vars > 20, mirroring Rust's SizeViolation."""
    from limen._qubo_spectrum_pyfallback import qubo_energy_spectrum as py_spectrum

    with pytest.raises(ValueError):
        py_spectrum([], 21)


def test_qubo_energy_spectrum_pyfallback_empty():
    """Pure-Python fallback handles the zero-variable edge case."""
    from limen._qubo_spectrum_pyfallback import qubo_energy_spectrum as py_spectrum

    bits, energy, distinct = py_spectrum([], 0)
    assert bits == []
    assert energy == 0.0
    assert distinct == [0.0]
