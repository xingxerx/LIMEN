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
"""Stackelberg co-design solver — Python interface to the limen_core Rust extension.

Drives the joint penalty-coefficient / embedding-quality optimisation loop.
The Rust StackelbergSolver scores each iteration; this module handles
recompilation when the chain strength recommendation changes.
"""

from dataclasses import dataclass, field
from typing import Any

from limen.core.compiler import PhysicalEncoding, compile_lexicographic
from limen.validator.validator import brute_force_solve, validate

try:
    from limen_core import EquilibriumScore, StackelbergSolver

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False


@dataclass
class CoDesignResult:
    """The result of a Stackelberg co-design run.

    Attributes:
        encoding: The best PhysicalEncoding found during the loop.
        score: The EquilibriumScore at convergence, or None if the Rust
            extension was unavailable (should not happen in practice).
        kappa: Calibration margin κ ∈ [0.0, 1.0] of the final encoding.
        iterations: Number of iterations executed.
        chain_strength_history: Chain strength value used at each iteration.
        confidence_history: Validator confidence at each iteration.
        converged: True if κ ≥ target_kappa before max_iterations.
        metadata: Arbitrary annotations from the solver run.
    """

    encoding: PhysicalEncoding
    score: Any
    kappa: float
    kappa_std: float
    iterations: int
    chain_strength_history: list[float]
    confidence_history: list[float]
    converged: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def _second_best_energy(qubo: dict, best_energy: float) -> float:
    """Return the second-distinct energy from brute force, or an approximation."""
    result = brute_force_solve(qubo)
    if result is None:
        return best_energy * 0.95

    variables = sorted({v for pair in qubo for v in pair})
    from itertools import product as iproduct

    energies = sorted(
        {
            sum(w * asgn[i] * asgn[j] for (i, j), w in qubo.items())
            for asgn in (dict(zip(variables, bits)) for bits in iproduct((0, 1), repeat=len(variables)))
        }
    )
    return energies[1] if len(energies) >= 2 else energies[0]


def _rebuild_hardware_graph(embedding: dict[str, list[str]]) -> dict[str, list[str]]:
    """Reconstruct a complete graph over the physical qubits used in an embedding."""
    nodes = [qubits[0] for qubits in embedding.values()]
    return {node: [other for other in nodes if other != node] for node in nodes}


def run_codesign(
    encoding: PhysicalEncoding,
    target_kappa: float = 0.85,
    max_iterations: int = 50,
    learning_rate: float = 0.1,
    runs_per_iteration: int = 500,
    seed: int = 42,
) -> CoDesignResult:
    """Run the Stackelberg co-design loop on a PhysicalEncoding.

    Each iteration validates the current encoding, scores it via the Rust
    StackelbergSolver, and recompiles with an adjusted chain strength if
    the calibration margin κ has not yet reached target_kappa.

    Args:
        encoding: Starting PhysicalEncoding from the LIMEN compiler.
        target_kappa: Convergence threshold for κ (default 0.85).
        max_iterations: Hard cap on the number of iterations (default 50).
        learning_rate: Multiplicative step size for chain-strength updates
            (default 0.1).
        runs_per_iteration: Simulated runs per validation call (default 500).
        seed: Base RNG seed; each iteration uses seed + i for independence.

    Returns:
        A CoDesignResult describing the best encoding found and the full
        convergence history.

    Raises:
        ImportError: If the limen_core Rust extension is not installed.
    """
    if not _RUST_AVAILABLE:
        raise ImportError(
            "limen_core Rust extension required. Run: maturin develop"
        )

    solver = StackelbergSolver(target_kappa, max_iterations, learning_rate)
    current_encoding = encoding

    confidences: list[float] = []
    best_energies: list[float] = []
    second_best_energies: list[float] = []
    chain_break_fractions: list[float] = []
    chain_strength_history: list[float] = []
    converged = False

    for i in range(max_iterations):
        vr = validate(current_encoding, runs=runs_per_iteration, seed=seed + i)
        s_best = _second_best_energy(current_encoding.qubo, vr.best_energy)

        confidences.append(vr.confidence)
        best_energies.append(vr.best_energy)
        second_best_energies.append(s_best)
        chain_break_fractions.append(0.0)
        chain_strength_history.append(current_encoding.chain_strength)

        recommended_cs, best_score = solver.solve(
            confidences,
            best_energies,
            second_best_energies,
            chain_break_fractions,
            current_encoding.chain_strength,
        )

        if best_score.kappa >= target_kappa:
            converged = True
            break

        if recommended_cs != current_encoding.chain_strength:
            hw = _rebuild_hardware_graph(current_encoding.embedding)
            current_encoding = compile_lexicographic(
                encoding.graph,
                hw,
                chain_strength=recommended_cs,
                seed=seed,
            )

    return CoDesignResult(
        encoding=current_encoding,
        score=best_score,
        kappa=best_score.kappa,
        kappa_std=best_score.kappa_std,
        iterations=len(confidences),
        chain_strength_history=chain_strength_history,
        confidence_history=confidences,
        converged=converged,
        metadata={
            "target_kappa": target_kappa,
            "max_iterations": max_iterations,
            "learning_rate": learning_rate,
            "kappa_std": best_score.kappa_std,
        },
    )
