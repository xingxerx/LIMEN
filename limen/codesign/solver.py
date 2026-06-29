# Copyright (C) 2026 Jemone McCubbin / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Stackelberg co-design solver — Python interface to the limen_core Rust extension.

Drives the joint penalty-coefficient / embedding-quality optimisation loop.
The Rust StackelbergSolver scores each iteration; this module handles
recompilation when the chain strength recommendation changes.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from limen.core.compiler import PhysicalEncoding, compile_lexicographic
from limen.qubo_spectrum import qubo_energy_spectrum
from limen.validator.validator import validate

try:
    from limen_core import EquilibriumScore, StackelbergSolver

    _RUST_AVAILABLE = True
except ImportError:
    # Pure-Python reference port of the Rust solver (same semantics).
    from limen.codesign._pyfallback import EquilibriumScore, StackelbergSolver

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
    spectrum = qubo_energy_spectrum(qubo)
    if spectrum is None:
        return best_energy * 0.95

    energies = spectrum.distinct_energies
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
    chain_break_fraction_fn: Callable[[PhysicalEncoding], float] | None = None,
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
        chain_break_fraction_fn: Optional callable (PhysicalEncoding) → float
            that returns the measured chain-break fraction for the current
            encoding. When None (default), chain-break fraction is 0.0
            (simulation mode). Pass dwave_chain_break_fn() to populate this
            from real D-Wave QPU hardware responses.

    Returns:
        A CoDesignResult describing the best encoding found and the full
        convergence history.

    Note:
        Uses the limen_core Rust extension when built; otherwise falls back
        to the pure-Python reference port in limen.codesign._pyfallback.
        The backend used is recorded in result.metadata["solver_backend"].
    """
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

        cbf = (
            chain_break_fraction_fn(current_encoding)
            if chain_break_fraction_fn is not None
            else 0.0
        )
        confidences.append(vr.confidence)
        best_energies.append(vr.best_energy)
        second_best_energies.append(s_best)
        chain_break_fractions.append(cbf)
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
            "solver_backend": "rust" if _RUST_AVAILABLE else "python",
        },
    )
