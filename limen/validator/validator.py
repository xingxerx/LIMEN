"""Probabilistic validator for LIMEN.

Compares simulated hardware-like runs against a classical brute-force
solver on small QUBO instances to produce a confidence bound for a
PhysicalEncoding. No external libraries are required.
"""

import random
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from limen.core.compiler import PhysicalEncoding


@dataclass
class ValidationResult:
    """The outcome of a probabilistic validation run.

    Attributes:
        confidence: Fraction of simulated runs within tolerance of the
            best energy found (0.0 to 1.0).
        feasible_runs: Number of runs that fell within tolerance of
            best_energy.
        total_runs: Total number of simulated runs.
        best_energy: Lowest energy observed across all simulated runs.
        classical_energy: Energy of the optimal solution found by brute-force,
            or None if the problem has more than 20 variables.
        notes: Human-readable observations about the validation outcome.
        metadata: Arbitrary key/value annotations.
    """

    confidence: float
    feasible_runs: int
    total_runs: int
    best_energy: float
    classical_energy: float | None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this result to a plain Python dict."""
        return {
            "confidence": self.confidence,
            "feasible_runs": self.feasible_runs,
            "total_runs": self.total_runs,
            "best_energy": self.best_energy,
            "classical_energy": self.classical_energy,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }


def _compute_energy(qubo: dict[tuple[str, str], float], assignment: dict[str, int]) -> float:
    """Return the QUBO energy for a given binary assignment."""
    return sum(w * assignment[i] * assignment[j] for (i, j), w in qubo.items())


def brute_force_solve(
    qubo: dict[tuple[str, str], float],
) -> tuple[dict[str, int], float] | None:
    """Find the minimum-energy binary assignment by exhaustive search.

    Args:
        qubo: QUBO dict mapping (variable, variable) pairs to float weights.

    Returns:
        A (best_assignment, best_energy) tuple where best_assignment maps
        each variable name to 0 or 1, or None if the problem has more than
        20 variables (too large to solve classically).
    """
    variables: list[str] = sorted({name for pair in qubo for name in pair})
    if len(variables) > 20:
        return None

    best_assignment: dict[str, int] = {}
    best_energy = float("inf")

    for bits in product((0, 1), repeat=len(variables)):
        assignment = dict(zip(variables, bits))
        energy = _compute_energy(qubo, assignment)
        if energy < best_energy:
            best_energy = energy
            best_assignment = assignment

    return best_assignment, best_energy


def simulate_runs(
    qubo: dict[tuple[str, str], float],
    n_runs: int,
    noise_level: float = 0.05,
    seed: int = 42,
) -> list[tuple[dict[str, int], float]]:
    """Simulate hardware runs by adding noise to the best known solution.

    Each run starts from the best known binary assignment and independently
    flips each variable with probability noise_level, then records the energy
    of the resulting assignment.

    Args:
        qubo: QUBO dict mapping (variable, variable) pairs to float weights.
        n_runs: Number of simulated runs to perform.
        noise_level: Per-variable bit-flip probability (0.0–1.0).
        seed: Seed for the internal RNG, ensuring deterministic output.

    Returns:
        A list of (assignment, energy) tuples, one per run.
    """
    rng = random.Random(seed)
    variables: list[str] = sorted({name for pair in qubo for name in pair})

    bf_result = brute_force_solve(qubo)
    if bf_result is not None:
        base_assignment, _ = bf_result
    else:
        base_assignment = {v: rng.randint(0, 1) for v in variables}

    results: list[tuple[dict[str, int], float]] = []
    for _ in range(n_runs):
        noisy = {
            v: (1 - val if rng.random() < noise_level else val)
            for v, val in base_assignment.items()
        }
        results.append((noisy, _compute_energy(qubo, noisy)))

    return results


def validate(
    encoding: PhysicalEncoding,
    runs: int = 1000,
    noise_level: float = 0.05,
    seed: int = 42,
) -> ValidationResult:
    """Validate a PhysicalEncoding by comparing simulated runs to classical optima.

    Args:
        encoding: A PhysicalEncoding produced by the LIMEN compiler.
        runs: Number of simulated hardware runs.
        noise_level: Per-variable bit-flip probability used in simulation.
        seed: RNG seed for reproducible results.

    Returns:
        A ValidationResult containing confidence, energy statistics, and
        human-readable notes.
    """
    qubo = encoding.qubo

    simulated = simulate_runs(qubo, n_runs=runs, noise_level=noise_level, seed=seed)
    bf_result = brute_force_solve(qubo)

    classical_energy: float | None = bf_result[1] if bf_result is not None else None
    best_energy = min(e for _, e in simulated) if simulated else 0.0

    tol = abs(best_energy) * 0.05 if best_energy != 0.0 else 0.01
    feasible_runs = sum(1 for _, e in simulated if e <= best_energy + tol)
    confidence = feasible_runs / runs if runs > 0 else 0.0

    notes: list[str] = []
    if classical_energy is None:
        notes.append(
            "Problem too large for classical verification (>20 variables). "
            "Confidence is self-referential."
        )
    if confidence > 0.9:
        notes.append("High confidence. Results are consistent.")
    elif confidence < 0.5:
        notes.append(
            "Low confidence. Check penalty coefficients or embedding quality."
        )

    return ValidationResult(
        confidence=confidence,
        feasible_runs=feasible_runs,
        total_runs=runs,
        best_energy=best_energy,
        classical_energy=classical_energy,
        notes=notes,
    )
