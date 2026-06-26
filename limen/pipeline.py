"""End-to-end gate-model pipeline: QUBO -> compile -> execute -> certify.

This module is the composition path the rest of LIMEN's gate-model track
was missing. It threads a single problem through every layer:

    QUBO dict
      -> LogicalGraph                      (limen.frontends)
      -> QAOA CircuitIR                    (limen.gates.qaoa)
      -> statevector execution + readout   (limen.gates.simulator)
      -> classical optimality check        (limen.validator)
      -> surface-code logical-error budget (limen.ecc)

and returns a single EndToEndCertificate that composes the optimization
result with the ECC LogicalErrorCertificate - the two certificate worlds
that previously had no shared output type.

Scope: the surface-code term certifies the logical error rate of running
the solution on distance-d-protected qubits at a given physical error
rate. It does not perform full fault-tolerant lattice-surgery compilation
of the QAOA circuit (out of scope; see limen/docs/architecture.md for how
LIMEN documents such gaps rather than half-implementing them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from limen.core.ir import LogicalGraph
from limen.ecc.certificate import certify_logical_qubit
from limen.ecc.decoder import LookupDecoder
from limen.ecc.encoder import verify_corrects_all_weight_one
from limen.ecc.surface_code import build_surface_code
from limen.frontends.pyqubo import from_qubo_dict
from limen.gates.qaoa import bitstring_to_assignment, compile_qaoa, variable_order
from limen.gates.simulator import probabilities
from limen.validator.validator import brute_force_solve


@dataclass
class EndToEndCertificate:
    """Composed result of the full QUBO -> gates -> ECC pipeline."""

    solution: dict[str, int]
    energy: float
    classical_energy: float | None
    is_optimal: bool | None
    success_probability: float
    qaoa_layers: int
    qaoa_params: dict[str, float]
    logical_error_rate: float | None
    aggregate_logical_error_rate: float | None
    physical_error_rate: float | None
    distance: int | None
    n_logical_qubits: int
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution": dict(self.solution),
            "energy": self.energy,
            "classical_energy": self.classical_energy,
            "is_optimal": self.is_optimal,
            "success_probability": self.success_probability,
            "qaoa_layers": self.qaoa_layers,
            "qaoa_params": dict(self.qaoa_params),
            "logical_error_rate": self.logical_error_rate,
            "aggregate_logical_error_rate": self.aggregate_logical_error_rate,
            "physical_error_rate": self.physical_error_rate,
            "distance": self.distance,
            "n_logical_qubits": self.n_logical_qubits,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }


def _graph_qubo(graph: LogicalGraph) -> dict[tuple[str, str], float]:
    return {(ix.i, ix.j): ix.weight for ix in graph.interactions}


def _energy(qubo: dict[tuple[str, str], float], assignment: dict[str, int]) -> float:
    return sum(w * assignment[i] * assignment[j] for (i, j), w in qubo.items())


def _grid_search(
    graph: LogicalGraph, layers: int, grid_size: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Search a 2D (gamma, beta) grid, shared across layers, for minimum <C>.

    Returns the best (gamma, beta) params and the resulting measurement
    distribution. A single shared angle pair per layer is a deliberately
    restricted QAOA schedule, sufficient for the small instances this
    offline pipeline targets.
    """
    order = variable_order(graph)
    qubo = _graph_qubo(graph)
    energies: dict[str, float] = {}

    def outcome_energy(bits: str) -> float:
        cached = energies.get(bits)
        if cached is None:
            cached = _energy(qubo, bitstring_to_assignment(bits, order))
            energies[bits] = cached
        return cached

    best_params = {"gamma": 0.0, "beta": 0.0}
    best_dist: dict[str, float] = {}
    best_expected = math.inf
    for gi in range(grid_size):
        gamma = 2.0 * math.pi * gi / grid_size
        for bi in range(grid_size):
            beta = math.pi * bi / grid_size
            circuit = compile_qaoa(graph, [gamma] * layers, [beta] * layers)
            dist = probabilities(circuit)
            expected = sum(p * outcome_energy(bits) for bits, p in dist.items())
            if expected < best_expected - 1e-12:
                best_expected = expected
                best_params = {"gamma": gamma, "beta": beta}
                best_dist = dist
    return best_params, best_dist


def run_pipeline(
    qubo: dict[tuple[str, str], float],
    *,
    qaoa_layers: int = 1,
    grid_size: int = 12,
    distance: int = 3,
    physical_error_rate: float | None = None,
    encode_logical: bool = True,
    seed: int = 42,
) -> EndToEndCertificate:
    """Run a QUBO end-to-end through the gate-model track and certify it.

    Args:
        qubo: QUBO dict mapping (var, var) pairs to weights.
        qaoa_layers: Number of QAOA cost+mixer layers (shared angles).
        grid_size: Resolution of the (gamma, beta) parameter grid.
        distance: Surface-code distance for the logical-qubit certificate.
        physical_error_rate: Per-qubit bit-flip rate for the ECC term;
            if None, the logical-qubit certificate is skipped.
        encode_logical: Whether to compute the surface-code certificate.
        seed: Reserved for reproducibility; the pipeline is deterministic.

    Returns:
        An EndToEndCertificate composing the QAOA solution with the
        surface-code logical-error budget.
    """
    graph = from_qubo_dict(qubo)
    order = variable_order(graph)
    n = len(order)
    canonical_qubo = _graph_qubo(graph)

    params, dist = _grid_search(graph, qaoa_layers, grid_size)
    solution_bits = max(dist, key=dist.get) if dist else "0" * n
    solution = bitstring_to_assignment(solution_bits, order)
    energy = _energy(canonical_qubo, solution)

    bf = brute_force_solve(canonical_qubo)
    classical_energy = bf[1] if bf is not None else None
    is_optimal = (
        None if classical_energy is None else abs(energy - classical_energy) < 1e-9
    )

    target = classical_energy if classical_energy is not None else energy
    success_probability = sum(
        p
        for bits, p in dist.items()
        if abs(_energy(canonical_qubo, bitstring_to_assignment(bits, order)) - target) < 1e-9
    )

    logical_rate: float | None = None
    aggregate_rate: float | None = None
    roundtrip_corrects_all_weight1: bool | None = None
    notes: list[str] = []
    if encode_logical and physical_error_rate is not None:
        patch = build_surface_code(distance)
        decoder = LookupDecoder(patch)
        cert = certify_logical_qubit(patch, decoder, physical_error_rate)
        logical_rate = cert.logical_error_rate
        aggregate_rate = 1.0 - (1.0 - logical_rate) ** n
        # Back the analytic certificate with executed gate circuits: run
        # the syndrome-extraction loop on the simulator for every weight-1
        # X error and confirm the code corrects them all.
        roundtrip_corrects_all_weight1 = verify_corrects_all_weight_one(patch, decoder)
        notes.append(
            f"Logical-qubit budget: distance-{distance} surface code at physical "
            f"error rate {physical_error_rate} gives per-qubit logical error "
            f"{logical_rate:.3e} ({n} logical qubits)."
        )
        notes.append(
            "Gate-executed round-trip "
            + ("corrects" if roundtrip_corrects_all_weight1 else "FAILS to correct")
            + " all weight-1 X errors."
        )
    elif encode_logical:
        notes.append("ECC certificate skipped: no physical_error_rate supplied.")

    if is_optimal:
        notes.append("QAOA most-likely outcome matches the classical optimum.")
    elif is_optimal is False:
        notes.append("QAOA most-likely outcome is sub-optimal; raise qaoa_layers/grid_size.")

    return EndToEndCertificate(
        solution=solution,
        energy=energy,
        classical_energy=classical_energy,
        is_optimal=is_optimal,
        success_probability=success_probability,
        qaoa_layers=qaoa_layers,
        qaoa_params=params,
        logical_error_rate=logical_rate,
        aggregate_logical_error_rate=aggregate_rate,
        physical_error_rate=physical_error_rate,
        distance=distance if logical_rate is not None else None,
        n_logical_qubits=n,
        notes=notes,
        metadata={
            "variable_order": order,
            "seed": seed,
            "roundtrip_corrects_all_weight1": roundtrip_corrects_all_weight1,
        },
    )
