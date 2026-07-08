# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Bridge a QUBO into limen.cutting's Pauli-observable reconstruction.

limen.cutting reconstructs a single Pauli observable's scalar expectation
value from a cut circuit's real sub-experiment counts -- it does not
produce a sampled solution bitstring the way limen.pipeline.run_pipeline
does. This module supplies the missing piece for QUBO problems too large
for a single backend: reconstruct each qubit's single-Z expectation value
<Z_i> via cutting, then decode a solution bitstring from those marginals
by threshold rounding (the standard "round the expectation value"
technique used throughout the QAOA literature to extract an approximate
solution without sampling a full bitstring).

This is deliberately scoped, not a substitute for exact optimality:

  - decoded_classical_energy (see limen.pipeline.run_cut_route_request)
    is the *exact* classical QUBO energy of the decoded bitstring -- no
    reconstruction error, since it is evaluated classically once decoding
    is done.
  - The decode itself is a heuristic (rounding), not a certified optimum
    -- there is no brute-force check at the sizes circuit cutting exists
    for, so a CuttingCertificate always reports is_optimal=None.
  - mean_field_expected_energy is an approximate cross-check
    (<Z_i Z_j> ~= <Z_i><Z_j>), not the true reconstructed <H>: computing
    the true two-qubit correlator would need one additional cutting
    dispatch per coupling, on top of the n already needed for the
    marginals themselves.

Reconstructing the *true* full probability distribution over bitstrings
via quasi-probability decomposition (rather than this expectation-value/
marginal-rounding approach) is a much larger, more uncertain research
problem (see docs/gap-analysis and CHANGELOG) -- explicitly out of scope
here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from limen.cutting.partition import CutPlan, find_cuts_and_partition
from limen.frontends.pyqubo import from_qubo_dict
from limen.gates.ir import CircuitIR
from limen.gates.qaoa import qubo_to_ising


def pauli_z_string(n: int, qubits: set[int]) -> str:
    """Build an n-qubit Pauli string with 'Z' on *qubits*, 'I' elsewhere.

    Follows Qiskit's convention that the rightmost character is qubit 0
    (empirically confirmed: ``Statevector.expectation_value(Pauli("IZ"))``
    reflects qubit 0's state, matching this codebase's 1-to-1
    CircuitIR-qubit-index -> Qiskit-qubit-index mapping in
    limen.gates.qiskit_exec.to_qiskit_circuit).
    """
    chars = ["I"] * n
    for q in qubits:
        chars[n - 1 - q] = "Z"
    return "".join(chars)


def qubo_ising_terms(
    qubo: dict[tuple[str, str], float]
) -> tuple[dict[int, float], dict[tuple[int, int], float], float, list[str]]:
    """QUBO dict -> (h, J, constant, order): the exact diagonal Ising form.

    ``constant + sum h[i] Z_i + sum J[(i, j)] Z_i Z_j`` reproduces the
    QUBO's classical energy exactly under ``x_i = (1 - z_i) / 2``
    (reuses limen.gates.qaoa.qubo_to_ising for h/J; the constant term
    that function discards is recovered here from the same substitution
    it documents: 0.5*w per linear term, 0.25*w per quadratic term).
    """
    graph = from_qubo_dict(qubo)
    h, j_coeffs, order = qubo_to_ising(graph)
    constant = 0.0
    for ix in graph.interactions:
        constant += 0.5 * ix.weight if ix.i == ix.j else 0.25 * ix.weight
    return h, j_coeffs, constant, order


@dataclass
class MarginalReconstructionResult:
    """Per-qubit <Z_i> expectation values reconstructed via circuit cutting,
    plus the cutting metadata accumulated along the way."""

    marginals: dict[int, float]
    num_cuts: int
    num_partitions: int
    job_ids: dict[str, str] = field(default_factory=dict)


def reconstruct_z_marginals_via_cutting(
    circuit: CircuitIR,
    qubits: list[int],
    max_subcircuit_qubits: int,
    dispatch_fn: Callable[[CutPlan], Any],
    reconstruct_fn: Callable[[Any], float],
) -> MarginalReconstructionResult:
    """Reconstruct <Z_i> for each qubit in *qubits* via circuit cutting.

    One find_cuts_and_partition + dispatch_fn + reconstruct_fn round trip
    per qubit -- O(len(qubits)) cutting dispatches, the cost of getting
    every marginal needed to decode a solution (see module docstring).

    *dispatch_fn*/*reconstruct_fn* are injected so this function is
    backend-agnostic: pass limen.cutting.dispatch.run_cut_circuit +
    limen.cutting.reconstruct.reconstruct_from_results for a real QPU, or
    limen.cutting.local_dispatch.run_cut_circuit_locally + the same
    reconstruct_from_results for a zero-credit local-sampler run (see
    limen.pipeline.run_cut_route_request).
    """
    n = circuit.n_qubits
    marginals: dict[int, float] = {}
    num_cuts = 0
    num_partitions = 0
    job_ids: dict[str, str] = {}
    for q in qubits:
        observable = pauli_z_string(n, {q})
        plan = find_cuts_and_partition(circuit, observable, max_subcircuit_qubits)
        num_cuts = max(num_cuts, plan.num_cuts)
        num_partitions = max(num_partitions, len(plan.subcircuits))
        dispatch_result = dispatch_fn(plan)
        marginals[q] = reconstruct_fn(dispatch_result)
        result_job_ids = getattr(dispatch_result, "job_ids", None)
        if result_job_ids:
            job_ids.update(
                {f"q{q}:{label}": jid for label, jid in result_job_ids.items()}
            )
    return MarginalReconstructionResult(
        marginals=marginals,
        num_cuts=num_cuts,
        num_partitions=num_partitions,
        job_ids=job_ids,
    )


def decode_bitstring_from_marginals(
    marginals: dict[int, float], order: list[str]
) -> dict[str, int]:
    """Threshold-round <Z_i> marginals into a {variable_name: bit} assignment.

    x_i = (1 - z_i)/2 exactly recovers a bit from an exact +-1 eigenvalue;
    for an approximate <Z_i> in [-1, 1] this rounds to the nearer
    eigenvalue first (<Z_i> >= 0 -> z=+1 -> x=0; <Z_i> < 0 -> z=-1 ->
    x=1) -- the standard expectation-value-rounding heuristic.
    """
    return {
        name: (0 if marginals.get(i, 1.0) >= 0.0 else 1)
        for i, name in enumerate(order)
    }


def classical_energy(
    qubo: dict[tuple[str, str], float], assignment: dict[str, int]
) -> float:
    """Exact classical QUBO energy of *assignment* -- no reconstruction error."""
    return sum(w * assignment[i] * assignment[j] for (i, j), w in qubo.items())


def mean_field_expected_energy(
    h: dict[int, float],
    j_coeffs: dict[tuple[int, int], float],
    constant: float,
    marginals: dict[int, float],
) -> float:
    """Mean-field cross-check: constant + sum h_i<Z_i> + sum J_ij<Z_i><Z_j>.

    Approximates the two-qubit correlator <Z_i Z_j> by the product of
    marginals <Z_i><Z_j> instead of reconstructing it directly (which
    would need one more cutting dispatch per coupling) -- see module
    docstring. Costs zero cutting dispatches beyond the marginals already
    computed by reconstruct_z_marginals_via_cutting.
    """
    total = constant
    for i, coeff in h.items():
        total += coeff * marginals.get(i, 0.0)
    for (a, b), coeff in j_coeffs.items():
        total += coeff * marginals.get(a, 0.0) * marginals.get(b, 0.0)
    return total
