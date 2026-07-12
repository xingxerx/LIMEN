# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""CuttingCertificate: the run_cut_route_request output shape.

Deliberately a different shape from limen.pipeline.EndToEndCertificate,
not a subclass or a padded-with-Nones version of it: circuit cutting
reconstructs Pauli-observable expectation values, not a sampled solution
bitstring, so a cutting-based certificate cannot honestly claim
``is_optimal`` the way a brute-force-checked EndToEndCertificate can (see
limen.cutting.qubo_bridge module docstring for the marginal-rounding
decode this certificate's ``solution`` comes from). ``is_optimal`` is
always ``None`` here -- not a sentinel for "not yet computed", but an
explicit statement that optimality was never checked at this problem
size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CuttingCertificate:
    """Composed result of the cut-circuit QUBO bridge.

    Attributes:
        solution: {variable_name: 0/1}, decoded from cutting-reconstructed
            single-qubit <Z_i> marginals via threshold rounding.
        decoded_classical_energy: Exact classical QUBO energy of
            ``solution`` -- no reconstruction error, since this is
            computed classically once decoding is done.
        reconstructed_expected_energy: A mean-field cross-check
            (constant + sum h_i<Z_i> + sum J_ij<Z_i><Z_j>), not the true
            reconstructed <H> -- see
            limen.cutting.qubo_bridge.mean_field_expected_energy.
        is_optimal: Always None. Circuit cutting is used precisely when
            the problem is too large to brute-force check, so this
            certificate never claims optimality either way.
        num_partitions: Number of sub-circuit partitions used (the
            largest across all per-qubit marginal reconstructions).
        num_cuts: Number of cuts inserted (likewise, the largest seen).
        job_ids: Real backend job ids, keyed "q<i>:<partition_label>";
            empty for an offline/local-sampler run.
        logical_error_rate: From limen.ecc.certificate.certify_logical_qubit,
            reused unmodified -- solution-agnostic, so no different from
            EndToEndCertificate's own ECC term.
        physical_error_rate: The physical_error_rate the ECC certificate
            was computed against.
        distance: Surface-code distance the ECC certificate used.
        notes: Human-readable caveats, always including an explicit
            statement of the marginal-rounding decode and the mean-field
            approximation (see class docstring).
        metadata: Arbitrary annotations (e.g. n_vars, max_subcircuit_qubits).
    """

    solution: dict[str, int]
    decoded_classical_energy: float
    reconstructed_expected_energy: float
    is_optimal: bool | None
    num_partitions: int
    num_cuts: int
    job_ids: dict[str, str]
    logical_error_rate: float | None
    physical_error_rate: float | None
    distance: int | None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution": dict(self.solution),
            "decoded_classical_energy": self.decoded_classical_energy,
            "reconstructed_expected_energy": self.reconstructed_expected_energy,
            "is_optimal": self.is_optimal,
            "num_partitions": self.num_partitions,
            "num_cuts": self.num_cuts,
            "job_ids": dict(self.job_ids),
            "logical_error_rate": self.logical_error_rate,
            "physical_error_rate": self.physical_error_rate,
            "distance": self.distance,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }
