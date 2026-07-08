# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Circuit cutting: run wider-than-any-backend QAOA circuits on real QPUs.

Cuts a circuit into sub-circuits that each fit on a validated LIMEN backend
(qiskit_addon_cutting), submits every sub-experiment as a real job
(dispatch.run_cut_circuit), and reconstructs the original circuit's
expectation value from real measurement counts in parallel Rust
(limen_core.cutting.reconstruct_expectation).
"""

from limen.cutting.certificate import CuttingCertificate
from limen.cutting.dispatch import CutDispatchResult, run_cut_circuit
from limen.cutting.local_dispatch import run_cut_circuit_locally
from limen.cutting.partition import CutPlan, find_cuts_and_partition
from limen.cutting.qubo_bridge import (
    MarginalReconstructionResult,
    classical_energy,
    decode_bitstring_from_marginals,
    mean_field_expected_energy,
    pauli_z_string,
    qubo_ising_terms,
    reconstruct_z_marginals_via_cutting,
)
from limen.cutting.reconstruct import reconstruct_from_results

__all__ = [
    "CutPlan",
    "find_cuts_and_partition",
    "CutDispatchResult",
    "run_cut_circuit",
    "run_cut_circuit_locally",
    "reconstruct_from_results",
    "CuttingCertificate",
    "MarginalReconstructionResult",
    "pauli_z_string",
    "qubo_ising_terms",
    "reconstruct_z_marginals_via_cutting",
    "decode_bitstring_from_marginals",
    "classical_energy",
    "mean_field_expected_energy",
]
