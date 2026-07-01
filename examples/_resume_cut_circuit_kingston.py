# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Resume the 160-qubit cut-circuit run after partition 0's job outlived
the 600s timeout in run_cut_circuit (it is real and still queued/running
server-side, not failed -- see job d91iuqeu9n7c73amqo50 on ibm_kingston).

Re-derives the same cut plan and QPD sample coefficients deterministically
(generate_cutting_experiments with num_samples=float("inf") performs the
exact expansion, not random sampling, so re-running it locally reproduces
the same subexperiments/coefficients with zero QPU cost). Waits on the
already-submitted partition-0 job rather than resubmitting it, then submits
partition 1 fresh, then reconstructs.
"""

from __future__ import annotations

import os

from qiskit.primitives.containers.bit_array import BitArray
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_addon_cutting import generate_cutting_experiments
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from limen.cutting.partition import find_cuts_and_partition
from limen.cutting.reconstruct import reconstruct_from_results
from limen.cutting.dispatch import CutDispatchResult
from examples.cut_circuit_ibm_kingston import (
    N_QUBITS,
    GAMMA,
    BETA,
    MAX_SUBCIRCUIT_QUBITS,
    SHOTS,
    build_chain_circuit,
)

PARTITION_0_JOB_ID = "d91iuqeu9n7c73amqo50"
BACKEND_NAME = "ibm_kingston"


def main() -> None:
    token = os.environ["IBM_QUANTUM_TOKEN"]
    crn = os.environ["IBM_QUANTUM_CRN"]

    circuit = build_chain_circuit(N_QUBITS, GAMMA, BETA)
    observable = "Z" * N_QUBITS
    plan = find_cuts_and_partition(circuit, observable, MAX_SUBCIRCUIT_QUBITS)

    subexperiments, coefficients = generate_cutting_experiments(
        plan.subcircuits, plan.subobservables, float("inf")
    )
    assert isinstance(subexperiments, dict)
    raw_labels = sorted(subexperiments.keys(), key=str)
    coefficients_out = [(i, float(c[0])) for i, c in enumerate(coefficients)]
    print(f"partitions: {[str(l) for l in raw_labels]}, samples: {len(coefficients)}")

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token, instance=crn
    )
    backend = service.backend(BACKEND_NAME)

    counts_out: list[tuple[int, str, dict, int]] = []
    job_ids: dict[str, str] = {}

    label0 = str(raw_labels[0])
    print(f"waiting on already-submitted job {PARTITION_0_JOB_ID} for partition {label0} ...")
    job0 = service.job(PARTITION_0_JOB_ID)
    pub_results0 = job0.result(timeout=None)
    job_ids[label0] = PARTITION_0_JOB_ID
    for sample_index, pub_result in enumerate(pub_results0):
        data = pub_result.data
        joint = BitArray.concatenate_bits(
            [data.observable_measurements, data.qpd_measurements]
        )
        counts_out.append((sample_index, label0, joint.get_counts(), joint.num_shots))
    print(f"partition {label0} done ({len(pub_results0)} sub-experiments recovered)")

    label1 = str(raw_labels[1])
    print(f"submitting partition {label1} fresh ...")
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    circuits1 = subexperiments[raw_labels[1]]
    transpiled1 = [pm.run(c) for c in circuits1]
    sampler = SamplerV2(mode=backend)
    job1 = sampler.run([(c,) for c in transpiled1], shots=SHOTS)
    job_ids[label1] = job1.job_id()
    print(f"partition {label1} job id: {job1.job_id()}, waiting ...")
    pub_results1 = job1.result(timeout=None)
    for sample_index, pub_result in enumerate(pub_results1):
        data = pub_result.data
        joint = BitArray.concatenate_bits(
            [data.observable_measurements, data.qpd_measurements]
        )
        counts_out.append((sample_index, label1, joint.get_counts(), joint.num_shots))
    print(f"partition {label1} done ({len(pub_results1)} sub-experiments recovered)")

    result = CutDispatchResult(
        counts=counts_out,
        coefficients=coefficients_out,
        subcircuit_labels=[str(l) for l in raw_labels],
        job_ids=job_ids,
    )

    value = reconstruct_from_results(result)
    print(f"real job ids: {job_ids}")
    print(f"reconstructed <Z...Z> ({N_QUBITS} qubits) from real ibm_kingston hardware: {value:.4f}")


if __name__ == "__main__":
    main()
