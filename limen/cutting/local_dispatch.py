# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Local AerSimulator dispatch for a CutPlan -- zero-credit, zero-network
mirror of limen.cutting.dispatch.run_cut_circuit.

Produces the identical CutDispatchResult shape (the same BitArray-derived
counts) so limen.cutting.reconstruct.reconstruct_from_results works
unchanged on either path. This is the same local-sampler pattern already
validated end-to-end in examples/cutting_smoke_test.py, factored out so
limen.pipeline.run_cut_route_request's offline=True mode and
tests/test_cutting_qubo_bridge.py can reuse it without spending real QPU
credits.
"""

from __future__ import annotations

from limen.cutting.dispatch import CutDispatchResult
from limen.cutting.partition import CutPlan

_INSTALL_MSG = (
    "qiskit-addon-cutting and qiskit-aer are required for local cut-circuit "
    "dispatch. Install with: pip install limen[cutting] qiskit-aer"
)


def run_cut_circuit_locally(plan: CutPlan, shots: int = 20000) -> CutDispatchResult:
    """Dispatch every sub-experiment in *plan* to a local AerSimulator.

    Args:
        plan: A CutPlan from find_cuts_and_partition.
        shots: Shots per sub-experiment.

    Returns:
        A CutDispatchResult with local simulator counts and empty job_ids
        (there is no remote job to track).

    Raises:
        ImportError: If qiskit-addon-cutting or qiskit-aer is not installed.
        ValueError: If a partition has more than one commuting observable
            group (see limen.cutting.dispatch.run_cut_circuit).
    """
    try:
        from qiskit.primitives import BackendSamplerV2
        from qiskit.primitives.containers.bit_array import BitArray
        from qiskit_addon_cutting import generate_cutting_experiments
        from qiskit_aer import AerSimulator
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc

    subexperiments, coefficients = generate_cutting_experiments(
        plan.subcircuits, plan.subobservables, float("inf")
    )
    if not isinstance(subexperiments, dict):
        raise ValueError("expected partitioned (dict) subexperiments")

    raw_labels = sorted(subexperiments.keys(), key=str)
    coefficients_out = [(i, float(c[0])) for i, c in enumerate(coefficients)]

    backend = AerSimulator()
    sampler = BackendSamplerV2(backend=backend)

    counts_out: list[tuple[int, str, dict[str, int], int]] = []
    for raw_label in raw_labels:
        label = str(raw_label)
        circuits = subexperiments[raw_label]
        if len(circuits) != len(coefficients):
            raise ValueError(
                f"partition {label!r} has {len(circuits)} sub-experiments but "
                f"there are {len(coefficients)} QPD samples; this dispatcher "
                "only supports observables that decompose into a single "
                "commuting group per partition"
            )
        job = sampler.run([(circ,) for circ in circuits], shots=shots)
        pub_results = job.result()
        for sample_index, pub_result in enumerate(pub_results):
            data = pub_result.data
            joint = BitArray.concatenate_bits(
                [data.observable_measurements, data.qpd_measurements]
            )
            counts_out.append(
                (sample_index, label, joint.get_counts(), joint.num_shots)
            )

    return CutDispatchResult(
        counts=counts_out,
        coefficients=coefficients_out,
        subcircuit_labels=[str(label) for label in raw_labels],
        job_ids={},
    )
