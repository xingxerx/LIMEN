# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Submit a CutPlan's sub-circuits to a real IBM QPU and collect real counts.

Reuses the exact QiskitRuntimeService/SamplerV2 client-construction pattern
already established in limen.backends.qiskit_backend.run_qiskit_qpu.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from limen.cutting.partition import CutPlan

_INSTALL_MSG = (
    "qiskit-addon-cutting and qiskit-ibm-runtime are required to dispatch "
    "cut circuits. Install with: pip install limen[cutting]"
)


def _check_deps() -> None:
    try:
        import qiskit_addon_cutting  # noqa: F401
        import qiskit_ibm_runtime  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc


@dataclass
class CutDispatchResult:
    """Real per-sub-circuit, per-sample measurement counts from real hardware.

    Attributes:
        counts: One (sample_index, subcircuit_label, counts, shots) tuple per
            real sub-experiment job result. ``counts`` maps a bitstring (the
            ``observable_measurements`` register concatenated with the
            ``qpd_measurements`` register, per-shot) to its real shot count.
        coefficients: One (sample_index, coefficient) tuple per joint QPD
            sample, as produced by generate_cutting_experiments.
        subcircuit_labels: Every partition label participating in the cut.
        job_ids: Partition label -> real IBM job id for that label's batch
            of sub-experiments.
    """

    counts: list[tuple[int, str, dict[str, int], int]]
    coefficients: list[tuple[int, float]]
    subcircuit_labels: list[str]
    job_ids: dict[str, str] = field(default_factory=dict)


def run_cut_circuit(
    plan: CutPlan,
    token: str,
    crn: str,
    backend_name: str = "ibm_kingston",
    shots: int = 1000,
    num_samples: float = float("inf"),
    timeout: float = 600.0,
) -> CutDispatchResult:
    """Submit every sub-circuit's QPD sub-experiments to a real IBM QPU.

    Args:
        plan: A CutPlan from find_cuts_and_partition.
        token: IBM Quantum Platform API token.
        crn: IBM Quantum service instance CRN.
        backend_name: IBM backend identifier. Must have at least
            plan's max sub-circuit width qubits available.
        shots: Shots per sub-experiment.
        num_samples: Forwarded to generate_cutting_experiments. Use
            float("inf") for the exact QPD expansion (no Monte Carlo
            sampling error) -- the right choice unless the number of
            cuts makes the exact expansion too large to submit.
        timeout: Seconds to wait for each partition's job (default 600).

    Returns:
        A CutDispatchResult with real counts, real coefficients, and the
        real job id submitted for each partition.

    Raises:
        ImportError: If qiskit-addon-cutting or qiskit-ibm-runtime is not installed.
        ValueError: If a partition has more than one commuting observable
            group -- this dispatcher assumes a single-Pauli-string observable
            per partition (true whenever the original observable passed to
            find_cuts_and_partition was a single Pauli string), so each
            partition's subexperiment count must equal len(coefficients).
        RuntimeError: If a job does not complete within timeout seconds.
    """
    _check_deps()
    from qiskit.primitives.containers.bit_array import BitArray
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_addon_cutting import generate_cutting_experiments
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    subexperiments, coefficients = generate_cutting_experiments(
        plan.subcircuits, plan.subobservables, num_samples
    )
    if not isinstance(subexperiments, dict):
        # Only happens if plan.subcircuits were a single QuantumCircuit rather
        # than the dict partition_problem always returns; guards against a
        # future caller bypassing find_cuts_and_partition.
        raise ValueError("expected partitioned (dict) subexperiments")

    # partition_problem auto-labels partitions with plain ints (0, 1, 2, ...);
    # the Rust side wants String labels, so str() once here and use the
    # original keys only to index back into the qiskit-side dicts.
    raw_labels = sorted(subexperiments.keys(), key=str)
    coefficients_out = [(i, float(c[0])) for i, c in enumerate(coefficients)]

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token, instance=crn
    )
    backend = service.backend(backend_name)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

    counts_out: list[tuple[int, str, dict[str, int], int]] = []
    job_ids: dict[str, str] = {}

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

        transpiled = [pm.run(circ) for circ in circuits]
        sampler = SamplerV2(mode=backend)
        job = sampler.run([(circ,) for circ in transpiled], shots=shots)
        job_ids[label] = job.job_id()

        pub_results = job.result(timeout=timeout)
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
        job_ids=job_ids,
    )
