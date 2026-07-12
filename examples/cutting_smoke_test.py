# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""End-to-end smoke test for limen.cutting, spending zero real QPU time.

Cuts a 4-qubit circuit into two 2-qubit sub-circuits (1 cut, exact QPD
expansion), runs every sub-experiment on a local AerSimulator via
BackendSamplerV2 (which produces the same PrimitiveResult/BitArray shape
as the real qiskit_ibm_runtime.SamplerV2 path in limen.cutting.dispatch),
and reconstructs the expectation value via limen_core.cutting. Compares
the reconstruction against a plain AerSimulator run of the same circuit,
uncut.

This validates the cut + reconstruct round-trip is correct before trusting
it on a circuit too wide to verify any other way. Run it with:
    python examples/cutting_smoke_test.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from qiskit.primitives import BackendSamplerV2
from qiskit.primitives.containers.bit_array import BitArray
from qiskit.quantum_info import PauliList
from qiskit_addon_cutting import (
    DeviceConstraints,
    OptimizationParameters,
    find_cuts,
    generate_cutting_experiments,
    partition_problem,
)
from qiskit_aer import AerSimulator

from limen.cutting.partition import _unmeasured_copy
from limen.gates.ir import CircuitIR, GateInstruction
from limen.gates.qiskit_exec import to_qiskit_circuit

OBSERVABLE = "ZZZZ"
SHOTS = 20000


def _plain_expectation(circuit: CircuitIR) -> float:
    """Plain AerSimulator expectation of Z⊗Z⊗Z⊗Z on the uncut circuit."""
    qc = to_qiskit_circuit(circuit)
    backend = AerSimulator()
    transpiled = backend_transpile(qc, backend)
    job = backend.run(transpiled, shots=SHOTS)
    counts = job.result().get_counts()
    total = sum(counts.values())
    acc = 0.0
    for bitstring, count in counts.items():
        ones = bitstring.count("1")
        parity = 1.0 if ones % 2 == 0 else -1.0
        acc += parity * count
    return acc / total


def backend_transpile(qc, backend):
    from qiskit import transpile

    return transpile(qc, backend)


def _cut_and_reconstruct(circuit: CircuitIR) -> float:
    from limen import limen_core

    qc = _unmeasured_copy(to_qiskit_circuit(circuit))
    constraints = DeviceConstraints(qubits_per_subcircuit=2)
    optimization = OptimizationParameters()
    cut_circuit, find_cuts_metadata = find_cuts(qc, optimization, constraints)
    print(f"cuts inserted: {find_cuts_metadata['cuts']}")

    partitioned = partition_problem(cut_circuit, observables=PauliList([OBSERVABLE]))
    assert partitioned.subobservables is not None

    subexperiments, coefficients = generate_cutting_experiments(
        dict(partitioned.subcircuits), dict(partitioned.subobservables), float("inf")
    )
    assert isinstance(subexperiments, dict)

    raw_labels = sorted(subexperiments.keys(), key=str)
    backend = AerSimulator()
    sampler = BackendSamplerV2(backend=backend)

    counts_objs = []
    for raw_label in raw_labels:
        label = str(raw_label)
        circuits = subexperiments[raw_label]
        assert len(circuits) == len(coefficients), (
            f"partition {label} has {len(circuits)} sub-experiments but there "
            f"are {len(coefficients)} QPD samples"
        )
        job = sampler.run([(circ,) for circ in circuits], shots=SHOTS)
        pub_results = job.result()
        for sample_index, pub_result in enumerate(pub_results):
            data = pub_result.data
            joint = BitArray.concatenate_bits(
                [data.observable_measurements, data.qpd_measurements]
            )
            counts_objs.append(
                limen_core.cutting.SubcircuitSampleCounts(
                    sample_index, label, joint.get_counts(), joint.num_shots
                )
            )

    coeff_objs = [
        limen_core.cutting.SampleCoefficient(i, float(c[0]))
        for i, c in enumerate(coefficients)
    ]
    subcircuit_labels = [str(label) for label in raw_labels]

    return limen_core.cutting.reconstruct_expectation(
        counts_objs, coeff_objs, subcircuit_labels
    )


def main() -> None:
    # GHZ state: (|0000> + |1111>)/sqrt(2), exact <ZZZZ> = 1.0 (no shot
    # noise floor), so a broken reconstruction has nowhere to hide behind
    # symmetry the way an all-H input (exact <ZZZZ> = 0) would.
    circuit = CircuitIR(
        n_qubits=4,
        instructions=[
            GateInstruction("h", [0]),
            GateInstruction("cx", [0, 1]),
            GateInstruction("cx", [1, 2]),
            GateInstruction("cx", [2, 3]),
        ],
    )

    plain = _plain_expectation(circuit)
    cut = _cut_and_reconstruct(circuit)

    print(f"plain AerSimulator <ZZZZ>:      {plain:.4f}")
    print(f"cut + reconstructed <ZZZZ>:     {cut:.4f}")
    print(f"absolute difference:            {abs(plain - cut):.4f}")

    tolerance = 0.05
    if abs(plain - cut) > tolerance:
        raise SystemExit(
            f"FAIL: reconstruction diverges from plain simulation by more than "
            f"{tolerance} (shot noise only, both runs use {SHOTS} shots)"
        )
    print("PASS: cut + reconstruct round-trip matches plain simulation")


if __name__ == "__main__":
    main()
