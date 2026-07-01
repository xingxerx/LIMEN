# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Run a genuinely-too-wide-for-156-qubits circuit through limen.cutting
against real ibm_kingston hardware.

Builds a 160-qubit chain circuit (1 qubit wider than ibm_kingston's 156),
derived from a Max-Cut QUBO over a 160-node path graph -- the same QUBO
convention as examples/cobyla_multi_backend.py's ring_max_cut_qubo, but
without the wraparound edge, so the chain has exactly one crossing edge
no matter where it's split into two <=80-qubit halves. That keeps the cut
count to exactly 1 (kappa^2 = 4 exact QPD samples for one CNOT gate cut),
which keeps the real-hardware job count small for this first checkpoint.

Gate decomposition note: the cost layer uses one CX per chain edge (rather
than the full CX-RZ-CX exp(-i*gamma*Z*Z) identity) specifically so each
edge is exactly one two-qubit gate -- with the RZ identity there would be
two CX gates crossing the boundary, doubling the cut count. The RZ local
bias terms (from the QUBO's diagonal coefficients) and the RX mixer are
still applied as real single-qubit rotations; they don't need cutting.

Observable: a single global Pauli string "Z"*160 (parity), the same
single-Pauli-observable convention validated in cutting_smoke_test.py.
This checkpoint is about exercising the real cut/dispatch/reconstruct
pipeline against real hardware noise, not reconstructing a full multi-term
QAOA cost Hamiltonian (which would need one cut-and-reconstruct call per
Pauli term in the cost Hamiltonian).

Two modes:
    python examples/cut_circuit_ibm_kingston.py --preview
        Builds the circuit and cut plan locally (no qiskit_ibm_runtime
        calls, zero QPU cost) and prints num_cuts, sub-circuit widths,
        sample count, and total real jobs/shots this would submit.

    python examples/cut_circuit_ibm_kingston.py --submit
        Actually calls limen.cutting.dispatch.run_cut_circuit against
        ibm_kingston and limen.cutting.reconstruct_from_results, then
        prints the reconstructed <Z...Z> and the real job ids.

Required environment variables for --submit:
    IBM_QUANTUM_TOKEN, IBM_QUANTUM_CRN
"""

from __future__ import annotations

import argparse
import os

from limen.cutting.partition import find_cuts_and_partition
from limen.gates.ir import CircuitIR, GateInstruction

N_QUBITS = 160
MAX_SUBCIRCUIT_QUBITS = 80
GAMMA = 0.1
BETA = 0.1
SHOTS = 1000


def chain_max_cut_qubo(n: int) -> dict[tuple[str, str], float]:
    """Same convention as cobyla_multi_backend.ring_max_cut_qubo, minus the
    wraparound edge -- a path instead of a cycle, so any contiguous split
    into two halves crosses exactly one edge."""
    qubo: dict[tuple[str, str], float] = {}
    names = [f"x{i:03d}" for i in range(n)]
    for i in range(n - 1):
        u, v = names[i], names[i + 1]
        key = (min(u, v), max(u, v))
        qubo[key] = qubo.get(key, 0.0) + 1.0
        qubo[(u, u)] = qubo.get((u, u), 0.0) - 1.0
        qubo[(v, v)] = qubo.get((v, v), 0.0) - 1.0
    return qubo


def build_chain_circuit(n: int, gamma: float, beta: float) -> CircuitIR:
    qubo = chain_max_cut_qubo(n)
    names = [f"x{i:03d}" for i in range(n)]
    index = {name: i for i, name in enumerate(names)}

    instructions: list[GateInstruction] = []
    for i in range(n):
        instructions.append(GateInstruction("h", [i]))

    # Local bias from each diagonal QUBO coefficient -- single-qubit, not
    # cut-relevant.
    for name in names:
        coeff = qubo.get((name, name), 0.0)
        if coeff:
            instructions.append(GateInstruction("rz", [index[name]], [2 * gamma * coeff]))

    # Chain coupling: one CX per edge. Only the edge crossing the partition
    # boundary needs a cut.
    for i in range(n - 1):
        instructions.append(GateInstruction("cx", [i, i + 1]))

    # QAOA mixer.
    for i in range(n):
        instructions.append(GateInstruction("rx", [i], [2 * beta]))

    return CircuitIR(n_qubits=n, instructions=instructions)


def preview() -> None:
    circuit = build_chain_circuit(N_QUBITS, GAMMA, BETA)
    observable = "Z" * N_QUBITS

    plan = find_cuts_and_partition(circuit, observable, MAX_SUBCIRCUIT_QUBITS)

    from qiskit_addon_cutting import generate_cutting_experiments

    subexperiments, coefficients = generate_cutting_experiments(
        plan.subcircuits, plan.subobservables, float("inf")
    )
    assert isinstance(subexperiments, dict)
    num_partitions = len(subexperiments)
    num_samples = len(coefficients)
    total_jobs = num_partitions  # one batched SamplerV2 job per partition
    total_subexperiments = num_partitions * num_samples
    total_shots = total_subexperiments * SHOTS

    print(f"original circuit: {circuit.n_qubits} qubits")
    print(f"cuts found: {plan.num_cuts}  ({plan.metadata['cuts']})")
    print(f"sampling_overhead (kappa^2): {plan.metadata['sampling_overhead']:.1f}")
    for label, subcirc in sorted(plan.subcircuits.items(), key=str):
        print(f"  partition {label}: {subcirc.num_qubits} qubits")
    print(f"QPD samples (exact expansion): {num_samples}")
    print(f"real IBM jobs this would submit: {total_jobs} (one per partition, batching all samples)")
    print(f"real sub-experiments total: {total_subexperiments}")
    print(f"shots per sub-experiment: {SHOTS}")
    print(f"total real shots across all jobs: {total_shots}")


def submit() -> None:
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")
    if not token or not crn:
        raise SystemExit("IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN must be set")

    from limen.cutting.dispatch import run_cut_circuit
    from limen.cutting.reconstruct import reconstruct_from_results

    circuit = build_chain_circuit(N_QUBITS, GAMMA, BETA)
    observable = "Z" * N_QUBITS
    plan = find_cuts_and_partition(circuit, observable, MAX_SUBCIRCUIT_QUBITS)

    print(f"submitting {plan.num_cuts} cut(s), {len(plan.subcircuits)} partitions "
          f"to ibm_kingston, {SHOTS} shots/sub-experiment ...")
    result = run_cut_circuit(
        plan, token=token, crn=crn, backend_name="ibm_kingston", shots=SHOTS
    )
    print(f"real job ids: {result.job_ids}")

    value = reconstruct_from_results(result)
    print(f"reconstructed <Z...Z> ({N_QUBITS} qubits) from real ibm_kingston hardware: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preview", action="store_true")
    group.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if args.preview:
        preview()
    else:
        submit()
