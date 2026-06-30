# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""IBM QPU demo for LIMEN.

Runs the trivial 2-variable QUBO

    {("x0","x0"): -1.0, ("x1","x1"): -1.0, ("x0","x1"): 2.0}

on a real IBM quantum processor (ibm_kingston) via QiskitRuntimeService and
SamplerV2, then compares results against a local exact simulator.

Required environment variables:
    IBM_QUANTUM_TOKEN  — IBM Quantum Platform API token
    IBM_QUANTUM_CRN    — IBM Quantum instance CRN (service instance identifier)

Usage::

    IBM_QUANTUM_TOKEN=<token> IBM_QUANTUM_CRN=<crn> python examples/ibm_qpu_demo.py
"""

import os
import pathlib
import sys

# Allow running directly from the project root without a full package install.
# The Rust extension (limen_core) is only needed for co-design; this demo
# does not use it.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Load .env from the project root if present (never required — real env vars
# take precedence via dotenv's override=False default).
try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass

from typing import Any

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.backends.qiskit_backend import run_qiskit

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------

QUBO: dict[tuple[str, str], float] = {
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x0", "x1"):  2.0,
}

_OPTIMAL_ENERGY = -1.0
_BACKEND_NAME = "ibm_kingston"
_NUM_SHOTS = 1000

# ---------------------------------------------------------------------------
# QUBO helpers
# ---------------------------------------------------------------------------

def _qubo_to_ising(
    qubo: dict[tuple[str, str], float],
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Convert a QUBO dict to Ising (h, J) via x_i = (1 + s_i) / 2.

    Args:
        qubo: QUBO dict mapping ``(var_i, var_j)`` pairs to float weights.

    Returns:
        ``(h, J)`` where ``h`` maps variable names to linear Ising biases
        and ``J`` maps ordered ``(i, j)`` pairs to quadratic couplings.
    """
    variables = sorted({name for pair in qubo for name in pair})
    h: dict[str, float] = {v: 0.0 for v in variables}
    J: dict[tuple[str, str], float] = {}
    for (i, j), w in qubo.items():
        if i == j:
            h[i] += w / 2.0
        else:
            h[i] += w / 4.0
            h[j] += w / 4.0
            key = (min(i, j), max(i, j))
            J[key] = J.get(key, 0.0) + w / 4.0
    return h, J


def _qubo_energy(qubo: dict[tuple[str, str], float], assignment: dict[str, int]) -> float:
    """Compute QUBO energy for a binary assignment."""
    return sum(w * assignment[i] * assignment[j] for (i, j), w in qubo.items())


def _bits_to_assignment(bitstring: str, variables: list[str]) -> dict[str, int]:
    """Convert a Qiskit bitstring to a variable→int dict.

    Qiskit orders bits right-to-left so q0 is the rightmost character.
    """
    bits = bitstring[::-1]
    return {v: int(bits[idx]) for idx, v in enumerate(variables)}


def _optimal_rate_from_list(energies: list[float]) -> float:
    """Fraction of shots at the optimal energy, as a percentage."""
    if not energies:
        return 0.0
    count = sum(1 for e in energies if abs(e - _OPTIMAL_ENERGY) < 1e-9)
    return count / len(energies) * 100.0


def _optimal_rate_from_counts(
    counts: dict[str, int],
    energy_map: dict[str, float],
    total: int,
) -> float:
    """Fraction of QPU shots at the optimal energy, as a percentage."""
    if total == 0:
        return 0.0
    count = sum(
        cnt for bs, cnt in counts.items()
        if abs(energy_map[bs] - _OPTIMAL_ENERGY) < 1e-9
    )
    return count / total * 100.0


# ---------------------------------------------------------------------------
# IBM QPU runner
# ---------------------------------------------------------------------------

def run_ibm_qpu(
    qubo: dict[tuple[str, str], float],
    token: str,
    crn: str,
    backend_name: str = _BACKEND_NAME,
    shots: int = _NUM_SHOTS,
) -> dict[str, Any]:
    """Build a 2-qubit QAOA circuit and execute it on a real IBM QPU.

    Constructs a depth-1 QAOAAnsatz from the QUBO's Ising Hamiltonian,
    transpiles it for the target backend, and runs it with SamplerV2.

    Args:
        qubo: QUBO problem dict (logical variable names as keys).
        token: IBM Quantum Platform API token.
        crn: Service instance CRN.
        backend_name: IBM backend identifier.
        shots: Number of measurement shots.

    Returns:
        Dict with keys ``backend_name``, ``shots``, ``counts``,
        ``energies_by_bitstring``, ``best_energy``, ``best_assignment``,
        and ``circuit_depth``.
    """
    try:
        from qiskit_ibm_runtime import (  # type: ignore[import]
            QiskitRuntimeService,
            SamplerV2 as Sampler,
        )
        from qiskit.circuit.library import QAOAAnsatz  # type: ignore[import]
        from qiskit.quantum_info import SparsePauliOp  # type: ignore[import]
        from qiskit.transpiler.preset_passmanagers import (  # type: ignore[import]
            generate_preset_pass_manager,
        )
    except ModuleNotFoundError as exc:
        print(
            "ERROR: Required packages not installed.\n"
            "Install with: pip install qiskit qiskit-ibm-runtime\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=crn,
    )
    backend = service.backend(backend_name)

    variables = sorted({name for pair in qubo for name in pair})
    n = len(variables)
    var_idx = {v: idx for idx, v in enumerate(variables)}

    h, J = _qubo_to_ising(qubo)

    # Build SparsePauliOp cost Hamiltonian from Ising biases.
    paulis: list[tuple[str, complex]] = []
    for v, bias in h.items():
        if bias != 0.0:
            label = ["I"] * n
            label[var_idx[v]] = "Z"
            paulis.append(("".join(reversed(label)), bias))
    for (vi, vj), coupling in J.items():
        if coupling != 0.0:
            label = ["I"] * n
            label[var_idx[vi]] = "Z"
            label[var_idx[vj]] = "Z"
            paulis.append(("".join(reversed(label)), coupling))
    if not paulis:
        paulis = [("I" * n, 0.0)]

    cost_op = SparsePauliOp.from_list(paulis)
    ansatz = QAOAAnsatz(cost_op, reps=1)
    ansatz.measure_all()

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled = pm.run(ansatz)
    circuit_depth: int = transpiled.depth()

    # Fixed initial parameters: β=0.1, γ=0.1 per QAOA layer.
    params = [0.1] * ansatz.num_parameters

    sampler = Sampler(mode=backend)
    job = sampler.run([(transpiled, params)], shots=shots)
    pub_result = job.result()[0]
    counts: dict[str, int] = pub_result.data.meas.get_counts()

    energy_map: dict[str, float] = {
        bs: _qubo_energy(qubo, _bits_to_assignment(bs, variables))
        for bs in counts
    }
    best_bs = min(counts, key=lambda b: energy_map[b])

    return {
        "backend_name": backend_name,
        "shots": shots,
        "counts": counts,
        "energies_by_bitstring": energy_map,
        "best_energy": energy_map[best_bs],
        "best_assignment": _bits_to_assignment(best_bs, variables),
        "circuit_depth": circuit_depth,
        "num_qubits": n,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    local_result: Any,
    local_best_logical: dict[str, int],
    qpu_result: dict[str, Any],
    num_shots: int,
) -> None:
    """Print the side-by-side simulator vs QPU comparison report.

    Args:
        local_result: ``QiskitResult`` from the exact simulator run.
        local_best_logical: Best assignment with logical variable names.
        qpu_result: Result dict from :func:`run_ibm_qpu`.
        num_shots: Total shot count used for both runs.
    """
    W = 63

    sim_rate = _optimal_rate_from_list(local_result.energies)
    qpu_rate = _optimal_rate_from_counts(
        qpu_result["counts"],
        qpu_result["energies_by_bitstring"],
        num_shots,
    )
    noise_rate = 100.0 - qpu_rate
    energy_diff = abs(local_result.best_energy - qpu_result["best_energy"])

    sim_assign_str = ", ".join(
        f"{k}={v}" for k, v in sorted(local_best_logical.items())
    )
    qpu_assign_str = ", ".join(
        f"{k}={v}" for k, v in sorted(qpu_result["best_assignment"].items())
    )
    top4 = sorted(qpu_result["counts"].items(), key=lambda x: -x[1])[:4]
    counts_str = ", ".join(f"{bs}: {cnt}" for bs, cnt in top4)

    print(f"── LIMEN IBM QPU Demo {'─' * (W - 21)}")
    print(f"  Problem : x0^2 - x0 + x1^2 - x1 + 2*x0*x1 (trivial 2-var QUBO)")
    print(f"  Optimal : x0=1,x1=0 or x0=0,x1=1  energy={_OPTIMAL_ENERGY:.1f}")
    print()
    print("  Local Exact Simulator")
    print(f"    Best energy    : {local_result.best_energy:.4f}")
    print(f"    Best assignment: {{{sim_assign_str}}}")
    print(f"    Confidence     : {sim_rate:.1f}%")
    print()
    print(f"  IBM QPU ({qpu_result['backend_name']} — real hardware)")
    print(f"    Backend        : {qpu_result['backend_name']}")
    print(f"    Qubits used    : {qpu_result['num_qubits']}")
    print(f"    Shots          : {num_shots}")
    print(f"    Best energy    : {qpu_result['best_energy']:.4f}")
    print(f"    Best assignment: {{{qpu_assign_str}}}")
    print(f"    Result counts  : {counts_str}")
    print(f"    QPU noise      : {noise_rate:.1f}% of shots on non-optimal assignments")
    print()
    print("  Comparison")
    print(f"    Simulator optimal rate : {sim_rate:.1f}%")
    print(f"    QPU optimal rate       : {qpu_rate:.1f}%")
    print(f"    Energy difference      : {energy_diff:.4f}")
    print(f"── End {'─' * (W - 6)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Compile the trivial QUBO, simulate locally, then run on real IBM QPU."""
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")

    if not token:
        print(
            "ERROR: IBM_QUANTUM_TOKEN environment variable is not set.\n"
            "Export your IBM Quantum Platform API token before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not crn:
        print(
            "ERROR: IBM_QUANTUM_CRN environment variable is not set.\n"
            "Export your IBM Quantum instance CRN before running this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Compile QUBO → PhysicalEncoding.
    graph = from_qubo_dict(QUBO)
    encoding = compile_lexicographic(graph, default_hardware_graph(2))

    # 2. Local exact simulation (encoding.qubo uses physical qubit names).
    print("Running local exact simulation ...")
    local_result = run_qiskit(encoding, num_shots=_NUM_SHOTS, algorithm="exact", seed=42)

    # Map physical qubit labels back to logical variable names for the report.
    phys_to_logical = {pq[0]: lv for lv, pq in encoding.embedding.items()}
    local_best_logical = {
        phys_to_logical.get(k, k): v
        for k, v in local_result.best_assignment.items()
    }

    # 3. Submit QAOA circuit to real QPU (uses original logical variable names).
    print(f"Submitting QAOA circuit to IBM QPU ({_BACKEND_NAME}) ...")
    qpu_result = run_ibm_qpu(
        QUBO,
        token=token,
        crn=crn,
        backend_name=_BACKEND_NAME,
        shots=_NUM_SHOTS,
    )

    # 4. Print side-by-side report.
    print()
    print_report(local_result, local_best_logical, qpu_result, _NUM_SHOTS)


if __name__ == "__main__":
    main()
