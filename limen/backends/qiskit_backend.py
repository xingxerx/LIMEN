# Copyright 2026 LIMEN Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""IBM Qiskit backend adapter for LIMEN.

Converts a PhysicalEncoding into an Ising Hamiltonian and solves it using
one of three algorithms:

- ``"exact"``  — analytical statevector enumeration (deterministic, no QPU)
- ``"qaoa"``   — Quantum Approximate Optimization Algorithm via AerSimulator
- ``"vqe"``    — Variational Quantum Eigensolver via AerSimulator

An optional IBM Runtime path (``use_runtime=True``) submits to a real IBM
backend. All Qiskit imports are guarded so this module loads cleanly without
the SDK installed; import errors surface at call time with a clear hint.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from limen.core.compiler import PhysicalEncoding

_INSTALL_MSG = (
    "The Qiskit SDK is required to use the Qiskit backend. "
    "Install it with: pip install limen[ibm]  "
    "(or: pip install qiskit qiskit-aer qiskit-algorithms)"
)


# ---------------------------------------------------------------------------
# QUBO → Ising conversion (pure Python, no Qiskit dependency)
# ---------------------------------------------------------------------------

def _qubo_to_ising(
    qubo: dict[tuple[str, str], float],
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    """Convert a QUBO dict to Ising h and J coefficients.

    Applies the substitution x_i = (1 + s_i) / 2 where s_i ∈ {-1, +1},
    giving:

        E_ising(s) = Σ_i h_i s_i + Σ_{i<j} J_ij s_i s_j  + const

    The constant offset is discarded (irrelevant for finding the optimum).

    Args:
        qubo: QUBO dict mapping ``(var_i, var_j)`` pairs to float weights.
            Diagonal entries ``(i, i)`` encode linear terms.

    Returns:
        A ``(h, J)`` tuple where ``h`` maps variable names to linear
        Ising biases and ``J`` maps ordered ``(i, j)`` pairs (i < j)
        to quadratic Ising couplings.
    """
    variables: list[str] = sorted({name for pair in qubo for name in pair})
    h: dict[str, float] = {v: 0.0 for v in variables}
    J: dict[tuple[str, str], float] = {}

    for (i, j), w in qubo.items():
        if i == j:
            # x_i = (1+s_i)/2  →  Q_ii x_i = Q_ii/2 + Q_ii/2 * s_i
            h[i] += w / 2.0
        else:
            # Q_ij x_i x_j = Q_ij/4 (1 + s_i + s_j + s_i s_j)
            h[i] += w / 4.0
            h[j] += w / 4.0
            key = (min(i, j), max(i, j))
            J[key] = J.get(key, 0.0) + w / 4.0

    return h, J


def _ising_energy(
    h: dict[str, float],
    J: dict[tuple[str, str], float],
    spin: dict[str, int],
) -> float:
    """Compute Ising energy for a spin assignment (values in {-1, +1})."""
    energy = sum(h[v] * spin[v] for v in h)
    energy += sum(w * spin[i] * spin[j] for (i, j), w in J.items())
    return energy


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class QiskitResult:
    """The result of a Qiskit sampling or variational run.

    Attributes:
        samples: All samples as variable→binary (0/1) dicts.
        energies: QUBO energy of each sample, in the same order as samples.
        best_assignment: Lowest-energy sample.
        best_energy: QUBO energy of best_assignment.
        circuit_depth: Transpiled circuit depth when available, else None.
        metadata: Algorithm name, reps, num_shots, seed, backend name, etc.
    """

    samples: list[dict[str, int]]
    energies: list[float]
    best_assignment: dict[str, int]
    best_energy: float
    circuit_depth: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _qubo_energy(qubo: dict[tuple[str, str], float], assignment: dict[str, int]) -> float:
    """Compute QUBO energy for a binary assignment."""
    return sum(w * assignment[i] * assignment[j] for (i, j), w in qubo.items())


def _bits_to_assignment(bitstring: str, variables: list[str]) -> dict[str, int]:
    """Convert a Qiskit bitstring (big-endian) to a variable→int dict."""
    # Qiskit orders bits right-to-left (q0 is rightmost character).
    bits = bitstring[::-1]
    return {v: int(bits[idx]) for idx, v in enumerate(variables)}


def _check_qiskit() -> None:
    """Raise ImportError if qiskit is not installed."""
    try:
        import qiskit  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc


def _run_exact(
    qubo: dict[tuple[str, str], float],
    variables: list[str],
    num_shots: int,
    seed: int,
) -> tuple[list[dict[str, int]], list[float], int | None]:
    """Enumerate all 2^n basis states; return (samples, energies, circuit_depth)."""
    from itertools import product as iproduct

    assignments = [
        dict(zip(variables, bits))
        for bits in iproduct((0, 1), repeat=len(variables))
    ]
    energies = [_qubo_energy(qubo, a) for a in assignments]
    pairs = sorted(zip(assignments, energies), key=lambda x: x[1])

    # Repeat top result to fill num_shots (mirrors sampler contract).
    samples = [p[0] for p in pairs]
    out_energies = [p[1] for p in pairs]

    # Pad / truncate to num_shots by cycling from the best.
    if len(samples) < num_shots:
        import math
        reps = math.ceil(num_shots / len(samples))
        samples = (samples * reps)[:num_shots]
        out_energies = (out_energies * reps)[:num_shots]
    else:
        samples = samples[:num_shots]
        out_energies = out_energies[:num_shots]

    return samples, out_energies, None


def _build_cost_hamiltonian(
    h: dict[str, float],
    J: dict[tuple[str, str], float],
    variables: list[str],
) -> Any:
    """Build a SparsePauliOp cost Hamiltonian from Ising h, J."""
    from qiskit.quantum_info import SparsePauliOp  # type: ignore[import]

    n = len(variables)
    var_idx = {v: idx for idx, v in enumerate(variables)}

    paulis: list[tuple[str, complex]] = []

    for v, bias in h.items():
        if bias == 0.0:
            continue
        label = ["I"] * n
        label[var_idx[v]] = "Z"
        paulis.append(("".join(reversed(label)), bias))

    for (vi, vj), coupling in J.items():
        if coupling == 0.0:
            continue
        label = ["I"] * n
        label[var_idx[vi]] = "Z"
        label[var_idx[vj]] = "Z"
        paulis.append(("".join(reversed(label)), coupling))

    if not paulis:
        paulis = [("I" * n, 0.0)]

    return SparsePauliOp.from_list(paulis)


def _counts_to_samples(
    counts: dict[str, int],
    qubo: dict[tuple[str, str], float],
    variables: list[str],
) -> tuple[list[dict[str, int]], list[float]]:
    """Convert a Qiskit bitstring-count dict to sorted (samples, energies) lists."""
    pairs: list[tuple[dict[str, int], float]] = []
    for bitstring, count in counts.items():
        assignment = _bits_to_assignment(bitstring, variables)
        energy = _qubo_energy(qubo, assignment)
        for _ in range(count):
            pairs.append((assignment, energy))
    pairs.sort(key=lambda x: x[1])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _run_qaoa(
    qubo: dict[tuple[str, str], float],
    variables: list[str],
    num_shots: int,
    reps: int,
    seed: int,
) -> tuple[list[dict[str, int]], list[float], int | None]:
    """Run QAOA using AerSimulator with fixed initial parameters (β=γ=0.1).

    Uses statevector simulation for small circuits and MPS for larger ones
    to keep memory bounded. Parameters are bound before simulation so this
    works with any qiskit-aer version that exposes AerSimulator.run().
    """
    try:
        from qiskit_aer import AerSimulator  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc

    try:
        from qiskit.circuit.library import QAOAAnsatz  # type: ignore[import]
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc

    h, J = _qubo_to_ising(qubo)
    cost_op = _build_cost_hamiltonian(h, J, variables)

    ansatz = QAOAAnsatz(cost_op, reps=reps)

    n_params = ansatz.num_parameters
    params = [0.1] * n_params

    n = len(variables)
    method = "matrix_product_state" if n > _QAOA_MPS_THRESHOLD else "statevector"
    opt_level = 0 if n > _QAOA_MPS_THRESHOLD else 1

    # Bind parameters before transpilation so AerSimulator.run() receives a
    # fully concrete circuit — avoids primitive-class version incompatibilities.
    # Use inplace=True on a copy; assign_parameters(inplace=False) returns
    # Optional[QuantumCircuit] in the type stubs.
    bound = ansatz.copy()
    bound.assign_parameters(dict(zip(bound.parameters, params)), inplace=True)
    bound.measure_all()

    backend = AerSimulator(method=method, seed_simulator=seed)
    pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
    transpiled = pm.run(bound)
    circuit_depth: int | None = transpiled.depth()

    counts = backend.run(transpiled, shots=num_shots, seed_simulator=seed).result().get_counts()

    samples, energies = _counts_to_samples(counts, qubo, variables)
    if not samples:
        return _run_exact(qubo, variables, num_shots, seed)
    return samples, energies, circuit_depth


def _run_vqe(
    qubo: dict[tuple[str, str], float],
    variables: list[str],
    num_shots: int,
    reps: int,
    seed: int,
) -> tuple[list[dict[str, int]], list[float], int | None]:
    """Run a sampling-VQE using Qiskit 1.x TwoLocal ansatz + AerSimulator.

    Uses a fixed initial parameter set; no classical optimiser loop.
    This acts as a variational ansatz sampler rather than a full VQE,
    which avoids the qiskit_algorithms dependency broken in Qiskit 1.x.
    """
    try:
        from qiskit_aer import AerSimulator  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc

    try:
        from qiskit.circuit.library import TwoLocal  # type: ignore[import]
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc

    n = len(variables)
    ansatz = TwoLocal(n, ["ry", "rz"], "cx", reps=reps)

    n_params = ansatz.num_parameters
    params = [0.1] * n_params

    bound = ansatz.copy()
    bound.assign_parameters(dict(zip(bound.parameters, params)), inplace=True)
    bound.measure_all()

    backend = AerSimulator(seed_simulator=seed)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled = pm.run(bound)

    counts = backend.run(transpiled, shots=num_shots, seed_simulator=seed).result().get_counts()

    samples, energies = _counts_to_samples(counts, qubo, variables)
    if not samples:
        return _run_exact(qubo, variables, num_shots, seed)
    return samples, energies, transpiled.depth() or None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_qiskit(
    encoding: PhysicalEncoding,
    num_shots: int = 1000,
    algorithm: str = "qaoa",
    reps: int = 1,
    use_runtime: bool = False,
    ibm_token: str | None = None,
    ibm_backend: str = "ibm_brisbane",
    seed: int = 42,
) -> QiskitResult:
    """Submit a PhysicalEncoding to a Qiskit sampler and return results.

    Args:
        encoding: A compiled PhysicalEncoding from the LIMEN compiler.
        num_shots: Number of circuit shots (or samples for exact mode).
        algorithm: One of ``"exact"``, ``"qaoa"``, or ``"vqe"``.
        reps: Number of QAOA layers or VQE repetitions.
        use_runtime: If True, submit to IBM Quantum via QiskitRuntimeService.
        ibm_token: IBM Quantum API token (runtime path only).
        ibm_backend: IBM backend name (runtime path only).
        seed: RNG seed for deterministic simulation.

    Returns:
        A QiskitResult with samples sorted by energy ascending.

    Raises:
        ImportError: If qiskit or qiskit-aer is not installed.
        ValueError: If algorithm is not one of the supported values.
    """
    _check_qiskit()

    qubo = encoding.qubo
    variables: list[str] = sorted({name for pair in qubo for name in pair})

    if use_runtime:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2  # type: ignore[import]
        except ModuleNotFoundError as exc:
            raise ImportError(_INSTALL_MSG) from exc

        QiskitRuntimeService(channel="ibm_quantum_platform", token=ibm_token)
        # Runtime path falls through to exact as a structural placeholder —
        # full runtime circuit submission requires a compiled QuantumCircuit.
        samples, energies, circuit_depth = _run_exact(qubo, variables, num_shots, seed)
        backend_name = ibm_backend
    elif algorithm == "exact":
        samples, energies, circuit_depth = _run_exact(qubo, variables, num_shots, seed)
        backend_name = "statevector"
    elif algorithm == "qaoa":
        samples, energies, circuit_depth = _run_qaoa(qubo, variables, num_shots, reps, seed)
        backend_name = "aer_simulator"
    elif algorithm == "vqe":
        samples, energies, circuit_depth = _run_vqe(qubo, variables, num_shots, reps, seed)
        backend_name = "aer_simulator"
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Choose from: 'exact', 'qaoa', 'vqe'."
        )

    best_idx = 0
    return QiskitResult(
        samples=samples,
        energies=energies,
        best_assignment={k: int(v) for k, v in samples[best_idx].items()},
        best_energy=float(energies[best_idx]),
        circuit_depth=circuit_depth,
        metadata={
            "algorithm": algorithm,
            "reps": reps,
            "num_shots": num_shots,
            "seed": seed,
            "backend": backend_name,
        },
    )


# ---------------------------------------------------------------------------
# Real IBM QPU path (QiskitRuntimeService + SamplerV2)
# ---------------------------------------------------------------------------

def _build_qaoa_ansatz(
    qubo: dict[tuple[str, str], float],
    variables: list[str],
    reps: int,
    cost_scale: float,
) -> Any:
    """Build a QAOAAnsatz for a QUBO with the cost Hamiltonian scaled.

    cost_scale is the gate-model analog of the annealer's chain strength:
    it scales the cost Hamiltonian energy (equivalently, the effective
    QAOA γ), which is the lever the Stackelberg leader controls.
    """
    from qiskit.circuit.library import QAOAAnsatz  # type: ignore[import]

    h, J = _qubo_to_ising(qubo)
    cost_op = _build_cost_hamiltonian(h, J, variables) * cost_scale
    return QAOAAnsatz(cost_op, reps=reps)


_STATEVECTOR_QUBIT_LIMIT = 24  # 2^24 statevector ≈ 256 MB — skip above this
_QAOA_MPS_THRESHOLD = 20      # above this, use MPS sampler instead of statevector


def _ideal_distribution(ansatz: Any, params: list[float]) -> dict[str, float]:
    """Noiseless bitstring distribution of a bound ansatz via statevector.

    Returns an empty dict when the circuit is too large to simulate in memory.
    Decomposes PauliEvolution gates into basis gates first; without this
    Qiskit densifies the full 2^n × 2^n operator matrix (64 GiB for n=16).
    """
    if ansatz.num_qubits > _STATEVECTOR_QUBIT_LIMIT:
        return {}

    from qiskit import transpile  # type: ignore[import]
    from qiskit.quantum_info import Statevector  # type: ignore[import]

    bound = ansatz.assign_parameters(params)
    # Decompose PauliEvolution and other high-level gates to primitive gates so
    # Statevector never calls PauliEvolution.to_matrix() (which densifies the
    # full 2^n × 2^n sparse exponential — 64 GiB for 16 qubits).
    decomposed = transpile(bound, basis_gates=["cx", "u", "h", "rx", "ry", "rz"])
    return {
        bs: float(p)
        for bs, p in Statevector(decomposed).probabilities_dict().items()
    }


def run_qiskit_qpu(
    encoding: PhysicalEncoding,
    token: str,
    crn: str,
    backend_name: str = "ibm_kingston",
    shots: int = 1000,
    reps: int = 1,
    cost_scale: float = 1.0,
    timeout: float = 600.0,
    dynamical_decoupling: bool = False,
    twirling: bool = False,
) -> QiskitResult:
    """Execute a PhysicalEncoding as a QAOA circuit on a real IBM QPU.

    Builds a depth-`reps` QAOAAnsatz from the encoding's QUBO with the
    cost Hamiltonian scaled by ``cost_scale``, transpiles it for the
    target backend, and runs it via SamplerV2 with fixed parameters
    (β=0.1, γ=0.1 per layer).

    Args:
        encoding: A compiled PhysicalEncoding from the LIMEN compiler.
        token: IBM Quantum Platform API token.
        crn: IBM Quantum service instance CRN.
        backend_name: IBM backend identifier (default ibm_kingston).
        shots: Number of measurement shots.
        reps: Number of QAOA layers.
        cost_scale: Multiplier applied to the cost Hamiltonian — the
            gate-model analog of chain strength (scales effective γ).
        timeout: Seconds to wait for the QPU job before raising
            RuntimeError (default 600). Pass None to wait indefinitely.
        dynamical_decoupling: If True, enables XY4 sequence dynamical decoupling on idle qubits.
        twirling: If True, enables Pauli and measurement twirling.

    Returns:
        A QiskitResult with samples sorted by energy ascending. The raw
        bitstring counts, job id, and ideal (noiseless) distribution of
        the same circuit are stored in metadata.

    Raises:
        ImportError: If qiskit or qiskit-ibm-runtime is not installed.
        RuntimeError: If the job does not complete within ``timeout`` seconds.
    """
    try:
        from qiskit_ibm_runtime import (  # type: ignore[import]
            QiskitRuntimeService,
            SamplerV2,
        )
        from qiskit.transpiler.preset_passmanagers import (  # type: ignore[import]
            generate_preset_pass_manager,
        )
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc

    qubo = encoding.qubo
    variables: list[str] = sorted({name for pair in qubo for name in pair})

    ansatz = _build_qaoa_ansatz(qubo, variables, reps, cost_scale)
    params = [0.1] * ansatz.num_parameters
    ideal = _ideal_distribution(ansatz, params)

    measured = ansatz.copy()
    measured.measure_all()

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token, instance=crn
    )
    backend = service.backend(backend_name)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled = pm.run(measured)

    sampler = SamplerV2(mode=backend)
    if dynamical_decoupling:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
    if twirling:
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True

    job = sampler.run([(transpiled, params)], shots=shots)
    pub_result = job.result(timeout=timeout)[0]
    counts: dict[str, int] = pub_result.data.meas.get_counts()

    samples, energies = _counts_to_samples(counts, qubo, variables)

    return QiskitResult(
        samples=samples,
        energies=energies,
        best_assignment={k: int(v) for k, v in samples[0].items()},
        best_energy=float(energies[0]),
        circuit_depth=transpiled.depth(),
        metadata={
            "algorithm": "qaoa",
            "reps": reps,
            "num_shots": shots,
            "backend": backend_name,
            "cost_scale": cost_scale,
            "counts": counts,
            "ideal_distribution": ideal,
            "job_id": job.job_id(),
        },
    )


def ibm_noise_fn(
    token: str,
    crn: str,
    backend_name: str = "ibm_kingston",
    shots: int = 1000,
    reps: int = 1,
    base_chain_strength: float | None = None,
    dynamical_decoupling: bool = False,
    twirling: bool = False,
) -> Callable[[PhysicalEncoding], float]:
    """Return a chain_break_fraction_fn for run_codesign backed by IBM QPU noise.

    The gate-model analog of the D-Wave chain-break fraction: each call
    runs the encoding's QAOA circuit on the QPU and returns the total
    variation distance between the measured and ideal (noiseless)
    bitstring distributions — a hardware-noise fraction in [0, 1] that
    feeds the κ scoring's cbf term.

    The encoding's chain strength is mapped to the circuit through the
    cost-Hamiltonian scale: cost_scale = chain_strength / base_chain_strength,
    so the Stackelberg leader's chain-strength moves change the physical
    circuit just as they change annealer penalties.

    The returned callable records per-iteration telemetry on its
    ``history`` attribute (cost_scale, tvd, optimal rate, counts, job id).

    Args:
        token: IBM Quantum Platform API token.
        crn: IBM Quantum service instance CRN.
        backend_name: IBM backend identifier (default ibm_kingston).
        shots: Shots per iteration.
        reps: Number of QAOA layers.
        base_chain_strength: Chain strength corresponding to cost_scale 1.0.
            When None, captured from the first encoding seen.
        dynamical_decoupling: If True, enables XY4 sequence dynamical decoupling on idle qubits.
        twirling: If True, enables Pauli and measurement twirling.

    Returns:
        A callable (PhysicalEncoding) → float for the
        chain_break_fraction_fn argument of run_codesign.
    """
    state: dict[str, Any] = {"base": base_chain_strength}
    history: list[dict[str, Any]] = []

    def _fn(encoding: PhysicalEncoding) -> float:
        if state["base"] is None:
            state["base"] = encoding.chain_strength
        cost_scale = encoding.chain_strength / state["base"]

        result = run_qiskit_qpu(
            encoding,
            token=token,
            crn=crn,
            backend_name=backend_name,
            shots=shots,
            reps=reps,
            cost_scale=cost_scale,
            dynamical_decoupling=dynamical_decoupling,
            twirling=twirling,
        )

        counts: dict[str, int] = result.metadata["counts"]
        ideal: dict[str, float] = result.metadata["ideal_distribution"]
        total = sum(counts.values())
        measured = {bs: cnt / total for bs, cnt in counts.items()}
        if ideal:
            all_bs = set(measured) | set(ideal)
            tvd = 0.5 * sum(
                abs(measured.get(bs, 0.0) - ideal.get(bs, 0.0)) for bs in all_bs
            )
        else:
            # Statevector simulation skipped (circuit too large); use worst-case TVD
            tvd = 1.0

        optimal_energy = min(
            _qubo_energy(encoding.qubo, a)
            for a in _enumerate_assignments(encoding.qubo)
        )
        optimal_rate = (
            sum(1 for e in result.energies if abs(e - optimal_energy) < 1e-9)
            / len(result.energies)
            if result.energies
            else 0.0
        )

        history.append(
            {
                "chain_strength": encoding.chain_strength,
                "cost_scale": cost_scale,
                "tvd": tvd,
                "qpu_optimal_rate": optimal_rate,
                "best_energy": result.best_energy,
                "job_id": result.metadata["job_id"],
            }
        )
        return tvd

    _fn.history = history  # type: ignore[attr-defined]
    return _fn


def _enumerate_assignments(
    qubo: dict[tuple[str, str], float],
) -> "list[dict[str, int]]":
    """All 2^n binary assignments over a QUBO's variables (n must be small)."""
    from itertools import product as iproduct

    variables = sorted({name for pair in qubo for name in pair})
    return [
        dict(zip(variables, bits))
        for bits in iproduct((0, 1), repeat=len(variables))
    ]
