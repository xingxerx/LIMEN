"""End-to-end gate-model pipeline: QUBO -> compile -> execute -> certify.

This module is the composition path the rest of LIMEN's gate-model track
was missing. It threads a single problem through every layer:

    QUBO dict
      -> LogicalGraph                      (limen.frontends)
      -> QAOA CircuitIR                    (limen.gates.qaoa)
      -> statevector execution + readout   (limen.gates.simulator)
      -> classical optimality check        (limen.validator)
      -> surface-code logical-error budget (limen.ecc)

and returns a single EndToEndCertificate that composes the optimization
result with the ECC LogicalErrorCertificate - the two certificate worlds
that previously had no shared output type.

Scope: the surface-code term certifies the logical error rate of running
the solution on distance-d-protected qubits at a given physical error
rate. It does not perform full fault-tolerant lattice-surgery compilation
of the QAOA circuit (out of scope; see limen/docs/architecture.md for how
LIMEN documents such gaps rather than half-implementing them).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from limen.core.ir import LogicalGraph
from limen.ecc.certificate import certify_logical_qubit
from limen.ecc.decoder import LookupDecoder
from limen.ecc.encoder import verify_corrects_all_weight_one
from limen.ecc.surface_code import build_surface_code
from limen.frontends.pyqubo import from_qubo_dict
from limen.gates.qaoa import bitstring_to_assignment, compile_qaoa, variable_order
from limen.gates.simulator import probabilities
from limen.validator.validator import brute_force_solve

_BACKEND_CHOICES = frozenset({"statevector", "aer", "qpu", "dwave"})


@dataclass
class EndToEndCertificate:
    """Composed result of the full QUBO -> gates -> ECC pipeline."""

    solution: dict[str, int]
    energy: float
    classical_energy: float | None
    is_optimal: bool | None
    success_probability: float
    qaoa_layers: int
    qaoa_params: dict[str, float]
    logical_error_rate: float | None
    aggregate_logical_error_rate: float | None
    physical_error_rate: float | None
    distance: int | None
    n_logical_qubits: int
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    distributed_compilation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution": dict(self.solution),
            "energy": self.energy,
            "classical_energy": self.classical_energy,
            "is_optimal": self.is_optimal,
            "success_probability": self.success_probability,
            "qaoa_layers": self.qaoa_layers,
            "qaoa_params": dict(self.qaoa_params),
            "logical_error_rate": self.logical_error_rate,
            "aggregate_logical_error_rate": self.aggregate_logical_error_rate,
            "physical_error_rate": self.physical_error_rate,
            "distance": self.distance,
            "n_logical_qubits": self.n_logical_qubits,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
            "distributed_compilation": (
                dict(self.distributed_compilation)
                if self.distributed_compilation is not None
                else None
            ),
        }


def _graph_qubo(graph: LogicalGraph) -> dict[tuple[str, str], float]:
    return {(ix.i, ix.j): ix.weight for ix in graph.interactions}


def _energy(qubo: dict[tuple[str, str], float], assignment: dict[str, int]) -> float:
    return sum(w * assignment[i] * assignment[j] for (i, j), w in qubo.items())


def _grid_search(
    graph: LogicalGraph,
    layers: int,
    grid_size: int,
    prob_fn: Any = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Search a 2D (gamma, beta) grid, shared across layers, for minimum <C>.

    Returns the best (gamma, beta) params and the resulting measurement
    distribution. A single shared angle pair per layer is a deliberately
    restricted QAOA schedule, sufficient for the small instances this
    offline pipeline targets.

    Args:
        graph: LogicalGraph to compile.
        layers: Number of QAOA cost+mixer layers.
        grid_size: Points per axis in the (gamma, beta) grid.
        prob_fn: Callable ``(CircuitIR) -> dict[str, float]`` that returns a
            qubit-0-first probability distribution.  Defaults to the
            pure-Python :func:`~limen.gates.simulator.probabilities`.
    """
    if prob_fn is None:
        prob_fn = probabilities

    order = variable_order(graph)
    qubo = _graph_qubo(graph)
    energies: dict[str, float] = {}

    def outcome_energy(bits: str) -> float:
        cached = energies.get(bits)
        if cached is None:
            cached = _energy(qubo, bitstring_to_assignment(bits, order))
            energies[bits] = cached
        return cached

    best_params = {"gamma": 0.0, "beta": 0.0}
    best_dist: dict[str, float] = {}
    best_expected = math.inf
    for gi in range(grid_size):
        gamma = 2.0 * math.pi * gi / grid_size
        for bi in range(grid_size):
            beta = math.pi * bi / grid_size
            circuit = compile_qaoa(graph, [gamma] * layers, [beta] * layers)
            dist = prob_fn(circuit)
            expected = sum(p * outcome_energy(bits) for bits, p in dist.items())
            if expected < best_expected - 1e-12:
                best_expected = expected
                best_params = {"gamma": gamma, "beta": beta}
                best_dist = dist
    return best_params, best_dist


def _aer_probabilities(
    circuit: Any,
    backend_name: str,
    shots: int,
) -> dict[str, float]:
    """Execute *circuit* on a local Aer simulator; return qubit-0-first probs.

    Qiskit's ``get_counts`` returns bitstrings with qubit 0 *rightmost*;
    we reverse them to match the simulator's qubit-0-first convention so
    that :func:`bitstring_to_assignment` works identically on both paths.

    Raises:
        ImportError: If qiskit or qiskit-aer are not installed.
    """
    try:
        from limen.gates.qiskit_exec import run_circuit
    except ImportError as exc:
        raise ImportError(
            "The 'aer' backend requires qiskit and qiskit-aer. "
            "Install with: pip install limen[ibm]"
        ) from exc

    result = run_circuit(circuit, backend_name=backend_name, shots=shots)
    total = sum(result.counts.values())
    if total == 0:
        return {}
    # Reverse: Qiskit is qubit-0 rightmost; we need qubit-0 leftmost.
    return {bits[::-1]: count / total for bits, count in result.counts.items()}


def _qpu_probabilities(
    circuit: Any,
    backend_name: str,
    shots: int,
    token: str,
    instance: str,
) -> dict[str, float]:
    """Execute *circuit* on a real IBM QPU; return qubit-0-first probs.

    Requires the ``qiskit`` and ``qiskit-ibm-runtime`` extras.

    Raises:
        ImportError: If qiskit or qiskit-ibm-runtime are not installed.
    """
    try:
        from limen.gates.qiskit_exec import to_qiskit_circuit
        from qiskit.transpiler.preset_passmanagers import (  # type: ignore[import]
            generate_preset_pass_manager,
        )
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'qpu' backend requires qiskit and qiskit-ibm-runtime. "
            "Install with: pip install limen[ibm]"
        ) from exc

    qc = to_qiskit_circuit(circuit)
    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
    )
    backend = service.backend(backend_name)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled = pm.run(qc)

    sampler = SamplerV2(mode=backend)
    job = sampler.run([transpiled], shots=shots)
    print(
        f"[limen] QPU job submitted — ID: {job.job_id()}  "
        f"backend: {backend_name}  shots: {shots}\n"
        f"[limen] Track at https://quantum.ibm.com/workloads/{job.job_id()}"
    )
    pub_result = job.result()[0]
    counts: dict[str, int] = pub_result.data.c.get_counts()

    total = sum(counts.values())
    if total == 0:
        return {}
    return {bits[::-1]: count / total for bits, count in counts.items()}


def _dwave_solve(
    graph: LogicalGraph,
    num_reads: int,
    use_qpu: bool,
    qpu_endpoint: str | None,
    qpu_token: str | None,
    seed: int,
) -> Any:
    """Compile *graph* to a PhysicalEncoding and submit it to D-Wave.

    Mirrors the lexicographic-compile pattern already used by
    :func:`_distributed_compile`, but for the single-node D-Wave path.
    Imports are deferred so the statevector/aer/qpu paths have no hard
    dependency on the Ocean SDK.

    The compiler's lexicographic embedding is 1-to-1 (one physical qubit
    per logical variable, no chains), so the returned DWaveResult's
    sample/assignment keys (physical qubit labels) are translated back to
    the original logical variable names before being handed back to the
    caller.

    Raises:
        ImportError: If the D-Wave Ocean SDK is not installed.
    """
    from limen.backends.dwave import run_dwave
    from limen.core.compiler import compile_lexicographic, default_hardware_graph

    n_vars = len(graph.variables)
    encoding = compile_lexicographic(graph, default_hardware_graph(n_vars))
    result = run_dwave(
        encoding,
        num_reads=num_reads,
        use_qpu=use_qpu,
        qpu_endpoint=qpu_endpoint,
        qpu_token=qpu_token,
        seed=seed,
    )

    # embedding: logical name -> [physical qubit label] (1-to-1).
    physical_to_logical = {
        phys[0]: logical for logical, phys in encoding.embedding.items()
    }
    result.samples = [
        {physical_to_logical[p]: v for p, v in sample.items()}
        for sample in result.samples
    ]
    result.best_assignment = {
        physical_to_logical[p]: v for p, v in result.best_assignment.items()
    }
    return result


def _distributed_compile(
    graph: LogicalGraph, server_addresses: list[str], num_partitions: int | None
) -> tuple[dict[str, Any], list[str]]:
    """Compile `graph` across peer nodes via the Coordination CompilePartition RPC.

    Partitions the graph, dispatches each partition to a peer over gRPC,
    merges the returned encodings, and verifies the merged encoding is
    energetically equivalent to a local single-shot compile. Requires the
    distributed extra (grpcio); imports are deferred so the local pipeline
    has no hard dependency on it.
    """
    from limen.core.compiler import compile_lexicographic, default_hardware_graph
    from limen.distributed.partition import (
        dispatch_partitions,
        merge_partition_results,
        partition_graph,
    )
    from limen.validator.validator import validate

    n_vars = len(graph.variables)
    k = max(1, min(num_partitions or len(server_addresses), n_vars))

    partitions = partition_graph(graph, k)
    encodings = dispatch_partitions(partitions, server_addresses)
    merged = merge_partition_results(partitions, encodings, graph)

    single_shot = compile_lexicographic(graph, default_hardware_graph(n_vars))
    merged_energy = validate(merged, runs=200).classical_energy
    single_energy = validate(single_shot, runs=200).classical_energy
    verified = (
        None
        if merged_energy is None or single_energy is None
        else abs(merged_energy - single_energy) < 1e-9
    )

    info: dict[str, Any] = {
        "num_partitions": len(partitions),
        "server_addresses": list(server_addresses),
        "n_physical_qubits": len(merged.embedding),
        "chain_strength": merged.chain_strength,
        "verified_equivalent_to_single_shot": verified,
    }
    notes = [
        f"Distributed compilation: graph split into {len(partitions)} partition(s) "
        f"dispatched to {len(server_addresses)} peer(s) via the CompilePartition RPC."
    ]
    if verified is True:
        notes.append(
            "Merged distributed encoding is energetically equivalent to a "
            "single-shot local compile."
        )
    elif verified is False:
        notes.append(
            "WARNING: merged distributed encoding does NOT match the "
            "single-shot compile energy."
        )
    return info, notes


def run_pipeline(
    qubo: dict[tuple[str, str], float],
    *,
    qaoa_layers: int = 1,
    grid_size: int = 12,
    distance: int = 3,
    physical_error_rate: float | None = None,
    encode_logical: bool = True,
    server_addresses: list[str] | None = None,
    num_partitions: int | None = None,
    seed: int = 42,
    backend: str = "statevector",
    qpu_backend_name: str = "aer_simulator",
    qpu_shots: int = 1000,
    qpu_token: str | None = None,
    qpu_instance: str | None = None,
    dwave_num_reads: int = 1000,
    dwave_use_qpu: bool = False,
    dwave_endpoint: str | None = None,
    dwave_token: str | None = None,
) -> EndToEndCertificate:
    """Run a QUBO end-to-end through the gate-model track and certify it.

    Args:
        qubo: QUBO dict mapping (var, var) pairs to weights.
        qaoa_layers: Number of QAOA cost+mixer layers (shared angles).
        grid_size: Resolution of the (gamma, beta) parameter grid.
        distance: Surface-code distance for the logical-qubit certificate.
        physical_error_rate: Per-qubit bit-flip rate for the ECC term;
            if None, the logical-qubit certificate is skipped.
        encode_logical: Whether to compute the surface-code certificate.
        server_addresses: Optional "host:port" peer addresses. If given, the
            logical graph is compiled across these peers via the Coordination
            CompilePartition RPC instead of only locally, and the merged
            encoding is recorded on the certificate. Requires the distributed
            extra (grpcio).
        num_partitions: Number of partitions to split the graph into when
            dispatching to peers; defaults to len(server_addresses).
        seed: Reserved for reproducibility; the pipeline is deterministic.
        backend: Execution backend for the QAOA circuit.  One of:

            - ``"statevector"`` *(default)* — pure-Python statevector
              simulator; no external dependencies; deterministic.
            - ``"aer"`` — Qiskit Aer simulator (shot-based); requires
              ``pip install limen[ibm]``.  Use *qpu_backend_name* to
              select the Aer method (e.g. ``"aer_simulator"``).
            - ``"qpu"`` — real IBM Quantum hardware via Qiskit Runtime;
              requires *qpu_token*, *qpu_instance*, and
              ``pip install limen[ibm]``.
            - ``"dwave"`` — direct QUBO annealing on a D-Wave sampler
              (simulated annealer by default, or a real D-Wave QPU via
              *dwave_use_qpu*); requires ``pip install limen[dwave]``.
              This path skips the QAOA grid-search and circuit execution
              entirely, since D-Wave samples the QUBO directly.

        qpu_backend_name: Backend name forwarded to Aer or IBM Runtime
            (e.g. ``"aer_simulator"``, ``"ibm_kingston"``).
        qpu_shots: Number of measurement shots when using the ``"aer"``
            or ``"qpu"`` backend.
        qpu_token: IBM Quantum Platform API token (``"qpu"`` only).
        qpu_instance: IBM Quantum CRN instance string (``"qpu"`` only).
        dwave_num_reads: Number of samples to draw (``"dwave"`` only).
        dwave_use_qpu: If True, submit to a real D-Wave QPU instead of
            the local simulated annealer (``"dwave"`` only).
        dwave_endpoint: D-Wave Leap API endpoint URL; required when
            *dwave_use_qpu* is True.
        dwave_token: D-Wave Leap API token; required when *dwave_use_qpu*
            is True.

    Returns:
        An EndToEndCertificate composing the QAOA solution with the
        surface-code logical-error budget.

    Raises:
        ValueError: If *backend* is not one of the supported choices.
        ValueError: If ``backend="qpu"`` but *qpu_token* or *qpu_instance*
            is not provided.
        ValueError: If ``backend="dwave"`` and *dwave_use_qpu* is True
            but *dwave_endpoint* or *dwave_token* is not provided.
        ImportError: If ``backend="aer"`` or ``"qpu"`` and qiskit is not
            installed, or ``backend="dwave"`` and the D-Wave Ocean SDK is
            not installed.
    """
    if backend not in _BACKEND_CHOICES:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose from: "
            + ", ".join(sorted(_BACKEND_CHOICES))
        )
    if backend == "qpu" and not (qpu_token and qpu_instance):
        raise ValueError(
            "backend='qpu' requires both qpu_token and qpu_instance."
        )
    if backend == "dwave" and dwave_use_qpu and not (dwave_endpoint and dwave_token):
        raise ValueError(
            "backend='dwave' with dwave_use_qpu=True requires both "
            "dwave_endpoint and dwave_token."
        )

    graph = from_qubo_dict(qubo)
    order = variable_order(graph)
    n = len(order)
    canonical_qubo = _graph_qubo(graph)

    dwave_result: Any = None
    params: dict[str, float]
    if backend == "dwave":
        # D-Wave anneals the QUBO directly — there is no QAOA circuit to
        # parametrise or execute, so the grid-search is skipped entirely.
        params = {}
        dwave_result = _dwave_solve(
            graph, dwave_num_reads, dwave_use_qpu, dwave_endpoint, dwave_token, seed
        )
        solution = {k: int(v) for k, v in dwave_result.best_assignment.items()}
        energy = _energy(canonical_qubo, solution)
    else:
        # Parameter optimisation always runs on the statevector simulator.
        # The grid search evaluates O(grid_size²) circuits; submitting that many
        # jobs to a QPU would be impractical and expensive.  We find the optimal
        # (gamma, beta) offline, then execute the final circuit once on the
        # chosen backend.
        params, sim_dist = _grid_search(graph, qaoa_layers, grid_size)

        # Final circuit execution — one shot on the chosen backend.
        if backend == "statevector":
            dist = sim_dist  # already computed above, no extra work
        else:
            final_circuit = compile_qaoa(
                graph,
                [params["gamma"]] * qaoa_layers,
                [params["beta"]] * qaoa_layers,
            )
            if backend == "aer":
                dist = _aer_probabilities(final_circuit, qpu_backend_name, qpu_shots)
            else:  # "qpu"
                dist = _qpu_probabilities(
                    final_circuit, qpu_backend_name, qpu_shots,
                    qpu_token, qpu_instance,  # type: ignore[arg-type]
                )
        solution_bits = max(dist, key=dist.get) if dist else "0" * n
        solution = bitstring_to_assignment(solution_bits, order)
        energy = _energy(canonical_qubo, solution)

    bf = brute_force_solve(canonical_qubo)
    classical_energy = bf[1] if bf is not None else None
    is_optimal = (
        None if classical_energy is None else abs(energy - classical_energy) < 1e-9
    )

    target = classical_energy if classical_energy is not None else energy
    if backend == "dwave":
        n_samples = len(dwave_result.samples)
        success_probability = (
            sum(
                1
                for sample, e in zip(dwave_result.samples, dwave_result.energies)
                if abs(e - target) < 1e-9
            )
            / n_samples
            if n_samples
            else 0.0
        )
    else:
        success_probability = sum(
            p
            for bits, p in dist.items()
            if abs(_energy(canonical_qubo, bitstring_to_assignment(bits, order)) - target) < 1e-9
        )

    logical_rate: float | None = None
    aggregate_rate: float | None = None
    roundtrip_corrects_all_weight1: bool | None = None
    notes: list[str] = []
    if encode_logical and physical_error_rate is not None:
        patch = build_surface_code(distance)
        decoder = LookupDecoder(patch)
        cert = certify_logical_qubit(patch, decoder, physical_error_rate)
        logical_rate = cert.logical_error_rate
        aggregate_rate = 1.0 - (1.0 - logical_rate) ** n
        # Back the analytic certificate with executed gate circuits: run
        # the syndrome-extraction loop on the simulator for every weight-1
        # X error and confirm the code corrects them all.
        roundtrip_corrects_all_weight1 = verify_corrects_all_weight_one(patch, decoder)
        notes.append(
            f"Logical-qubit budget: distance-{distance} surface code at physical "
            f"error rate {physical_error_rate} gives per-qubit logical error "
            f"{logical_rate:.3e} ({n} logical qubits)."
        )
        notes.append(
            "Gate-executed round-trip "
            + ("corrects" if roundtrip_corrects_all_weight1 else "FAILS to correct")
            + " all weight-1 X errors."
        )
    elif encode_logical:
        notes.append("ECC certificate skipped: no physical_error_rate supplied.")

    if backend == "dwave":
        if is_optimal:
            notes.append("D-Wave best sample matches the classical optimum.")
        elif is_optimal is False:
            notes.append("D-Wave best sample is sub-optimal; raise dwave_num_reads.")
    else:
        if is_optimal:
            notes.append("QAOA most-likely outcome matches the classical optimum.")
        elif is_optimal is False:
            notes.append("QAOA most-likely outcome is sub-optimal; raise qaoa_layers/grid_size.")

    distributed_compilation: dict[str, Any] | None = None
    if server_addresses:
        distributed_compilation, dist_notes = _distributed_compile(
            graph, server_addresses, num_partitions
        )
        notes.extend(dist_notes)

    metadata: dict[str, Any] = {
        "variable_order": order,
        "seed": seed,
        "roundtrip_corrects_all_weight1": roundtrip_corrects_all_weight1,
        "execution_backend": backend,
    }

    if backend == "dwave":
        notes.append(
            f"Executed on backend 'dwave' "
            f"(use_qpu={dwave_use_qpu}, num_reads={dwave_num_reads}, "
            f"chain_break_fraction={dwave_result.chain_break_fraction:.4f})."
        )
        metadata["chain_break_fraction"] = dwave_result.chain_break_fraction
        metadata["dwave_timing"] = dict(dwave_result.timing)
    elif backend != "statevector":
        notes.append(
            f"Executed on backend '{backend}' "
            f"(backend_name={qpu_backend_name!r}, shots={qpu_shots})."
        )

    return EndToEndCertificate(
        solution=solution,
        energy=energy,
        classical_energy=classical_energy,
        is_optimal=is_optimal,
        success_probability=success_probability,
        qaoa_layers=0 if backend == "dwave" else qaoa_layers,
        qaoa_params=params,
        logical_error_rate=logical_rate,
        aggregate_logical_error_rate=aggregate_rate,
        physical_error_rate=physical_error_rate,
        distance=distance if logical_rate is not None else None,
        n_logical_qubits=n,
        notes=notes,
        metadata=metadata,
        distributed_compilation=distributed_compilation,
    )
