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

_BACKEND_CHOICES = frozenset({"statevector", "aer", "qpu", "dwave", "braket", "openquantum"})


@dataclass
class EndToEndCertificate:
    """Composed result of the full QUBO -> gates -> ECC pipeline.

    ``aggregate_logical_error_rate`` is always the surface-code model's
    own prediction, untouched by any empirical data.
    ``measured_logical_error_prior`` is an optional empirical prior from
    run history on the same backend (limen.router.history), and
    ``predicted_logical_error_bound`` is the conservative envelope
    ``max(model, prior)`` — the number "did the measured deficit land
    within prediction?" checks should compare against. The two inputs
    are deliberately never averaged: a blend would obscure that the
    model is a proxy while the prior is a measurement.
    """

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
    measured_logical_error_prior: float | None = None
    predicted_logical_error_bound: float | None = None

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
            "measured_logical_error_prior": self.measured_logical_error_prior,
            "predicted_logical_error_bound": self.predicted_logical_error_bound,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
            "distributed_compilation": (
                dict(self.distributed_compilation)
                if self.distributed_compilation is not None
                else None
            ),
        }


def _known_peers_from_env() -> list[str] | None:
    """Auto-discover gRPC peers from this node's own configuration.

    Only activates when ``LIMEN_NODE_ID`` is set — i.e. when this process
    is itself a configured LIMEN node, not an ad-hoc script — so a caller
    that never opted into distributed mode still gets purely local
    compilation, unchanged from before this fallback existed.
    """
    try:
        from limen.distributed.config import NodeConfig
    except ImportError:
        return None
    try:
        node_config = NodeConfig.from_env()
    except ValueError:
        return None
    return node_config.known_peers or None


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
            "Install with: pip install limen-compiler[ibm]"
        ) from exc

    result = run_circuit(circuit, backend_name=backend_name, shots=shots)
    total = sum(result.counts.values())
    if total == 0:
        return {}
    # Reverse: Qiskit is qubit-0 rightmost; we need qubit-0 leftmost.
    return {bits[::-1]: count / total for bits, count in result.counts.items()}


def _transpile_and_submit_qpu_job(
    circuit: Any,
    backend_name: str,
    shots: int,
    token: str,
    instance: str,
) -> Any:
    """Transpile *circuit* for *backend_name* and submit it via SamplerV2.

    Returns the (unresolved) qiskit-ibm-runtime job immediately — the
    caller decides whether to block on ``job.result()`` or persist the
    job id and return, letting a separate process poll for completion
    later (see examples/router_tier2_kingston_fetch.py).

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
            "Install with: pip install limen-compiler[ibm]"
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
    return sampler.run([transpiled], shots=shots)


def _get_counts_from_pub_result(pub_result: Any) -> dict[str, int]:
    """Pull the counts BitArray out of a SamplerV2 pub result.

    The classical register name varies with how the circuit was built
    (``c``, ``meas``, ...), so take whichever register is actually present
    instead of assuming a fixed name.
    """
    regs = [
        name
        for name in dir(pub_result.data)
        if not name.startswith("_")
        and hasattr(getattr(pub_result.data, name), "get_counts")
    ]
    if not regs:
        raise RuntimeError(f"no classical registers in result: {pub_result.data}")
    return getattr(pub_result.data, regs[0]).get_counts()


def _qpu_probabilities(
    circuit: Any,
    backend_name: str,
    shots: int,
    token: str,
    instance: str,
) -> dict[str, float]:
    """Submit *circuit* to a real IBM QPU, block for the result, and return
    qubit-0-first probs.

    Requires the ``qiskit`` and ``qiskit-ibm-runtime`` extras.

    Raises:
        ImportError: If qiskit or qiskit-ibm-runtime are not installed.
    """
    job = _transpile_and_submit_qpu_job(circuit, backend_name, shots, token, instance)
    print(
        f"[limen] QPU job submitted — ID: {job.job_id()}  "
        f"backend: {backend_name}  shots: {shots}\n"
        f"[limen] Track at https://quantum.ibm.com/workloads/{job.job_id()}"
    )
    pub_result = job.result()[0]
    counts: dict[str, int] = _get_counts_from_pub_result(pub_result)

    total = sum(counts.values())
    if total == 0:
        return {}
    return {bits[::-1]: count / total for bits, count in counts.items()}


def submit_qpu_job(
    qubo: dict[tuple[str, str], float],
    *,
    qaoa_layers: int = 1,
    grid_size: int = 12,
    qpu_backend_name: str,
    qpu_shots: int = 1000,
    qpu_token: str,
    qpu_instance: str,
) -> str:
    """Compile *qubo*'s QAOA circuit and submit it to an IBM QPU without
    waiting for a result; return the job id immediately.

    Runs the identical deterministic offline grid-search
    ``run_pipeline(backend="qpu")`` performs internally, so a later
    ``run_pipeline(qubo, ..., qpu_counts=<fetched counts>)`` call
    reproduces the same QAOA parameters without needing them persisted
    alongside the job id — decoupling "submit a job" from "wait for and
    certify its result" into two independently restartable steps.

    Raises:
        ImportError: If qiskit or qiskit-ibm-runtime are not installed.
    """
    graph = from_qubo_dict(qubo)
    params, _ = _grid_search(graph, qaoa_layers, grid_size)
    circuit = compile_qaoa(
        graph, [params["gamma"]] * qaoa_layers, [params["beta"]] * qaoa_layers
    )
    job = _transpile_and_submit_qpu_job(
        circuit, qpu_backend_name, qpu_shots, qpu_token, qpu_instance
    )
    return job.job_id()


def run_pipeline_from_plan(
    qubo: dict[tuple[str, str], float],
    plan: Any,
    *,
    cut_offline: bool = False,
    cut_token: str | None = None,
    cut_crn: str | None = None,
) -> Any:
    """Dispatch a QUBO through :func:`run_pipeline` using a router RoutePlan.

    ``plan.pipeline_kwargs`` (see limen.router.budget_router.route) are
    already exact run_pipeline keyword arguments, so this is the single
    call site every router-planned execution should go through instead
    of each caller manually unpacking the plan:
    ``run_pipeline_from_plan(qubo, plan)`` in place of
    ``run_pipeline(qubo, **plan.pipeline_kwargs)``.

    Args:
        qubo: The same QUBO dict passed to ``route()`` to produce *plan*.
        plan: A limen.router.RoutePlan.
        cut_offline: Only consulted when ``plan.use_cutting`` is True.
            True dispatches every cutting sub-experiment to a local
            AerSimulator (zero credits) instead of ``plan.backend``'s
            real hardware. See :func:`run_cut_route_request`.
        cut_token: IBM Quantum Platform API token, forwarded to
            :func:`run_cut_route_request`. Required unless
            ``cut_offline=True``.
        cut_crn: IBM Quantum CRN, forwarded to :func:`run_cut_route_request`.
            Required unless ``cut_offline=True``.

    Returns:
        An :class:`EndToEndCertificate`, or — when ``plan.use_cutting`` is
        True — a :class:`~limen.cutting.certificate.CuttingCertificate`
        from :func:`run_cut_route_request`. These are deliberately
        different shapes (see that class's docstring): circuit cutting
        reconstructs Pauli-observable expectation values, not a sampled
        solution bitstring, so a cutting-based certificate cannot claim
        brute-force-verified optimality the way EndToEndCertificate does.
    """
    if plan.use_cutting:
        return run_cut_route_request(
            qubo, plan, offline=cut_offline, token=cut_token, crn=cut_crn
        )
    return run_pipeline(qubo, **plan.pipeline_kwargs)


def run_cut_route_request(
    qubo: dict[tuple[str, str], float],
    plan: Any,
    *,
    token: str | None = None,
    crn: str | None = None,
    offline: bool = False,
    shots: int | None = None,
) -> Any:
    """Execute an oversized QUBO's RoutePlan via circuit cutting.

    The cut-circuit counterpart to :func:`run_pipeline`: compiles the QUBO
    to the same QAOA circuit run_pipeline would use, reconstructs every
    qubit's single-Z expectation value <Z_i> via
    limen.cutting (one cut + dispatch + reconstruct round trip per
    qubit — see limen.cutting.qubo_bridge module docstring for why),
    decodes a solution bitstring from those marginals by threshold
    rounding, and certifies it with the same surface-code ECC term
    run_pipeline uses (limen.ecc.certificate.certify_logical_qubit,
    reused unmodified — it is solution-agnostic).

    Args:
        qubo: The QUBO dict ``plan`` was routed for.
        plan: A limen.router.RoutePlan with ``use_cutting=True``.
        token: IBM Quantum Platform API token. Required unless *offline*.
        crn: IBM Quantum CRN. Required unless *offline*.
        offline: If True, every cutting sub-experiment runs on a local
            AerSimulator (limen.cutting.local_dispatch) instead of
            ``plan.backend``'s real hardware — zero credits, the same
            loopback semantics RouteRequest.offline uses elsewhere.
        shots: Shots per cutting sub-experiment; defaults to ``plan.shots``.

    Returns:
        A :class:`~limen.cutting.certificate.CuttingCertificate`. Its
        ``solution``/``decoded_classical_energy`` are the real, exactly
        (classically) evaluated answer; ``is_optimal`` is always None —
        see that class's docstring for why a cutting-based certificate
        cannot claim optimality the way EndToEndCertificate does.

    Raises:
        ValueError: If neither *offline* nor both *token* and *crn* are given.
        ImportError: If qiskit-addon-cutting (and, for a real dispatch,
            qiskit-ibm-runtime; for an offline run, qiskit-aer) is not
            installed.
    """
    from limen.cutting.certificate import CuttingCertificate
    from limen.cutting.qubo_bridge import (
        classical_energy,
        decode_bitstring_from_marginals,
        mean_field_expected_energy,
        qubo_ising_terms,
        reconstruct_z_marginals_via_cutting,
    )
    from limen.cutting.reconstruct import reconstruct_from_results

    if not offline and not (token and crn):
        raise ValueError(
            "run_cut_route_request targeting real hardware requires both "
            "token and crn (or pass offline=True for a local-sampler run)."
        )

    graph = from_qubo_dict(qubo)
    order = variable_order(graph)
    n = len(order)
    h, j_coeffs, constant, _ = qubo_ising_terms(qubo)

    params, _ = _grid_search(graph, 1, 8)
    circuit = compile_qaoa(graph, [params["gamma"]], [params["beta"]])

    max_subcircuit_qubits = plan.backend.max_qubits
    run_shots = shots if shots is not None else plan.shots

    if offline:
        from limen.cutting.local_dispatch import run_cut_circuit_locally

        def dispatch_fn(cut_plan: Any) -> Any:
            return run_cut_circuit_locally(cut_plan, shots=run_shots)
    else:
        from limen.cutting.dispatch import run_cut_circuit

        assert token is not None and crn is not None  # validated above
        real_token, real_crn = token, crn

        def dispatch_fn(cut_plan: Any) -> Any:
            return run_cut_circuit(
                cut_plan,
                real_token,
                real_crn,
                backend_name=plan.backend.name,
                shots=run_shots,
            )

    reconstruction = reconstruct_z_marginals_via_cutting(
        circuit, list(range(n)), max_subcircuit_qubits, dispatch_fn, reconstruct_from_results
    )

    solution = decode_bitstring_from_marginals(reconstruction.marginals, order)
    decoded_energy = classical_energy(qubo, solution)
    expected_energy = mean_field_expected_energy(
        h, j_coeffs, constant, reconstruction.marginals
    )

    distance = plan.pipeline_kwargs.get("distance", 3)
    physical_error_rate = plan.pipeline_kwargs.get("physical_error_rate", 1e-3)
    patch = build_surface_code(distance)
    decoder = LookupDecoder(patch)
    cert = certify_logical_qubit(patch, decoder, physical_error_rate)

    notes = [
        f"{n} vars exceeded backend capacity ({max_subcircuit_qubits}q): solved "
        f"via circuit cutting into up to {reconstruction.num_partitions} "
        f"partition(s), {reconstruction.num_cuts} cut(s).",
        "solution decoded via cutting-reconstructed single-qubit <Z_i> "
        "marginal threshold rounding, then evaluated exactly and "
        "classically (decoded_classical_energy) -- not brute-force "
        "verified optimal, hence is_optimal=None.",
        "reconstructed_expected_energy is a mean-field cross-check "
        "(<Z_i Z_j> approximated by <Z_i><Z_j>), not the true "
        "reconstructed <H>.",
    ]

    return CuttingCertificate(
        solution=solution,
        decoded_classical_energy=decoded_energy,
        reconstructed_expected_energy=expected_energy,
        is_optimal=None,
        num_partitions=reconstruction.num_partitions,
        num_cuts=reconstruction.num_cuts,
        job_ids=reconstruction.job_ids,
        logical_error_rate=cert.logical_error_rate,
        physical_error_rate=physical_error_rate,
        distance=distance,
        notes=notes,
        metadata={"n_vars": n, "max_subcircuit_qubits": max_subcircuit_qubits},
    )


def _resolve_router_memory(memory: Any, results_dir: Any) -> Any:
    """Normalize the four accepted *memory* shapes into ``None`` or a
    :class:`~limen.router.RouterMemory`.

    Kept separate from :func:`run_route_request` so each shape is
    independently testable without going through the whole routing path.
    """
    from limen.router import RouterMemory

    if memory is None or isinstance(memory, RouterMemory):
        return memory
    if memory is True:
        if results_dir is None:
            raise ValueError(
                "memory=True requires results_dir to know where to place "
                "router_memory.sqlite3."
            )
        return RouterMemory.in_results_dir(results_dir)
    return RouterMemory(memory)


def run_route_request(
    request: Any,
    *,
    results_dir: Any = None,
    fleet: Any = None,
    memory: Any = None,
    qpu_token: str | None = None,
    qpu_instance: str | None = None,
    poll_initial_seconds: float = 30.0,
    poll_backoff_cap_seconds: float = 300.0,
    poll_ceiling_seconds: float = 24 * 3600.0,
    server_addresses: list[str] | None = None,
    num_partitions: int | None = None,
) -> EndToEndCertificate:
    """QUBO + budget in, certified answer out — route and execute in one call.

    The zero-manual-steps composition of the router and the pipeline:
    builds an informed fleet from *results_dir* (run history + calibration
    snapshots, see :func:`limen.router.informed_fleet`), routes *request*
    through :func:`limen.router.route`, and executes the resulting plan.

    Simulator and synchronous-hardware plans dispatch straight through
    :func:`run_pipeline_from_plan`. IBM QPU plans go through the
    deliberately decoupled submit -> poll -> certify chain (see
    examples/router_tier2_kingston.py for why submission and waiting are
    separate concerns): the job id and lifecycle state are persisted to
    *results_dir* at every poll when one is given, so a crash or timeout
    here loses nothing — examples/router_tier2_kingston_fetch.py (or a
    rerun) can re-attach by job id and finish the certification.

    Args:
        request: A limen.router.RouteRequest.
        results_dir: Directory holding past certs and calibration
            snapshots. When given, it seeds the fleet (unless *fleet*
            overrides it) and receives QPU job-state files. When None,
            the DEFAULT_FLEET static profiles are used and no state is
            persisted.
        fleet: Explicit fleet override; skips informed_fleet entirely.
        memory: Persistent router memory (see limen.router.RouterMemory),
            folded into the informed fleet after history/calibration so
            its ledger estimates take priority where samples exist.
            Accepts four shapes: None (default) leaves today's behavior
            exactly unchanged — no ledger is opened or consulted; a
            RouterMemory instance is used as-is; a path (str or Path) to
            a SQLite file opens or creates a RouterMemory there; and True
            auto-places router_memory.sqlite3 inside results_dir (which
            must then not be None). Ignored when fleet is given, since
            that skips informed_fleet entirely.
        qpu_token: IBM Quantum Platform API token; falls back to the
            IBM_QUANTUM_TOKEN environment variable. Only consulted for
            IBM QPU plans.
        qpu_instance: IBM Quantum CRN; falls back to IBM_QUANTUM_CRN.
            Only consulted for IBM QPU plans.
        poll_initial_seconds: First poll delay for QPU jobs; doubles each
            poll up to *poll_backoff_cap_seconds*.
        poll_backoff_cap_seconds: Maximum delay between polls.
        poll_ceiling_seconds: Give up polling (TimeoutError) after this
            long; the job keeps running on IBM's side and the persisted
            job id can be re-attached later.
        server_addresses: Optional list of ``"host:port"`` gRPC peer
            addresses. When given, the LogicalGraph is compiled across
            these peers via the :class:`~limen.distributed.server.CoordinationServicer`
            ``CompilePartition`` RPC instead of only locally, and the
            merged encoding is recorded on the certificate. Requires
            the distributed extra (grpcio). When None (default), this
            node's own configuration is consulted:
            ``NodeConfig.from_env().known_peers`` (the ``LIMEN_KNOWN_PEERS``
            env var) is used automatically if ``LIMEN_NODE_ID`` is set,
            so a deployed LIMEN cluster gets distributed compilation
            without every caller re-wiring the peer list by hand. If
            ``LIMEN_NODE_ID`` is unset, compilation stays local exactly
            as before.
        num_partitions: Number of partitions to split the graph into
            when dispatching to peers; defaults to
            ``len(server_addresses)``.

    Returns:
        The EndToEndCertificate for the executed plan, or — when the plan
        targets IBM hardware and exceeds its qubit capacity
        (``plan.use_cutting``) — a
        :class:`~limen.cutting.certificate.CuttingCertificate` from
        :func:`run_cut_route_request`, a deliberately different shape
        (see that class's docstring for why).

    Raises:
        ValueError: If the plan needs IBM credentials and none are
            available (including for a cutting dispatch, which also
            targets real IBM hardware).
        RuntimeError: If the QPU job ends in ERROR or CANCELLED.
        TimeoutError: If *poll_ceiling_seconds* elapses before the QPU
            job reaches a terminal status.
    """
    import dataclasses
    import os

    from limen.router import DEFAULT_FLEET, informed_fleet, route

    if server_addresses is None:
        server_addresses = _known_peers_from_env()

    if fleet is None:
        mem = _resolve_router_memory(memory, results_dir)
        fleet = (
            informed_fleet(results_dir, memory=mem)
            if results_dir is not None
            else DEFAULT_FLEET
        )
    plan = route(request, fleet=fleet)
    if server_addresses:
        plan = dataclasses.replace(plan, server_addresses=tuple(server_addresses))

    if plan.pipeline_kwargs.get("backend") != "qpu":
        return run_pipeline(
            request.qubo,
            **plan.pipeline_kwargs,
            server_addresses=server_addresses,
            num_partitions=num_partitions,
        )

    token = qpu_token or os.environ.get("IBM_QUANTUM_TOKEN")
    instance = qpu_instance or os.environ.get("IBM_QUANTUM_CRN")
    if not (token and instance):
        raise ValueError(
            "This plan targets IBM hardware "
            f"({plan.backend.name}); pass qpu_token/qpu_instance or set "
            "IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN."
        )

    if plan.use_cutting:
        return run_cut_route_request(request.qubo, plan, token=token, crn=instance)

    job_id = submit_qpu_job(
        request.qubo,
        qpu_backend_name=plan.pipeline_kwargs["qpu_backend_name"],
        qpu_shots=plan.shots,
        qpu_token=token,
        qpu_instance=instance,
    )
    counts = _poll_qpu_counts(
        job_id,
        plan,
        token,
        instance,
        results_dir=results_dir,
        poll_initial_seconds=poll_initial_seconds,
        poll_backoff_cap_seconds=poll_backoff_cap_seconds,
        poll_ceiling_seconds=poll_ceiling_seconds,
    )
    return run_pipeline(
        request.qubo,
        **plan.pipeline_kwargs,
        qpu_counts=counts,
        server_addresses=server_addresses,
        num_partitions=num_partitions,
    )


def _poll_qpu_counts(
    job_id: str,
    plan: Any,
    token: str,
    instance: str,
    *,
    results_dir: Any,
    poll_initial_seconds: float,
    poll_backoff_cap_seconds: float,
    poll_ceiling_seconds: float,
) -> dict[str, int]:
    """Poll an IBM Runtime job to a terminal status and return its counts.

    Persists a limen.router.job_state.JobState to *results_dir* (when
    given) at submission and every poll, mirroring
    examples/router_tier2_kingston_fetch.py's crash-resilience contract:
    the job id never expires, so any non-DONE exit here can be resumed by
    a later process.
    """
    import pathlib
    import time

    from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore[import]

    from limen.router import JobState, JobStatus
    from limen.router.job_state import now_iso, save_state

    if results_dir is not None:
        results_dir = pathlib.Path(results_dir)

    terminal = {
        "DONE": JobStatus.DONE,
        "ERROR": JobStatus.ERROR,
        "CANCELLED": JobStatus.CANCELLED,
    }

    state = JobState(
        job_id=job_id,
        status=JobStatus.SUBMITTED,
        plan=plan.to_dict(),
        submitted_at=now_iso(),
    )
    if results_dir is not None:
        save_state(results_dir, state)

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token, instance=instance
    )
    job = service.job(job_id)

    start = time.monotonic()
    backoff = poll_initial_seconds
    while True:
        raw_status = str(job.status())
        mapped = terminal.get(raw_status)
        state.status = mapped if mapped is not None else JobStatus.QUEUED
        state.last_polled_at = now_iso()
        if results_dir is not None:
            save_state(results_dir, state)

        if mapped is JobStatus.DONE:
            return _get_counts_from_pub_result(job.result()[0])
        if mapped is not None:
            raise RuntimeError(
                f"QPU job {job_id} ended with status {raw_status}; not "
                "resubmitting automatically — a hardware-side failure "
                "would burn credits on a problem this pipeline can't fix. "
                "Submit a fresh request deliberately if you want to retry."
            )
        if time.monotonic() - start >= poll_ceiling_seconds:
            state.status = JobStatus.TIMED_OUT
            if results_dir is not None:
                save_state(results_dir, state)
            raise TimeoutError(
                f"QPU job {job_id} still {raw_status} after "
                f"{poll_ceiling_seconds:.0f}s; it keeps running on IBM's "
                "side — re-attach later by job id (e.g. "
                f"examples/router_tier2_kingston_fetch.py {job_id})."
            )
        time.sleep(backoff)
        backoff = min(backoff * 2, poll_backoff_cap_seconds)


def _counts_to_probabilities(counts: dict[str, int]) -> dict[str, float]:
    """Convert raw Qiskit counts (qubit-0 rightmost) to qubit-0-first probs."""
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


def _braket_solve(
    graph: LogicalGraph,
    shots: int,
    use_qpu: bool,
    device_arn: str,
) -> Any:
    """Compile *graph* to a PhysicalEncoding and submit it to QuEra Aquila.

    Mirrors the single-node D-Wave path in :func:`_dwave_solve`: a 1-to-1
    lexicographic embedding means the returned BraketResult's sample and
    best_assignment keys (physical qubit labels) are translated back to
    the original logical variable names before being handed to the caller.

    Raises:
        ImportError: If the Amazon Braket SDK is not installed.
    """
    from limen.backends.braket import run_braket
    from limen.core.compiler import compile_lexicographic, default_hardware_graph

    n_vars = len(graph.variables)
    encoding = compile_lexicographic(graph, default_hardware_graph(n_vars))
    result = run_braket(encoding, device_arn=device_arn, shots=shots, use_qpu=use_qpu)

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


def _openquantum_solve(
    graph: LogicalGraph,
    client_id: str,
    client_secret: str,
    backend_class_id: str,
    shots: int,
    reps: int,
) -> Any:
    """Compile *graph* to a PhysicalEncoding and submit it through Open Quantum.

    Mirrors the single-node D-Wave/Braket paths: a 1-to-1 lexicographic
    embedding means the returned OpenQuantumResult's sample and
    best_assignment keys (physical qubit labels) are translated back to
    the original logical variable names before being handed to the caller.

    Raises:
        ImportError: If the Open Quantum SDK is not installed.
    """
    from limen.backends.openquantum import run_openquantum
    from limen.core.compiler import compile_lexicographic, default_hardware_graph

    n_vars = len(graph.variables)
    encoding = compile_lexicographic(graph, default_hardware_graph(n_vars))
    result = run_openquantum(
        encoding,
        client_id=client_id,
        client_secret=client_secret,
        backend_class_id=backend_class_id,
        shots=shots,
        reps=reps,
    )

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
    measured_logical_error: float | None = None,
    encode_logical: bool = True,
    server_addresses: list[str] | None = None,
    num_partitions: int | None = None,
    seed: int = 42,
    backend: str = "statevector",
    qpu_backend_name: str = "aer_simulator",
    qpu_shots: int = 1000,
    qpu_token: str | None = None,
    qpu_instance: str | None = None,
    qpu_counts: dict[str, int] | None = None,
    dwave_num_reads: int = 1000,
    dwave_use_qpu: bool = False,
    dwave_endpoint: str | None = None,
    dwave_token: str | None = None,
    braket_device_arn: str = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
    braket_shots: int = 100,
    braket_use_qpu: bool = False,
    openquantum_client_id: str | None = None,
    openquantum_client_secret: str | None = None,
    openquantum_backend_class_id: str = "ionq:forte-1",
    openquantum_shots: int = 1000,
    openquantum_reps: int = 1,
) -> EndToEndCertificate:
    """Run a QUBO end-to-end through the gate-model track and certify it.

    Args:
        qubo: QUBO dict mapping (var, var) pairs to weights.
        qaoa_layers: Number of QAOA cost+mixer layers (shared angles).
        grid_size: Resolution of the (gamma, beta) parameter grid.
        distance: Surface-code distance for the logical-qubit certificate.
        physical_error_rate: Per-qubit bit-flip rate for the ECC term;
            if None, the logical-qubit certificate is skipped.
        measured_logical_error: Optional empirical logical-error prior
            for the target backend, measured from previous certified
            runs (see limen.router.history). Recorded on the certificate
            as ``measured_logical_error_prior`` and folded into
            ``predicted_logical_error_bound = max(model, prior)`` — it
            never alters ``aggregate_logical_error_rate``, which stays
            the surface-code model's own prediction.
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
              ``pip install limen-compiler[ibm]``.  Use *qpu_backend_name* to
              select the Aer method (e.g. ``"aer_simulator"``).
            - ``"qpu"`` — real IBM Quantum hardware via Qiskit Runtime;
              requires *qpu_token*, *qpu_instance*, and
              ``pip install limen-compiler[ibm]``.
            - ``"dwave"`` — direct QUBO annealing on a D-Wave sampler
              (simulated annealer by default, or a real D-Wave QPU via
              *dwave_use_qpu*); requires ``pip install limen-compiler[dwave]``.
              This path skips the QAOA grid-search and circuit execution
              entirely, since D-Wave samples the QUBO directly.
            - ``"braket"`` — analog Hamiltonian simulation on QuEra's
              Aquila neutral-atom device via Amazon Braket (its local
              simulator by default, or the real QPU via *braket_use_qpu*);
              requires ``pip install limen-compiler[braket]``. Like ``"dwave"``,
              this path skips the QAOA grid-search entirely. See
              limen/backends/braket.py for the QUBO-to-Rydberg-array
              approximation and its limitations.
            - ``"openquantum"`` — submits a fixed-parameter QAOA circuit,
              exported to OpenQASM 2.0, through Open Quantum's unified
              credential to IonQ, Rigetti, IQM, or AQT hardware (selected
              via *openquantum_backend_class_id*); requires ``pip install
              limen[openquantum]`` and *openquantum_client_id*/
              *openquantum_client_secret*. Like ``"dwave"``/``"braket"``,
              this path skips the local grid-search entirely.

        qpu_backend_name: Backend name forwarded to Aer or IBM Runtime
            (e.g. ``"aer_simulator"``, ``"ibm_kingston"``).
        qpu_shots: Number of measurement shots when using the ``"aer"``
            or ``"qpu"`` backend.
        qpu_token: IBM Quantum Platform API token (``"qpu"`` only).
        qpu_instance: IBM Quantum CRN instance string (``"qpu"`` only).
        qpu_counts: Raw Qiskit counts (qubit-0 rightmost) from a job
            already submitted and fetched separately (``"qpu"`` only).
            When given, no new job is submitted — *qpu_token*/
            *qpu_instance* are not required — and these counts are
            certified through the same grid-search/energy/ECC path a
            live QPU run would use.
        dwave_num_reads: Number of samples to draw (``"dwave"`` only).
        dwave_use_qpu: If True, submit to a real D-Wave QPU instead of
            the local simulated annealer (``"dwave"`` only).
        dwave_endpoint: D-Wave Leap API endpoint URL; required when
            *dwave_use_qpu* is True.
        dwave_token: D-Wave Leap API token; required when *dwave_use_qpu*
            is True.
        braket_device_arn: Braket device ARN to target (``"braket"``
            only); defaults to Aquila. Only used when *braket_use_qpu*
            is True — otherwise the local AHS simulator is used.
        braket_shots: Number of shots to run (``"braket"`` only).
        braket_use_qpu: If True, submit to the real AwsDevice at
            *braket_device_arn* instead of Braket's local AHS simulator
            (``"braket"`` only); requires AWS credentials with Braket
            access.
        openquantum_client_id: Open Quantum SDK client id (``"openquantum"``
            only).
        openquantum_client_secret: Open Quantum SDK client secret
            (``"openquantum"`` only).
        openquantum_backend_class_id: Target backend, e.g. "ionq:forte-1",
            "rigetti:cepheus-1", "iqm:emerald", "iqm:garnet", "aqt:ibex-q1"
            (``"openquantum"`` only).
        openquantum_shots: Number of shots (``"openquantum"`` only).
        openquantum_reps: Number of QAOA layers (``"openquantum"`` only).

    Returns:
        An EndToEndCertificate composing the QAOA solution with the
        surface-code logical-error budget.

    Raises:
        ValueError: If *backend* is not one of the supported choices.
        ValueError: If ``backend="qpu"`` but *qpu_token* or *qpu_instance*
            is not provided.
        ValueError: If ``backend="dwave"`` and *dwave_use_qpu* is True
            but *dwave_endpoint* or *dwave_token* is not provided.
        ValueError: If ``backend="openquantum"`` but
            *openquantum_client_id* or *openquantum_client_secret* is not
            provided.
        ImportError: If ``backend="aer"`` or ``"qpu"`` and qiskit is not
            installed, ``backend="dwave"`` and the D-Wave Ocean SDK is not
            installed, ``backend="braket"`` and the Amazon Braket SDK is
            not installed, or ``backend="openquantum"`` and the Open
            Quantum SDK is not installed.
    """
    if backend not in _BACKEND_CHOICES:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose from: "
            + ", ".join(sorted(_BACKEND_CHOICES))
        )
    if backend == "qpu" and qpu_counts is None and not (qpu_token and qpu_instance):
        raise ValueError(
            "backend='qpu' requires either qpu_counts, or both qpu_token "
            "and qpu_instance."
        )
    if backend == "dwave" and dwave_use_qpu and not (dwave_endpoint and dwave_token):
        raise ValueError(
            "backend='dwave' with dwave_use_qpu=True requires both "
            "dwave_endpoint and dwave_token."
        )
    if backend == "openquantum" and not (openquantum_client_id and openquantum_client_secret):
        raise ValueError(
            "backend='openquantum' requires both openquantum_client_id "
            "and openquantum_client_secret."
        )

    graph = from_qubo_dict(qubo)
    order = variable_order(graph)
    n = len(order)
    canonical_qubo = _graph_qubo(graph)

    dwave_result: Any = None
    braket_result: Any = None
    openquantum_result: Any = None
    params: dict[str, float]
    dist: dict[str, float] = {}
    if backend == "dwave":
        # D-Wave anneals the QUBO directly — there is no QAOA circuit to
        # parametrise or execute, so the grid-search is skipped entirely.
        params = {}
        dwave_result = _dwave_solve(
            graph, dwave_num_reads, dwave_use_qpu, dwave_endpoint, dwave_token, seed
        )
        solution = {k: int(v) for k, v in dwave_result.best_assignment.items()}
        energy = _energy(canonical_qubo, solution)
    elif backend == "braket":
        # Aquila is an analog device sampled directly — there is no QAOA
        # circuit to parametrise or execute, so the grid-search is skipped.
        params = {}
        braket_result = _braket_solve(graph, braket_shots, braket_use_qpu, braket_device_arn)
        solution = {k: int(v) for k, v in braket_result.best_assignment.items()}
        energy = _energy(canonical_qubo, solution)
    elif backend == "openquantum":
        # Open Quantum runs a fixed-parameter QAOA circuit on real hardware
        # directly — the local grid-search is skipped entirely.
        params = {}
        openquantum_result = _openquantum_solve(
            graph,
            openquantum_client_id,  # type: ignore[arg-type]
            openquantum_client_secret,  # type: ignore[arg-type]
            openquantum_backend_class_id,
            openquantum_shots,
            openquantum_reps,
        )
        solution = {k: int(v) for k, v in openquantum_result.best_assignment.items()}
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
            elif qpu_counts is not None:  # "qpu", already-fetched job
                dist = _counts_to_probabilities(qpu_counts)
            else:  # "qpu", submit a new job
                dist = _qpu_probabilities(
                    final_circuit, qpu_backend_name, qpu_shots,
                    qpu_token, qpu_instance,  # type: ignore[arg-type]
                )
        solution_bits = max(dist, key=lambda bits: dist[bits]) if dist else "0" * n
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
    elif backend == "braket":
        n_samples = len(braket_result.samples)
        success_probability = (
            sum(
                1
                for sample, e in zip(braket_result.samples, braket_result.energies)
                if abs(e - target) < 1e-9
            )
            / n_samples
            if n_samples
            else 0.0
        )
    elif backend == "openquantum":
        n_samples = len(openquantum_result.samples)
        success_probability = (
            sum(
                1
                for sample, e in zip(openquantum_result.samples, openquantum_result.energies)
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
    predicted_bound: float | None = None
    roundtrip_corrects_all_weight1: bool | None = None
    notes: list[str] = []
    if encode_logical and physical_error_rate is not None:
        patch = build_surface_code(distance)
        decoder = LookupDecoder(patch)
        cert = certify_logical_qubit(patch, decoder, physical_error_rate)
        logical_rate = cert.logical_error_rate
        aggregate_rate = 1.0 - (1.0 - logical_rate) ** n
        # Conservative envelope, not a blend: the model prediction stays
        # reported untouched in aggregate_logical_error_rate; the bound
        # takes whichever of (model, measured prior) is worse, so repeat
        # runs on a backend with a known deficit land within prediction
        # without pretending the model predicted it.
        predicted_bound = aggregate_rate
        if measured_logical_error is not None:
            predicted_bound = max(aggregate_rate, measured_logical_error)
            if measured_logical_error > aggregate_rate:
                notes.append(
                    f"Measured logical-error prior {measured_logical_error:.3e} "
                    f"from run history exceeds the surface-code model's "
                    f"aggregate prediction {aggregate_rate:.3e}; "
                    f"predicted_logical_error_bound uses the prior."
                )
            else:
                notes.append(
                    f"Measured logical-error prior {measured_logical_error:.3e} "
                    f"from run history is within the surface-code model's "
                    f"aggregate prediction {aggregate_rate:.3e}; "
                    f"predicted_logical_error_bound uses the model."
                )
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
    elif backend == "braket":
        if is_optimal:
            notes.append("Aquila best sample matches the classical optimum.")
        elif is_optimal is False:
            notes.append("Aquila best sample is sub-optimal; raise braket_shots.")
    elif backend == "openquantum":
        if is_optimal:
            notes.append("Open Quantum best sample matches the classical optimum.")
        elif is_optimal is False:
            notes.append(
                "Open Quantum best sample is sub-optimal; raise openquantum_shots/openquantum_reps."
            )
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
    elif backend == "braket":
        notes.append(
            f"Executed on backend 'braket' "
            f"(device_arn={braket_device_arn!r}, use_qpu={braket_use_qpu}, "
            f"shots={braket_shots}, valid_shots={len(braket_result.samples)})."
        )
        metadata["braket_metadata"] = dict(braket_result.metadata)
    elif backend == "openquantum":
        notes.append(
            f"Executed on backend 'openquantum' "
            f"(backend_class_id={openquantum_backend_class_id!r}, "
            f"shots={openquantum_shots}, job_id={openquantum_result.job_id!r})."
        )
        metadata["openquantum_job_id"] = openquantum_result.job_id
        metadata["openquantum_circuit_depth"] = openquantum_result.circuit_depth
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
        qaoa_layers=0 if backend in ("dwave", "braket", "openquantum") else qaoa_layers,
        qaoa_params=params,
        logical_error_rate=logical_rate,
        aggregate_logical_error_rate=aggregate_rate,
        physical_error_rate=physical_error_rate,
        distance=distance if logical_rate is not None else None,
        n_logical_qubits=n,
        notes=notes,
        metadata=metadata,
        distributed_compilation=distributed_compilation,
        measured_logical_error_prior=measured_logical_error,
        predicted_logical_error_bound=predicted_bound,
    )
