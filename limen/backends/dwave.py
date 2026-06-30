# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""D-Wave backend adapter for LIMEN.

Converts a PhysicalEncoding into a BinaryQuadraticModel and submits it to
either a simulated annealer (default, no credentials required) or a real
D-Wave QPU (opt-in via use_qpu=True).

All D-Wave Ocean SDK imports are guarded so this module loads cleanly even
when the SDK is not installed. Import errors surface at call time with a
clear install hint.
"""

from dataclasses import dataclass, field
from typing import Any

from limen.core.compiler import PhysicalEncoding

_INSTALL_MSG = (
    "The D-Wave Ocean SDK is required to use the D-Wave backend. "
    "Install it with: pip install limen[dwave]  "
    "(or: pip install dwave-ocean-sdk)"
)


@dataclass
class DWaveResult:
    """The result of a D-Wave sampling run.

    Attributes:
        samples: All samples returned by the sampler, each a dict mapping
            variable name to binary value (0 or 1).
        energies: Energy of each sample, in the same order as samples.
        timing: Timing and diagnostic information from sampleset.info
            (may be an empty dict when using the simulator).
        best_assignment: The lowest-energy sample as a variable→value dict.
        best_energy: The energy of best_assignment.
        chain_break_fraction: Mean chain-break fraction across all reads.
            Non-zero only when using a real QPU (0.0 for the simulator).
    """

    samples: list[dict[str, int]]
    energies: list[float]
    timing: dict[str, Any]
    best_assignment: dict[str, int]
    best_energy: float
    chain_break_fraction: float = 0.0
    embedding: dict[str, list] | None = None


def _import_bqm():
    """Return dimod.BinaryQuadraticModel, raising ImportError if absent."""
    try:
        from dimod import BinaryQuadraticModel  # type: ignore[import]
        return BinaryQuadraticModel
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc


def _import_simulator():
    """Return a SimulatedAnnealingSampler instance, trying two import paths."""
    try:
        from dwave.samplers import SimulatedAnnealingSampler  # type: ignore[import]
        return SimulatedAnnealingSampler()
    except ModuleNotFoundError:
        pass
    try:
        from neal import SimulatedAnnealingSampler  # type: ignore[import]
        return SimulatedAnnealingSampler()
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc


def _import_raw_qpu_sampler(endpoint: str | None, token: str | None):
    """Return a bare DWaveSampler (no embedding composite)."""
    try:
        from dwave.system import DWaveSampler  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc

    kwargs: dict[str, Any] = {}
    if endpoint:
        kwargs["endpoint"] = endpoint
    if token:
        kwargs["token"] = token
    return DWaveSampler(**kwargs)


def _find_pegasus_embedding(bqm: Any, raw_sampler: Any) -> "tuple[Any, dict[str, list]]":
    """Find a Pegasus minor-embedding for bqm using minorminer.

    Args:
        bqm: A dimod BinaryQuadraticModel whose variables need embedding.
        raw_sampler: A bare DWaveSampler providing the hardware edge list.

    Returns:
        (FixedEmbeddingComposite, embedding) where embedding maps each BQM
        variable to a list of physical Pegasus qubit labels.

    Raises:
        ImportError: If minorminer or dwave-system is not installed.
        RuntimeError: If no embedding can be found for the given problem size.
    """
    try:
        import minorminer  # type: ignore[import]
        from dwave.system import FixedEmbeddingComposite  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc

    source_edges = list(bqm.quadratic)
    target_edges = raw_sampler.edgelist
    embedding = minorminer.find_embedding(source_edges, target_edges)
    if not embedding:
        raise RuntimeError(
            f"minorminer could not find a Pegasus embedding for "
            f"{len(bqm.variables)} variables. "
            "Reduce the problem size or switch to the simulator."
        )
    return FixedEmbeddingComposite(raw_sampler, embedding), embedding


def pegasus_hardware_graph(m: int = 16) -> "dict[str, list[str]]":
    """Return the Pegasus-m hardware graph as an adjacency dict.

    The returned dict is suitable for passing directly to
    compile_lexicographic() so the compiler knows the actual hardware
    connectivity rather than assuming a complete graph.

    Args:
        m: Pegasus parameter. Pegasus-16 (the default) matches the
            Advantage QPU topology (~5000 qubits).

    Returns:
        Adjacency dict mapping str(qubit_label) -> [str(neighbor), ...].

    Raises:
        ImportError: If dwave-networkx is not installed.
    """
    try:
        import dwave_networkx as dnx  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "dwave-networkx is required for Pegasus graph generation. "
            "Install it with: pip install dwave-networkx"
        ) from exc

    G = dnx.pegasus_graph(m)
    return {str(node): [str(n) for n in G.neighbors(node)] for node in G.nodes()}


def run_dwave(
    encoding: PhysicalEncoding,
    num_reads: int = 1000,
    use_qpu: bool = False,
    qpu_endpoint: str | None = None,
    qpu_token: str | None = None,
    seed: int = 42,
) -> DWaveResult:
    """Submit a PhysicalEncoding to a D-Wave sampler and return results.

    Args:
        encoding: A compiled PhysicalEncoding from the LIMEN compiler.
        num_reads: Number of samples to draw.
        use_qpu: If True, submit to a real D-Wave QPU instead of the
            simulator. Requires a valid endpoint and token.
        qpu_endpoint: D-Wave Leap API endpoint URL (QPU path only).
        qpu_token: D-Wave Leap API token (QPU path only).
        seed: RNG seed for the simulator (ignored on QPU path).

    Returns:
        A DWaveResult containing all samples sorted by energy, the best
        assignment, and timing metadata.

    Raises:
        ImportError: If the D-Wave Ocean SDK is not installed.
    """
    BinaryQuadraticModel = _import_bqm()

    bqm = BinaryQuadraticModel.from_qubo(encoding.qubo)

    phys_embedding: dict[str, list] | None = None
    if use_qpu:
        raw = _import_raw_qpu_sampler(qpu_endpoint, qpu_token)
        sampler, phys_embedding = _find_pegasus_embedding(bqm, raw)
        sampleset = sampler.sample(
            bqm,
            num_reads=num_reads,
            chain_strength=encoding.chain_strength,
        )
    else:
        sampler = _import_simulator()
        sampleset = sampler.sample(bqm, num_reads=num_reads, seed=seed)

    # Collect and sort by energy ascending.
    pairs = sorted(
        zip(sampleset.samples(), sampleset.data_vectors["energy"]),
        key=lambda x: x[1],
    )
    samples = [dict(s) for s, _ in pairs]
    energies = [float(e) for _, e in pairs]

    # Chain break fraction is available from QPU samplesets only.
    cbf = 0.0
    if use_qpu:
        try:
            cbf_values = sampleset.record.chain_break_fraction
            if len(cbf_values) > 0:
                cbf = float(sum(cbf_values) / len(cbf_values))
        except AttributeError:
            pass

    return DWaveResult(
        samples=samples,
        energies=energies,
        timing=dict(sampleset.info),
        best_assignment={k: int(v) for k, v in samples[0].items()},
        best_energy=energies[0],
        chain_break_fraction=cbf,
        embedding=phys_embedding,
    )


def dwave_chain_break_fn(
    num_reads: int = 100,
    use_qpu: bool = False,
    qpu_endpoint: str | None = None,
    qpu_token: str | None = None,
) -> "Callable[[PhysicalEncoding], float]":
    """Return a chain_break_fraction_fn suitable for run_codesign.

    The returned callable submits the encoding to a D-Wave sampler each
    iteration and extracts the mean chain-break fraction. With use_qpu=False
    (the default) this always returns 0.0; set use_qpu=True when Leap
    credentials are available. This is the D-Wave analog of
    `limen.backends.qiskit_backend.ibm_noise_fn` — see
    `examples/dwave_codesign_qpu.py` for a closed-loop Stackelberg run
    against a real D-Wave QPU.

    The returned callable records per-iteration telemetry on its
    ``history`` attribute (chain_strength, chain_break_fraction,
    best_energy, sampler timing info).

    Args:
        num_reads: Samples per iteration (kept low to reduce QPU time).
        use_qpu: Route through a real D-Wave QPU instead of the simulator.
        qpu_endpoint: D-Wave Leap endpoint URL.
        qpu_token: D-Wave Leap API token.

    Returns:
        A callable (PhysicalEncoding) → float for use as the
        chain_break_fraction_fn argument of run_codesign.
    """
    history: list[dict[str, Any]] = []

    def _fn(encoding: PhysicalEncoding) -> float:
        result = run_dwave(
            encoding,
            num_reads=num_reads,
            use_qpu=use_qpu,
            qpu_endpoint=qpu_endpoint,
            qpu_token=qpu_token,
        )
        history.append(
            {
                "chain_strength": encoding.chain_strength,
                "chain_break_fraction": result.chain_break_fraction,
                "best_energy": result.best_energy,
                "timing": result.timing,
            }
        )
        return result.chain_break_fraction

    _fn.history = history  # type: ignore[attr-defined]
    return _fn
