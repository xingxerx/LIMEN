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


def _import_qpu_sampler(endpoint: str | None, token: str | None):
    """Return an EmbeddingComposite-wrapped DWaveSampler."""
    try:
        from dwave.system import DWaveSampler, EmbeddingComposite  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc

    kwargs: dict[str, Any] = {}
    if endpoint:
        kwargs["endpoint"] = endpoint
    if token:
        kwargs["token"] = token
    return EmbeddingComposite(DWaveSampler(**kwargs))


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

    if use_qpu:
        sampler = _import_qpu_sampler(qpu_endpoint, qpu_token)
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
    credentials are available.

    Args:
        num_reads: Samples per iteration (kept low to reduce QPU time).
        use_qpu: Route through a real D-Wave QPU instead of the simulator.
        qpu_endpoint: D-Wave Leap endpoint URL.
        qpu_token: D-Wave Leap API token.

    Returns:
        A callable (PhysicalEncoding) → float for use as the
        chain_break_fraction_fn argument of run_codesign.
    """
    from typing import Callable as _Callable

    def _fn(encoding: PhysicalEncoding) -> float:
        result = run_dwave(
            encoding,
            num_reads=num_reads,
            use_qpu=use_qpu,
            qpu_endpoint=qpu_endpoint,
            qpu_token=qpu_token,
        )
        return result.chain_break_fraction

    return _fn
