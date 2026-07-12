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
"""Atom Computing backend adapter for LIMEN, via Azure Quantum.

STATUS: DORMANT. This adapter has never been exercised against a live
Azure Quantum workspace — no Atom Computing hardware run has validated
it, and it is not part of results/fleet_certificate.json or the budget
router's DEFAULT_FLEET. The code is kept import-clean and unit-testable,
but treat it as unverified until a real target id and hardware run
confirm it.

Atom Computing's machines are *gate-model* neutral-atom devices (nuclear-spin
qubits in Yb-171 held in optical tweezers), unlike QuEra Aquila's analog
Hamiltonian mode. No LHZ parity fallback is needed: a compiled QUBO is run
as a standard QAOA circuit, built with the same ansatz machinery as the
qiskit backend, and submitted through the Azure Quantum Qiskit provider.

Access path: Azure Quantum workspace (a third credential path alongside IBM
Runtime and Open Quantum). Requires:

    pip install azure-quantum[qiskit]

and an Azure Quantum workspace (resource id + location), authenticated via
``az login`` / DefaultAzureCredential or a connection string.

NOTE: the default target id below (``atom-computing.qpu``) follows Azure's
``provider.target`` convention but has NOT been verified against the live
Azure catalog from this environment. Call :func:`list_azure_targets` with
real credentials to discover the actual target ids before hardware runs;
pass the correct id via ``target=``.

All Azure SDK imports are guarded so this module loads cleanly even when
the SDK is not installed. Import errors surface at call time with a clear
install hint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from limen.core.compiler import PhysicalEncoding

_INSTALL_MSG = (
    "The Azure Quantum SDK is required to use the Atom Computing backend. "
    "Install it with: pip install limen[azure]  "
    "(or: pip install 'azure-quantum[qiskit]')"
)

DEFAULT_ATOM_TARGET = "atom-computing.qpu"  # unverified; see module docstring
DEFAULT_SIM_TARGET = "atom-computing.sim"   # unverified


@dataclass
class AzureAtomResult:
    """The result of a gate-model run on Atom Computing via Azure Quantum.

    Attributes:
        samples: One dict per shot outcome, mapping variable name to binary
            value (0 or 1), sorted by energy ascending.
        energies: QUBO energy of each sample, same order as samples.
        best_assignment: The lowest-energy sample.
        best_energy: The energy of best_assignment.
        target: The Azure Quantum target id that was run against.
        shots: Number of shots requested.
        job_id: The Azure Quantum job id, when available.
        metadata: Diagnostic info: num_qubits, qaoa reps, workspace name.
    """

    samples: list[dict[str, int]]
    energies: list[float]
    best_assignment: dict[str, int]
    best_energy: float
    target: str
    shots: int
    job_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _check_azure() -> None:
    try:
        import azure.quantum  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc


def _workspace(
    resource_id: str | None,
    location: str | None,
    connection_string: str | None,
):
    """Build an Azure Quantum Workspace from either auth style."""
    from azure.quantum import Workspace

    if connection_string:
        return Workspace.from_connection_string(connection_string)
    if not resource_id or not location:
        raise ValueError(
            "Provide either connection_string, or both resource_id and "
            "location, for the Azure Quantum workspace."
        )
    return Workspace(resource_id=resource_id, location=location)


def list_azure_targets(
    resource_id: str | None = None,
    location: str | None = None,
    connection_string: str | None = None,
) -> list[str]:
    """Return the target ids visible in the workspace (e.g. to find the
    real Atom Computing target id). Requires valid Azure credentials."""
    _check_azure()
    ws = _workspace(resource_id, location, connection_string)
    return [t.name for t in ws.get_targets()]


def run_azure_atom(
    encoding: PhysicalEncoding,
    num_shots: int = 1000,
    reps: int = 1,
    target: str = DEFAULT_ATOM_TARGET,
    resource_id: str | None = None,
    location: str | None = None,
    connection_string: str | None = None,
    cost_scale: float = 1.0,
    params: list[float] | None = None,
    timeout_s: int = 600,
) -> AzureAtomResult:
    """Submit a PhysicalEncoding to Atom Computing via Azure Quantum.

    Builds the same QAOA ansatz as the qiskit backend and routes it through
    the Azure Quantum Qiskit provider instead of IBM Runtime.

    Args:
        encoding: A compiled PhysicalEncoding from the LIMEN compiler.
        num_shots: Number of circuit shots.
        reps: Number of QAOA layers.
        target: Azure Quantum target id. Verify with list_azure_targets().
        resource_id: Azure Quantum workspace resource id.
        location: Azure region of the workspace (e.g. "westus").
        connection_string: Alternative auth; overrides resource_id/location.
        cost_scale: Multiplier applied to the cost Hamiltonian, matching
            run_qiskit_qpu's chain-strength analog.
        params: QAOA parameter vector (gamma/beta per layer). Defaults to
            flat 0.1 per parameter when None, matching run_qiskit_qpu.
        timeout_s: Seconds to wait for the job result.

    Returns:
        An AzureAtomResult with samples sorted by energy ascending.

    Raises:
        ImportError: If azure-quantum[qiskit] or qiskit is not installed.
    """
    _check_azure()
    # Reuse the qiskit backend's machinery; imported lazily so this module
    # stays importable without qiskit installed.
    from limen.backends.qiskit_backend import (
        _build_qaoa_ansatz,
        _check_qiskit,
        _counts_to_samples,
    )

    _check_qiskit()
    try:
        from azure.quantum.qiskit import AzureQuantumProvider
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc

    qubo = encoding.qubo
    variables: list[str] = sorted({name for pair in qubo for name in pair})

    ansatz = _build_qaoa_ansatz(qubo, variables, reps, cost_scale)
    bound_params = params if params is not None else [0.1] * ansatz.num_parameters
    if len(bound_params) != ansatz.num_parameters:
        raise ValueError(
            f"params has {len(bound_params)} entries but the ansatz needs "
            f"{ansatz.num_parameters}"
        )

    measured = ansatz.assign_parameters(bound_params)
    measured.measure_all()

    ws = _workspace(resource_id, location, connection_string)
    provider = AzureQuantumProvider(workspace=ws)
    backend = provider.get_backend(target)

    from qiskit import transpile

    transpiled = transpile(measured, backend=backend)
    job = backend.run(transpiled, shots=num_shots)
    result = job.result(timeout=timeout_s) if _accepts_timeout(job) else job.result()
    counts = result.get_counts()

    samples, energies = _counts_to_samples(counts, qubo, variables)

    return AzureAtomResult(
        samples=samples,
        energies=energies,
        best_assignment=samples[0] if samples else {},
        best_energy=energies[0] if energies else float("nan"),
        target=target,
        shots=num_shots,
        job_id=getattr(job, "id", None) or getattr(job, "job_id", lambda: None)(),
        metadata={
            "num_qubits": len(variables),
            "reps": reps,
            "cost_scale": cost_scale,
            "workspace": getattr(ws, "name", None),
            "provider": "atom-computing",
            "counts": counts,
        },
    )


def _accepts_timeout(job: Any) -> bool:
    """Azure's Qiskit job wrapper has varied on timeout kwargs across
    SDK versions; probe instead of pinning."""
    import inspect

    try:
        return "timeout" in inspect.signature(job.result).parameters
    except (TypeError, ValueError):
        return False
