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
"""Open Quantum backend adapter for LIMEN.

Open Quantum (operated by Quantum Rings Inc., https://docs.openquantum.com/)
exposes a single credential that reaches IonQ, Rigetti, IQM, and AQT
hardware. Submission is OpenQASM-based and Qiskit-compatible: any
``qiskit.QuantumCircuit`` can be exported to OpenQASM 2.0 and submitted
as-is, with no per-vendor account or SDK needed.

This module reuses limen.backends.qiskit_backend's existing QUBO -> Ising
-> QAOAAnsatz machinery to build the circuit, exports it to OpenQASM 2.0,
and submits it via openquantum-sdk's SchedulerClient. The job-submission
and result-retrieval call shapes below (ClientCredentialsAuth,
SchedulerClient, JobSubmissionConfig, submit_job, download_job_output) are
taken from the SDK's published quickstart and job-submission docs. The
exact JSON schema of a job's output is not documented in detail there, so
_extract_counts() probes a few plausible shapes (a top-level "counts" or
"histogram" key, or the output dict itself already being a counts
mapping) and raises a clear, inspectable error if none match — see that
function's docstring before depending on a specific provider's output
format.

All openquantum_sdk imports are guarded so this module loads cleanly when
the SDK is not installed; import errors surface at call time with a clear
install hint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from limen.backends.qiskit_backend import (
    _build_cost_hamiltonian,
    _counts_to_samples,
    _qubo_to_ising,
)
from limen.core.compiler import PhysicalEncoding

_INSTALL_MSG = (
    "The Open Quantum SDK is required to use the Open Quantum backend. "
    "Install it with: pip install limen[openquantum]  "
    "(or: pip install openquantum-sdk)"
)

# A few of the backend_class_id values documented at
# https://docs.openquantum.com/ — pass any other valid id directly;
# call list_backend_classes() for the live, authoritative list.
DEFAULT_BACKEND_CLASS_ID = "ionq:forte-1"
KNOWN_BACKEND_CLASS_IDS = (
    "ionq:forte-1",
    "rigetti:cepheus-1",
    "iqm:emerald",
    "iqm:garnet",
    "aqt:ibex-q1",
)

_TERMINAL_FAILURE_STATUSES = ("Failed", "Cancelled")


@dataclass
class OpenQuantumResult:
    """The result of a QAOA run submitted through Open Quantum.

    Attributes:
        samples: All samples as variable→binary (0/1) dicts, sorted by
            energy ascending.
        energies: QUBO energy of each sample, same order as samples.
        best_assignment: Lowest-energy sample.
        best_energy: QUBO energy of best_assignment.
        circuit_depth: Transpiled circuit depth.
        job_id: The Open Quantum job id.
        backend_class_id: The backend the job ran on (e.g. "ionq:forte-1").
        metadata: reps, shots, seed, and the raw job output for debugging.
    """

    samples: list[dict[str, int]]
    energies: list[float]
    best_assignment: dict[str, int]
    best_energy: float
    circuit_depth: int | None
    job_id: str
    backend_class_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _check_sdk() -> None:
    try:
        import openquantum_sdk  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc


def _build_qasm_circuit(
    qubo: dict[tuple[str, str], float],
    variables: list[str],
    reps: int,
) -> tuple[str, int | None]:
    """Build a fixed-parameter QAOA circuit and export it to OpenQASM 2.0.

    Mirrors limen.backends.qiskit_backend._run_qaoa's circuit construction
    (β=γ=0.1 per layer, no classical optimiser loop) but exports OpenQASM
    text instead of running locally on AerSimulator.
    """
    from qiskit import transpile  # type: ignore[import]
    from qiskit.circuit.library import QAOAAnsatz  # type: ignore[import]
    from qiskit.qasm2 import dumps as qasm2_dumps  # type: ignore[import]

    h, J = _qubo_to_ising(qubo)
    cost_op = _build_cost_hamiltonian(h, J, variables)
    ansatz = QAOAAnsatz(cost_op, reps=reps)

    bound = ansatz.copy()
    bound.assign_parameters({p: 0.1 for p in bound.parameters}, inplace=True)
    bound.measure_all()

    # qelib1.inc (OpenQASM 2.0's standard gate library) covers this basis.
    transpiled = transpile(bound, basis_gates=["cx", "u", "h", "rx", "ry", "rz"])
    return qasm2_dumps(transpiled), transpiled.depth()


def _extract_counts(output: Any) -> dict[str, int]:
    """Pull a bitstring->count histogram out of a job's downloaded output.

    Open Quantum's published docs describe download_job_output() as
    "fetches ... and parses it as JSON" without specifying the histogram's
    exact key per provider. This checks the shapes that are common across
    quantum-cloud APIs (a "counts" or "histogram" key, or the output dict
    itself already being bitstring->count) and raises with the raw output
    attached in the exception message if none match, so a real run against
    an unanticipated shape fails loudly instead of silently mis-parsing.
    """
    if isinstance(output, dict):
        for key in ("counts", "histogram", "results"):
            value = output.get(key)
            if isinstance(value, dict) and value:
                return {str(k): int(v) for k, v in value.items()}
        if output and all(isinstance(v, (int, float)) for v in output.values()):
            return {str(k): int(v) for k, v in output.items()}

    raise RuntimeError(
        "Could not find a bitstring counts histogram in the Open Quantum "
        f"job output (got: {output!r}). The SDK's output schema may differ "
        "by provider; inspect the raw output and extend _extract_counts()."
    )


def list_backend_classes(client_id: str, client_secret: str, limit: int = 20) -> list[Any]:
    """List Open Quantum's available backend classes (IonQ, Rigetti, IQM, AQT, ...).

    Thin wrapper around openquantum_sdk.clients.ManagementClient, useful for
    discovering current backend_class_id values and online/accepting_jobs
    status before submitting a job.

    Raises:
        ImportError: If the Open Quantum SDK is not installed.
    """
    _check_sdk()
    from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
    from openquantum_sdk.clients import ManagementClient

    auth = ClientCredentialsAuth(
        creds=ClientCredentials(client_id=client_id, client_secret=client_secret),
    )
    management = ManagementClient(auth=auth)
    result = management.list_backend_classes(limit=limit)
    return list(result.backend_classes)


def run_openquantum(
    encoding: PhysicalEncoding,
    client_id: str,
    client_secret: str,
    backend_class_id: str = DEFAULT_BACKEND_CLASS_ID,
    shots: int = 1000,
    reps: int = 1,
    job_name: str = "limen-qaoa",
    job_timeout_seconds: float = 86400.0,
) -> OpenQuantumResult:
    """Submit a PhysicalEncoding as a QAOA circuit through Open Quantum.

    Builds a fixed-parameter QAOA circuit from the encoding's QUBO,
    exports it to OpenQASM 2.0, and submits it via the openquantum-sdk's
    SchedulerClient — routing to IonQ, Rigetti, IQM, or AQT hardware
    depending on backend_class_id, all under one credential.

    Args:
        encoding: A compiled PhysicalEncoding from the LIMEN compiler.
        client_id: Open Quantum SDK client id (from an SDK Key).
        client_secret: Open Quantum SDK client secret (from an SDK Key).
        backend_class_id: Target backend, e.g. "ionq:forte-1",
            "rigetti:cepheus-1", "iqm:emerald", "iqm:garnet",
            "aqt:ibex-q1". Call list_backend_classes() for the live list.
        shots: Number of measurement shots.
        reps: Number of QAOA layers.
        job_name: Human-readable job name shown in the Open Quantum portal.
        job_timeout_seconds: Forwarded to JobSubmissionConfig; submit_job()
            blocks until the job reaches a terminal status or this elapses.

    Returns:
        An OpenQuantumResult with samples sorted by energy ascending.

    Raises:
        ImportError: If the Open Quantum SDK is not installed.
        RuntimeError: If the job ends in a non-"Completed" status, or its
            output does not contain a recognizable counts histogram.
    """
    _check_sdk()
    from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
    from openquantum_sdk.clients import JobSubmissionConfig, SchedulerClient

    qubo = encoding.qubo
    variables: list[str] = sorted({name for pair in qubo for name in pair})
    qasm_text, circuit_depth = _build_qasm_circuit(qubo, variables, reps)

    auth = ClientCredentialsAuth(
        creds=ClientCredentials(client_id=client_id, client_secret=client_secret),
    )
    scheduler = SchedulerClient(auth=auth)
    try:
        config = JobSubmissionConfig(
            backend_class_id=backend_class_id,
            name=job_name,
            job_subcategory_id="phys:oth",
            shots=shots,
            job_timeout_seconds=job_timeout_seconds,
        )
        job = scheduler.submit_job(config, file_content=qasm_text.encode("utf-8"))
        if job.status in _TERMINAL_FAILURE_STATUSES:
            raise RuntimeError(
                f"Open Quantum job {job.id} ended with status {job.status!r} "
                f"on backend {backend_class_id!r}."
            )
        output = scheduler.download_job_output(job)
    finally:
        scheduler.close()

    counts = _extract_counts(output)
    samples, energies = _counts_to_samples(counts, qubo, variables)
    if not samples:
        raise RuntimeError(
            f"Open Quantum job {job.id} returned an empty counts histogram."
        )

    return OpenQuantumResult(
        samples=samples,
        energies=energies,
        best_assignment=samples[0],
        best_energy=energies[0],
        circuit_depth=circuit_depth,
        job_id=str(job.id),
        backend_class_id=backend_class_id,
        metadata={
            "reps": reps,
            "shots": shots,
            "raw_output": output,
        },
    )
