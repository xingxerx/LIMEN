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
"""Fetch results for one or more already-submitted IBM Quantum jobs.

IBM Runtime jobs run on IBM's servers independent of your machine — closing
the terminal or shutting down your PC does not cancel a pending/running job,
but it does stop anything from receiving the result when it completes.
This script re-attaches to a job by id and pulls its result whenever you
happen to be online, regardless of how long ago it was submitted.

Required environment variables:
    IBM_QUANTUM_TOKEN  — IBM Quantum Platform API token
    IBM_QUANTUM_CRN    — IBM Quantum instance CRN (service instance identifier)

Usage::

    python examples/fetch_qpu_job.py <job_id> [<job_id> ...]
"""

import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass


def _physical_qubits(job) -> list[int] | None:
    """Extract the physical qubits a job's (transpiled) circuit was mapped to."""
    try:
        circuit = job.inputs["pubs"][0][0]
        virtual_bits = circuit.layout.initial_layout.get_virtual_bits()
        return sorted(
            int(physical)
            for virtual, physical in virtual_bits.items()
            if virtual._register.name == "q"
        )
    except (KeyError, IndexError, AttributeError):
        return None


def fetch_job(service, job_id: str) -> dict:
    """Fetch status, timestamps, layout, and (if completed) counts for a job id."""
    job = service.job(job_id)
    status = job.status()
    record: dict = {"job_id": job_id, "status": str(status)}

    metrics = job.metrics()
    if metrics and "timestamps" in metrics:
        record["timestamps"] = metrics["timestamps"]

    physical_qubits = _physical_qubits(job)
    if physical_qubits is not None:
        record["physical_qubits"] = physical_qubits

    if str(status) != "DONE":
        return record

    result = job.result()
    pubs = []
    for pub_result in result:
        register_name = next(iter(pub_result.data.__dict__))
        counts = getattr(pub_result.data, register_name).get_counts()
        pubs.append({"register": register_name, "counts": counts})
    record["pubs"] = pubs
    return record


def main() -> None:
    job_ids = sys.argv[1:]
    if not job_ids:
        print(
            "Usage: python examples/fetch_qpu_job.py <job_id> [<job_id> ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")
    if not token or not crn:
        print(
            "ERROR: IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN must be set.",
            file=sys.stderr,
        )
        sys.exit(1)

    from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore[import]

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token, instance=crn
    )

    records = []
    for job_id in job_ids:
        print(f"Fetching {job_id} ...")
        record = fetch_job(service, job_id)
        records.append(record)
        print(f"  status: {record['status']}")
        if "pubs" in record:
            for pub in record["pubs"]:
                top = sorted(pub["counts"].items(), key=lambda x: -x[1])[:8]
                print(f"  register {pub['register']} top counts: {top}")

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"fetched_jobs_{'_'.join(job_ids)[:60]}.json"
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
