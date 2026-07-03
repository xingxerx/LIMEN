# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Fetch results for one or more already-submitted Open Quantum jobs.

Mirrors examples/fetch_qpu_job.py's re-attach pattern but for jobs routed
through Open Quantum (IonQ, Rigetti, IQM, AQT under one credential).

Required environment variables:
    OPENQUANTUM_CLIENT_ID      — Open Quantum SDK client id
    OPENQUANTUM_CLIENT_SECRET  — Open Quantum SDK client secret

Usage::

    python examples/fetch_openquantum_job.py <job_id> [<job_id> ...]
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

from limen.backends.openquantum import _extract_counts  # noqa: E402


def fetch_job(scheduler, job_id: str) -> dict:
    """Fetch status, metadata, and (if completed) counts for a job id."""
    job = scheduler.get_job(job_id)
    record: dict = {
        "job_id": job.id,
        "status": job.status,
        "message": job.message,
        "submitted_at": job.submitted_at,
    }

    if job.status != "Completed":
        return record

    output = scheduler.download_job_output(job)
    try:
        record["counts"] = _extract_counts(output)
    except RuntimeError as exc:
        record["counts_error"] = str(exc)
        record["raw_output"] = output
    return record


def main() -> None:
    job_ids = sys.argv[1:]
    if not job_ids:
        print(
            "Usage: python examples/fetch_openquantum_job.py <job_id> [<job_id> ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    client_id = os.environ.get("OPENQUANTUM_CLIENT_ID")
    client_secret = os.environ.get("OPENQUANTUM_CLIENT_SECRET")
    if not client_id or not client_secret:
        print(
            "ERROR: OPENQUANTUM_CLIENT_ID and OPENQUANTUM_CLIENT_SECRET must be set.",
            file=sys.stderr,
        )
        sys.exit(1)

    from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
    from openquantum_sdk.clients import SchedulerClient

    auth = ClientCredentialsAuth(
        creds=ClientCredentials(client_id=client_id, client_secret=client_secret),
    )
    scheduler = SchedulerClient(auth=auth)

    records = []
    try:
        for job_id in job_ids:
            print(f"Fetching {job_id} ...")
            record = fetch_job(scheduler, job_id)
            records.append(record)
            print(f"  status: {record['status']}")
            if "counts" in record:
                top = sorted(record["counts"].items(), key=lambda x: -x[1])[:8]
                print(f"  top counts: {top}")
    finally:
        scheduler.close()

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"fetched_openquantum_jobs_{'_'.join(job_ids)[:60]}.json"
    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
