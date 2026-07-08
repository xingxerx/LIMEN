# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Local persistence for a long-running QPU job's lifecycle.

IBM Runtime jobs run on IBM's servers independent of any local process —
closing the terminal that submitted a job, or crashing mid-poll, does not
touch the job itself. The failure mode this module targets is losing
track of a job id (and the plan needed to certify it) when the local
process dies, not the job failing on the provider's side.

Every stage transition (SUBMITTED -> QUEUED -> RUNNING ->
DONE/ERROR/CANCELLED/TIMED_OUT) is written to a small JSON state file
next to where the eventual certificate lands in results/, so restarting
the poller just resumes from the persisted job id and submitted_at —
no manual reconstruction, and the 24h polling ceiling survives restarts
because submitted_at is anchored at first submission, not at whichever
process most recently started polling.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import sys
import time
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class JobStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"


TERMINAL_STATUSES = frozenset(
    {JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED, JobStatus.TIMED_OUT}
)


@dataclasses.dataclass
class JobState:
    """Persisted lifecycle record for one submitted QPU job."""

    job_id: str
    status: JobStatus
    plan: dict[str, Any]
    submitted_at: str
    last_polled_at: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "plan": self.plan,
            "submitted_at": self.submitted_at,
            "last_polled_at": _normalize_iso(self.last_polled_at),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JobState:
        return cls(
            job_id=d["job_id"],
            status=JobStatus(d["status"]),
            plan=d["plan"],
            submitted_at=d["submitted_at"],
            last_polled_at=d.get("last_polled_at"),
            error=d.get("error"),
        )


def now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _normalize_iso(ts: str | None) -> str | None:
    """Reformat a ``now_iso()``-style string to ISO-8601 with a ``+00:00`` offset.

    Only applied to ``last_polled_at`` — ``submitted_at`` is the 24h-polling-
    ceiling anchor and must round-trip byte-for-byte (see module docstring).
    """
    if ts is None:
        return None
    try:
        dt = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S UTC").replace(
            tzinfo=datetime.timezone.utc
        )
    except ValueError:
        # Already normalized (e.g. re-saving a state that was previously
        # loaded) -- pass through unchanged instead of raising.
        return ts
    return dt.isoformat()


def state_path(results_dir: pathlib.Path, job_id: str) -> pathlib.Path:
    return results_dir / f"router_tier2_kingston_{job_id}.state.json"


def cert_path(results_dir: pathlib.Path, job_id: str) -> pathlib.Path:
    return results_dir / f"router_tier2_kingston_{job_id}.json"


def save_state(results_dir: pathlib.Path, state: JobState) -> None:
    results_dir.mkdir(exist_ok=True)
    state_path(results_dir, state.job_id).write_text(
        json.dumps(state.to_dict(), indent=2)
    )


def load_state(results_dir: pathlib.Path, job_id: str) -> JobState | None:
    path = state_path(results_dir, job_id)
    if not path.exists():
        return None
    return JobState.from_dict(json.loads(path.read_text()))


def retry_transient(
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 5.0,
    retryable: tuple[type[BaseException], ...] | None = None,
) -> T:
    """Retry ``fn()`` on transient network errors only, with linear backoff.

    Only exception types in *retryable* are retried (defaults to
    ``requests``'s connection/timeout errors) — a real submission
    failure (bad credentials, unknown backend, calibration fault
    reported back synchronously) must surface immediately rather than
    silently burn retries or, worse, risk a duplicate submission.
    """
    if retryable is None:
        import requests

        retryable = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except retryable as exc:
            if attempt == attempts:
                raise
            delay = base_delay * attempt
            print(
                f"[limen] transient error on attempt {attempt}/{attempts}: "
                f"{exc!r}; retrying in {delay:.0f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")
