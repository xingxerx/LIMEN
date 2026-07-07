# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Seed a learned cost model from finished certs in results/.

BackendProfile.cost_per_shot/avg_queue_seconds/measured_logical_error are
currently hardcoded guesses (see budget_router.DEFAULT_FLEET). This module
closes that loop: it scans results/ for any cert this codebase produces,
extracts whatever backend/shots/timing/error data each shape carries, and
folds it into an updated fleet so the next route() call is informed by
real run history instead of priors.

Recognizes two cert shapes today (more can be added as new example
scripts land; unrecognized files are skipped, not errors):

    - tsp_eil51 benchmark certs (top-level "qpu_run": backend/shots/
      elapsed_seconds) -> contributes a cost_per_shot (seconds-per-shot)
      sample.
    - router_tier2_* certs (top-level "plan"/"measured_success_deficit"/
      "timestamps") -> contributes a measured_logical_error sample, and
      a queue-seconds sample when job timestamps were captured.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import pathlib
import statistics
from typing import Any

from limen.router.budget_router import BackendProfile


@dataclasses.dataclass
class BackendHistory:
    """Accumulated history samples for one backend, before averaging."""

    name: str
    seconds_per_shot: list[float] = dataclasses.field(default_factory=list)
    queue_seconds: list[float] = dataclasses.field(default_factory=list)
    logical_errors: list[float] = dataclasses.field(default_factory=list)

    @property
    def n_certs(self) -> int:
        return len(self.seconds_per_shot) + len(self.queue_seconds) + len(
            self.logical_errors
        )


def _parse_timestamp(value: Any) -> datetime.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _queue_seconds_from_timestamps(timestamps: Any) -> float | None:
    """IBM Runtime job metrics timestamps: queued -> running gap, in seconds."""
    if not isinstance(timestamps, dict):
        return None
    created = _parse_timestamp(timestamps.get("created"))
    running = _parse_timestamp(timestamps.get("running"))
    if created is None or running is None:
        return None
    return max(0.0, (running - created).total_seconds())


def _scan_tsp_eil51(doc: dict[str, Any], history: dict[str, BackendHistory]) -> bool:
    qpu_run = doc.get("qpu_run")
    if not isinstance(qpu_run, dict):
        return False
    backend = qpu_run.get("backend")
    shots = qpu_run.get("shots")
    elapsed = qpu_run.get("elapsed_seconds")
    if not backend or not shots:
        return False
    if elapsed is not None:
        history.setdefault(backend, BackendHistory(backend)).seconds_per_shot.append(
            elapsed / shots
        )
    return True


def _scan_router_tier2(doc: dict[str, Any], history: dict[str, BackendHistory]) -> bool:
    plan = doc.get("plan")
    if not isinstance(plan, dict):
        return False
    backend_block = plan.get("backend")
    if not isinstance(backend_block, dict):
        return False
    backend = backend_block.get("name")
    if not backend:
        return False

    entry = history.setdefault(backend, BackendHistory(backend))
    measured = doc.get("measured_success_deficit")
    if measured is not None:
        entry.logical_errors.append(float(measured))
    queue_seconds = _queue_seconds_from_timestamps(doc.get("timestamps"))
    if queue_seconds is not None:
        entry.queue_seconds.append(queue_seconds)
    return True


_SCANNERS = (_scan_router_tier2, _scan_tsp_eil51)


def scan_results(results_dir: pathlib.Path | str) -> dict[str, BackendHistory]:
    """Scan every *.json in results_dir, returning per-backend history.

    Files that don't match a recognized cert shape are silently skipped
    (results/ also holds fleet_certificate.json, fetched_jobs_*.json,
    benchmark summaries, etc. that carry no per-backend timing/error
    signal this model uses).
    """
    history: dict[str, BackendHistory] = {}
    for path in sorted(pathlib.Path(results_dir).glob("*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        for scanner in _SCANNERS:
            if scanner(doc, history):
                break
    return history


def apply_history(
    fleet: tuple[BackendProfile, ...], history: dict[str, BackendHistory]
) -> tuple[BackendProfile, ...]:
    """Fold scanned history into a fleet, replacing hardcoded guesses.

    A backend absent from history is returned unchanged. cost_per_shot is
    only overwritten when at least one seconds-per-shot sample exists
    (mean of samples); avg_queue_seconds/measured_logical_error are set
    from their respective sample means, or left None if no cert supplied
    them.
    """
    updated: list[BackendProfile] = []
    for profile in fleet:
        entry = history.get(profile.name)
        if entry is None:
            updated.append(profile)
            continue
        updated.append(
            dataclasses.replace(
                profile,
                cost_per_shot=(
                    statistics.fmean(entry.seconds_per_shot)
                    if entry.seconds_per_shot
                    else profile.cost_per_shot
                ),
                avg_queue_seconds=(
                    statistics.fmean(entry.queue_seconds)
                    if entry.queue_seconds
                    else profile.avg_queue_seconds
                ),
                measured_logical_error=(
                    statistics.fmean(entry.logical_errors)
                    if entry.logical_errors
                    else profile.measured_logical_error
                ),
            )
        )
    return tuple(updated)
