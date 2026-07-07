"""Budget-aware fidelity-tier routing: pick a tier, backend, cutting
strategy, ECC allocation, and shot count for a QUBO before spending a credit."""

import pathlib

from limen.router.budget_router import (
    DEFAULT_FLEET,
    BackendProfile,
    RoutePlan,
    RouteRequest,
    Tier,
    route,
)
from limen.router.calibration import (
    apply_calibration,
    fetch_backend_calibration,
    scan_calibration,
)
from limen.router.history import BackendHistory, apply_history, scan_results
from limen.router.job_state import JobState, JobStatus, retry_transient


def informed_fleet(
    results_dir: pathlib.Path | str,
    fleet: tuple[BackendProfile, ...] = DEFAULT_FLEET,
) -> tuple[BackendProfile, ...]:
    """Fold both run history and calibration into a fleet in one call.

    Every caller that wants a fleet informed by past results (see
    limen.router.history) and live calibration snapshots (see
    limen.router.calibration) was hand-wiring the same two-step chain:
    ``apply_calibration(apply_history(fleet, scan_results(d)),
    scan_calibration(d))``. This is that chain, in the order history
    then calibration so a backend with both a scanned
    measured_logical_error and a scanned physical_error_rate ends up
    with both fields populated.
    """
    fleet = apply_history(fleet, scan_results(results_dir))
    fleet = apply_calibration(fleet, scan_calibration(results_dir))
    return fleet


__all__ = [
    "Tier",
    "BackendProfile",
    "RouteRequest",
    "RoutePlan",
    "DEFAULT_FLEET",
    "route",
    "BackendHistory",
    "scan_results",
    "apply_history",
    "fetch_backend_calibration",
    "scan_calibration",
    "apply_calibration",
    "informed_fleet",
    "JobState",
    "JobStatus",
    "retry_transient",
]
