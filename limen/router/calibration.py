# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Seed BackendProfile.physical_error_rate from live IBM calibration data.

RouteRequest.physical_error_rate (limen.router.budget_router) defaults to
a hardcoded 1e-3 guess fed into the Tier 2 surface-code certificate. That
guess is what tests/results (see router_tier2_kingston_*.json certs) have
shown to be off by ~2 orders of magnitude from measured behavior on real
hardware — not because the certificate math is wrong, but because 1e-3
was never a measurement of anything.

This module closes that gap the same way limen.router.history closes the
cost-model gap: a live-fetch half (network call, requires credentials)
and an offline scan/apply half (no network, safe to run in tests and CI)
that folds cached calibration snapshots into a fleet.

    live query  -> fetch_backend_calibration() -> results/calibration_*.json
    offline load -> scan_calibration() -> apply_calibration() -> fleet

Scope limit: the surface-code certificate models a single scalar
independent per-qubit bit-flip probability (see limen.ecc.certificate).
Real device calibration reports many distinct error channels (per-pair
two-qubit gate error, per-qubit readout error, T1/T2 decoherence, ...).
physical_error_rate here is a single-number proxy — the mean of two-qubit
gate error, readout error, and a T1-relaxation estimate scaled by the
caller's expected circuit depth — not a faithful reproduction of the
device's full noise profile. It replaces one guess (1e-3) with a better,
measured guess; it is not itself ground truth. The T1/T2 term is
unvalidated against a measured deficit as of this writing — see
docs/architecture.md and results/ for the calibrated-vs-measured
comparison once a second hardware run lands.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import math
import pathlib
import statistics
from typing import Any

from limen.router.budget_router import BackendProfile


def _collect(fn: Any, args_list: list[tuple[Any, ...]]) -> list[float]:
    """Call ``fn(*args)`` for each entry, skipping qubits/gates the provider
    doesn't report calibration data for (observed on ibm_fez: some qubits
    raise instead of returning None for a missing T2 reading) rather than
    letting one gap abort the whole fetch.
    """
    values = []
    for args in args_list:
        try:
            values.append(fn(*args))
        except Exception:
            continue
    return values


def fetch_backend_calibration(
    service: Any, backend_name: str, *, expected_two_qubit_depth: int = 1
) -> dict[str, Any]:
    """Query live calibration data for one IBM backend.

    Requires a connected ``QiskitRuntimeService`` (network + credentials).
    Returns a JSON-serializable record; callers are expected to write it
    to ``results/calibration_<backend_name>_<timestamp>.json`` themselves
    (see examples/fetch_backend_calibration.py) so :func:`scan_calibration`
    can pick it up later without a live connection.

    Args:
        service: A connected ``QiskitRuntimeService``.
        backend_name: The backend to query.
        expected_two_qubit_depth: Number of sequential two-qubit-gate
            layers the caller expects the circuit to have (e.g. QAOA
            layers times the ECC patch's internal gate depth). Used only
            to scale the T1 decoherence estimate below; defaults to 1
            (a single layer) when the caller has no better estimate.

    Raises:
        RuntimeError: If the backend exposes no calibration properties
            (e.g. a simulator backend).
    """
    backend = service.backend(backend_name)
    props = backend.properties()
    if props is None:
        raise RuntimeError(
            f"{backend_name!r} has no calibration properties "
            "(simulators and some fake backends don't)."
        )

    two_qubit_gates = [gate for gate in props.gates if len(gate.qubits) == 2]
    gate_args = [(g.gate, g.qubits) for g in two_qubit_gates]
    two_qubit_gate_errors = _collect(props.gate_error, gate_args)
    two_qubit_gate_lengths = _collect(props.gate_length, gate_args)
    qubit_args = [(q,) for q in range(backend.num_qubits)]
    readout_errors = _collect(props.readout_error, qubit_args)
    t1_times = _collect(props.t1, qubit_args)
    t2_times = _collect(props.t2, qubit_args)

    avg_two_qubit_gate_error = (
        statistics.fmean(two_qubit_gate_errors) if two_qubit_gate_errors else None
    )
    avg_readout_error = (
        statistics.fmean(readout_errors) if readout_errors else None
    )
    avg_two_qubit_gate_length = (
        statistics.fmean(two_qubit_gate_lengths) if two_qubit_gate_lengths else None
    )
    avg_t1 = statistics.fmean(t1_times) if t1_times else None
    avg_t2 = statistics.fmean(t2_times) if t2_times else None

    # Exponential T1 relaxation over the expected circuit duration: the
    # probability a qubit has decohered by the time the circuit finishes,
    # independent of (and additional to) per-gate error above. This is
    # what's missing from the pre-existing gate+readout average — it's
    # the piece that scales with circuit depth rather than being a fixed
    # per-gate constant. Not yet validated against a measured deficit;
    # see limen/router/calibration.py module docstring.
    decoherence_prob = None
    if avg_t1 is not None and avg_two_qubit_gate_length is not None:
        circuit_duration = expected_two_qubit_depth * avg_two_qubit_gate_length
        decoherence_prob = 1.0 - math.exp(-circuit_duration / avg_t1)

    components = [
        e
        for e in (avg_two_qubit_gate_error, avg_readout_error, decoherence_prob)
        if e is not None
    ]
    if not components:
        raise RuntimeError(
            f"{backend_name!r} properties() carried no gate, readout, or "
            "T1/T2 data."
        )
    physical_error_rate = statistics.fmean(components)

    return {
        "backend": backend_name,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "avg_two_qubit_gate_error": avg_two_qubit_gate_error,
        "avg_readout_error": avg_readout_error,
        "avg_two_qubit_gate_length": avg_two_qubit_gate_length,
        "avg_t1": avg_t1,
        "avg_t2": avg_t2,
        "expected_two_qubit_depth": expected_two_qubit_depth,
        "decoherence_prob": decoherence_prob,
        "physical_error_rate": physical_error_rate,
    }


def scan_calibration(results_dir: pathlib.Path | str) -> dict[str, float]:
    """Load the most recent calibration_*.json snapshot per backend.

    Multiple snapshots for the same backend may exist (re-fetched over
    time); the one with the latest ``generated_at`` wins. Files that
    don't match the calibration record shape are silently skipped.
    """
    latest: dict[str, tuple[str, float]] = {}
    for path in sorted(pathlib.Path(results_dir).glob("calibration_*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        backend = doc.get("backend")
        rate = doc.get("physical_error_rate")
        generated_at = doc.get("generated_at")
        if not backend or rate is None or not isinstance(generated_at, str):
            continue
        current = latest.get(backend)
        if current is None or generated_at > current[0]:
            latest[backend] = (generated_at, float(rate))
    return {backend: rate for backend, (_, rate) in latest.items()}


def apply_calibration(
    fleet: tuple[BackendProfile, ...], calibration: dict[str, float]
) -> tuple[BackendProfile, ...]:
    """Fold scanned calibration into a fleet's physical_error_rate field.

    A backend absent from ``calibration`` is returned unchanged.
    """
    updated: list[BackendProfile] = []
    for profile in fleet:
        rate = calibration.get(profile.name)
        if rate is None:
            updated.append(profile)
            continue
        updated.append(dataclasses.replace(profile, physical_error_rate=rate))
    return tuple(updated)
