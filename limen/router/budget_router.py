# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Budget router: deterministic fidelity-tier planning for a QUBO run.

Given a QUBO, a fidelity target, and a credit budget, produce a
:class:`RoutePlan` that fully determines the run *before* a credit is
spent: which tier, which backend, whether circuit cutting is required,
how the ECC patch budget is allocated, and how many shots the budget
buys. The plan's ``pipeline_kwargs`` feed :func:`limen.pipeline.run_pipeline`
directly.

Tiers:

    Tier 0 (SIM)          — free, offline, exact statevector simulation.
    Tier 1 (HW_STANDARD)  — QPU shots, no error correction.
    Tier 2 (HW_CERTIFIED) — QPU shots plus a surface-code patch budget
                            allocated by criticality (limen.ecc.budget).

Routing is deterministic — a pure function of the request and the fleet.
The same QUBO with the same request always yields an identical plan;
there is no learned model or randomness here (a run-history cost model
is the planned successor, per docs/architecture.md).

The fidelity signal between Tier 1 and Tier 2 is the criticality spread
max/mean over :func:`limen.ecc.budget.rank_criticality`: a flat spectrum
means every variable is equally error-sensitive and plain QPU sampling
suffices; a heavy-tailed spectrum means a few variables dominate error
sensitivity, so protecting exactly those with surface-code patches (the
thing the ECC budget allocator does) is where the credits should go.

Credentials never appear in a RoutePlan: ``pipeline_kwargs`` carries
backend names and shot counts only, and the caller supplies
``qpu_token``/``qpu_instance``/client secrets at execution time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from limen.ecc.budget import PatchAssignment, allocate_ecc_budget, rank_criticality
from limen.frontends.pyqubo import from_qubo_dict
from limen.gates.qaoa import variable_order

# Below this fidelity target the caller has asked for so little that the
# free exact simulator answers it; no reason to spend hardware credits.
MIN_HW_FIDELITY_TARGET = 0.5

# max/mean criticality at or above which the spectrum counts as
# heavy-tailed: a few variables dominate error sensitivity and Tier 2's
# targeted surface-code patches pay off. A perfectly flat spectrum has
# spread exactly 1.0.
CRITICALITY_SPREAD_THRESHOLD = 2.0

# Fidelity target at or above which the surface code steps up from
# distance 3 to distance 5.
DISTANCE5_FIDELITY_TARGET = 0.99

MIN_SHOTS = 100
MAX_SHOTS = 100_000
DEFAULT_SIM_SHOTS = 1000

_BACKEND_KINDS = frozenset({"ibm", "openquantum", "braket", "sim"})


class Tier(IntEnum):
    """Fidelity tiers, cheapest first."""

    SIM = 0
    HW_STANDARD = 1
    HW_CERTIFIED = 2


@dataclass(frozen=True)
class BackendProfile:
    """Static routing profile for one execution backend.

    ``cost_per_shot`` is a rough credit estimate used only to convert a
    credit budget into a shot count; it is not a billing quote.
    ``validated`` mirrors results/fleet_certificate.json: True means the
    backend has returned certified jobs from this stack before, which is
    a prerequisite for Tier 2.

    ``avg_queue_seconds`` and ``measured_logical_error`` are optional
    history-derived fields (see limen.router.history): None until at
    least one finished cert for this backend has been scanned from
    results/.

    ``physical_error_rate`` is an optional calibration-derived field (see
    limen.router.calibration): None until a live calibration snapshot
    for this backend has been fetched and scanned. When set, route()
    prefers it over RouteRequest.physical_error_rate's hardcoded default
    for Tier 2 planning.
    """

    name: str
    kind: str
    max_qubits: int
    cost_per_shot: float
    validated: bool = False
    avg_queue_seconds: float | None = None
    measured_logical_error: float | None = None
    physical_error_rate: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _BACKEND_KINDS:
            raise ValueError(
                f"Unknown backend kind {self.kind!r}. Choose from: "
                + ", ".join(sorted(_BACKEND_KINDS))
            )
        if self.max_qubits <= 0:
            raise ValueError("max_qubits must be positive")
        if self.cost_per_shot < 0:
            raise ValueError("cost_per_shot must be non-negative")


# Seeded from results/fleet_certificate.json (2026-06-29 live query).
DEFAULT_FLEET: tuple[BackendProfile, ...] = (
    BackendProfile("statevector", "sim", 24, 0.0, validated=True),
    BackendProfile("aer_simulator", "sim", 30, 0.0, validated=True),
    BackendProfile("ibm_kingston", "ibm", 156, 0.002, validated=True),
    BackendProfile("ibm_fez", "ibm", 156, 0.002, validated=True),
    BackendProfile("ibm_marrakesh", "ibm", 156, 0.002, validated=True),
    BackendProfile("ionq:forte-1", "openquantum", 36, 0.01, validated=False),
)


@dataclass(frozen=True)
class RouteRequest:
    """What the caller wants: a QUBO answered at a fidelity, within a budget.

    Attributes:
        qubo: QUBO dict mapping (var, var) pairs to weights — the same
            shape :func:`limen.pipeline.run_pipeline` accepts.
        fidelity_target: Desired solution fidelity in [0, 1].
        credit_budget: Credits available for hardware shots; 0 forces
            the free simulator tier.
        force_tier: Pin the tier instead of letting the criticality
            signal choose (used by loopback tests and manual overrides).
        physical_error_rate: Fallback per-qubit bit-flip rate for the
            Tier 2 surface-code certificate, used only when the chosen
            backend carries no calibration-derived
            ``BackendProfile.physical_error_rate`` (see
            limen.router.calibration).
        offline: If True, the plan's ``pipeline_kwargs`` execute on the
            local simulator regardless of tier, while every other
            planning decision (backend choice, cutting, ECC allocation,
            shots) is still made against the hardware profile. This is
            the loopback mode: full plan, zero credits.
    """

    qubo: dict[tuple[str, str], float]
    fidelity_target: float
    credit_budget: float
    force_tier: Tier | None = None
    physical_error_rate: float = 1e-3
    offline: bool = False

    def __post_init__(self) -> None:
        if not self.qubo:
            raise ValueError("qubo must be non-empty")
        if not 0.0 <= self.fidelity_target <= 1.0:
            raise ValueError("fidelity_target must be in [0, 1]")
        if self.credit_budget < 0:
            raise ValueError("credit_budget must be non-negative")


@dataclass(frozen=True)
class RoutePlan:
    """A fully-determined execution plan for one QUBO run.

    ``pipeline_kwargs`` are exact keyword arguments for
    :func:`limen.pipeline.run_pipeline` (the QUBO itself stays on the
    request: ``run_pipeline(request.qubo, **plan.pipeline_kwargs)``).

    When ``use_cutting`` is True the problem exceeds the backend and
    must be split into ``num_partitions`` sub-circuits routed through
    limen.cutting (find_cuts_and_partition + dispatch.run_cut_circuit)
    instead of a single run_pipeline call; the plan records the decision
    but does not build the CutPlan.
    """

    tier: Tier
    backend: BackendProfile
    pipeline_kwargs: dict[str, Any]
    use_cutting: bool
    num_partitions: int | None
    ecc_distance: int | None
    physical_qubit_budget: int | None
    patch_assignments: tuple[PatchAssignment, ...]
    n_vars: int
    shots: int
    criticality_spread: float
    notes: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": int(self.tier),
            "backend": {
                "name": self.backend.name,
                "kind": self.backend.kind,
                "max_qubits": self.backend.max_qubits,
                "cost_per_shot": self.backend.cost_per_shot,
                "validated": self.backend.validated,
                "avg_queue_seconds": self.backend.avg_queue_seconds,
                "measured_logical_error": self.backend.measured_logical_error,
            },
            "pipeline_kwargs": dict(self.pipeline_kwargs),
            "use_cutting": self.use_cutting,
            "num_partitions": self.num_partitions,
            "ecc_distance": self.ecc_distance,
            "physical_qubit_budget": self.physical_qubit_budget,
            "patch_assignments": [
                {
                    "logical_var": a.logical_var,
                    "distance": a.distance,
                    "physical_start": a.physical_start,
                    "physical_end": a.physical_end,
                }
                for a in self.patch_assignments
            ],
            "n_vars": self.n_vars,
            "shots": self.shots,
            "criticality_spread": self.criticality_spread,
            "notes": list(self.notes),
        }


def _int_qubo(
    qubo: dict[tuple[str, str], float], order: list[str]
) -> list[tuple[tuple[int, int], float]]:
    """Re-key a name-keyed QUBO onto the canonical sorted-name qubit indices."""
    index = {name: i for i, name in enumerate(order)}
    return [((index[i], index[j]), w) for (i, j), w in qubo.items()]


def criticality_spread(
    qubo: list[tuple[tuple[int, int], float]], n_vars: int
) -> float:
    """max/mean criticality over rank_criticality — the Tier 1 vs 2 signal.

    1.0 means a perfectly flat spectrum (every variable equally
    error-sensitive); large values mean a few variables dominate.
    Degenerate inputs (no variables, all-zero weights) report 1.0, i.e.
    flat.
    """
    ranked = rank_criticality(qubo, n_vars)
    if not ranked:
        return 1.0
    weights = [w for _, w in ranked]
    mean = sum(weights) / len(weights)
    if mean <= 0.0:
        return 1.0
    return max(weights) / mean


def distance_for_fidelity(fidelity_target: float) -> int:
    """Surface-code distance for a fidelity target: d=3 baseline, d=5 at >=0.99."""
    return 5 if fidelity_target >= DISTANCE5_FIDELITY_TARGET else 3


def _select_tier(request: RouteRequest, spread: float) -> Tier:
    if request.force_tier is not None:
        return request.force_tier
    if request.credit_budget == 0 or request.fidelity_target < MIN_HW_FIDELITY_TARGET:
        return Tier.SIM
    if spread >= CRITICALITY_SPREAD_THRESHOLD:
        return Tier.HW_CERTIFIED
    return Tier.HW_STANDARD


def _select_backend(
    tier: Tier, n_vars: int, fleet: tuple[BackendProfile, ...]
) -> BackendProfile:
    """Deterministically pick the backend for a tier.

    Tier 0 picks the cheapest simulator that fits (largest simulator if
    none fits). Tier 1 prefers validated backends that fit without
    cutting, then lowest cost, then name. Tier 2 restricts to validated
    hardware and prefers the largest device (more ECC headroom).
    """
    if tier == Tier.SIM:
        sims = [p for p in fleet if p.kind == "sim"]
        if not sims:
            raise ValueError("fleet contains no simulator profile")
        fitting = [p for p in sims if p.max_qubits >= n_vars]
        if fitting:
            return min(fitting, key=lambda p: (p.cost_per_shot, p.max_qubits, p.name))
        return max(sims, key=lambda p: (p.max_qubits, p.name))

    hardware = [p for p in fleet if p.kind != "sim"]
    if tier == Tier.HW_CERTIFIED:
        hardware = [p for p in hardware if p.validated]
        if not hardware:
            raise ValueError("fleet contains no validated hardware for Tier 2")
        return min(
            hardware,
            key=lambda p: (p.max_qubits < n_vars, -p.max_qubits, p.cost_per_shot, p.name),
        )

    if not hardware:
        raise ValueError("fleet contains no hardware profile for Tier 1")
    return min(
        hardware,
        key=lambda p: (not p.validated, p.max_qubits < n_vars, p.cost_per_shot, p.name),
    )


def _shots_for_budget(credit_budget: float, cost_per_shot: float) -> int:
    if cost_per_shot <= 0.0:
        return DEFAULT_SIM_SHOTS
    return max(MIN_SHOTS, min(MAX_SHOTS, int(credit_budget / cost_per_shot)))


def route(
    request: RouteRequest, fleet: tuple[BackendProfile, ...] = DEFAULT_FLEET
) -> RoutePlan:
    """Plan a QUBO run: tier, backend, cutting, ECC allocation, shots.

    Pure function of (request, fleet) — identical inputs always produce
    an identical plan.

    Args:
        request: The QUBO plus fidelity target, budget, and overrides.
        fleet: Backend profiles to route across; defaults to
            :data:`DEFAULT_FLEET`.

    Returns:
        A RoutePlan whose ``pipeline_kwargs`` feed run_pipeline as
        ``run_pipeline(request.qubo, **plan.pipeline_kwargs)`` (unless
        ``use_cutting`` is set, in which case execution goes through
        limen.cutting instead).

    Raises:
        ValueError: If the fleet has no profile usable for the chosen tier.
    """
    graph = from_qubo_dict(request.qubo)
    order = variable_order(graph)
    n_vars = len(order)
    int_qubo = _int_qubo(request.qubo, order)
    spread = criticality_spread(int_qubo, n_vars)

    tier = _select_tier(request, spread)
    backend = _select_backend(tier, n_vars, fleet)
    shots = _shots_for_budget(request.credit_budget, backend.cost_per_shot)

    use_cutting = n_vars > backend.max_qubits
    num_partitions = math.ceil(n_vars / backend.max_qubits) if use_cutting else None

    notes: list[str] = [
        f"criticality spread {spread:.3f} "
        f"(threshold {CRITICALITY_SPREAD_THRESHOLD}) -> tier {int(tier)}"
        + (" (forced)" if request.force_tier is not None else ""),
        f"backend {backend.name} ({backend.kind}, {backend.max_qubits}q, "
        f"validated={backend.validated}), {shots} shots",
    ]

    ecc_distance: int | None = None
    physical_qubit_budget: int | None = None
    patches: tuple[PatchAssignment, ...] = ()
    if tier == Tier.HW_CERTIFIED:
        ecc_distance = distance_for_fidelity(request.fidelity_target)
        physical_qubit_budget = max(backend.max_qubits - n_vars, 0)
        patches = tuple(
            allocate_ecc_budget(int_qubo, n_vars, physical_qubit_budget, ecc_distance)
        )
        notes.append(
            f"ECC: distance {ecc_distance}, {physical_qubit_budget} physical qubits "
            f"budgeted, {len(patches)} of {n_vars} variables patched"
        )
    if use_cutting:
        notes.append(
            f"{n_vars} vars exceed {backend.max_qubits}q: cut into "
            f"{num_partitions} partitions via limen.cutting"
        )

    kwargs: dict[str, Any] = {}
    if request.offline or tier == Tier.SIM:
        kwargs["backend"] = "statevector"
        kwargs["qpu_backend_name"] = "aer_simulator"
        kwargs["qpu_shots"] = shots
        if request.offline and tier != Tier.SIM:
            notes.append("offline loopback: executing plan on the local simulator")
    elif backend.kind == "ibm":
        kwargs["backend"] = "qpu"
        kwargs["qpu_backend_name"] = backend.name
        kwargs["qpu_shots"] = shots
    elif backend.kind == "openquantum":
        kwargs["backend"] = "openquantum"
        kwargs["openquantum_backend_class_id"] = backend.name
        kwargs["openquantum_shots"] = shots
    else:  # "braket"
        kwargs["backend"] = "braket"
        kwargs["braket_shots"] = shots
        kwargs["braket_use_qpu"] = True

    if tier == Tier.HW_CERTIFIED:
        kwargs["encode_logical"] = True
        kwargs["distance"] = ecc_distance
        if backend.physical_error_rate is not None:
            kwargs["physical_error_rate"] = backend.physical_error_rate
            notes.append(
                f"physical_error_rate {backend.physical_error_rate:.3e} from "
                f"{backend.name} calibration (overrides the "
                f"{request.physical_error_rate:.3e} request default)"
            )
        else:
            kwargs["physical_error_rate"] = request.physical_error_rate
    else:
        kwargs["encode_logical"] = False

    return RoutePlan(
        tier=tier,
        backend=backend,
        pipeline_kwargs=kwargs,
        use_cutting=use_cutting,
        num_partitions=num_partitions,
        ecc_distance=ecc_distance,
        physical_qubit_budget=physical_qubit_budget,
        patch_assignments=patches,
        n_vars=n_vars,
        shots=shots,
        criticality_spread=spread,
        notes=tuple(notes),
    )
