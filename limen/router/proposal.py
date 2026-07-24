# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Self-improvement loop (harness-roadmap/03, ATRIUM AGENTS.md section 5):
gates routing-policy changes behind a ledger-backed non-regression replay
before they can merge.

Scope guard (harness-roadmap/03, "out of scope"): this module gates a
named, closed set of routing-policy tunables only -- never arbitrary
code. Widening that set is a deliberate, separately proposed decision,
not something a proposal or this loop can do on its own.

The loop, per the roadmap:

    1. Ledger baseline  -- :func:`baseline_snapshot` records the fleet's
       current ledger-adjusted state before any proposal is considered.
    2. Proposal          -- :class:`Proposal`, loaded from a plain-English
       claim file (what changes, what it unlocks, what it does not).
    3. Replay + verdict  -- :func:`evaluate_proposal` replays a fixed set
       of representative routing scenarios under the current policy value
       and the proposed one, against the *same* ledger-adjusted fleet, and
       produces a :class:`Verdict`. Non-regression, not "looks better": a
       proposal that changes behavior without making cost or logical-error
       exposure worse is accepted; anything else is rejected.
    4. Ledger learns     -- accepted verdicts are appended to
       :class:`~limen.router.memory.RouterMemory`'s append-only
       ``policy_proposals`` table via :meth:`RouterMemory.record_proposal`,
       so the whole history of what was proposed and decided is witnessed
       and replayable, exactly like the certificate ledger.

fidelity_estimate / success_probability here are always the empirical
values already on the certificates the ledger was built from -- this
module never invents a predicted number (ATRIUM AGENTS.md: "fidelity_estimate
remains empirical success_probability, never a prediction").
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time
from typing import Any

import limen.router.budget_router as budget_router
from limen.router.budget_router import DEFAULT_FLEET, RouteRequest, Tier
from limen.router.memory import RouterMemory

# The closed set of routing-policy knobs this loop is allowed to gate.
# Adding to this set is itself a policy decision, not something a
# proposal makes for itself.
GATED_POLICIES: dict[str, str] = {
    "CRITICALITY_SPREAD_THRESHOLD": (
        "Tier 1 (HW_STANDARD) vs Tier 2 (HW_CERTIFIED) criticality-spread "
        "cutoff -- see budget_router.py module docstring."
    ),
}


def _flat_qubo(n: int, weight: float = 1.0) -> dict[tuple[str, str], float]:
    """A perfectly flat QUBO: every variable equally weighted (criticality
    spread ~= 1.0, i.e. Tier 1 territory)."""
    return {(f"x{i}", f"x{i}"): weight for i in range(n)}


def _skewed_qubo(n: int, dominant: float = 20.0, rest: float = 1.0) -> dict[tuple[str, str], float]:
    """One dominant variable, the rest flat: a heavy-tailed criticality
    spectrum (Tier 2 territory)."""
    q: dict[tuple[str, str], float] = {(f"x{i}", f"x{i}"): rest for i in range(n)}
    q[("x0", "x0")] = dominant
    return q


@dataclasses.dataclass(frozen=True)
class ReplayScenario:
    """One representative routing decision the loop replays under both
    the current and proposed policy value. Fixed, not proposal-specific,
    so every proposal against the same policy is judged on the same
    ground -- see :data:`DEFAULT_SCENARIOS`."""

    name: str
    qubo: dict[tuple[str, str], float]
    fidelity_target: float
    credit_budget: float


DEFAULT_SCENARIOS: tuple[ReplayScenario, ...] = (
    ReplayScenario("flat_small", _flat_qubo(8), fidelity_target=0.9, credit_budget=50.0),
    ReplayScenario("flat_large", _flat_qubo(40), fidelity_target=0.95, credit_budget=500.0),
    ReplayScenario("skewed_small", _skewed_qubo(8), fidelity_target=0.9, credit_budget=50.0),
    ReplayScenario("skewed_large", _skewed_qubo(40), fidelity_target=0.97, credit_budget=800.0),
    ReplayScenario("skewed_high_fidelity", _skewed_qubo(20), fidelity_target=0.995, credit_budget=1000.0),
)


@dataclasses.dataclass(frozen=True)
class Proposal:
    """A claim file (ATRIUM AGENTS.md section 5): what changes, what it
    unlocks, what it does not unlock -- stated before any work runs."""

    id: str
    title: str
    what_changes: str
    what_it_unlocks: str
    what_it_does_not_unlock: str
    policy_name: str
    proposed_value: float
    created_at: float = dataclasses.field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.policy_name not in GATED_POLICIES:
            raise ValueError(
                f"{self.policy_name!r} is not a gated routing-policy tunable. "
                f"Gated policies: {sorted(GATED_POLICIES)}"
            )

    @classmethod
    def from_claim_file(cls, path: pathlib.Path | str) -> "Proposal":
        doc = json.loads(pathlib.Path(path).read_text())
        return cls(
            id=doc["id"],
            title=doc["title"],
            what_changes=doc["what_changes"],
            what_it_unlocks=doc["what_it_unlocks"],
            what_it_does_not_unlock=doc["what_it_does_not_unlock"],
            policy_name=doc["policy_name"],
            proposed_value=float(doc["proposed_value"]),
            created_at=float(doc.get("created_at", time.time())),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ScenarioOutcome:
    """One scenario's outcome. ``physical_error_exposure`` is a
    deliberately coarse proxy -- ledger-adjusted ``physical_error_rate``
    times shots, counted only for Tier 2 (HW_CERTIFIED), since that is
    the only tier this repo allocates a surface-code patch budget for.
    It is NOT a decoded logical-error-rate estimate (that would require
    the distance-dependent decoder model in limen.ecc.decoder) -- it is
    the raw physical-error budget a proposal is exposing more or fewer
    shots to, which is exactly the quantity a criticality-threshold
    change trades against cost."""

    scenario_name: str
    tier: int
    backend: str
    shots: int
    estimated_cost: float
    physical_error_exposure: float


@dataclasses.dataclass(frozen=True)
class FleetMetrics:
    """Aggregate, ledger-adjusted outcome of replaying every scenario in
    :data:`DEFAULT_SCENARIOS` under one policy value."""

    outcomes: tuple[ScenarioOutcome, ...]

    @property
    def total_estimated_cost(self) -> float:
        return sum(o.estimated_cost for o in self.outcomes)

    @property
    def total_physical_error_exposure(self) -> float:
        return sum(o.physical_error_exposure for o in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_estimated_cost": self.total_estimated_cost,
            "total_physical_error_exposure": self.total_physical_error_exposure,
            "scenarios": [dataclasses.asdict(o) for o in self.outcomes],
        }


def baseline_snapshot(
    mem: RouterMemory,
    *,
    fleet: tuple[Any, ...] = DEFAULT_FLEET,
    now: float | None = None,
) -> FleetMetrics:
    """Record the ledger-adjusted fleet's current routing behavior across
    :data:`DEFAULT_SCENARIOS`, using whatever policy value is live in
    ``budget_router`` right now. This is step 1 of the loop -- called
    before a proposal is evaluated, never after."""
    return _replay(fleet, mem, now=now)


def _replay(
    fleet: tuple[Any, ...],
    mem: RouterMemory,
    *,
    now: float | None,
) -> FleetMetrics:
    adjusted_fleet = mem.apply_memory(fleet, now=now) if mem is not None else fleet
    outcomes: list[ScenarioOutcome] = []
    for scenario in DEFAULT_SCENARIOS:
        request = RouteRequest(
            qubo=scenario.qubo,
            fidelity_target=scenario.fidelity_target,
            credit_budget=scenario.credit_budget,
        )
        plan = budget_router.route(request, adjusted_fleet)
        physical_error_exposure = 0.0
        if plan.tier == Tier.HW_CERTIFIED and plan.backend.physical_error_rate is not None:
            physical_error_exposure = plan.backend.physical_error_rate * plan.shots
        outcomes.append(
            ScenarioOutcome(
                scenario_name=scenario.name,
                tier=int(plan.tier),
                backend=plan.backend.name,
                shots=plan.shots,
                estimated_cost=plan.shots * plan.backend.cost_per_shot,
                physical_error_exposure=physical_error_exposure,
            )
        )
    return FleetMetrics(outcomes=tuple(outcomes))


@dataclasses.dataclass(frozen=True)
class Verdict:
    """The CI verdict for one proposal: accepted only on non-regression
    against the ledger baseline (ATRIUM AGENTS.md section 5: "Agents
    propose; the harness disposes")."""

    proposal: Proposal
    accepted: bool
    reason: str
    baseline: FleetMetrics
    proposed: FleetMetrics
    recorded_at: float = dataclasses.field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal": self.proposal.to_dict(),
            "accepted": self.accepted,
            "reason": self.reason,
            "baseline_metrics": self.baseline.to_dict(),
            "new_metrics": self.proposed.to_dict(),
            "recorded_at": self.recorded_at,
        }


def evaluate_proposal(
    proposal: Proposal,
    mem: RouterMemory,
    *,
    fleet: tuple[Any, ...] = DEFAULT_FLEET,
    now: float | None = None,
) -> Verdict:
    """Replay :data:`DEFAULT_SCENARIOS` under the current policy value
    (baseline) and the proposal's value (proposed), against the same
    ledger-adjusted fleet, and produce a non-regression verdict.

    Accepted iff the proposed policy value does not increase total
    estimated credit cost AND does not increase total physical-error
    exposure across the fixed scenario set. Anything else -- including
    "it's cheaper but riskier" -- is rejected; a proposal that wants to
    trade one for the other needs a separate, explicit policy decision,
    not a pass through this loop.
    """
    baseline = baseline_snapshot(mem, fleet=fleet, now=now)

    original_value = getattr(budget_router, proposal.policy_name)
    setattr(budget_router, proposal.policy_name, proposal.proposed_value)
    try:
        proposed = _replay(fleet, mem, now=now)
    finally:
        setattr(budget_router, proposal.policy_name, original_value)

    cost_delta = proposed.total_estimated_cost - baseline.total_estimated_cost
    error_delta = proposed.total_physical_error_exposure - baseline.total_physical_error_exposure

    if cost_delta > 0 and error_delta > 0:
        reason = (
            f"regression: total estimated cost rose by {cost_delta:.4g} credits "
            f"AND physical-error exposure rose by {error_delta:.4g}"
        )
        accepted = False
    elif cost_delta > 0:
        reason = f"regression: total estimated cost rose by {cost_delta:.4g} credits"
        accepted = False
    elif error_delta > 0:
        reason = f"regression: physical-error exposure rose by {error_delta:.4g}"
        accepted = False
    else:
        reason = (
            f"non-regression: cost delta {cost_delta:.4g}, "
            f"physical-error exposure delta {error_delta:.4g}"
        )
        accepted = True

    return Verdict(
        proposal=proposal,
        accepted=accepted,
        reason=reason,
        baseline=baseline,
        proposed=proposed,
        recorded_at=time.time() if now is None else now,
    )
