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
import logging
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

# Cost non-regression threshold applied after the per-scenario meet gate.
# Plain English: proposals that keep every currently-meeting scenario meeting
# may not increase total estimated cost by any amount above zero.
TOTAL_COST_INCREASE_EPSILON = 0.0


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

# Held-out scenarios: a second replay set that evaluate_proposal also gates
# on. Criticality spread is max/mean (see budget_router.criticality_spread),
# which for n variables with one dominant weight d and the rest 1.0 is
# d*n/(d+n-1). The default set certifies at spreads ~10.26 and ~13.6 and
# leaves ~5.9 in Tier 1; these sit in other bands (~6.7 Tier 1, ~11.1,
# ~15.6 and ~20.2 certified) so a proposal that only "works" by fitting the
# five visible scenarios can still fail here. Nothing that authors or tunes
# a proposal should read these; they exist only to be judged against (the
# same discipline as brain's held-out corrections).
HELDOUT_SCENARIOS: tuple[ReplayScenario, ...] = (
    ReplayScenario("heldout_flat_mid", _flat_qubo(20), fidelity_target=0.92, credit_budget=120.0),
    ReplayScenario(
        "heldout_skewed_low", _skewed_qubo(14, dominant=12.0), fidelity_target=0.96, credit_budget=300.0
    ),
    ReplayScenario(
        "heldout_certified_11", _skewed_qubo(30, dominant=17.0), fidelity_target=0.96, credit_budget=600.0
    ),
    ReplayScenario(
        "heldout_certified_15", _skewed_qubo(40, dominant=25.0), fidelity_target=0.975, credit_budget=800.0
    ),
    ReplayScenario(
        "heldout_certified_20", _skewed_qubo(30, dominant=60.0), fidelity_target=0.98, credit_budget=900.0
    ),
)

# Per-scenario cost non-regression (the Yule-Simpson guard). The aggregate
# check alone lets a large saving on one scenario hide a cost increase on
# several others, so no single scenario may cost more than it did under the
# live policy, even when the total goes down.
PER_SCENARIO_COST_INCREASE_EPSILON = 0.0
_COST_TOLERANCE = 1e-9


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
    fidelity_target: float
    tier: int
    backend: str
    shots: int
    estimated_cost: float
    physical_error_exposure: float

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "ScenarioOutcome":
        return cls(
            scenario_name=str(doc["scenario_name"]),
            fidelity_target=float(doc["fidelity_target"]),
            tier=int(doc["tier"]),
            backend=str(doc["backend"]),
            shots=int(doc["shots"]),
            estimated_cost=float(doc["estimated_cost"]),
            physical_error_exposure=float(doc["physical_error_exposure"]),
        )


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

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "FleetMetrics":
        return cls(outcomes=tuple(ScenarioOutcome.from_dict(s) for s in doc["scenarios"]))

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
    scenarios: tuple[ReplayScenario, ...] = DEFAULT_SCENARIOS,
) -> FleetMetrics:
    adjusted_fleet = mem.apply_memory(fleet, now=now) if mem is not None else fleet
    outcomes: list[ScenarioOutcome] = []
    for scenario in scenarios:
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
                fidelity_target=scenario.fidelity_target,
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
    # Value of the gated policy that was live before this proposal was evaluated.
    # Needed so a post-land rollback can restore the exact prior constant.
    previous_value: float | None = None
    # Last certificate_ledger sequence number observed at evaluation time.
    # The rollback window closes once 20 new rows are recorded after this seq.
    last_ledger_seq_at_eval: int | None = None
    recorded_at: float = dataclasses.field(default_factory=time.time)
    # Held-out replay (see HELDOUT_SCENARIOS). None on verdicts recorded
    # before the held-out gate existed.
    heldout_baseline: FleetMetrics | None = None
    heldout_proposed: FleetMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "proposal": self.proposal.to_dict(),
            "accepted": self.accepted,
            "reason": self.reason,
            "baseline_metrics": self.baseline.to_dict(),
            "new_metrics": self.proposed.to_dict(),
            "previous_value": self.previous_value,
            "last_ledger_seq_at_eval": self.last_ledger_seq_at_eval,
            "recorded_at": self.recorded_at,
        }
        if self.heldout_baseline is not None and self.heldout_proposed is not None:
            doc["heldout_baseline_metrics"] = self.heldout_baseline.to_dict()
            doc["heldout_new_metrics"] = self.heldout_proposed.to_dict()
        return doc

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "Verdict":
        """Inverse of :meth:`to_dict`, for reading a witnessed verdict back
        from policy_proposals/ or the proposal ledger."""
        p = doc["proposal"]
        proposal = Proposal(
            id=p["id"],
            title=p["title"],
            what_changes=p["what_changes"],
            what_it_unlocks=p["what_it_unlocks"],
            what_it_does_not_unlock=p["what_it_does_not_unlock"],
            policy_name=p["policy_name"],
            proposed_value=float(p["proposed_value"]),
            created_at=float(p.get("created_at", time.time())),
        )
        prior = doc.get("previous_value")
        seq = doc.get("last_ledger_seq_at_eval")
        heldout_base = doc.get("heldout_baseline_metrics")
        heldout_new = doc.get("heldout_new_metrics")
        return cls(
            proposal=proposal,
            accepted=bool(doc["accepted"]),
            reason=str(doc["reason"]),
            baseline=FleetMetrics.from_dict(doc["baseline_metrics"]),
            proposed=FleetMetrics.from_dict(doc["new_metrics"]),
            previous_value=float(prior) if prior is not None else None,
            last_ledger_seq_at_eval=int(seq) if seq is not None else None,
            recorded_at=float(doc.get("recorded_at", time.time())),
            heldout_baseline=FleetMetrics.from_dict(heldout_base) if heldout_base else None,
            heldout_proposed=FleetMetrics.from_dict(heldout_new) if heldout_new else None,
        )


def evaluate_proposal(
    proposal: Proposal,
    mem: RouterMemory,
    *,
    fleet: tuple[Any, ...] = DEFAULT_FLEET,
    now: float | None = None,
) -> Verdict:
    """Replay :data:`DEFAULT_SCENARIOS` under the current policy value
    (baseline) and the proposal's value (proposed), against the same
    ledger-adjusted fleet, and produce a frozen per-scenario verdict.

    Per-scenario meet gate (frozen residual):
      - Every scenario that currently meets its fidelity_target within its
        credit_budget must still meet under the proposal. In practice, for
        this repo's tiers, any scenario that routes to Tier 2 (HW_CERTIFIED)
        under the live policy is treated as meeting by design and must remain
        in Tier 2 under the proposal. A dump-to-cheapest policy that moves a
        certified scenario to Tier 1 is rejected even if aggregate sums
        improve, because empirical fidelity comes from certificates and is
        never predicted in this loop.
      - Among scenarios that still meet, total estimated cost may not rise
        more than TOTAL_COST_INCREASE_EPSILON (zero by default).
    """
    # One clock reading for the whole evaluation. The ledger's trend
    # projection is a function of time, so replaying baseline and proposal
    # (and the held-out set) at different instants made even a no-op proposal
    # report a ~1e-10 exposure delta.
    now = time.time() if now is None else now
    baseline = baseline_snapshot(mem, fleet=fleet, now=now)
    # Record the last certificate_ledger seq at evaluation time for rollback windowing.
    last_seq = 0
    try:
        last_seen = None
        for entry in mem.certificates():
            last_seen = entry
        if last_seen is not None:
            last_seq = int(last_seen.seq)
    except Exception:
        last_seq = 0

    original_value = getattr(budget_router, proposal.policy_name)
    setattr(budget_router, proposal.policy_name, proposal.proposed_value)
    try:
        proposed = _replay(fleet, mem, now=now)
        heldout_proposed = _replay(fleet, mem, now=now, scenarios=HELDOUT_SCENARIOS)
    finally:
        setattr(budget_router, proposal.policy_name, original_value)
    heldout_baseline = _replay(fleet, mem, now=now, scenarios=HELDOUT_SCENARIOS)

    accepted, detail = _non_regression(baseline, proposed)
    if accepted:
        held_ok, held_detail = _non_regression(heldout_baseline, heldout_proposed, label="held-out ")
        if held_ok:
            detail = f"{detail}; {held_detail}"
        else:
            accepted, detail = False, held_detail
    reason = f"accepted: {detail}" if accepted else detail

    return Verdict(
        proposal=proposal,
        accepted=accepted,
        reason=reason,
        baseline=baseline,
        proposed=proposed,
        previous_value=float(original_value) if original_value is not None else None,
        last_ledger_seq_at_eval=last_seq,
        recorded_at=time.time() if now is None else now,
        heldout_baseline=heldout_baseline,
        heldout_proposed=heldout_proposed,
    )


def _non_regression(base: FleetMetrics, new: FleetMetrics, *, label: str = "") -> tuple[bool, str]:
    """One non-regression check of *new* against *base*. Returns
    ``(ok, detail)``; on failure *detail* is the full ``rejected: ...``
    reason, on success it is the plain-English accept detail.

    Checks, in order (first failure wins):
      1. Per-scenario meet gate: a scenario certified at baseline must stay
         HW_CERTIFIED.
      2. Per-scenario cost: no single scenario may cost more than at
         baseline (PER_SCENARIO_COST_INCREASE_EPSILON), so a saving on one
         scenario cannot hide an increase on another.
      3. Aggregate cost: the total may not rise (TOTAL_COST_INCREASE_EPSILON).
    """
    left = [
        b.scenario_name
        for b, n in zip(base.outcomes, new.outcomes)
        if b.tier == int(Tier.HW_CERTIFIED) and n.tier != int(Tier.HW_CERTIFIED)
    ]
    if left:
        return False, (
            "rejected: per-scenario meet gate failed; "
            f"the following {label}scenario(s) left HW_CERTIFIED under the proposal: "
            + ", ".join(sorted(left))
        )

    costlier = [
        b.scenario_name
        for b, n in zip(base.outcomes, new.outcomes)
        if n.estimated_cost - b.estimated_cost > PER_SCENARIO_COST_INCREASE_EPSILON + _COST_TOLERANCE
    ]
    if costlier:
        return False, (
            f"rejected: per-scenario cost rose on {label}scenario(s) "
            + ", ".join(sorted(costlier))
            + " (an aggregate saving may not hide a per-scenario increase)"
        )

    cost_delta = new.total_estimated_cost - base.total_estimated_cost
    if cost_delta > TOTAL_COST_INCREASE_EPSILON:
        return False, (
            f"rejected: total estimated {label}cost rose by {cost_delta:.4g} credits "
            f"(allowed increase {TOTAL_COST_INCREASE_EPSILON:.4g})"
        )

    # Physical-error exposure is reported but gates acceptance only
    # indirectly, via the Tier 2 requirement above.
    error_delta = new.total_physical_error_exposure - base.total_physical_error_exposure
    return True, (
        f"all previously meeting {label}scenarios still meet; "
        f"cost delta {cost_delta:.4g}, physical-error exposure delta {error_delta:.4g}"
    )

def maybe_rollback_after_land(
    mem: RouterMemory,
    verdict: Verdict,
    *,
    monitor_next_n_certificates: int = 20,
    now: float | None = None,
) -> int | None:
    """Post-land rollback guard.

    Plain English thresholds:
      - Monitor window: the next 20 certificate_ledger rows recorded after land.
      - Rollback trigger: any single currently-meeting scenario that misses compared to the
        frozen baseline snapshot is considered a failure.

    On a miss, restore the live routing constant to the value that was in force
    before the accepted proposal landed, then append a rejected verdict via the
    append-only policy_proposals table. Failed jobs stay archived and are never
    deleted. Returns the recorded sequence number, or None when no miss was
    reported.
    """
    # Windowing: require at least N new certificate_ledger rows after evaluation time.
    start_seq = int(verdict.last_ledger_seq_at_eval or 0)
    new_count = 0
    for entry in mem.certificates():
        if int(entry.seq) > start_seq:
            new_count += 1
    if new_count < monitor_next_n_certificates:
        # Window not closed; do nothing yet.
        return None

    # Replay the current live policy against the ledger-adjusted fleet.
    current = baseline_snapshot(mem, now=now)
    # Per-scenario meet regression relative to the frozen baseline on the verdict.
    failed: list[str] = []
    for base_o, cur_o in zip(verdict.baseline.outcomes, current.outcomes):
        was_certified = base_o.tier == int(Tier.HW_CERTIFIED)
        now_certified = cur_o.tier == int(Tier.HW_CERTIFIED)
        if was_certified and not now_certified:
            failed.append(base_o.scenario_name)
    if not failed:
        return None

    policy_name = verdict.proposal.policy_name
    prior = verdict.previous_value
    if prior is None:
        # Without a prior value we cannot restore deterministically; witness a reject.
        reject = Verdict(
            proposal=verdict.proposal,
            accepted=False,
            reason=("rollback: miss in the 20-entry monitor window; no prior value recorded to restore"),
            baseline=verdict.baseline,
            proposed=current,
            previous_value=None,
            last_ledger_seq_at_eval=verdict.last_ledger_seq_at_eval,
            recorded_at=time.time() if now is None else now,
        )
        return mem.record_proposal(verdict.proposal.id, policy_name, False, reject.to_dict())

    # Restore the live module constant.
    setattr(budget_router, policy_name, prior)
    reject = Verdict(
        proposal=verdict.proposal,
        accepted=False,
        reason=(f"rollback: miss in the 20-entry monitor window; restored {policy_name} to its prior value"),
        baseline=verdict.baseline,
        proposed=current,
        previous_value=prior,
        last_ledger_seq_at_eval=verdict.last_ledger_seq_at_eval,
        recorded_at=time.time() if now is None else now,
    )
    return mem.record_proposal(verdict.proposal.id, policy_name, False, reject.to_dict())


# ---------------------------------------------------------------------------
# Post-land monitoring (harness-roadmap/03, rollback guard)
# ---------------------------------------------------------------------------

_logger = logging.getLogger("limen.router.proposal")

MONITOR_WINDOW = 20
_LANDED_PREFIX = "landed:"
_CLOSED_PREFIX = "monitor window closed:"
_ROLLBACK_PREFIX = "rollback:"


def write_pending_claim(proposal: Proposal, proposals_dir: pathlib.Path | str) -> pathlib.Path | None:
    """Write *proposal* to ``<proposals_dir>/pending/<id>.claim.json`` for the
    CI verdict job to judge. Returns the path, or ``None`` without writing
    when a claim with this id already exists in pending/, accepted/ or
    failed/ (idempotent; a witnessed decision is never re-opened)."""
    root = pathlib.Path(proposals_dir)
    name = f"{proposal.id}.claim.json"
    for sub in ("pending", "accepted", "failed"):
        if (root / sub / name).exists():
            return None
    pending = root / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    path = pending / name
    path.write_text(json.dumps(proposal.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def _revert_claim(verdict: Verdict, *, seq: int, stamp: float) -> Proposal | None:
    """The claim that makes an in-process rollback permanent: set the policy
    back to the value it had before *verdict* landed. It goes through the same
    CI verdict as any other change, so a revert is witnessed, not assumed."""
    prior = verdict.previous_value
    if prior is None:
        return None
    name = verdict.proposal.policy_name
    pid = verdict.proposal.id
    day = time.strftime("%Y-%m-%d", time.gmtime(stamp))
    return Proposal(
        id=f"{day}-revert-{pid}",
        title=f"Revert {name} to {prior:g} after {pid} regressed",
        what_changes=(
            f"Sets {name} back to {prior:g}, the value in force before proposal {pid} landed."
        ),
        what_it_unlocks=(
            f"Makes the rollback recorded at proposal-ledger seq {seq} permanent in source, so "
            "every process (CLI runs, CI replay, other daemons) routes with the restored value "
            "instead of only the daemon that detected the regression."
        ),
        what_it_does_not_unlock=(
            "It does not say why the proposal regressed, change any other constant, or touch "
            "the failed proposal's witnessed record."
        ),
        policy_name=name,
        proposed_value=prior,
        created_at=stamp,
    )


def _load_accepted_verdicts(proposals_dir: pathlib.Path | str) -> list[Verdict]:
    """Every accepted verdict file under ``<proposals_dir>/accepted/``.
    Malformed files are skipped, never fatal: the daemon must not crash
    because of a bad witness file."""
    accepted_dir = pathlib.Path(proposals_dir) / "accepted"
    verdicts: list[Verdict] = []
    if not accepted_dir.is_dir():
        return verdicts
    for path in sorted(accepted_dir.glob("*.json")):
        if path.name.endswith(".claim.json"):
            continue
        try:
            verdict = Verdict.from_dict(json.loads(path.read_text()))
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if verdict.accepted:
            verdicts.append(verdict)
    return verdicts


def monitor_landed_proposals(
    mem: RouterMemory,
    proposals_dir: pathlib.Path | str,
    *,
    window: int = MONITOR_WINDOW,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Run the post-land rollback guard against the live ledger *mem*.

    Which proposal is monitored: per gated policy, the latest accepted
    verdict that actually changed the value and whose proposed value is the
    one live in ``budget_router`` right now. That condition is what "landed"
    means here, and it is the safety interlock: an accepted proposal that was
    never merged, or that a newer one superseded, is never rolled back (a
    rollback would otherwise restore a stale value over a newer one).

    The window is anchored in *mem*'s own certificate sequence, not the CI
    ledger's: the first time a landed proposal is seen, a ``landed:`` row is
    appended to the proposal ledger recording the live sequence number, and
    the next *window* live certificates are counted from there. When the
    window closes, :func:`maybe_rollback_after_land` decides; either a
    ``rollback:`` row or a ``monitor window closed:`` row is appended, so a
    proposal is resolved exactly once and everything is witnessed.

    A rollback only restores the in-process constant. After a restart the
    source value is live again, so a recorded rollback is re-applied on the
    next call (action ``reapplied``). A rollback also files a ``-revert-``
    claim in ``<proposals_dir>/pending/``; once CI accepts it,
    ``scripts/apply_proposal.py`` makes the source edit permanent and a human
    commits it through the normal PR gate.

    Returns one action dict per proposal acted on (empty if nothing to do).
    """
    actions: list[dict[str, Any]] = []

    landed: dict[str, Verdict] = {}
    for verdict in _load_accepted_verdicts(proposals_dir):
        prop = verdict.proposal
        if verdict.previous_value is None or verdict.previous_value == prop.proposed_value:
            continue  # a no-change claim has nothing to land or roll back
        if getattr(budget_router, prop.policy_name, None) != prop.proposed_value:
            continue  # not live: unmerged, superseded, or already rolled back
        current = landed.get(prop.policy_name)
        if current is None or verdict.recorded_at >= current.recorded_at:
            landed[prop.policy_name] = verdict

    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in mem.proposals():
        rows_by_id.setdefault(row["proposal_id"], []).append(row)

    def _reason(row: dict[str, Any]) -> str:
        return str(row["verdict"].get("reason", ""))

    for policy_name, verdict in sorted(landed.items()):
        pid = verdict.proposal.id
        rows = rows_by_id.get(pid, [])

        if any(_reason(r).startswith(_CLOSED_PREFIX) for r in rows):
            continue  # window closed cleanly, nothing more to watch

        if any(_reason(r).startswith(_ROLLBACK_PREFIX) for r in rows):
            # Rolled back earlier, but the source value is live again
            # (typically after a restart). Re-apply the witnessed rollback.
            setattr(budget_router, policy_name, verdict.previous_value)
            _logger.warning(
                "re-applied recorded rollback of %s for proposal %s (prior value %r); "
                "revert the constant in budget_router.py to make this permanent",
                policy_name, pid, verdict.previous_value,
            )
            actions.append({"proposal_id": pid, "policy_name": policy_name, "action": "reapplied"})
            continue

        marker_rows = [r for r in rows if _reason(r).startswith(_LANDED_PREFIX)]
        if not marker_rows:
            last_seq = 0
            for entry in mem.certificates():
                last_seq = int(entry.seq)
            stamp = time.time() if now is None else now
            marker = dataclasses.replace(
                verdict,
                reason=(
                    f"{_LANDED_PREFIX} observed live; monitoring the next {window} "
                    f"live certificates from ledger seq {last_seq}"
                ),
                last_ledger_seq_at_eval=last_seq,
                recorded_at=stamp,
            )
            mem.record_proposal(pid, policy_name, True, marker.to_dict(), recorded_at=stamp)
            actions.append({"proposal_id": pid, "policy_name": policy_name, "action": "landed"})
            continue

        marker_verdict = Verdict.from_dict(marker_rows[-1]["verdict"])
        start_seq = int(marker_verdict.last_ledger_seq_at_eval or 0)
        new_count = sum(1 for entry in mem.certificates() if int(entry.seq) > start_seq)
        if new_count < window:
            continue  # window still open

        seq = maybe_rollback_after_land(
            mem, marker_verdict, monitor_next_n_certificates=window, now=now
        )
        if seq is not None:
            stamp = time.time() if now is None else now
            revert = _revert_claim(marker_verdict, seq=seq, stamp=stamp)
            revert_path = write_pending_claim(revert, proposals_dir) if revert is not None else None
            _logger.warning(
                "rolled back %s for proposal %s after a miss in the %d-certificate "
                "window (restored %r); %s",
                policy_name, pid, window, marker_verdict.previous_value,
                (
                    f"revert claim written to {revert_path}; once CI accepts it, "
                    f"`python scripts/apply_proposal.py {revert.id}` edits the source"
                    if revert_path is not None
                    else "no new revert claim was written (none possible or one already exists); "
                    "revert the constant in budget_router.py to make this permanent"
                ),
            )
            actions.append(
                {
                    "proposal_id": pid,
                    "policy_name": policy_name,
                    "action": "rolled_back",
                    "seq": seq,
                    "revert_claim": str(revert_path) if revert_path is not None else None,
                }
            )
            continue

        stamp = time.time() if now is None else now
        closed = dataclasses.replace(
            marker_verdict,
            reason=(
                f"{_CLOSED_PREFIX} no regression across {new_count} live certificates "
                f"(window {window})"
            ),
            recorded_at=stamp,
        )
        mem.record_proposal(pid, policy_name, True, closed.to_dict(), recorded_at=stamp)
        actions.append({"proposal_id": pid, "policy_name": policy_name, "action": "confirmed"})

    return actions
