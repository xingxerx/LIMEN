# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Proposer for the self-improvement loop (harness-roadmap/03).

The loop's gate (:mod:`limen.router.proposal`) judges a claim file; until
now every claim was written by hand. This module writes them from the
ledger: it sweeps a gated routing-policy knob over a grid, replays the
visible :data:`~limen.router.proposal.DEFAULT_SCENARIOS` at each value, and
proposes the best value that

  1. passes the same non-regression rules the gate applies, and
  2. strictly improves total estimated cost or physical-error exposure
     without making the other worse.

If no value qualifies it proposes nothing, and says so. Today that is the
answer for ``CRITICALITY_SPREAD_THRESHOLD``: the meet gate forbids any
certified scenario from leaving Tier 2, which rules out every move that
could lower exposure, and cost is the same in both tiers, so the replay has
no gradient on this knob. The proposer reports that fixed point rather than
manufacture a claim the evidence does not support.

Held-out discipline: this module never reads
:data:`~limen.router.proposal.HELDOUT_SCENARIOS` and never calls
:func:`~limen.router.proposal.evaluate_proposal`, because an accept/reject
answer from the full gate would leak the held-out result back to the thing
doing the proposing. CI judges the claim it writes, held-out set included
(``scripts/run_proposal.py`` on ``policy_proposals/pending/``).

Scope guard: only knobs in :data:`~limen.router.proposal.GATED_POLICIES` can
be proposed, and this module cannot widen that set. Widening it is a
separate, deliberate decision (see proposal.py).
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from collections.abc import Mapping
from typing import Any

import limen.router.budget_router as budget_router
from limen.router.budget_router import DEFAULT_FLEET
from limen.router.memory import RouterMemory
from limen.router.proposal import (
    DEFAULT_SCENARIOS,
    GATED_POLICIES,
    FleetMetrics,
    Proposal,
    ReplayScenario,
    _non_regression,
    _replay,
    write_pending_claim,
)

# Grid the knob is swept over. 0.25 matches the precision of the values
# already proposed by hand (8.0, 10.25).
CANDIDATE_STEP = 0.25
CANDIDATE_CEILING = 64.0

# A move counts as an improvement only if it beats the baseline by more than
# this on cost or exposure (guards against float noise).
MIN_IMPROVEMENT = 1e-6


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One knob value that passed the gates and strictly improved the replay."""

    value: float
    metrics: FleetMetrics
    cost_delta: float
    exposure_delta: float


@dataclasses.dataclass(frozen=True)
class Search:
    """What a sweep found, including the nothing-to-propose case."""

    policy_name: str
    live_value: float
    baseline: FleetMetrics
    tested: int
    passing: int
    candidate: Candidate | None


def candidate_values(live: float, *, step: float = CANDIDATE_STEP) -> list[float]:
    """Grid values to try: every multiple of *step* up to the larger of
    :data:`CANDIDATE_CEILING` and four times the live value, excluding the
    live value itself."""
    ceiling = max(CANDIDATE_CEILING, 4.0 * live)
    count = int(round(ceiling / step))
    return [round(step * i, 6) for i in range(1, count + 1) if round(step * i, 6) != live]


def select_candidate(
    baseline: FleetMetrics,
    replays: Mapping[float, FleetMetrics],
    live: float,
) -> tuple[int, Candidate | None]:
    """Pick the best value from *replays* (value -> metrics). Returns
    ``(passing, candidate)``, where *passing* counts values that cleared the
    non-regression rules whether or not they improved anything.

    Ranking among improving values: lowest cost delta, then lowest exposure
    delta, then closest to the live value (smallest change), then smallest
    value. A value that lowers one axis by raising the other is not an
    improvement and is skipped.
    """
    passing = 0
    best: Candidate | None = None
    best_key: tuple[float, float, float, float] | None = None
    for value, metrics in replays.items():
        if value == live:
            continue
        ok, _ = _non_regression(baseline, metrics)
        if not ok:
            continue
        passing += 1
        cost_delta = metrics.total_estimated_cost - baseline.total_estimated_cost
        exposure_delta = (
            metrics.total_physical_error_exposure - baseline.total_physical_error_exposure
        )
        if cost_delta > MIN_IMPROVEMENT or exposure_delta > MIN_IMPROVEMENT:
            continue  # trades one axis for the other
        if cost_delta > -MIN_IMPROVEMENT and exposure_delta > -MIN_IMPROVEMENT:
            continue  # no strict gain on either axis
        key = (cost_delta, exposure_delta, abs(value - live), value)
        if best_key is None or key < best_key:
            best_key = key
            best = Candidate(
                value=value, metrics=metrics, cost_delta=cost_delta, exposure_delta=exposure_delta
            )
    return passing, best


def find_candidate(
    mem: RouterMemory,
    *,
    policy_name: str = "CRITICALITY_SPREAD_THRESHOLD",
    fleet: tuple[Any, ...] = DEFAULT_FLEET,
    scenarios: tuple[ReplayScenario, ...] = DEFAULT_SCENARIOS,
    now: float | None = None,
) -> Search:
    """Sweep *policy_name* and return what the visible scenarios support.

    The live constant is restored before returning, even on error. Pass
    *scenarios* only in tests; production use is always the visible set.
    """
    if policy_name not in GATED_POLICIES:
        raise ValueError(
            f"{policy_name!r} is not a gated routing-policy tunable. "
            f"Gated policies: {sorted(GATED_POLICIES)}"
        )
    now = time.time() if now is None else now
    live = float(getattr(budget_router, policy_name))
    baseline = _replay(fleet, mem, now=now, scenarios=scenarios)

    replays: dict[float, FleetMetrics] = {}
    values = candidate_values(live)
    try:
        for value in values:
            setattr(budget_router, policy_name, value)
            replays[value] = _replay(fleet, mem, now=now, scenarios=scenarios)
    finally:
        setattr(budget_router, policy_name, live)

    passing, candidate = select_candidate(baseline, replays, live)
    return Search(
        policy_name=policy_name,
        live_value=live,
        baseline=baseline,
        tested=len(values),
        passing=passing,
        candidate=candidate,
    )


def build_claim(search: Search, *, now: float | None = None) -> Proposal | None:
    """Turn a successful :class:`Search` into a plain-English claim, or
    ``None`` when the sweep found nothing."""
    cand = search.candidate
    if cand is None:
        return None
    now = time.time() if now is None else now
    name = search.policy_name
    stamp = time.strftime("%Y-%m-%d", time.gmtime(now))
    slug = name.lower().replace("_", "-")
    value_text = f"{cand.value:g}".replace(".", "p")
    return Proposal(
        id=f"{stamp}-auto-{slug}-{value_text}",
        title=f"Move {name} from {search.live_value:g} to {cand.value:g} (proposer)",
        what_changes=f"Sets {name} to {cand.value:g} (currently {search.live_value:g}).",
        what_it_unlocks=(
            f"Replaying the {len(search.baseline.outcomes)} visible scenarios against the "
            f"ledger-adjusted fleet: total estimated cost changes by {cand.cost_delta:+.4g} "
            f"credits and physical-error exposure by {cand.exposure_delta:+.4g}, with every "
            "scenario that meets its fidelity target today still meeting it and no single "
            "scenario costing more."
        ),
        what_it_does_not_unlock=(
            "It was not judged against the held-out scenarios (the proposer never reads "
            "them; CI does). It changes no other routing constant, does not change backend "
            "selection or shot counts, and does not widen the set of gated policies."
        ),
        policy_name=name,
        proposed_value=cand.value,
        created_at=now,
    )


def write_claim(proposal: Proposal, proposals_dir: pathlib.Path | str) -> pathlib.Path | None:
    """Write *proposal* to ``<proposals_dir>/pending/<id>.claim.json`` for CI
    to judge. Returns the path, or ``None`` without writing when a claim
    with this id already exists in pending/, accepted/ or failed/ (so the
    proposer is idempotent and never re-opens a witnessed decision)."""
    return write_pending_claim(proposal, proposals_dir)


def propose(
    mem: RouterMemory,
    proposals_dir: pathlib.Path | str,
    *,
    policy_name: str = "CRITICALITY_SPREAD_THRESHOLD",
    fleet: tuple[Any, ...] = DEFAULT_FLEET,
    now: float | None = None,
    write: bool = True,
) -> tuple[Search, Proposal | None, pathlib.Path | None]:
    """Search, build the claim, and (unless *write* is false) write it."""
    search = find_candidate(mem, policy_name=policy_name, fleet=fleet, now=now)
    proposal = build_claim(search, now=now)
    path = write_claim(proposal, proposals_dir) if (proposal is not None and write) else None
    return search, proposal, path
