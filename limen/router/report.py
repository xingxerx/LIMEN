# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Human-readable route reports: what was chosen, why, and what the
ledger says about it.

Milestone 2 of the harness-R&D program (see Milestone 1,
``_record_route_outcome`` in ``limen.pipeline``, which closed the write
side of the loop). Every routing decision already carries its own
rationale (``RoutePlan.notes``, built during :func:`route`) and every
completed run already produces a signed, hash-chained
``EndToEndCertificate``. Nothing before this module turned those two
pieces of evidence into one artifact a non-engineer could read and
verify without opening a debugger.

A ``RouteReport`` is deliberately *not* a new source of truth -- it
never computes a number that isn't already on the plan, the
certificate, or the ledger. It is a formatter over evidence that
already exists, which is what makes it a report and not a claim.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any


@dataclasses.dataclass(frozen=True)
class LedgerComparison:
    """What the ledger says about the chosen backend's *metric*, and how
    that compares to the static (pre-memory) fleet value that would have
    been used had no ledger existed.

    ``static_value`` is the field on the un-adjusted ``BackendProfile``
    (i.e. before :meth:`~limen.router.memory.RouterMemory.apply_memory`
    ran). ``ledger_value`` is the conservative trend-aware estimate that
    actually informed this route. ``moved`` is True only when the two
    differ, so a report never claims memory "helped" when the backend
    simply had no samples yet.
    """

    metric: str
    static_value: float | None
    ledger_value: float | None
    sample_count: int

    @property
    def moved(self) -> bool:
        if self.static_value is None or self.ledger_value is None:
            return False
        return self.static_value != self.ledger_value


@dataclasses.dataclass(frozen=True)
class RouteReport:
    """A completed run, explained.

    Built entirely from evidence already produced elsewhere:
    ``RouteRequest`` (what was asked), ``RoutePlan`` (what was decided,
    including its own ``notes``), ``EndToEndCertificate`` (what actually
    happened), and optionally the ``RouterMemory`` ledger comparison
    (whether history changed the decision).
    """

    backend: str
    tier: int
    n_vars: int
    shots: int
    use_cutting: bool
    num_partitions: int | None
    routing_notes: tuple[str, ...]
    ledger_comparisons: tuple[LedgerComparison, ...]
    is_optimal: bool | None
    success_probability: float
    energy: float
    physical_error_rate: float | None
    aggregate_logical_error_rate: float | None
    certificate_sha256: str | None
    generated_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "tier": self.tier,
            "n_vars": self.n_vars,
            "shots": self.shots,
            "use_cutting": self.use_cutting,
            "num_partitions": self.num_partitions,
            "routing_notes": list(self.routing_notes),
            "ledger_comparisons": [
                dataclasses.asdict(c) for c in self.ledger_comparisons
            ],
            "is_optimal": self.is_optimal,
            "success_probability": self.success_probability,
            "energy": self.energy,
            "physical_error_rate": self.physical_error_rate,
            "aggregate_logical_error_rate": self.aggregate_logical_error_rate,
            "certificate_sha256": self.certificate_sha256,
            "generated_at": self.generated_at,
        }

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append(f"# Route report: {self.backend} (Tier {self.tier})")
        lines.append("")
        lines.append(
            f"- **Problem size:** {self.n_vars} variables, {self.shots} shots"
            + (f", split across {self.num_partitions} partitions" if self.use_cutting else "")
        )
        optimal = (
            "optimal" if self.is_optimal else
            "not confirmed optimal" if self.is_optimal is False else
            "optimality not evaluated"
        )
        lines.append(
            f"- **Result:** energy {self.energy:.4g}, {optimal}, "
            f"success probability {self.success_probability:.1%}"
        )
        if self.physical_error_rate is not None:
            lines.append(f"- **Physical error rate:** {self.physical_error_rate:.2e}")
        if self.aggregate_logical_error_rate is not None:
            lines.append(
                f"- **Logical error rate:** {self.aggregate_logical_error_rate:.2e}"
            )
        if self.certificate_sha256:
            lines.append(f"- **Certificate hash:** `{self.certificate_sha256[:16]}...`")
        lines.append("")

        lines.append("## Why this backend")
        if self.routing_notes:
            for note in self.routing_notes:
                lines.append(f"- {note}")
        else:
            lines.append("- No routing notes recorded for this plan.")
        lines.append("")

        if self.ledger_comparisons:
            lines.append("## What history changed")
            any_moved = False
            for cmp in self.ledger_comparisons:
                if cmp.sample_count == 0:
                    continue
                if cmp.moved:
                    any_moved = True
                    lines.append(
                        f"- **{cmp.metric}**: static estimate was "
                        f"{cmp.static_value:.4g}, {cmp.sample_count} recorded "
                        f"run(s) moved it to {cmp.ledger_value:.4g}"
                    )
                else:
                    lines.append(
                        f"- **{cmp.metric}**: {cmp.sample_count} recorded run(s), "
                        f"no change from the static estimate ({cmp.static_value:.4g})"
                    )
            if not any_moved:
                lines.append(
                    "- (History is on, but no metric has diverged from the "
                    "static fleet profile yet.)"
                )
            lines.append("")

        return "\n".join(lines)


def build_proposal_report(verdict: Any) -> str:
    """Plain-English report for one self-improvement-loop verdict
    (harness-roadmap/03, ATRIUM AGENTS.md section 4, witness requirement).

    *verdict* is a :class:`limen.router.proposal.Verdict`. Extends this
    module's existing report format rather than inventing a new one:
    same "what/why, then evidence" shape as :meth:`RouteReport.to_markdown`.
    Every accepted or rejected proposal gets one of these -- rejection is
    witnessed and archived, never silently dropped (CANON: "no punishment
    realms").
    """
    p = verdict.proposal
    lines: list[str] = []
    lines.append(f"# Routing-policy proposal: {p.title}")
    lines.append("")
    lines.append(f"- **Proposal id:** `{p.id}`")
    lines.append(f"- **Policy:** `{p.policy_name}` -> {p.proposed_value!r}")
    lines.append(f"- **Verdict:** {'ACCEPTED' if verdict.accepted else 'REJECTED'} -- {verdict.reason}")
    lines.append("")

    lines.append("## What this changes")
    lines.append(f"- {p.what_changes}")
    lines.append("")
    lines.append("## What it unlocks")
    lines.append(f"- {p.what_it_unlocks}")
    lines.append("")
    lines.append("## What it does not unlock")
    lines.append(f"- {p.what_it_does_not_unlock}")
    lines.append("")

    lines.append("## Ledger-backed replay")
    lines.append(
        f"- **Baseline** total estimated cost: {verdict.baseline.total_estimated_cost:.4g} credits, "
        f"physical-error exposure: {verdict.baseline.total_physical_error_exposure:.4g}"
    )
    lines.append(
        f"- **Proposed** total estimated cost: {verdict.proposed.total_estimated_cost:.4g} credits, "
        f"physical-error exposure: {verdict.proposed.total_physical_error_exposure:.4g}"
    )
    lines.append("")
    lines.append("| scenario | baseline tier/backend | proposed tier/backend | changed |")
    lines.append("|---|---|---|---|")
    for base_o, new_o in zip(verdict.baseline.outcomes, verdict.proposed.outcomes):
        changed = "yes" if (base_o.tier, base_o.backend) != (new_o.tier, new_o.backend) else "no"
        lines.append(
            f"| {base_o.scenario_name} | T{base_o.tier}/{base_o.backend} "
            f"| T{new_o.tier}/{new_o.backend} | {changed} |"
        )
    lines.append("")

    return "\n".join(lines)


def _static_profile_for(backend_name: str) -> Any:
    """The un-adjusted BackendProfile for *backend_name* from the static
    default fleet, or None if it isn't one of the known defaults (e.g. a
    caller-supplied custom fleet)."""
    from limen.router.budget_router import DEFAULT_FLEET

    for profile in DEFAULT_FLEET:
        if profile.name == backend_name:
            return profile
    return None


def build_route_report(
    request: Any,
    plan: Any,
    cert: Any,
    *,
    mem: Any = None,
    certificate_sha256: str | None = None,
    now: float | None = None,
) -> RouteReport:
    """Build a :class:`RouteReport` from a completed run.

    *mem*, if given, is consulted read-only for a before/after comparison
    on the chosen backend's metrics -- it is never written to here (the
    write side is `limen.pipeline._record_route_outcome`, already run by
    the time a caller has a certificate to report on).
    """
    from limen.router.memory import _METRICS  # field-name/metric-name map lives here

    now = time.time() if now is None else now
    backend_name = plan.backend.name
    static_profile = _static_profile_for(backend_name)

    comparisons: list[LedgerComparison] = []
    if mem is not None:
        field_metric = (
            ("cost_per_shot", "seconds_per_shot"),
            ("avg_queue_seconds", "queue_seconds"),
            ("measured_logical_error", "logical_error"),
            ("physical_error_rate", "physical_error_rate"),
        )
        for field_name, metric in field_metric:
            if metric not in _METRICS:
                continue
            stats = mem.stats(backend_name, metric, now=now)
            static_value = (
                getattr(static_profile, field_name, None)
                if static_profile is not None
                else None
            )
            comparisons.append(
                LedgerComparison(
                    metric=metric,
                    static_value=static_value,
                    ledger_value=stats.conservative_estimate(now) if stats else None,
                    sample_count=stats.n if stats else 0,
                )
            )

    return RouteReport(
        backend=backend_name,
        tier=int(plan.tier),
        n_vars=plan.n_vars,
        shots=plan.shots,
        use_cutting=plan.use_cutting,
        num_partitions=plan.num_partitions,
        routing_notes=tuple(plan.notes),
        ledger_comparisons=tuple(comparisons),
        is_optimal=cert.is_optimal,
        success_probability=cert.success_probability,
        energy=cert.energy,
        physical_error_rate=cert.physical_error_rate,
        aggregate_logical_error_rate=cert.aggregate_logical_error_rate,
        certificate_sha256=certificate_sha256,
        generated_at=now,
    )
