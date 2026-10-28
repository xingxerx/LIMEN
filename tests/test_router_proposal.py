"""Tests for limen.router.proposal / the self-improvement-loop replay
(harness-roadmap/03): claim-file parsing, the closed policy scope guard,
non-regression verdicts, and the append-only proposal ledger round-trip
through limen.router.memory and limen.router.report."""

import dataclasses
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from limen.router import budget_router
from limen.router import proposal as proposal_module
from limen.router.memory import RouterMemory
from limen.router.proposal import (
    GATED_POLICIES,
    FleetMetrics,
    Proposal,
    ReplayScenario,
    ScenarioOutcome,
    Verdict,
    _non_regression,
    _skewed_qubo,
    baseline_snapshot,
    evaluate_proposal,
    maybe_rollback_after_land,
    monitor_landed_proposals,
)
from limen.router.report import build_proposal_report

_NOW = 1_800_000_000.0


def _claim(**overrides) -> dict:
    doc = {
        "id": "test-proposal",
        "title": "Test proposal",
        "what_changes": "a routing constant",
        "what_it_unlocks": "nothing real, this is a test",
        "what_it_does_not_unlock": "everything else",
        "policy_name": "CRITICALITY_SPREAD_THRESHOLD",
        "proposed_value": 2.0,
    }
    doc.update(overrides)
    return doc


class ProposalScopeGuardTests(unittest.TestCase):
    def test_rejects_ungated_policy_name(self):
        with self.assertRaises(ValueError):
            Proposal(
                id="x", title="x", what_changes="x", what_it_unlocks="x",
                what_it_does_not_unlock="x", policy_name="NOT_A_GATED_POLICY",
                proposed_value=1.0,
            )

    def test_from_claim_file_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "claim.json"
            path.write_text(json.dumps(_claim(proposed_value=3.5)))
            proposal = Proposal.from_claim_file(path)
            self.assertEqual(proposal.policy_name, "CRITICALITY_SPREAD_THRESHOLD")
            self.assertEqual(proposal.proposed_value, 3.5)
            self.assertIn(proposal.policy_name, GATED_POLICIES)


class ProposalReplayTests(unittest.TestCase):
    def setUp(self):
        # Some tests below set the live constant to simulate a landing and
        # never restore it; put it back so later tests see the real value.
        self.addCleanup(
            setattr, budget_router, "CRITICALITY_SPREAD_THRESHOLD",
            budget_router.CRITICALITY_SPREAD_THRESHOLD,
        )
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)
        self.mem = RouterMemory(self.dir / "memory.sqlite3")
        self.addCleanup(self.mem.close)

    def test_no_op_proposal_is_a_non_regression_accept(self):
        # Proposing the value already in force must always be a no-op
        # accept: baseline and proposed are computed by literally the
        # same replay.
        from limen.router import budget_router

        proposal = Proposal(
            id="noop", title="noop", what_changes="x", what_it_unlocks="x",
            what_it_does_not_unlock="x", policy_name="CRITICALITY_SPREAD_THRESHOLD",
            proposed_value=budget_router.CRITICALITY_SPREAD_THRESHOLD,
        )
        verdict = evaluate_proposal(proposal, self.mem, now=_NOW)
        self.assertTrue(verdict.accepted)
        self.assertEqual(
            verdict.baseline.total_estimated_cost, verdict.proposed.total_estimated_cost
        )
        self.assertEqual(
            verdict.baseline.total_physical_error_exposure,
            verdict.proposed.total_physical_error_exposure,
        )

    def test_policy_value_is_restored_after_evaluation(self):
        from limen.router import budget_router

        original = budget_router.CRITICALITY_SPREAD_THRESHOLD
        proposal = Proposal(
            id="restore-check", title="x", what_changes="x", what_it_unlocks="x",
            what_it_does_not_unlock="x", policy_name="CRITICALITY_SPREAD_THRESHOLD",
            proposed_value=original + 100.0,
        )
        evaluate_proposal(proposal, self.mem, now=_NOW)
        self.assertEqual(budget_router.CRITICALITY_SPREAD_THRESHOLD, original)

    def test_raising_threshold_past_a_sample_is_rejected_by_per_scenario_gate(self):
        # Previously this asserted that a large threshold raise was accepted
        # because physical-error exposure dropped and cost stayed flat.
        # The new gate freezes per-scenario meets: any scenario that was
        # certified must remain certified. With a very high threshold,
        # skewed_high_fidelity (fidelity_target 0.995) would leave Tier 2,
        # so the proposal must be rejected.
        self.mem.record_sample(
            "ibm_marrakesh", "physical_error_rate", 0.03, observed_at=_NOW
        )
        proposal = Proposal(
            id="raise-threshold", title="x", what_changes="x", what_it_unlocks="x",
            what_it_does_not_unlock="x", policy_name="CRITICALITY_SPREAD_THRESHOLD",
            proposed_value=50.0,  # high enough that every scenario moves to Tier 1
        )
        verdict = evaluate_proposal(proposal, self.mem, now=_NOW)
        self.assertFalse(verdict.accepted)
        self.assertIn("left HW_CERTIFIED", verdict.reason)

    def test_verdict_report_is_witnessed_in_the_append_only_ledger(self):
        proposal = Proposal(
            id="ledger-witness", title="x", what_changes="x", what_it_unlocks="x",
            what_it_does_not_unlock="x", policy_name="CRITICALITY_SPREAD_THRESHOLD",
            proposed_value=2.0,
        )
        verdict = evaluate_proposal(proposal, self.mem, now=_NOW)
        self.mem.record_proposal(
            proposal.id, proposal.policy_name, verdict.accepted, verdict.to_dict()
        )
        recorded = list(self.mem.proposals())
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["proposal_id"], "ledger-witness")
        self.assertEqual(recorded[0]["accepted"], verdict.accepted)

        # Append-only: rejected or accepted, nothing is ever deleted or
        # updated (mirrors certificate_ledger's triggers).
        with self.assertRaises(Exception):
            self.mem._conn.execute("DELETE FROM policy_proposals WHERE seq = 1")
            self.mem._conn.commit()

    def test_report_mentions_verdict_and_both_metrics(self):
        proposal = Proposal(
            id="report-check", title="Report check", what_changes="a change",
            what_it_unlocks="an unlock", what_it_does_not_unlock="not everything",
            policy_name="CRITICALITY_SPREAD_THRESHOLD", proposed_value=2.0,
        )
        verdict = evaluate_proposal(proposal, self.mem, now=_NOW)
        report = build_proposal_report(verdict)
        self.assertIn("ACCEPTED" if verdict.accepted else "REJECTED", report)
        self.assertIn("a change", report)
        self.assertIn("an unlock", report)
        self.assertIn("not everything", report)

    def test_dump_to_cheapest_policy_is_rejected_due_to_high_fidelity_scenario(self):
        # Proposed threshold +inf dumps every scenario into Tier 1.
        # Old behavior would accept because exposure drops to zero,
        # but the new per-scenario gate must reject because
        # skewed_high_fidelity leaves HW_CERTIFIED.
        proposal = Proposal(
            id="dump-to-cheapest", title="x", what_changes="x", what_it_unlocks="x",
            what_it_does_not_unlock="x", policy_name="CRITICALITY_SPREAD_THRESHOLD",
            proposed_value=float("inf"),
        )
        verdict = evaluate_proposal(proposal, self.mem, now=_NOW)
        self.assertFalse(verdict.accepted)
        self.assertIn("left HW_CERTIFIED", verdict.reason)

    def test_policy_that_keeps_meets_and_does_not_raise_cost_is_accepted(self):
        # Lowering the threshold keeps previously certified scenarios certified
        # and should not increase total estimated cost.
        proposal = Proposal(
            id="lower-threshold", title="x", what_changes="x", what_it_unlocks="x",
            what_it_does_not_unlock="x", policy_name="CRITICALITY_SPREAD_THRESHOLD",
            proposed_value=2.0,
        )
        verdict = evaluate_proposal(proposal, self.mem, now=_NOW)
        self.assertTrue(verdict.accepted)
        self.assertLessEqual(
            verdict.proposed.total_estimated_cost, verdict.baseline.total_estimated_cost
        )

    def test_rollback_restores_previous_value_and_records_rejected_verdict(self):
        from limen.router import budget_router
        original = budget_router.CRITICALITY_SPREAD_THRESHOLD
        new_value = 50.0
        proposal = Proposal(
            id="land-and-rollback", title="x", what_changes="x", what_it_unlocks="x",
            what_it_does_not_unlock="x", policy_name="CRITICALITY_SPREAD_THRESHOLD",
            proposed_value=new_value,
        )
        verdict = evaluate_proposal(proposal, self.mem, now=_NOW)
        # Simulate land: set the live constant to the proposed value.
        setattr(budget_router, proposal.policy_name, new_value)
        # Append fewer than N=20 certificate_ledger rows: window not closed yet.
        for i in range(19):
            self.mem.append_certificate({"kind": "test-cert", "i": i}, backend=None, recorded_at=_NOW + i)
        # No action taken; returns None; policy unchanged.
        none_seq = maybe_rollback_after_land(self.mem, verdict, now=_NOW)
        self.assertIsNone(none_seq)
        self.assertEqual(getattr(budget_router, proposal.policy_name), new_value)
        prev_rows = list(self.mem.proposals())
        # Now append the 20th row to close the window.
        self.mem.append_certificate({"kind": "test-cert", "i": 19}, backend=None, recorded_at=_NOW + 19)
        # Trigger rollback evaluation; for this test, we force a miss by having chosen
        # a policy that can change tiering compared to baseline in DEFAULT_SCENARIOS.
        seq = maybe_rollback_after_land(self.mem, verdict, now=_NOW)
        self.assertIsInstance(seq, int)
        # The live constant must be restored to the pre-land value.
        self.assertEqual(getattr(budget_router, proposal.policy_name), original)
        # A rejected verdict is appended to the proposals ledger.
        rows = list(self.mem.proposals())
        self.assertGreater(len(rows), len(prev_rows))
        self.assertFalse(rows[-1]["accepted"])
        # Must not leave the proposed value live.
        self.assertNotEqual(getattr(budget_router, proposal.policy_name), new_value)

    def test_rollback_window_open_no_action(self):
        from limen.router import budget_router
        original = budget_router.CRITICALITY_SPREAD_THRESHOLD
        new_value = 50.0
        proposal = Proposal(
            id="window-open", title="x", what_changes="x", what_it_unlocks="x",
            what_it_does_not_unlock="x", policy_name="CRITICALITY_SPREAD_THRESHOLD",
            proposed_value=new_value,
        )
        verdict = evaluate_proposal(proposal, self.mem, now=_NOW)
        # Simulate land
        setattr(budget_router, proposal.policy_name, new_value)
        before_rows = list(self.mem.proposals())
        # Append fewer than 20 rows
        for i in range(10):
            self.mem.append_certificate({"kind": "test-cert", "case": "open", "i": i}, backend=None, recorded_at=_NOW + i)
        seq = maybe_rollback_after_land(self.mem, verdict, now=_NOW)
        self.assertIsNone(seq)
        # No restore yet
        self.assertEqual(getattr(budget_router, proposal.policy_name), new_value)
        # No new proposal row
        after_rows = list(self.mem.proposals())
        self.assertEqual(len(after_rows), len(before_rows))


class BaselineSnapshotTests(unittest.TestCase):
    def test_baseline_is_deterministic_for_fixed_ledger_and_now(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        mem = RouterMemory(pathlib.Path(tmp.name) / "memory.sqlite3")
        self.addCleanup(mem.close)
        a = baseline_snapshot(mem, now=_NOW)
        b = baseline_snapshot(mem, now=_NOW)
        self.assertEqual(a.to_dict(), b.to_dict())


def _metrics(costs: list[tuple[str, float]], tier: int = 1) -> FleetMetrics:
    return FleetMetrics(
        outcomes=tuple(
            ScenarioOutcome(
                scenario_name=name, fidelity_target=0.9, tier=tier, backend="b",
                shots=1, estimated_cost=cost, physical_error_exposure=0.0,
            )
            for name, cost in costs
        )
    )


class PerScenarioCostGateTests(unittest.TestCase):
    def test_aggregate_saving_cannot_hide_a_per_scenario_increase(self):
        # Yule-Simpson shape: the total goes down, one scenario goes up.
        base = _metrics([("a", 100.0), ("b", 100.0)])
        new = _metrics([("a", 10.0), ("b", 120.0)])
        self.assertLess(new.total_estimated_cost, base.total_estimated_cost)
        ok, detail = _non_regression(base, new)
        self.assertFalse(ok)
        self.assertIn("per-scenario cost rose on scenario(s) b (", detail)

    def test_equal_or_lower_everywhere_passes(self):
        base = _metrics([("a", 100.0), ("b", 100.0)])
        new = _metrics([("a", 100.0), ("b", 60.0)])
        ok, detail = _non_regression(base, new)
        self.assertTrue(ok)
        self.assertIn("still meet", detail)

    def test_heldout_label_is_named_in_the_reason(self):
        base = _metrics([("h1", 10.0)])
        new = _metrics([("h1", 11.0)])
        ok, detail = _non_regression(base, new, label="held-out ")
        self.assertFalse(ok)
        self.assertIn("held-out scenario(s) h1", detail)

    def test_meet_gate_still_runs_before_the_cost_gates(self):
        base = _metrics([("a", 10.0)], tier=2)
        new = _metrics([("a", 5.0)], tier=1)
        ok, detail = _non_regression(base, new)
        self.assertFalse(ok)
        self.assertIn("left HW_CERTIFIED", detail)


class HeldoutGateTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(budget_router, "CRITICALITY_SPREAD_THRESHOLD", 10.25)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.mem = RouterMemory(pathlib.Path(self._tmp.name) / "memory.sqlite3")
        self.addCleanup(self.mem.close)

    def _proposal(self, pid: str, value: float) -> Proposal:
        return Proposal(
            id=pid, title="x", what_changes="x", what_it_unlocks="x",
            what_it_does_not_unlock="x", policy_name="CRITICALITY_SPREAD_THRESHOLD",
            proposed_value=value,
        )

    def test_a_proposal_that_fits_the_visible_scenarios_can_still_fail_held_out(self):
        # skewed_high_fidelity has spread 20*20/39 = 10.2564, so 10.255 keeps
        # every visible scenario certified. This held-out scenario has spread
        # 19.99*20/38.99 = 10.2539: certified at the live 10.25, but not at 10.255.
        near = ReplayScenario(
            "near_threshold", _skewed_qubo(20, dominant=19.99),
            fidelity_target=0.995, credit_budget=1000.0,
        )
        with mock.patch.object(proposal_module, "HELDOUT_SCENARIOS", (near,)):
            verdict = evaluate_proposal(self._proposal("fit-visible", 10.255), self.mem, now=_NOW)
        visible_ok, _ = _non_regression(verdict.baseline, verdict.proposed)
        self.assertTrue(visible_ok)
        self.assertFalse(verdict.accepted)
        self.assertIn("held-out scenario(s) left HW_CERTIFIED", verdict.reason)
        self.assertIn("near_threshold", verdict.reason)

    def test_real_heldout_set_accepts_a_no_op(self):
        verdict = evaluate_proposal(
            self._proposal("noop", budget_router.CRITICALITY_SPREAD_THRESHOLD), self.mem, now=_NOW
        )
        self.assertTrue(verdict.accepted)
        self.assertIsNotNone(verdict.heldout_baseline)
        self.assertEqual(
            len(verdict.heldout_baseline.outcomes), len(proposal_module.HELDOUT_SCENARIOS)
        )

    def test_heldout_scenarios_are_distinct_from_the_visible_ones(self):
        visible = {s.name for s in proposal_module.DEFAULT_SCENARIOS}
        held = {s.name for s in proposal_module.HELDOUT_SCENARIOS}
        self.assertFalse(visible & held)

    def test_verdict_round_trips_through_json(self):
        verdict = evaluate_proposal(self._proposal("round-trip", 9.0), self.mem, now=_NOW)
        again = Verdict.from_dict(json.loads(json.dumps(verdict.to_dict())))
        self.assertEqual(again.to_dict(), verdict.to_dict())

    def test_legacy_verdict_without_heldout_keys_still_loads(self):
        doc = evaluate_proposal(self._proposal("legacy", 9.0), self.mem, now=_NOW).to_dict()
        doc.pop("heldout_baseline_metrics")
        doc.pop("heldout_new_metrics")
        legacy = Verdict.from_dict(doc)
        self.assertIsNone(legacy.heldout_baseline)
        self.assertNotIn("heldout_baseline_metrics", legacy.to_dict())

    def test_report_includes_the_heldout_table(self):
        verdict = evaluate_proposal(self._proposal("report", 9.0), self.mem, now=_NOW)
        report = build_proposal_report(verdict)
        self.assertIn("## Held-out replay", report)
        self.assertIn("heldout_certified_11", report)


class MonitorLandedProposalsTests(unittest.TestCase):
    LIVE = 10.25
    PRIOR = 8.0

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)
        self.proposals_dir = self.dir / "policy_proposals"
        (self.proposals_dir / "accepted").mkdir(parents=True)
        self.mem = RouterMemory(self.dir / "live.sqlite3")
        self.addCleanup(self.mem.close)
        # Every test runs with the live constant pinned (and restored after).
        patcher = mock.patch.object(budget_router, "CRITICALITY_SPREAD_THRESHOLD", self.LIVE)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write_accepted(self, pid: str, proposed: float, previous: float | None = PRIOR) -> Verdict:
        proposal = Proposal(
            id=pid, title="x", what_changes="x", what_it_unlocks="x",
            what_it_does_not_unlock="x", policy_name="CRITICALITY_SPREAD_THRESHOLD",
            proposed_value=proposed,
        )
        # Baseline replayed under the value live before the proposal landed.
        with mock.patch.object(budget_router, "CRITICALITY_SPREAD_THRESHOLD", self.PRIOR):
            baseline = baseline_snapshot(self.mem, now=_NOW)
        verdict = Verdict(
            proposal=proposal, accepted=True, reason="accepted: test",
            baseline=baseline, proposed=baseline, previous_value=previous,
            last_ledger_seq_at_eval=0, recorded_at=_NOW,
        )
        (self.proposals_dir / "accepted" / f"{pid}.json").write_text(json.dumps(verdict.to_dict()))
        (self.proposals_dir / "accepted" / f"{pid}.claim.json").write_text(json.dumps(proposal.to_dict()))
        return verdict

    def _certs(self, n: int, start: int = 0) -> None:
        for i in range(start, start + n):
            self.mem.append_certificate({"kind": "live-cert", "i": i}, backend=None, recorded_at=_NOW + i)

    def _reasons(self) -> list[str]:
        return [row["verdict"]["reason"] for row in self.mem.proposals()]

    def test_first_sighting_anchors_the_window_then_waits(self):
        self._write_accepted("landed-one", self.LIVE)
        self._certs(3)
        actions = monitor_landed_proposals(self.mem, self.proposals_dir, now=_NOW)
        self.assertEqual([a["action"] for a in actions], ["landed"])
        self.assertTrue(self._reasons()[-1].startswith("landed:"))
        self.assertIn("seq 3", self._reasons()[-1])
        # Window still open (fewer than 20 certs since the anchor): nothing to do.
        self._certs(19, start=3)
        self.assertEqual(monitor_landed_proposals(self.mem, self.proposals_dir, now=_NOW), [])

    def test_clean_window_is_confirmed_exactly_once(self):
        self._write_accepted("clean", self.LIVE)
        monitor_landed_proposals(self.mem, self.proposals_dir, now=_NOW)
        self._certs(20)
        actions = monitor_landed_proposals(self.mem, self.proposals_dir, now=_NOW)
        self.assertEqual([a["action"] for a in actions], ["confirmed"])
        self.assertTrue(self._reasons()[-1].startswith("monitor window closed:"))
        self.assertEqual(budget_router.CRITICALITY_SPREAD_THRESHOLD, self.LIVE)
        self.assertEqual(monitor_landed_proposals(self.mem, self.proposals_dir, now=_NOW), [])

    def test_regression_rolls_back_and_the_rollback_survives_a_restart(self):
        # A landed value that drops every certified scenario to Tier 1.
        # The file's baseline is replayed under PRIOR (8.0), where scenarios
        # are certified; under 50.0 they are not, so the window ends in a miss.
        with mock.patch.object(budget_router, "CRITICALITY_SPREAD_THRESHOLD", 50.0):
            self._write_accepted("bad-land", 50.0, previous=self.LIVE)
            monitor_landed_proposals(self.mem, self.proposals_dir, now=_NOW)  # anchors
            self._certs(20)
            actions = monitor_landed_proposals(self.mem, self.proposals_dir, now=_NOW)
            self.assertEqual([a["action"] for a in actions], ["rolled_back"])
            self.assertEqual(budget_router.CRITICALITY_SPREAD_THRESHOLD, self.LIVE)
            self.assertTrue(self._reasons()[-1].startswith("rollback:"))
            self.assertFalse(list(self.mem.proposals())[-1]["accepted"])

            # The rollback also files a revert claim so it can become permanent
            # through the normal CI verdict (and scripts/apply_proposal.py).
            claim_path = pathlib.Path(actions[0]["revert_claim"])
            self.assertEqual(claim_path.parent, self.proposals_dir / "pending")
            revert = Proposal.from_claim_file(claim_path)
            self.assertIn("revert-bad-land", revert.id)
            self.assertEqual(revert.proposed_value, self.LIVE)
            # As CI would see it: the bad value is what the source holds.
            budget_router.CRITICALITY_SPREAD_THRESHOLD = 50.0
            self.assertTrue(evaluate_proposal(revert, self.mem, now=_NOW).accepted)
            budget_router.CRITICALITY_SPREAD_THRESHOLD = self.LIVE

            # Restart: the source value (50.0) is live again.
            budget_router.CRITICALITY_SPREAD_THRESHOLD = 50.0
            actions = monitor_landed_proposals(self.mem, self.proposals_dir, now=_NOW)
            self.assertEqual([a["action"] for a in actions], ["reapplied"])
            self.assertEqual(budget_router.CRITICALITY_SPREAD_THRESHOLD, self.LIVE)
            # Re-applying must not file a second revert claim.
            self.assertEqual(len(list((self.proposals_dir / "pending").glob("*.claim.json"))), 1)

    def test_accepted_but_not_live_is_never_touched(self):
        self._write_accepted("unmerged", 12.0)  # live is 10.25
        self._certs(25)
        self.assertEqual(monitor_landed_proposals(self.mem, self.proposals_dir, now=_NOW), [])
        self.assertEqual(list(self.mem.proposals()), [])

    def test_no_change_claim_is_ignored(self):
        self._write_accepted("no-change", self.LIVE, previous=self.LIVE)
        self.assertEqual(monitor_landed_proposals(self.mem, self.proposals_dir, now=_NOW), [])

    def test_malformed_witness_file_is_skipped(self):
        (self.proposals_dir / "accepted" / "junk.json").write_text("{not json")
        (self.proposals_dir / "accepted" / "partial.json").write_text(json.dumps({"accepted": True}))
        self.assertEqual(monitor_landed_proposals(self.mem, self.proposals_dir, now=_NOW), [])

    def test_only_the_latest_landed_proposal_per_policy_is_monitored(self):
        older = self._write_accepted("older", self.LIVE)
        newer = dataclasses.replace(self._write_accepted("newer", self.LIVE), recorded_at=_NOW + 1)
        (self.proposals_dir / "accepted" / "newer.json").write_text(json.dumps(newer.to_dict()))
        actions = monitor_landed_proposals(self.mem, self.proposals_dir, now=_NOW)
        self.assertEqual([a["proposal_id"] for a in actions], ["newer"])
        self.assertEqual(older.proposal.id, "older")


class LimendMonitorPolicyTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)
        self.proposals_dir = self.dir / "policy_proposals"
        (self.proposals_dir / "accepted").mkdir(parents=True)
        patcher = mock.patch.object(budget_router, "CRITICALITY_SPREAD_THRESHOLD", 10.25)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _land(self, pid: str = "daemon-landed") -> None:
        mem = RouterMemory(self.dir / "baseline.sqlite3")
        try:
            baseline = baseline_snapshot(mem, now=_NOW)
        finally:
            mem.close()
        verdict = Verdict(
            proposal=Proposal(
                id=pid, title="x", what_changes="x", what_it_unlocks="x",
                what_it_does_not_unlock="x", policy_name="CRITICALITY_SPREAD_THRESHOLD",
                proposed_value=10.25,
            ),
            accepted=True, reason="accepted: test", baseline=baseline, proposed=baseline,
            previous_value=8.0, last_ledger_seq_at_eval=0, recorded_at=_NOW,
        )
        (self.proposals_dir / "accepted" / f"{pid}.json").write_text(json.dumps(verdict.to_dict()))

    def test_off_unless_a_proposals_dir_is_given(self):
        from limen.limend.daemon import monitor_policy

        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("LIMEN_PROPOSALS_DIR", None)
            self.assertEqual(monitor_policy(self.dir / "live.sqlite3", None), [])

    def test_missing_dir_is_skipped_without_raising(self):
        from limen.limend.daemon import monitor_policy

        self.assertEqual(monitor_policy(self.dir / "live.sqlite3", None, self.dir / "nope"), [])

    def test_a_guard_failure_never_propagates(self):
        from limen.limend.daemon import monitor_policy

        with mock.patch.object(proposal_module, "monitor_landed_proposals", side_effect=RuntimeError("boom")):
            self.assertEqual(
                monitor_policy(self.dir / "live.sqlite3", None, self.proposals_dir), []
            )

    def test_leaves_a_caller_supplied_ledger_open(self):
        from limen.limend.daemon import monitor_policy

        self._land()
        mem = RouterMemory(self.dir / "live.sqlite3")
        self.addCleanup(mem.close)
        actions = monitor_policy(mem, None, self.proposals_dir)
        self.assertEqual([a["action"] for a in actions], ["landed"])
        self.assertEqual(len(list(mem.proposals())), 1)  # connection still usable

    def test_run_forever_anchors_the_window_at_startup(self):
        from limen.limend.daemon import run_forever

        self._land()
        results = self.dir / "results"
        results.mkdir()
        run_forever(self.dir / "spool", results_dir=results, once=True, proposals_dir=self.proposals_dir)
        with RouterMemory.in_results_dir(results) as mem:
            reasons = [r["verdict"]["reason"] for r in mem.proposals()]
        self.assertEqual(len(reasons), 1)
        self.assertTrue(reasons[0].startswith("landed:"))


if __name__ == "__main__":
    unittest.main()
