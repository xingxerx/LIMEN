"""Tests for limen.router.proposal / the self-improvement-loop replay
(harness-roadmap/03): claim-file parsing, the closed policy scope guard,
non-regression verdicts, and the append-only proposal ledger round-trip
through limen.router.memory and limen.router.report."""

import json
import pathlib
import tempfile
import unittest

from limen.router.memory import RouterMemory
from limen.router.proposal import (
    GATED_POLICIES,
    Proposal,
    baseline_snapshot,
    evaluate_proposal,
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

    def test_raising_threshold_past_a_sample_drops_its_error_exposure_without_raising_cost(self):
        # Seed one sample so ibm_marrakesh has a non-trivial ledger-adjusted
        # physical_error_rate, then confirm a threshold raise that moves a
        # skewed scenario from Tier 2 to Tier 1 can only ever drop (never
        # raise) that scenario's physical-error exposure, with cost held
        # exactly constant (same backend, same shots either tier).
        self.mem.record_sample(
            "ibm_marrakesh", "physical_error_rate", 0.03, observed_at=_NOW
        )
        proposal = Proposal(
            id="raise-threshold", title="x", what_changes="x", what_it_unlocks="x",
            what_it_does_not_unlock="x", policy_name="CRITICALITY_SPREAD_THRESHOLD",
            proposed_value=50.0,  # high enough that every scenario stays/moves to Tier 1
        )
        verdict = evaluate_proposal(proposal, self.mem, now=_NOW)
        self.assertTrue(verdict.accepted)
        self.assertEqual(
            verdict.baseline.total_estimated_cost, verdict.proposed.total_estimated_cost
        )
        self.assertLessEqual(
            verdict.proposed.total_physical_error_exposure,
            verdict.baseline.total_physical_error_exposure,
        )
        # Every scenario now Tier 1, so exposure is exactly zero.
        self.assertEqual(verdict.proposed.total_physical_error_exposure, 0.0)

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


class BaselineSnapshotTests(unittest.TestCase):
    def test_baseline_is_deterministic_for_fixed_ledger_and_now(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        mem = RouterMemory(pathlib.Path(tmp.name) / "memory.sqlite3")
        self.addCleanup(mem.close)
        a = baseline_snapshot(mem, now=_NOW)
        b = baseline_snapshot(mem, now=_NOW)
        self.assertEqual(a.to_dict(), b.to_dict())


if __name__ == "__main__":
    unittest.main()
