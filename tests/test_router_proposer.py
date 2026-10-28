"""Tests for limen.router.proposer: the grid, the selection rules, the
fixed-point answer on the live knob, claim writing, and the held-out
discipline (the proposer must never see the held-out scenarios)."""

import ast
import inspect
import json
import pathlib
import tempfile
import unittest
from unittest import mock

from limen.router import budget_router
from limen.router import proposer
from limen.router.memory import RouterMemory
from limen.router.proposal import (
    FleetMetrics,
    Proposal,
    ScenarioOutcome,
    evaluate_proposal,
)
from limen.router.proposer import (
    Candidate,
    Search,
    build_claim,
    candidate_values,
    find_candidate,
    propose,
    select_candidate,
    write_claim,
)

_NOW = 1_800_000_000.0  # 2027-01-15 UTC


def _fleet(rows: list[tuple[str, int, float, float]]) -> FleetMetrics:
    """rows: (scenario_name, tier, estimated_cost, physical_error_exposure)."""
    return FleetMetrics(
        outcomes=tuple(
            ScenarioOutcome(
                scenario_name=name, fidelity_target=0.9, tier=tier, backend="b",
                shots=1, estimated_cost=cost, physical_error_exposure=exposure,
            )
            for name, tier, cost, exposure in rows
        )
    )


class CandidateGridTests(unittest.TestCase):
    def test_grid_excludes_the_live_value_and_covers_it_in_steps(self):
        values = candidate_values(10.25)
        self.assertNotIn(10.25, values)
        self.assertIn(10.0, values)
        self.assertIn(10.5, values)
        self.assertEqual(values[0], 0.25)
        self.assertGreaterEqual(values[-1], 64.0)

    def test_grid_grows_with_a_large_live_value(self):
        self.assertGreaterEqual(candidate_values(100.0)[-1], 400.0)


class SelectCandidateTests(unittest.TestCase):
    BASE = _fleet([("a", 2, 100.0, 50.0), ("b", 1, 100.0, 0.0)])

    def test_picks_the_value_that_lowers_cost(self):
        replays = {
            5.0: _fleet([("a", 2, 90.0, 50.0), ("b", 1, 100.0, 0.0)]),
            6.0: _fleet([("a", 2, 100.0, 50.0), ("b", 1, 100.0, 0.0)]),  # no gain
        }
        passing, cand = select_candidate(self.BASE, replays, live=10.0)
        self.assertEqual(passing, 2)
        self.assertEqual(cand.value, 5.0)
        self.assertAlmostEqual(cand.cost_delta, -10.0)

    def test_a_per_scenario_cost_increase_is_never_proposed(self):
        # Aggregate cost falls 70 but scenario b rises: Yule-Simpson shape.
        replays = {5.0: _fleet([("a", 2, 10.0, 50.0), ("b", 1, 120.0, 0.0)])}
        passing, cand = select_candidate(self.BASE, replays, live=10.0)
        self.assertEqual(passing, 0)
        self.assertIsNone(cand)

    def test_a_certified_scenario_leaving_tier_2_is_never_proposed(self):
        replays = {5.0: _fleet([("a", 1, 50.0, 0.0), ("b", 1, 100.0, 0.0)])}
        passing, cand = select_candidate(self.BASE, replays, live=10.0)
        self.assertEqual(passing, 0)
        self.assertIsNone(cand)

    def test_trading_exposure_for_cost_is_not_an_improvement(self):
        replays = {5.0: _fleet([("a", 2, 90.0, 80.0), ("b", 1, 100.0, 0.0)])}
        passing, cand = select_candidate(self.BASE, replays, live=10.0)
        self.assertEqual(passing, 1)
        self.assertIsNone(cand)

    def test_exposure_gain_alone_counts(self):
        replays = {5.0: _fleet([("a", 2, 100.0, 40.0), ("b", 1, 100.0, 0.0)])}
        _, cand = select_candidate(self.BASE, replays, live=10.0)
        self.assertEqual(cand.value, 5.0)
        self.assertAlmostEqual(cand.exposure_delta, -10.0)

    def test_ties_go_to_the_smallest_change_from_live(self):
        better = _fleet([("a", 2, 90.0, 50.0), ("b", 1, 100.0, 0.0)])
        replays = {2.0: better, 8.0: better, 9.5: better}
        _, cand = select_candidate(self.BASE, replays, live=10.0)
        self.assertEqual(cand.value, 9.5)

    def test_the_live_value_itself_is_ignored(self):
        replays = {10.0: _fleet([("a", 2, 1.0, 1.0), ("b", 1, 1.0, 0.0)])}
        passing, cand = select_candidate(self.BASE, replays, live=10.0)
        self.assertEqual(passing, 0)
        self.assertIsNone(cand)


class LiveKnobTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(budget_router, "CRITICALITY_SPREAD_THRESHOLD", 10.25)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)
        self.mem = RouterMemory(self.dir / "memory.sqlite3")
        self.addCleanup(self.mem.close)

    def test_the_live_threshold_is_a_fixed_point(self):
        # Under the meet gate no move can lower exposure, and cost is the same
        # in both tiers, so the replay has no gradient on this knob.
        search = find_candidate(self.mem, now=_NOW)
        self.assertIsNone(search.candidate)
        self.assertGreater(search.tested, 200)
        self.assertGreater(search.passing, 0)  # values that merely do no harm exist
        self.assertIsNone(build_claim(search, now=_NOW))

    def test_the_live_constant_is_restored_after_a_sweep(self):
        find_candidate(self.mem, now=_NOW)
        self.assertEqual(budget_router.CRITICALITY_SPREAD_THRESHOLD, 10.25)

    def test_the_live_constant_is_restored_when_a_replay_fails(self):
        # First call is the baseline replay; the second (first swept value) fails.
        with mock.patch.object(proposer, "_replay") as replay:
            replay.side_effect = [_fleet([("a", 1, 1.0, 0.0)]), RuntimeError("boom")]
            with self.assertRaises(RuntimeError):
                find_candidate(self.mem, now=_NOW)
        self.assertEqual(budget_router.CRITICALITY_SPREAD_THRESHOLD, 10.25)

    def test_an_ungated_policy_is_refused(self):
        with self.assertRaises(ValueError):
            find_candidate(self.mem, policy_name="MAX_SHOTS", now=_NOW)

    def test_propose_writes_nothing_at_a_fixed_point(self):
        proposals = self.dir / "policy_proposals"
        search, proposal, path = propose(self.mem, proposals, now=_NOW)
        self.assertIsNone(proposal)
        self.assertIsNone(path)
        self.assertFalse((proposals / "pending").exists())


class ClaimWritingTests(unittest.TestCase):
    def _search(self) -> Search:
        baseline = _fleet([("a", 2, 100.0, 50.0), ("b", 1, 100.0, 0.0)])
        better = _fleet([("a", 2, 90.0, 40.0), ("b", 1, 100.0, 0.0)])
        return Search(
            policy_name="CRITICALITY_SPREAD_THRESHOLD",
            live_value=10.25,
            baseline=baseline,
            tested=255,
            passing=3,
            candidate=Candidate(value=9.5, metrics=better, cost_delta=-10.0, exposure_delta=-10.0),
        )

    def test_claim_is_plain_english_and_names_what_it_does_not_unlock(self):
        proposal = build_claim(self._search(), now=_NOW)
        self.assertEqual(proposal.id, "2027-01-15-auto-criticality-spread-threshold-9p5")
        self.assertEqual(proposal.proposed_value, 9.5)
        self.assertIn("9.5", proposal.what_changes)
        self.assertIn("10.25", proposal.what_changes)
        self.assertIn("-10", proposal.what_it_unlocks)
        self.assertIn("held-out", proposal.what_it_does_not_unlock)

    def test_written_claim_round_trips_through_the_gate_loader(self):
        with tempfile.TemporaryDirectory() as d:
            proposal = build_claim(self._search(), now=_NOW)
            path = write_claim(proposal, d)
            self.assertEqual(path.parent.name, "pending")
            self.assertTrue(path.name.endswith(".claim.json"))
            loaded = Proposal.from_claim_file(path)
            self.assertEqual(loaded.proposed_value, 9.5)
            self.assertEqual(loaded.policy_name, "CRITICALITY_SPREAD_THRESHOLD")
            json.loads(path.read_text())  # valid JSON

    def test_write_is_idempotent_and_never_reopens_a_witnessed_decision(self):
        with tempfile.TemporaryDirectory() as d:
            proposal = build_claim(self._search(), now=_NOW)
            self.assertIsNotNone(write_claim(proposal, d))
            self.assertIsNone(write_claim(proposal, d))  # already pending
            for sub in ("accepted", "failed"):
                root = pathlib.Path(d) / sub
                root.mkdir()
                (root / f"{proposal.id}.claim.json").write_text("{}")
            (pathlib.Path(d) / "pending" / f"{proposal.id}.claim.json").unlink()
            self.assertIsNone(write_claim(proposal, d))  # witnessed elsewhere


class HeldoutDisciplineTests(unittest.TestCase):
    def test_the_proposer_never_reads_the_heldout_set_or_the_full_gate(self):
        # AST, not text: the docstring is allowed to name them, the code is not.
        tree = ast.parse(inspect.getsource(proposer))
        referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        referenced |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                referenced |= {alias.name for alias in node.names}
        for forbidden in ("HELDOUT_SCENARIOS", "evaluate_proposal", "heldout_baseline", "heldout_proposed"):
            self.assertNotIn(forbidden, referenced)

    def test_the_full_gate_still_adds_held_out_evidence_to_the_claim(self):
        # The proposer's claim carries no held-out evidence; the gate adds it.
        with tempfile.TemporaryDirectory() as d:
            mem = RouterMemory(pathlib.Path(d) / "memory.sqlite3")
            try:  # close before the temp dir goes: Windows will not delete an open sqlite file
                with mock.patch.object(budget_router, "CRITICALITY_SPREAD_THRESHOLD", 10.25):
                    verdict = evaluate_proposal(
                        Proposal(
                            id="x", title="x", what_changes="x", what_it_unlocks="x",
                            what_it_does_not_unlock="x", policy_name="CRITICALITY_SPREAD_THRESHOLD",
                            proposed_value=9.5,
                        ),
                        mem, now=_NOW,
                    )
            finally:
                mem.close()
        self.assertIsNotNone(verdict.heldout_baseline)


if __name__ == "__main__":
    unittest.main()
