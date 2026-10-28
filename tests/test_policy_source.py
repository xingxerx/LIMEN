"""Tests for limen.router.policy_source: turning an accepted proposal into a
guarded one-line source edit (the durable half of the rollback guard)."""

import json
import pathlib
import shutil
import tempfile
import unittest

from limen.router.memory import RouterMemory
from limen.router.policy_source import PolicyApplyError, apply_accepted, set_constant
from limen.router.proposal import FleetMetrics, Proposal, ScenarioOutcome, Verdict

NAME = "CRITICALITY_SPREAD_THRESHOLD"
REPO = pathlib.Path(__file__).resolve().parent.parent


def _fleet() -> FleetMetrics:
    return FleetMetrics(
        outcomes=(
            ScenarioOutcome(
                scenario_name="s", fidelity_target=0.9, tier=1, backend="b",
                shots=1, estimated_cost=1.0, physical_error_exposure=0.0,
            ),
        )
    )


def _witness(proposals: pathlib.Path, pid: str, verdict: Verdict) -> None:
    """Record *verdict* in the proposal ledger the way run_proposal.py does."""
    mem = RouterMemory(proposals / "ledger" / "router_memory.sqlite3")
    try:
        mem.record_proposal(pid, NAME, verdict.accepted, verdict.to_dict())
    finally:
        mem.close()


def _verdict(pid: str, proposed: float, previous: float | None, accepted: bool = True) -> Verdict:
    return Verdict(
        proposal=Proposal(
            id=pid, title="t", what_changes="x", what_it_unlocks="x", what_it_does_not_unlock="x",
            policy_name=NAME, proposed_value=proposed,
        ),
        accepted=accepted, reason="accepted: test" if accepted else "rejected: test",
        baseline=_fleet(), proposed=_fleet(), previous_value=previous,
        last_ledger_seq_at_eval=0, recorded_at=1.0,
    )


class SetConstantTests(unittest.TestCase):
    def test_replaces_the_value_and_keeps_the_trailing_comment(self):
        text = f"A = 1\n{NAME} = 10.25  # live\nB = 2\n"
        new, old = set_constant(text, NAME, 8.0)
        self.assertEqual(old, 10.25)
        self.assertEqual(new, f"A = 1\n{NAME} = 8.0  # live\nB = 2\n")

    def test_handles_an_annotated_assignment(self):
        new, old = set_constant(f"{NAME}: float = 3.5\n", NAME, 4.0)
        self.assertEqual((new, old), (f"{NAME}: float = 4.0\n", 3.5))

    def test_ignores_comments_and_indented_uses(self):
        text = f"# {NAME} = 99.0\ndef f():\n    return {NAME} >= 1\n{NAME} = 2.0\n"
        new, _ = set_constant(text, NAME, 5.0)
        self.assertIn(f"{NAME} = 5.0\n", new)
        self.assertIn(f"# {NAME} = 99.0", new)

    def test_refuses_zero_or_several_assignments(self):
        with self.assertRaises(PolicyApplyError):
            set_constant("X = 1\n", NAME, 1.0)
        with self.assertRaises(PolicyApplyError):
            set_constant(f"{NAME} = 1.0\n{NAME} = 2.0\n", NAME, 3.0)

    def test_refuses_a_stale_source(self):
        with self.assertRaises(PolicyApplyError) as ctx:
            set_constant(f"{NAME} = 10.25\n", NAME, 8.0, expected_old=2.0)
        self.assertIn("stale", str(ctx.exception))

    def test_refuses_a_non_numeric_value(self):
        with self.assertRaises(PolicyApplyError):
            set_constant(f"{NAME} = compute()\n", NAME, 1.0)


class ApplyAcceptedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)
        self.proposals = self.dir / "policy_proposals"
        (self.proposals / "accepted").mkdir(parents=True)
        self.source = self.dir / "budget_router.py"
        self.source.write_text(f"{NAME} = 10.25  # live\n")

    def _accept(self, pid: str, proposed: float, previous: float | None, accepted: bool = True,
                witness: bool = True) -> Verdict:
        verdict = _verdict(pid, proposed, previous, accepted)
        (self.proposals / "accepted" / f"{pid}.json").write_text(json.dumps(verdict.to_dict()))
        if witness:
            _witness(self.proposals, pid, verdict)
        return verdict

    def test_applies_an_accepted_revert(self):
        self._accept("revert-one", 8.0, 10.25)
        result = apply_accepted("revert-one", self.proposals, self.source)
        self.assertTrue(result.changed)
        self.assertEqual((result.old_value, result.new_value), (10.25, 8.0))
        self.assertEqual(self.source.read_text(), f"{NAME} = 8.0  # live\n")

    def test_dry_run_does_not_write(self):
        self._accept("revert-one", 8.0, 10.25)
        result = apply_accepted("revert-one", self.proposals, self.source, dry_run=True)
        self.assertTrue(result.changed)
        self.assertEqual(self.source.read_text(), f"{NAME} = 10.25  # live\n")

    def test_preserves_crlf_line_endings(self):
        self.source.write_bytes(f"A = 1\r\n{NAME} = 10.25\r\n".encode())
        self._accept("revert-one", 8.0, 10.25)
        apply_accepted("revert-one", self.proposals, self.source)
        self.assertEqual(self.source.read_bytes(), f"A = 1\r\n{NAME} = 8.0\r\n".encode())

    def test_already_applied_is_a_no_op(self):
        self._accept("landed", 10.25, 8.0)
        result = apply_accepted("landed", self.proposals, self.source)
        self.assertFalse(result.changed)
        self.assertEqual(self.source.read_text(), f"{NAME} = 10.25  # live\n")

    def test_refuses_when_the_source_moved_since_the_verdict(self):
        self._accept("stale", 8.0, 2.0)  # verdict assumed the source held 2.0
        with self.assertRaises(PolicyApplyError):
            apply_accepted("stale", self.proposals, self.source)
        self.assertEqual(self.source.read_text(), f"{NAME} = 10.25  # live\n")

    def test_refuses_without_an_accepted_verdict(self):
        with self.assertRaises(PolicyApplyError) as ctx:
            apply_accepted("never-judged", self.proposals, self.source)
        self.assertIn("only a proposal the CI verdict accepted", str(ctx.exception))

    def test_refuses_a_verdict_with_no_recorded_previous_value(self):
        # Without it there is nothing to check the source against; applying
        # blind could silently overwrite a newer value.
        self._accept("legacy", 8.0, None)
        with self.assertRaises(PolicyApplyError) as ctx:
            apply_accepted("legacy", self.proposals, self.source)
        self.assertIn("previous_value", str(ctx.exception))
        self.assertEqual(self.source.read_text(), f"{NAME} = 10.25  # live\n")

    def test_refuses_a_verdict_that_is_not_accepted(self):
        self._accept("rejected", 8.0, 10.25, accepted=False)
        with self.assertRaises(PolicyApplyError):
            apply_accepted("rejected", self.proposals, self.source)

    def test_refuses_an_ungated_policy_or_a_corrupt_verdict(self):
        verdict = _verdict("bad", 8.0, 10.25).to_dict()
        verdict["proposal"]["policy_name"] = "MAX_SHOTS"
        (self.proposals / "accepted" / "bad.json").write_text(json.dumps(verdict))
        with self.assertRaises(PolicyApplyError):
            apply_accepted("bad", self.proposals, self.source)
        (self.proposals / "accepted" / "junk.json").write_text("{not json")
        with self.assertRaises(PolicyApplyError):
            apply_accepted("junk", self.proposals, self.source)


class WitnessTests(unittest.TestCase):
    """The verdict file alone is not enough: the ledger must have recorded it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)
        self.proposals = self.dir / "policy_proposals"
        (self.proposals / "accepted").mkdir(parents=True)
        self.source = self.dir / "budget_router.py"
        self.source.write_text(f"{NAME} = 10.25\n")

    def _file(self, verdict: Verdict) -> None:
        (self.proposals / "accepted" / f"{verdict.proposal.id}.json").write_text(
            json.dumps(verdict.to_dict())
        )

    def test_a_verdict_dropped_in_by_hand_is_refused(self):
        verdict = _verdict("forged", 8.0, 10.25)
        self._file(verdict)  # never recorded by the harness
        with self.assertRaises(PolicyApplyError):
            apply_accepted("forged", self.proposals, self.source)
        self.assertEqual(self.source.read_text(), f"{NAME} = 10.25\n")

    def test_a_missing_ledger_is_refused(self):
        self._file(_verdict("no-ledger", 8.0, 10.25))
        with self.assertRaises(PolicyApplyError) as ctx:
            apply_accepted("no-ledger", self.proposals, self.source)
        self.assertIn("no proposal ledger", str(ctx.exception))

    def test_a_verdict_edited_after_it_was_witnessed_is_refused(self):
        verdict = _verdict("edited", 8.0, 10.25)
        _witness(self.proposals, "edited", verdict)
        tampered = _verdict("edited", 1.0, 10.25)  # same id, different proposed value
        self._file(tampered)
        with self.assertRaises(PolicyApplyError) as ctx:
            apply_accepted("edited", self.proposals, self.source)
        self.assertIn("differs from what the ledger recorded", str(ctx.exception))
        self.assertEqual(self.source.read_text(), f"{NAME} = 10.25\n")

    def test_a_later_rejection_in_the_ledger_wins(self):
        accepted = _verdict("flipped", 8.0, 10.25)
        _witness(self.proposals, "flipped", accepted)
        _witness(self.proposals, "flipped", _verdict("flipped", 8.0, 10.25, accepted=False))
        self._file(accepted)
        with self.assertRaises(PolicyApplyError):
            apply_accepted("flipped", self.proposals, self.source)

    def test_the_latest_acceptance_is_what_counts(self):
        _witness(self.proposals, "reeval", _verdict("reeval", 8.0, 2.0))
        latest = _verdict("reeval", 8.0, 10.25)  # re-evaluated later against the current source
        _witness(self.proposals, "reeval", latest)
        self._file(latest)
        self.assertTrue(apply_accepted("reeval", self.proposals, self.source).changed)

    def test_checking_the_ledger_never_writes_to_the_repository(self):
        verdict = _verdict("ro", 8.0, 10.25)
        _witness(self.proposals, "ro", verdict)
        self._file(verdict)
        ledger = self.proposals / "ledger" / "router_memory.sqlite3"
        before_bytes = ledger.read_bytes()
        before_dir = sorted(p.name for p in ledger.parent.iterdir())
        apply_accepted("ro", self.proposals, self.source, dry_run=True)
        self.assertEqual(ledger.read_bytes(), before_bytes)
        self.assertEqual(sorted(p.name for p in ledger.parent.iterdir()), before_dir)  # no -wal/-shm


class RealRepoTests(unittest.TestCase):
    """Against copies of the real source and accepted verdicts, never the repo files."""

    def test_the_live_proposal_is_already_applied_in_the_real_source(self):
        result = apply_accepted(
            "2026-09-01-raise-criticality-threshold-10p25",
            REPO / "policy_proposals",
            REPO / "limen/router/budget_router.py",
            dry_run=True,
        )
        self.assertFalse(result.changed)
        self.assertEqual(result.new_value, 10.25)

    def test_a_revert_edits_exactly_one_line_of_a_copy_of_the_real_source(self):
        with tempfile.TemporaryDirectory() as d:
            copy = pathlib.Path(d) / "budget_router.py"
            shutil.copy2(REPO / "limen/router/budget_router.py", copy)
            proposals = pathlib.Path(d) / "pp"
            (proposals / "accepted").mkdir(parents=True)
            verdict = _verdict("r", 8.0, 10.25)
            (proposals / "accepted" / "r.json").write_text(json.dumps(verdict.to_dict()))
            _witness(proposals, "r", verdict)
            before = copy.read_text().splitlines()
            apply_accepted("r", proposals, copy)
            after = copy.read_text().splitlines()
            changed = [(a, b) for a, b in zip(before, after) if a != b]
            self.assertEqual(len(before), len(after))
            self.assertEqual(changed, [(f"{NAME} = 10.25", f"{NAME} = 8.0")])


if __name__ == "__main__":
    unittest.main()
