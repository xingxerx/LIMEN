"""Tests for limen.router.history: seeding the router's cost model from
finished certs in results/ (the eil51 benchmark certs already on disk,
plus synthetic router_tier2-shaped certs for the newer cert type)."""

import json
import pathlib
import tempfile
import unittest

from limen.router import (
    DEFAULT_FLEET,
    RouteRequest,
    Tier,
    apply_history,
    route,
    scan_results,
)
from limen.router.history import _queue_seconds_from_timestamps

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


class TestScanRealResults(unittest.TestCase):
    """Scans the actual results/ directory checked into the repo."""

    def test_scans_eil51_certs_without_crashing(self):
        history = scan_results(RESULTS_DIR)
        self.assertIn("ibm_kingston", history)
        self.assertIn("ibm_fez", history)

    def test_seconds_per_shot_matches_known_eil51_cert(self):
        history = scan_results(RESULTS_DIR)
        # results/tsp_eil51_20260626_215323.json: ibm_kingston,
        # 1000 shots, 21.755569219589233 elapsed seconds.
        expected = 21.755569219589233 / 1000
        self.assertIn(expected, history["ibm_kingston"].seconds_per_shot)

    def test_unrecognized_files_are_skipped(self):
        # fleet_certificate.json, fetched_jobs_*.json, teleport_summary.json,
        # etc. carry no per-backend timing/error signal; scanning must not
        # raise even though they don't match either cert shape.
        history = scan_results(RESULTS_DIR)
        self.assertIsInstance(history, dict)


class TestApplyHistory(unittest.TestCase):

    def test_backend_without_history_is_unchanged(self):
        history = scan_results(RESULTS_DIR)
        updated = apply_history(DEFAULT_FLEET, history)
        by_name = {p.name: p for p in updated}
        # ionq:forte-1 never appears in any results/ cert.
        original = next(p for p in DEFAULT_FLEET if p.name == "ionq:forte-1")
        self.assertEqual(by_name["ionq:forte-1"], original)

    def test_backend_with_history_updates_cost_per_shot(self):
        history = scan_results(RESULTS_DIR)
        updated = apply_history(DEFAULT_FLEET, history)
        by_name = {p.name: p for p in updated}
        original = next(p for p in DEFAULT_FLEET if p.name == "ibm_kingston")
        self.assertNotEqual(by_name["ibm_kingston"].cost_per_shot, original.cost_per_shot)
        self.assertAlmostEqual(
            by_name["ibm_kingston"].cost_per_shot, 21.755569219589233 / 1000
        )

    def test_fleet_size_and_order_preserved(self):
        history = scan_results(RESULTS_DIR)
        updated = apply_history(DEFAULT_FLEET, history)
        self.assertEqual([p.name for p in updated], [p.name for p in DEFAULT_FLEET])


class TestRouterTier2Shape(unittest.TestCase):
    """Synthetic router_tier2_kingston_fetch.py-shaped cert, since no real
    one exists on disk until a Tier 2 hardware job actually completes."""

    def _write_cert(self, tmp_path: pathlib.Path, **overrides) -> None:
        doc = {
            "job_id": "synthetic123",
            "timestamps": {
                "created": "2026-07-06T20:00:00Z",
                "running": "2026-07-06T20:05:30Z",
                "finished": "2026-07-06T20:05:45Z",
            },
            "plan": {"backend": {"name": "ibm_kingston"}, "shots": 1000},
            "measured_success_deficit": 0.0123,
        }
        doc.update(overrides)
        (tmp_path / "router_tier2_kingston_synthetic123.json").write_text(
            json.dumps(doc)
        )

    def test_measured_logical_error_and_queue_seconds_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            self._write_cert(tmp_path)
            history = scan_results(tmp_path)
            entry = history["ibm_kingston"]
            self.assertIn(0.0123, entry.logical_errors)
            self.assertAlmostEqual(entry.queue_seconds[0], 330.0)

    def test_apply_history_sets_measured_logical_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            self._write_cert(tmp_path)
            history = scan_results(tmp_path)
            updated = apply_history(DEFAULT_FLEET, history)
            by_name = {p.name: p for p in updated}
            self.assertAlmostEqual(by_name["ibm_kingston"].measured_logical_error, 0.0123)
            self.assertAlmostEqual(by_name["ibm_kingston"].avg_queue_seconds, 330.0)

    def test_missing_timestamps_leaves_queue_seconds_none(self):
        self.assertIsNone(_queue_seconds_from_timestamps(None))
        self.assertIsNone(_queue_seconds_from_timestamps({"created": "bad"}))

    def test_route_reports_measured_logical_error_without_blending(self):
        # measured_logical_error is reported in plan.notes for visibility,
        # but must never override physical_error_rate/pipeline_kwargs: the
        # surface-code certificate's prediction and the empirical history
        # prior are deliberately kept separate (see budget_router.route).
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            self._write_cert(tmp_path, measured_success_deficit=0.0123)
            history = scan_results(tmp_path)
            fleet = tuple(
                p
                for p in apply_history(DEFAULT_FLEET, history)
                if p.kind == "sim" or p.name == "ibm_kingston"
            )
            qubo: dict[tuple[str, str], float] = {}
            hub = "hub"
            # 16 leaves -> criticality spread 8.5, above the accepted-policy
            # Tier1/Tier2 cutoff of 8.0, so this routes to Tier 2 on its own.
            for i in range(16):
                leaf = f"leaf{i}"
                qubo[(hub, leaf)] = 2.0
                qubo[(hub, hub)] = qubo.get((hub, hub), 0.0) - 1.0
                qubo[(leaf, leaf)] = -1.0
            plan = route(
                RouteRequest(qubo, fidelity_target=0.9, credit_budget=10.0),
                fleet=fleet,
            )
            self.assertEqual(plan.tier, Tier.HW_CERTIFIED)
            self.assertEqual(plan.backend.name, "ibm_kingston")
            self.assertTrue(
                any("measured_logical_error" in note for note in plan.notes)
            )
            self.assertNotEqual(plan.pipeline_kwargs["physical_error_rate"], 0.0123)


if __name__ == "__main__":
    unittest.main()
