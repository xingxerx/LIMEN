"""Tests for limen.router.calibration: seeding BackendProfile.physical_error_rate
from cached live-calibration snapshots, and route() preferring that measured
value over RouteRequest's hardcoded 1e-3 default."""

import json
import pathlib
import tempfile
import unittest

from limen.router import DEFAULT_FLEET, RouteRequest, Tier, apply_calibration, route
from limen.router.calibration import scan_calibration


def _write_snapshot(tmp_path: pathlib.Path, backend: str, rate: float, when: str) -> None:
    doc = {
        "backend": backend,
        "generated_at": when,
        "avg_two_qubit_gate_error": rate * 1.2,
        "avg_readout_error": rate * 0.8,
        "physical_error_rate": rate,
    }
    (tmp_path / f"calibration_{backend}_{when.replace(':', '')}.json").write_text(
        json.dumps(doc)
    )


class TestScanCalibration(unittest.TestCase):

    def test_single_snapshot_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            _write_snapshot(tmp_path, "ibm_kingston", 0.0614, "2026-07-07T00:00:00+00:00")
            calibration = scan_calibration(tmp_path)
            self.assertAlmostEqual(calibration["ibm_kingston"], 0.0614)

    def test_latest_snapshot_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            _write_snapshot(tmp_path, "ibm_kingston", 0.001, "2026-07-01T00:00:00+00:00")
            _write_snapshot(tmp_path, "ibm_kingston", 0.0614, "2026-07-07T00:00:00+00:00")
            calibration = scan_calibration(tmp_path)
            self.assertAlmostEqual(calibration["ibm_kingston"], 0.0614)

    def test_unrecognized_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            (tmp_path / "calibration_bogus.json").write_text("not json")
            (tmp_path / "other.json").write_text(json.dumps({"unrelated": True}))
            calibration = scan_calibration(tmp_path)
            self.assertEqual(calibration, {})


class TestApplyCalibration(unittest.TestCase):

    def test_backend_without_calibration_is_unchanged(self):
        updated = apply_calibration(DEFAULT_FLEET, {})
        self.assertEqual(updated, DEFAULT_FLEET)

    def test_backend_with_calibration_sets_physical_error_rate(self):
        updated = apply_calibration(DEFAULT_FLEET, {"ibm_kingston": 0.0614})
        by_name = {p.name: p for p in updated}
        self.assertAlmostEqual(by_name["ibm_kingston"].physical_error_rate, 0.0614)
        self.assertIsNone(by_name["ibm_fez"].physical_error_rate)


class TestRoutePrefersCalibration(unittest.TestCase):

    def _star_qubo(self, n_leaves: int) -> dict[tuple[str, str], float]:
        qubo: dict[tuple[str, str], float] = {}
        hub = "hub"
        for i in range(n_leaves):
            leaf = f"leaf{i}"
            qubo[(hub, leaf)] = 2.0
            qubo[(hub, hub)] = qubo.get((hub, hub), 0.0) - 1.0
            qubo[(leaf, leaf)] = -1.0
        return qubo

    def test_uncalibrated_fleet_uses_request_default(self):
        plan = route(
            RouteRequest(self._star_qubo(5), fidelity_target=0.9, credit_budget=10.0)
        )
        self.assertEqual(plan.tier, Tier.HW_CERTIFIED)
        self.assertEqual(plan.pipeline_kwargs["physical_error_rate"], 1e-3)

    def test_calibrated_fleet_overrides_request_default(self):
        # Restrict to sim + ibm_kingston so backend selection is unambiguous
        # (all three IBM profiles are otherwise tied on max_qubits/cost).
        fleet = tuple(
            p
            for p in apply_calibration(DEFAULT_FLEET, {"ibm_kingston": 0.0614})
            if p.kind == "sim" or p.name == "ibm_kingston"
        )
        plan = route(
            RouteRequest(self._star_qubo(5), fidelity_target=0.9, credit_budget=10.0),
            fleet=fleet,
        )
        self.assertEqual(plan.tier, Tier.HW_CERTIFIED)
        self.assertEqual(plan.backend.name, "ibm_kingston")
        self.assertAlmostEqual(plan.pipeline_kwargs["physical_error_rate"], 0.0614)
        self.assertTrue(any("calibration" in note for note in plan.notes))


if __name__ == "__main__":
    unittest.main()
