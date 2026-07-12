"""Tests for limen.router.informed_fleet: the single call that folds both
run history (limen.router.history) and live calibration
(limen.router.calibration) into a fleet, replacing the two-step chain every
caller was hand-wiring."""

import pathlib
import unittest

from limen.router import (
    DEFAULT_FLEET,
    apply_calibration,
    apply_history,
    informed_fleet,
    scan_calibration,
    scan_results,
)

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


class TestInformedFleet(unittest.TestCase):

    def test_matches_manual_composition(self):
        expected = apply_calibration(
            apply_history(DEFAULT_FLEET, scan_results(RESULTS_DIR)),
            scan_calibration(RESULTS_DIR),
        )
        actual = informed_fleet(RESULTS_DIR)
        self.assertEqual(actual, expected)

    def test_reflects_both_history_and_calibration(self):
        updated = informed_fleet(RESULTS_DIR)
        by_name = {p.name: p for p in updated}
        kingston = by_name["ibm_kingston"]
        self.assertIsNotNone(kingston.measured_logical_error)
        self.assertIsNotNone(kingston.physical_error_rate)

    def test_default_fleet_argument(self):
        self.assertEqual(
            informed_fleet(RESULTS_DIR),
            informed_fleet(RESULTS_DIR, DEFAULT_FLEET),
        )


if __name__ == "__main__":
    unittest.main()
