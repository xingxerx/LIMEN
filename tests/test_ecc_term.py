# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""The shared surface-code ECC term and every place it now reaches.

run_pipeline and run_cut_route_request certify through one helper
(limen.pipeline._certify_ecc); a CuttingCertificate carries the same
aggregate/bound/round-trip fields as an EndToEndCertificate; and the
router ledger + route report accept either shape, so cut-circuit
hardware runs are witnessed and explained exactly like single-circuit
ones.
"""

from __future__ import annotations

import pathlib
import tempfile
import unittest

from limen.cutting.certificate import CuttingCertificate
from limen.pipeline import _certify_ecc, _record_route_outcome, run_pipeline
from limen.router import RouteRequest, RouterMemory, Tier, route
from limen.router.report import build_route_report


def cycle_maxcut(n: int) -> dict[tuple[str, str], float]:
    qubo: dict[tuple[str, str], float] = {}
    for i in range(n):
        a, b = f"x{i}", f"x{(i + 1) % n}"
        qubo[(a, b)] = 2.0
        qubo[(a, a)] = qubo.get((a, a), 0.0) - 1.0
        qubo[(b, b)] = qubo.get((b, b), 0.0) - 1.0
    return qubo


class TestCertifyEcc(unittest.TestCase):

    def test_disabled_returns_empty_term(self):
        term = _certify_ecc(4, 3, 1e-3, encode_logical=False)
        self.assertIsNone(term.logical_error_rate)
        self.assertIsNone(term.aggregate_logical_error_rate)
        self.assertEqual(term.notes, [])

    def test_missing_physical_rate_only_notes_the_skip(self):
        term = _certify_ecc(4, 3, None)
        self.assertIsNone(term.logical_error_rate)
        self.assertEqual(len(term.notes), 1)
        self.assertIn("skipped", term.notes[0])

    def test_aggregate_scales_with_logical_qubits(self):
        one = _certify_ecc(1, 3, 1e-3)
        four = _certify_ecc(4, 3, 1e-3)
        self.assertEqual(one.logical_error_rate, four.logical_error_rate)
        p = one.logical_error_rate
        self.assertAlmostEqual(one.aggregate_logical_error_rate, p)
        self.assertAlmostEqual(four.aggregate_logical_error_rate, 1 - (1 - p) ** 4)
        self.assertEqual(four.distance, 3)
        self.assertTrue(four.roundtrip_corrects_all_weight1)

    def test_bound_is_max_of_model_and_prior_never_a_blend(self):
        model = _certify_ecc(4, 3, 1e-3)
        agg = model.aggregate_logical_error_rate
        self.assertEqual(model.predicted_logical_error_bound, agg)

        worse = _certify_ecc(4, 3, 1e-3, measured_logical_error=agg * 10)
        self.assertEqual(worse.predicted_logical_error_bound, agg * 10)
        self.assertEqual(worse.aggregate_logical_error_rate, agg)

        better = _certify_ecc(4, 3, 1e-3, measured_logical_error=agg / 10)
        self.assertEqual(better.predicted_logical_error_bound, agg)

    def test_matches_run_pipeline_certificate(self):
        qubo = cycle_maxcut(4)
        cert = run_pipeline(qubo, physical_error_rate=1e-3, distance=3)
        term = _certify_ecc(4, 3, 1e-3)
        self.assertEqual(cert.logical_error_rate, term.logical_error_rate)
        self.assertEqual(cert.aggregate_logical_error_rate, term.aggregate_logical_error_rate)
        self.assertEqual(cert.predicted_logical_error_bound, term.predicted_logical_error_bound)


def _cutting_cert(term, *, physical_error_rate=1e-3, measured_prior=None) -> CuttingCertificate:
    return CuttingCertificate(
        solution={"x0": 1, "x1": 0, "x2": 1, "x3": 0},
        decoded_classical_energy=-4.0,
        reconstructed_expected_energy=-3.9,
        is_optimal=None,
        num_partitions=2,
        num_cuts=1,
        job_ids={},
        logical_error_rate=term.logical_error_rate,
        physical_error_rate=physical_error_rate,
        distance=term.distance,
        aggregate_logical_error_rate=term.aggregate_logical_error_rate,
        predicted_logical_error_bound=term.predicted_logical_error_bound,
        measured_logical_error_prior=measured_prior,
        roundtrip_corrects_all_weight1=term.roundtrip_corrects_all_weight1,
        notes=["synthetic"],
    )


class TestCuttingCertificateEcc(unittest.TestCase):

    def test_to_dict_carries_the_full_ecc_term(self):
        term = _certify_ecc(4, 3, 1e-3, measured_logical_error=0.5)
        d = _cutting_cert(term, measured_prior=0.5).to_dict()
        for key in (
            "logical_error_rate",
            "aggregate_logical_error_rate",
            "predicted_logical_error_bound",
            "measured_logical_error_prior",
            "roundtrip_corrects_all_weight1",
            "distance",
        ):
            self.assertIn(key, d)
        self.assertEqual(d["predicted_logical_error_bound"], 0.5)
        self.assertEqual(d["distance"], 3)
        self.assertTrue(d["roundtrip_corrects_all_weight1"])

    def test_new_fields_default_to_none(self):
        # Older constructors that only pass the original fields still work.
        cert = CuttingCertificate(
            solution={}, decoded_classical_energy=0.0, reconstructed_expected_energy=0.0,
            is_optimal=None, num_partitions=1, num_cuts=0, job_ids={},
            logical_error_rate=None, physical_error_rate=None, distance=None,
        )
        self.assertIsNone(cert.aggregate_logical_error_rate)
        self.assertIsNone(cert.predicted_logical_error_bound)


class TestCuttingCertificateReachesLedgerAndReport(unittest.TestCase):

    def setUp(self):
        self.request = RouteRequest(
            cycle_maxcut(4), fidelity_target=0.9, credit_budget=2.0,
            force_tier=Tier.HW_STANDARD, offline=True,
        )
        self.plan = route(self.request)
        self.term = _certify_ecc(4, 3, 1e-3)
        self.cert = _cutting_cert(self.term)

    def test_ledger_records_cutting_run_ecc_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = RouterMemory(pathlib.Path(tmp) / "router_memory.sqlite3")
            try:
                sha = _record_route_outcome(mem, self.plan, self.cert)
                self.assertIsNotNone(sha)
                backend = self.plan.backend.name
                logical = mem.stats(backend, "logical_error")
                physical = mem.stats(backend, "physical_error_rate")
                self.assertIsNotNone(logical)
                self.assertEqual(logical.n, 1)
                self.assertIsNotNone(physical)
                self.assertEqual(physical.n, 1)
            finally:
                mem.close()

    def test_report_builds_from_cutting_certificate(self):
        report = build_route_report(self.request, self.plan, self.cert)
        self.assertIsNone(report.success_probability)
        self.assertEqual(report.energy, -4.0)
        self.assertEqual(report.ecc_distance, 3)
        self.assertEqual(
            report.aggregate_logical_error_rate, self.term.aggregate_logical_error_rate
        )
        self.assertEqual(
            report.predicted_logical_error_bound, self.term.predicted_logical_error_bound
        )
        md = report.to_markdown()
        self.assertIn("distance-3 surface code", md)
        self.assertIn("Predicted logical error bound", md)
        self.assertNotIn("success probability", md)
        d = report.to_dict()
        self.assertEqual(d["ecc_distance"], 3)
        self.assertIn("predicted_logical_error_bound", d)

    def test_report_from_end_to_end_certificate_keeps_success_probability(self):
        cert = run_pipeline(self.request.qubo, **self.plan.pipeline_kwargs)
        report = build_route_report(self.request, self.plan, cert)
        self.assertIsNotNone(report.success_probability)
        self.assertEqual(report.ecc_distance, 3)
        self.assertIn("success probability", report.to_markdown())


if __name__ == "__main__":
    unittest.main()
