"""Tests for limen.pipeline.run_pipeline_from_plan: dispatching a RoutePlan
in one call instead of every caller manually unpacking
plan.pipeline_kwargs into run_pipeline."""

import dataclasses
import unittest

import pytest

from limen.pipeline import EndToEndCertificate, run_pipeline, run_pipeline_from_plan
from limen.router import RouteRequest, Tier, route


def cycle_maxcut(n: int) -> dict[tuple[str, str], float]:
    qubo: dict[tuple[str, str], float] = {}
    for i in range(n):
        j = (i + 1) % n
        a, b = f"x{i}", f"x{j}"
        qubo[(a, b)] = qubo.get((a, b), 0.0) + 2.0
        qubo[(a, a)] = qubo.get((a, a), 0.0) - 1.0
        qubo[(b, b)] = qubo.get((b, b), 0.0) - 1.0
    return qubo


class TestRunPipelineFromPlan(unittest.TestCase):

    QUBO = cycle_maxcut(4)

    def _plan(self, tier: Tier):
        request = RouteRequest(
            self.QUBO,
            fidelity_target=0.9,
            credit_budget=2.0,
            force_tier=tier,
            offline=True,
        )
        return route(request)

    def test_matches_manual_dispatch(self):
        for tier in Tier:
            plan = self._plan(tier)
            expected = run_pipeline(self.QUBO, **plan.pipeline_kwargs)
            actual = run_pipeline_from_plan(self.QUBO, plan)
            self.assertIsInstance(actual, EndToEndCertificate)
            self.assertEqual(actual.solution, expected.solution)

    def test_cutting_plan_requires_credentials_or_offline(self):
        # use_cutting=True dispatches through run_cut_route_request, which
        # (like the rest of the QPU path) refuses to guess credentials.
        plan = self._plan(Tier.HW_CERTIFIED)
        cut_plan = dataclasses.replace(plan, use_cutting=True, num_partitions=2)
        with self.assertRaises(ValueError):
            run_pipeline_from_plan(self.QUBO, cut_plan)

    def test_cutting_plan_offline_returns_cutting_certificate(self):
        pytest.importorskip(
            "qiskit_addon_cutting", reason="qiskit-addon-cutting not installed"
        )
        pytest.importorskip("qiskit_aer", reason="qiskit-aer not installed")
        pytest.importorskip("limen_core", reason="limen_core Rust extension not installed")
        from limen.cutting.certificate import CuttingCertificate
        from limen.cutting.qubo_bridge import classical_energy

        # A 4-var QUBO with backend.max_qubits monkeypatched down to 2 so
        # cutting is actually exercised, without needing a 150+ variable
        # QUBO to trigger it naturally.
        plan = self._plan(Tier.HW_STANDARD)
        small_backend = dataclasses.replace(plan.backend, max_qubits=2)
        cut_plan = dataclasses.replace(
            plan, use_cutting=True, num_partitions=2, backend=small_backend
        )
        cert = run_pipeline_from_plan(self.QUBO, cut_plan, cut_offline=True)

        self.assertIsInstance(cert, CuttingCertificate)
        self.assertIsNone(cert.is_optimal)
        self.assertEqual(set(cert.solution.keys()), {"x0", "x1", "x2", "x3"})
        self.assertEqual(
            cert.decoded_classical_energy,
            classical_energy(self.QUBO, cert.solution),
        )


if __name__ == "__main__":
    unittest.main()
