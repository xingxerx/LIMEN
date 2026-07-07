"""Tests for limen.pipeline.run_pipeline_from_plan: dispatching a RoutePlan
in one call instead of every caller manually unpacking
plan.pipeline_kwargs into run_pipeline."""

import dataclasses
import unittest

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

    def test_cutting_plan_raises_not_implemented(self):
        plan = self._plan(Tier.HW_CERTIFIED)
        cut_plan = dataclasses.replace(plan, use_cutting=True, num_partitions=2)
        with self.assertRaises(NotImplementedError):
            run_pipeline_from_plan(self.QUBO, cut_plan)


if __name__ == "__main__":
    unittest.main()
