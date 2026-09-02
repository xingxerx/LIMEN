"""Tests for the measured-logical-error prior in the end-to-end certificate:
run_pipeline's measured_logical_error kwarg, the max(model, prior) bound
rule, and route() forwarding a backend's history-derived prior into
pipeline_kwargs."""

import unittest

from limen.pipeline import run_pipeline
from limen.router import DEFAULT_FLEET, RouteRequest, Tier, apply_history, route
from limen.router.history import BackendHistory


def cycle_maxcut(n: int) -> dict[tuple[str, str], float]:
    qubo: dict[tuple[str, str], float] = {}
    for i in range(n):
        j = (i + 1) % n
        a, b = f"x{i}", f"x{j}"
        qubo[(a, b)] = qubo.get((a, b), 0.0) + 2.0
        qubo[(a, a)] = qubo.get((a, a), 0.0) - 1.0
        qubo[(b, b)] = qubo.get((b, b), 0.0) - 1.0
    return qubo


def star_maxcut(n_leaves: int) -> dict[tuple[str, str], float]:
    qubo: dict[tuple[str, str], float] = {}
    for i in range(n_leaves):
        leaf = f"leaf{i}"
        qubo[("hub", leaf)] = 2.0
        qubo[("hub", "hub")] = qubo.get(("hub", "hub"), 0.0) - 1.0
        qubo[(leaf, leaf)] = -1.0
    return qubo


class TestMeasuredPriorBound(unittest.TestCase):
    """Blend rule: never averaged. aggregate_logical_error_rate is always
    the surface-code model's own prediction; predicted_logical_error_bound
    is max(model, prior)."""

    QUBO = cycle_maxcut(4)

    def _run(self, **kwargs):
        return run_pipeline(
            self.QUBO, physical_error_rate=1e-3, distance=3, **kwargs
        )

    def test_no_prior_bound_equals_model(self):
        cert = self._run()
        self.assertIsNone(cert.measured_logical_error_prior)
        self.assertEqual(
            cert.predicted_logical_error_bound, cert.aggregate_logical_error_rate
        )

    def test_prior_above_model_becomes_the_bound(self):
        cert = self._run(measured_logical_error=0.5)
        self.assertEqual(cert.measured_logical_error_prior, 0.5)
        self.assertEqual(cert.predicted_logical_error_bound, 0.5)
        # The model's own prediction is untouched by the prior.
        baseline = self._run()
        self.assertEqual(
            cert.aggregate_logical_error_rate, baseline.aggregate_logical_error_rate
        )
        self.assertTrue(any("exceeds" in note for note in cert.notes))

    def test_prior_below_model_keeps_model_as_bound(self):
        cert = self._run(measured_logical_error=1e-12)
        self.assertEqual(cert.measured_logical_error_prior, 1e-12)
        self.assertEqual(
            cert.predicted_logical_error_bound, cert.aggregate_logical_error_rate
        )
        self.assertTrue(any("within" in note for note in cert.notes))

    def test_no_ecc_no_bound(self):
        cert = run_pipeline(
            self.QUBO, encode_logical=False, measured_logical_error=0.5
        )
        self.assertIsNone(cert.predicted_logical_error_bound)
        self.assertEqual(cert.measured_logical_error_prior, 0.5)

    def test_to_dict_carries_both_fields(self):
        doc = self._run(measured_logical_error=0.5).to_dict()
        self.assertEqual(doc["measured_logical_error_prior"], 0.5)
        self.assertEqual(doc["predicted_logical_error_bound"], 0.5)


class TestRouteForwardsPrior(unittest.TestCase):

    def test_history_prior_lands_in_pipeline_kwargs(self):
        history = {
            "ibm_kingston": BackendHistory(
                name="ibm_kingston",
                seconds_per_shot=[0.02],
                queue_seconds=[330.0],
                logical_errors=[0.0613],
            )
        }
        fleet = tuple(
            p
            for p in apply_history(DEFAULT_FLEET, history)
            if p.kind == "sim" or p.name == "ibm_kingston"
        )
        plan = route(
            # star(20)'s spread is 10.5, above the accepted-policy cutoff of
            # 10.25, so this routes to Tier 2 on its own (a 4-leaf star at
            # spread 2.5 no longer does).
            RouteRequest(star_maxcut(20), fidelity_target=0.9, credit_budget=10.0),
            fleet=fleet,
        )
        self.assertEqual(plan.tier, Tier.HW_CERTIFIED)
        self.assertAlmostEqual(
            plan.pipeline_kwargs["measured_logical_error"], 0.0613
        )

    def test_no_history_no_kwarg(self):
        plan = route(
            RouteRequest(star_maxcut(20), fidelity_target=0.9, credit_budget=10.0)
        )
        self.assertEqual(plan.tier, Tier.HW_CERTIFIED)
        self.assertNotIn("measured_logical_error", plan.pipeline_kwargs)


if __name__ == "__main__":
    unittest.main()
