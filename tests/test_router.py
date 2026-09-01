"""Budget-router loopback tests: plan offline, execute on the simulator."""

import unittest

from limen.pipeline import EndToEndCertificate, run_pipeline
from limen.router import DEFAULT_FLEET, BackendProfile, RouteRequest, Tier, route
from limen.router.budget_router import CRITICALITY_SPREAD_THRESHOLD


def cycle_maxcut(n: int) -> dict[tuple[str, str], float]:
    """Max-Cut QUBO for an n-cycle: minimize sum over edges of 2*xi*xj - xi - xj.

    Every vertex touches exactly two edges, so the criticality spectrum
    is perfectly flat (spread 1.0).
    """
    qubo: dict[tuple[str, str], float] = {}
    for i in range(n):
        j = (i + 1) % n
        a, b = f"x{i:03d}", f"x{j:03d}"
        qubo[(a, b)] = qubo.get((a, b), 0.0) + 2.0
        qubo[(a, a)] = qubo.get((a, a), 0.0) - 1.0
        qubo[(b, b)] = qubo.get((b, b), 0.0) - 1.0
    return qubo


def star_maxcut(n_leaves: int) -> dict[tuple[str, str], float]:
    """Max-Cut QUBO for a star graph: the hub dominates the criticality
    spectrum (heavy tail), the signal that routes to Tier 2."""
    qubo: dict[tuple[str, str], float] = {}
    hub = "hub"
    for i in range(n_leaves):
        leaf = f"leaf{i}"
        qubo[(hub, leaf)] = 2.0
        qubo[(hub, hub)] = qubo.get((hub, hub), 0.0) - 1.0
        qubo[(leaf, leaf)] = -1.0
    return qubo


class TestRoutingDecisions(unittest.TestCase):

    def test_zero_budget_routes_to_sim(self):
        plan = route(RouteRequest(cycle_maxcut(6), fidelity_target=0.9, credit_budget=0))
        self.assertEqual(plan.tier, Tier.SIM)
        self.assertEqual(plan.backend.kind, "sim")
        self.assertEqual(plan.backend.cost_per_shot, 0.0)

    def test_low_fidelity_target_routes_to_sim(self):
        plan = route(RouteRequest(cycle_maxcut(6), fidelity_target=0.3, credit_budget=10.0))
        self.assertEqual(plan.tier, Tier.SIM)

    def test_flat_spectrum_routes_to_tier1(self):
        # Every cycle vertex has identical criticality -> spread exactly 1.0.
        plan = route(RouteRequest(cycle_maxcut(6), fidelity_target=0.9, credit_budget=10.0))
        self.assertAlmostEqual(plan.criticality_spread, 1.0)
        self.assertEqual(plan.tier, Tier.HW_STANDARD)
        self.assertNotEqual(plan.backend.kind, "sim")
        self.assertFalse(plan.patch_assignments)

    def test_heavy_tailed_spectrum_routes_to_tier2(self):
        # star(n)'s spread is (n+1)/2; 20 leaves -> 10.5, above the 10.25
        # cutoff set by accepted policy 2026-09-01-raise-criticality-threshold-10p25.
        # Smaller stars are only moderately skewed now and belong in Tier 1.
        plan = route(RouteRequest(star_maxcut(20), fidelity_target=0.9, credit_budget=10.0))
        self.assertGreaterEqual(plan.criticality_spread, CRITICALITY_SPREAD_THRESHOLD)
        self.assertEqual(plan.tier, Tier.HW_CERTIFIED)
        self.assertTrue(plan.backend.validated)
        self.assertEqual(plan.ecc_distance, 3)
        self.assertTrue(plan.patch_assignments)
        # The hub is the most critical variable and must be patched first.
        order = sorted({v for pair in star_maxcut(20) for v in pair})
        self.assertEqual(order[plan.patch_assignments[0].logical_var], "hub")

    def test_distance_scales_with_fidelity_target(self):
        # star(20) routes to Tier 2 on its own (spread 10.5 >= the 10.25
        # accepted-policy cutoff); a d=5 surface code is selected at
        # fidelity targets >= 0.99.
        req = RouteRequest(
            star_maxcut(20), fidelity_target=0.99, credit_budget=10.0
        )
        self.assertEqual(route(req).ecc_distance, 5)

    def test_shots_derived_from_budget_and_clamped(self):
        qubo = cycle_maxcut(6)
        # 1.0 credit at 0.002/shot -> 500 shots.
        plan = route(RouteRequest(qubo, fidelity_target=0.9, credit_budget=1.0))
        self.assertEqual(plan.shots, 500)
        # Huge budget clamps at MAX_SHOTS.
        plan = route(RouteRequest(qubo, fidelity_target=0.9, credit_budget=1e9))
        self.assertEqual(plan.shots, 100_000)
        # Tiny budget clamps at MIN_SHOTS.
        plan = route(RouteRequest(qubo, fidelity_target=0.9, credit_budget=0.01))
        self.assertEqual(plan.shots, 100)

    def test_invalid_requests_raise(self):
        with self.assertRaises(ValueError):
            RouteRequest({}, fidelity_target=0.9, credit_budget=1.0)
        with self.assertRaises(ValueError):
            RouteRequest(cycle_maxcut(4), fidelity_target=1.5, credit_budget=1.0)
        with self.assertRaises(ValueError):
            RouteRequest(cycle_maxcut(4), fidelity_target=0.9, credit_budget=-1.0)
        with self.assertRaises(ValueError):
            BackendProfile("bogus", "quantum-stuff", 10, 0.0)


class TestLoopback(unittest.TestCase):
    """Force each tier on the same QUBO and run the resulting plans offline."""

    QUBO = cycle_maxcut(6)

    def _request(self, tier: Tier) -> RouteRequest:
        return RouteRequest(
            self.QUBO,
            fidelity_target=0.9,
            credit_budget=2.0,
            force_tier=tier,
            offline=True,
        )

    def test_plans_are_deterministic(self):
        for tier in Tier:
            first = route(self._request(tier))
            second = route(self._request(tier))
            self.assertEqual(first, second)
            self.assertEqual(first.to_dict(), second.to_dict())

    def test_tier2_plan_has_patch_assignments(self):
        plan = route(self._request(Tier.HW_CERTIFIED))
        self.assertEqual(plan.ecc_distance, 3)
        self.assertTrue(plan.patch_assignments)
        # 6 logical vars on a 156q device: every variable gets a patch.
        self.assertEqual(len(plan.patch_assignments), plan.n_vars)
        self.assertEqual(
            plan.physical_qubit_budget, plan.backend.max_qubits - plan.n_vars
        )
        # Patches are contiguous, non-overlapping blocks of d^2 qubits.
        for a in plan.patch_assignments:
            self.assertEqual(a.physical_end - a.physical_start, 9)

    def test_each_tier_plan_runs_and_certifies_optimal(self):
        for tier in Tier:
            with self.subTest(tier=tier.name):
                plan = route(self._request(tier))
                self.assertEqual(plan.tier, tier)
                self.assertEqual(plan.pipeline_kwargs["backend"], "statevector")
                self.assertEqual(
                    plan.pipeline_kwargs["qpu_backend_name"], "aer_simulator"
                )
                cert = run_pipeline(self.QUBO, **plan.pipeline_kwargs)
                self.assertIsInstance(cert, EndToEndCertificate)
                self.assertTrue(cert.is_optimal)
                self.assertAlmostEqual(cert.energy, -6.0)
                if tier == Tier.HW_CERTIFIED:
                    self.assertIsNotNone(cert.logical_error_rate)
                    self.assertEqual(cert.distance, plan.ecc_distance)
                else:
                    self.assertIsNone(cert.logical_error_rate)


class TestCutting(unittest.TestCase):

    def test_oversized_qubo_triggers_cutting_plan_only(self):
        # 200 vars against the 156q IBM profile: ceil(200/156) = 2 partitions.
        plan = route(
            RouteRequest(
                cycle_maxcut(200),
                fidelity_target=0.9,
                credit_budget=2.0,
                force_tier=Tier.HW_STANDARD,
            )
        )
        self.assertEqual(plan.n_vars, 200)
        self.assertEqual(plan.backend.max_qubits, 156)
        self.assertTrue(plan.use_cutting)
        self.assertEqual(plan.num_partitions, 2)

    def test_small_qubo_does_not_cut(self):
        plan = route(
            RouteRequest(
                cycle_maxcut(6),
                fidelity_target=0.9,
                credit_budget=2.0,
                force_tier=Tier.HW_STANDARD,
            )
        )
        self.assertFalse(plan.use_cutting)
        self.assertIsNone(plan.num_partitions)


class TestFleetSeed(unittest.TestCase):

    def test_default_fleet_matches_fleet_certificate(self):
        by_name = {p.name: p for p in DEFAULT_FLEET}
        for ibm in ("ibm_kingston", "ibm_fez", "ibm_marrakesh"):
            self.assertEqual(by_name[ibm].max_qubits, 156)
            self.assertTrue(by_name[ibm].validated)
        self.assertEqual(by_name["ionq:forte-1"].max_qubits, 36)
        self.assertIn("statevector", by_name)


class TestSubstrateAffinityTiebreak(unittest.TestCase):
    """substrate_affinity must only ever decide between backends already
    tied on cost/capacity/validation -- never override those criteria."""

    def test_no_affinity_falls_back_to_name_unchanged(self):
        # Two backends identical except name: default (empty) affinity
        # must reproduce the pre-existing name tiebreak.
        fleet = (
            BackendProfile("alpha", "ibm", 156, 0.002, validated=True),
            BackendProfile("beta", "ibm", 156, 0.002, validated=True),
        )
        plan = route(
            RouteRequest(
                cycle_maxcut(6),
                fidelity_target=0.9,
                credit_budget=2.0,
                force_tier=Tier.HW_STANDARD,
            ),
            fleet=fleet,
        )
        self.assertEqual(plan.backend.name, "alpha")

    def test_affinity_breaks_a_genuine_tie(self):
        # Same tie as above, but "beta" prefers high frustration_index --
        # cycle_maxcut has a nonzero, deterministic frustration_index, so
        # a positive affinity weight must swing the tie to "beta".
        fleet = (
            BackendProfile("alpha", "ibm", 156, 0.002, validated=True),
            BackendProfile(
                "beta", "ibm", 156, 0.002, validated=True,
                substrate_affinity={"frustration_index": 1.0},
            ),
        )
        plan = route(
            RouteRequest(
                cycle_maxcut(6),
                fidelity_target=0.9,
                credit_budget=2.0,
                force_tier=Tier.HW_STANDARD,
            ),
            fleet=fleet,
        )
        self.assertEqual(plan.backend.name, "beta")

    def test_affinity_never_overrides_cost(self):
        # "beta" scores far better on substrate affinity but costs more --
        # cost must still win; affinity must not override it.
        fleet = (
            BackendProfile("alpha", "ibm", 156, 0.002, validated=True),
            BackendProfile(
                "beta", "ibm", 156, 0.01, validated=True,
                substrate_affinity={"frustration_index": 1000.0},
            ),
        )
        plan = route(
            RouteRequest(
                cycle_maxcut(6),
                fidelity_target=0.9,
                credit_budget=2.0,
                force_tier=Tier.HW_STANDARD,
            ),
            fleet=fleet,
        )
        self.assertEqual(plan.backend.name, "alpha")

    def test_affinity_never_overrides_validation_for_tier2(self):
        # "beta" scores far better on substrate affinity but is unvalidated
        # -- Tier 2 must still exclude it entirely.
        fleet = (
            BackendProfile("alpha", "ibm", 156, 0.002, validated=True),
            BackendProfile(
                "beta", "ibm", 156, 0.002, validated=False,
                substrate_affinity={"frustration_index": 1000.0},
            ),
        )
        plan = route(
            RouteRequest(
                star_maxcut(6),
                fidelity_target=0.99,
                credit_budget=2.0,
                force_tier=Tier.HW_CERTIFIED,
            ),
            fleet=fleet,
        )
        self.assertEqual(plan.backend.name, "alpha")


if __name__ == "__main__":
    unittest.main()
