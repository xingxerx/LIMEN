"""Tests for limen.pipeline.run_route_request: RouteRequest in, certified
answer out, with no manual plan unpacking. QPU submission itself is
exercised by the examples/router_tier2_kingston.py hardware workflow, not
here — these tests cover the offline dispatch path and the guard rails in
front of spending credits."""

import os
import pathlib
import unittest
from unittest import mock

from limen.pipeline import run_pipeline_from_plan, run_route_request
from limen.router import DEFAULT_FLEET, RouteRequest, Tier, informed_fleet, route

RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"


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


class TestOfflineDispatch(unittest.TestCase):

    def test_matches_manual_route_and_dispatch(self):
        for tier in Tier:
            request = RouteRequest(
                cycle_maxcut(4),
                fidelity_target=0.9,
                credit_budget=2.0,
                force_tier=tier,
                offline=True,
            )
            manual_plan = route(request, fleet=informed_fleet(RESULTS_DIR))
            expected = run_pipeline_from_plan(request.qubo, manual_plan)
            actual = run_route_request(request, results_dir=RESULTS_DIR)
            self.assertEqual(actual.to_dict(), expected.to_dict())

    def test_default_fleet_when_no_results_dir(self):
        request = RouteRequest(
            cycle_maxcut(4), fidelity_target=0.9, credit_budget=0.0
        )
        manual_plan = route(request, fleet=DEFAULT_FLEET)
        expected = run_pipeline_from_plan(request.qubo, manual_plan)
        actual = run_route_request(request)
        self.assertEqual(actual.to_dict(), expected.to_dict())

    def test_explicit_fleet_override(self):
        request = RouteRequest(
            cycle_maxcut(4), fidelity_target=0.9, credit_budget=0.0
        )
        actual = run_route_request(request, fleet=DEFAULT_FLEET)
        self.assertTrue(actual.is_optimal)


class TestMemoryLoopClosure(unittest.TestCase):
    """A completed run must land in the ledger before run_route_request
    returns -- not merely be readable by some future call that rescans
    results_dir. See _record_route_outcome in limen/pipeline.py."""

    def test_outcome_and_certificate_recorded_within_one_call(self):
        import tempfile

        from limen.router import RouterMemory

        request = RouteRequest(
            cycle_maxcut(4), fidelity_target=0.9, credit_budget=0.0
        )
        with tempfile.TemporaryDirectory() as tmp:
            db_path = pathlib.Path(tmp) / "router_memory.sqlite3"
            mem = RouterMemory(db_path)
            try:
                cert = run_route_request(request, fleet=DEFAULT_FLEET, memory=mem)

                entries = list(mem.certificates())
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0].payload["energy"], cert.energy)
                self.assertTrue(mem.verify_ledger())

                # physical_error_rate is the metric this offline plan can
                # actually produce; assert at least one sample landed
                # rather than asserting on a metric this fixture has no
                # opinion about.
                stats = mem.stats(DEFAULT_FLEET[0].name, "physical_error_rate")
                if cert.physical_error_rate is not None:
                    self.assertIsNotNone(stats)
            finally:
                mem.close()

    def test_no_memory_no_write_and_no_error(self):
        # memory=None (the default) must leave today's behavior exactly
        # unchanged -- no ledger opened, no write attempted.
        request = RouteRequest(
            cycle_maxcut(4), fidelity_target=0.9, credit_budget=0.0
        )
        actual = run_route_request(request, fleet=DEFAULT_FLEET)
        self.assertTrue(actual.is_optimal)


class TestPeerAutoDiscovery(unittest.TestCase):

    def test_known_peers_from_env_used_when_server_addresses_omitted(self):
        request = RouteRequest(cycle_maxcut(4), fidelity_target=0.9, credit_budget=0.0)
        with mock.patch.dict(
            os.environ,
            {"LIMEN_NODE_ID": "test-node", "LIMEN_KNOWN_PEERS": "peer-a:50051,peer-b:50051"},
        ):
            with mock.patch("limen.pipeline._distributed_compile") as compile_mock:
                compile_mock.return_value = ({"num_partitions": 2}, ["note"])
                run_route_request(request, fleet=DEFAULT_FLEET)
        compile_mock.assert_called_once()
        called_addresses = compile_mock.call_args[0][1]
        self.assertEqual(list(called_addresses), ["peer-a:50051", "peer-b:50051"])

    def test_no_node_id_stays_local(self):
        request = RouteRequest(cycle_maxcut(4), fidelity_target=0.9, credit_budget=0.0)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LIMEN_NODE_ID", None)
            with mock.patch("limen.pipeline._distributed_compile") as compile_mock:
                actual = run_route_request(request, fleet=DEFAULT_FLEET)
        compile_mock.assert_not_called()
        self.assertIsNone(actual.distributed_compilation)


class TestQpuGuardRails(unittest.TestCase):

    def test_missing_credentials_raise_before_submitting(self):
        # Star Max-Cut at fidelity 0.9 with budget routes to Tier 2 on an
        # IBM backend -> pipeline_kwargs backend "qpu". With no token
        # available this must fail fast, before any network call.
        request = RouteRequest(
            star_maxcut(4), fidelity_target=0.9, credit_budget=10.0
        )
        with mock.patch.dict(
            os.environ, {"IBM_QUANTUM_TOKEN": "", "IBM_QUANTUM_CRN": ""}
        ):
            with self.assertRaises(ValueError):
                run_route_request(request, fleet=DEFAULT_FLEET)


if __name__ == "__main__":
    unittest.main()
