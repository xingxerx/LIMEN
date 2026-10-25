"""Local GPU statevector tier 0: VRAM-sized cap, prepend-only routing."""

import importlib.util
import os
import unittest
from unittest import mock

from limen.router import DEFAULT_FLEET, RouteRequest, Tier, route
from limen.router.budget_router import (
    LOCAL_GPU_BACKEND_NAME,
    LOCAL_GPU_ENV,
    gpu_qubit_cap,
    local_gpu_enabled,
    route_local_gpu_first,
)


def chain(n: int) -> dict[tuple[str, str], float]:
    return {(f"x{i}", f"x{i + 1}"): 1.0 for i in range(n - 1)}


class TestGpuQubitCap(unittest.TestCase):
    def test_no_gpu(self):
        self.assertEqual(gpu_qubit_cap(None), 0)
        self.assertEqual(gpu_qubit_cap(0), 0)
        self.assertEqual(gpu_qubit_cap(-5), 0)

    def test_formula_with_20pct_reserve_and_headroom(self):
        # usable = 0.8 * free; n = floor(log2(usable / 8)); headroom 0 by default
        self.assertEqual(gpu_qubit_cap(1024), 26)   # 1 GiB  -> 26
        self.assertEqual(gpu_qubit_cap(8192), 29)   # 8 GiB  -> 29
        self.assertEqual(gpu_qubit_cap(14574), 30)  # idle RTX 5080 -> 30 (cap reachable)
        self.assertEqual(gpu_qubit_cap(16384), 30)  # capped
        self.assertEqual(gpu_qubit_cap(10**9), 30)  # capped

    def test_headroom_argument(self):
        self.assertEqual(gpu_qubit_cap(14574, headroom_qubits=1), 29)

    def test_tenant_holding_vram_lowers_cap(self):
        # LM Studio holding ~7.4 GB on a 16 GB card leaves ~8.2 GB free.
        self.assertEqual(gpu_qubit_cap(8400), 29)
        self.assertEqual(gpu_qubit_cap(1), 16)

    def test_cap_argument(self):
        self.assertEqual(gpu_qubit_cap(10**9, cap=24), 24)


class TestLocalGpuEnabled(unittest.TestCase):
    def test_env_values(self):
        for val, expected in [("1", True), ("true", True), ("YES", True), ("on", True),
                              ("0", False), ("", False), ("off", False)]:
            with mock.patch.dict(os.environ, {LOCAL_GPU_ENV: val}):
                self.assertEqual(local_gpu_enabled(), expected, val)
        env = {k: v for k, v in os.environ.items() if k != LOCAL_GPU_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(local_gpu_enabled())


class TestRouteLocalGpuFirst(unittest.TestCase):
    def test_fits_routes_to_gpu_tier0_end_to_end_on_aer(self):
        req = RouteRequest(chain(6), fidelity_target=0.9, credit_budget=10.0)
        plan = route_local_gpu_first(req, DEFAULT_FLEET, free_vram_mib=lambda: 8192)
        self.assertEqual(plan.tier, Tier.SIM)
        self.assertEqual(plan.backend.name, LOCAL_GPU_BACKEND_NAME)
        self.assertEqual(plan.backend.max_qubits, 29)  # 8 GiB free -> 29
        # Both the grid search and the final circuit must go through Aer on
        # the GPU backend name; "aer_simulator" would silently be CPU.
        self.assertEqual(plan.pipeline_kwargs["backend"], "aer")
        self.assertEqual(plan.pipeline_kwargs["qpu_backend_name"], LOCAL_GPU_BACKEND_NAME)
        self.assertFalse(plan.use_cutting)
        self.assertTrue(any("tier 0 aer_gpu" in n for n in plan.notes))

    def test_no_gpu_falls_through_to_plain_route(self):
        req = RouteRequest(chain(6), fidelity_target=0.9, credit_budget=10.0)
        base = route(req, DEFAULT_FLEET)
        plan = route_local_gpu_first(req, DEFAULT_FLEET, free_vram_mib=lambda: None)
        self.assertEqual(plan.backend, base.backend)
        self.assertEqual(plan.tier, base.tier)
        self.assertEqual(plan.pipeline_kwargs, base.pipeline_kwargs)
        self.assertTrue(any("fallthrough" in n for n in plan.notes))

    def test_zero_budget_without_gpu_stays_on_free_sim(self):
        # The original diff forced ibm_fez on fallthrough, which would have
        # sent a zero-credit request to hardware. Fallthrough must be the
        # plain route() decision.
        req = RouteRequest(chain(6), fidelity_target=0.9, credit_budget=0.0)
        plan = route_local_gpu_first(req, DEFAULT_FLEET, free_vram_mib=None)
        self.assertEqual(plan.tier, Tier.SIM)
        self.assertEqual(plan.backend.kind, "sim")

    def test_above_cap_falls_through(self):
        req = RouteRequest(chain(20), fidelity_target=0.9, credit_budget=10.0)
        base = route(req, DEFAULT_FLEET)
        plan = route_local_gpu_first(req, DEFAULT_FLEET, free_vram_mib=1)  # cap 16
        self.assertEqual(plan.backend, base.backend)
        self.assertTrue(any("exceeds cap" in n for n in plan.notes))

    def test_pinned_hardware_tier_is_kept(self):
        for tier in (Tier.HW_STANDARD, Tier.HW_CERTIFIED):
            req = RouteRequest(chain(6), fidelity_target=0.9, credit_budget=10.0, force_tier=tier)
            base = route(req, DEFAULT_FLEET)
            plan = route_local_gpu_first(req, DEFAULT_FLEET, free_vram_mib=8192)
            self.assertEqual(plan.tier, tier)
            self.assertEqual(plan.backend, base.backend)
            self.assertTrue(any("hardware tier kept" in n for n in plan.notes))

    def test_pinned_sim_tier_takes_gpu(self):
        req = RouteRequest(chain(6), fidelity_target=0.9, credit_budget=10.0, force_tier=Tier.SIM)
        plan = route_local_gpu_first(req, DEFAULT_FLEET, free_vram_mib=8192)
        self.assertEqual(plan.backend.name, LOCAL_GPU_BACKEND_NAME)

    def test_plan_is_serializable(self):
        req = RouteRequest(chain(4), fidelity_target=0.9, credit_budget=1.0)
        plan = route_local_gpu_first(req, DEFAULT_FLEET, free_vram_mib=8192)
        d = plan.to_dict()
        self.assertEqual(d["backend"]["name"], LOCAL_GPU_BACKEND_NAME)


@unittest.skipUnless(
    importlib.util.find_spec("qiskit") is not None
    and importlib.util.find_spec("qiskit_aer") is not None,
    "qiskit and qiskit-aer required",
)
class TestAerGpuBackendName(unittest.TestCase):
    def test_run_circuit_aer_gpu_runs_or_falls_back(self):
        from limen.gates.ir import CircuitIR, GateInstruction
        from limen.gates.qiskit_exec import GPU_BACKEND_NAME, run_circuit

        self.assertEqual(GPU_BACKEND_NAME, LOCAL_GPU_BACKEND_NAME)
        bell = CircuitIR(
            n_qubits=2,
            instructions=[GateInstruction("h", [0]), GateInstruction("cx", [0, 1])],
        )
        result = run_circuit(bell, backend_name=GPU_BACKEND_NAME, shots=500)
        self.assertEqual(sum(result.counts.values()), 500)
        self.assertIn(result.metadata["device"], {"GPU", "CPU"})
        if result.metadata["device"] == "CPU":
            self.assertIn("gpu_fallback_reason", result.metadata)
        # A Bell pair only ever yields 00 or 11.
        self.assertTrue(set(result.counts) <= {"00", "11"}, result.counts)


if __name__ == "__main__":
    unittest.main()
