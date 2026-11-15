"""limen.study: the tier-0 measurements, run at toy sizes on the CPU."""

import importlib.util
import pathlib
import tempfile
import unittest

from limen.frontends.maxcut import build_max_cut_qubo, count_cut_edges, ring_edges
from limen.study import StudyContext, context
from limen.study._common import baseline_backend, z_marginals
from limen.study.cutting_study import random_field_ring

HAS_AER = (
    importlib.util.find_spec("qiskit") is not None
    and importlib.util.find_spec("qiskit_aer") is not None
)
HAS_CUTTING = HAS_AER and importlib.util.find_spec("qiskit_addon_cutting") is not None


def _cpu_ctx(results_dir=None) -> StudyContext:
    return StudyContext(
        results_dir=pathlib.Path(results_dir) if results_dir else None,
        local_gpu=False,
        free_vram_mib=None,
        gpu_cap=0,
    )


class TestCommon(unittest.TestCase):
    def test_context_without_gpu(self):
        ctx = context(None, local_gpu=False)
        self.assertFalse(ctx.local_gpu)
        self.assertEqual(ctx.gpu_cap, 0)
        self.assertEqual(ctx.exact_max_qubits, 26)

    def test_baseline_backend_by_width(self):
        cpu = _cpu_ctx()
        self.assertEqual(baseline_backend(10, cpu), ("statevector", "aer_simulator"))
        self.assertEqual(baseline_backend(20, cpu), ("aer", "aer_simulator"))
        with self.assertRaises(ValueError):
            baseline_backend(27, cpu)
        gpu = StudyContext(None, True, 12000, 30)
        self.assertEqual(baseline_backend(10, gpu), ("aer", "aer_gpu"))
        self.assertEqual(baseline_backend(30, gpu), ("aer", "aer_gpu"))
        with self.assertRaises(ValueError):
            baseline_backend(31, gpu)

    def test_z_marginals(self):
        # qubit-0-first strings: "10" means qubit 0 = 1
        self.assertEqual(z_marginals({"10": 1.0}, 2), {0: -1.0, 1: 1.0})
        self.assertEqual(z_marginals({"00": 0.5, "11": 0.5}, 2), {0: 0.0, 1: 0.0})

    def test_maxcut_qubo_energy_is_minus_cut(self):
        from limen.cutting.qubo_bridge import classical_energy

        edges = ring_edges(6)
        qubo = build_max_cut_qubo(edges)
        x = {f"v{i}": i % 2 for i in range(6)}
        self.assertEqual(classical_energy(qubo, x), -count_cut_edges(x, edges))
        self.assertEqual(count_cut_edges(x, edges), 6)

    def test_random_field_ring_breaks_symmetry(self):
        qubo = random_field_ring(6, 0)
        diag = [w for (u, v), w in qubo.items() if u == v]
        self.assertEqual(len(diag), 6)
        self.assertTrue(all(0.5 <= abs(w) <= 1.5 for w in diag))


class TestBench(unittest.TestCase):
    def test_tsp_two_cities_optimal_on_statevector(self):
        from limen.study import bench_tsp

        row = bench_tsp(2, ctx=_cpu_ctx())
        self.assertEqual(row["n_vars"], 4)
        self.assertTrue(row["feasible"])
        self.assertEqual(row["approx_ratio"], 1.0)
        self.assertEqual(row["tier"], 0)

    def test_tsp_three_cities_records_feasibility_honestly(self):
        # p=1 QAOA with a single shared angle pair does not reliably land in
        # the one-hot feasible subspace at 9 variables; the row must say so
        # (feasible False, approx_ratio None) rather than invent a tour.
        from limen.study import bench_tsp

        row = bench_tsp(3, ctx=_cpu_ctx())
        self.assertEqual(row["n_vars"], 9)
        self.assertIn(row["feasible"], (True, False))
        if not row["feasible"]:
            self.assertIsNone(row["approx_ratio"])
            self.assertIsNone(row["tour"])

    def test_maxcut_ring_scores_against_known_optimum(self):
        from limen.study import bench_maxcut

        row = bench_maxcut(6, ctx=_cpu_ctx(), kind="ring")
        self.assertEqual(row["known_optimum"], 6)
        self.assertIsNotNone(row["approx_ratio"])
        self.assertLessEqual(row["cut_edges"], 6)


@unittest.skipUnless(HAS_AER, "qiskit and qiskit-aer required")
class TestVerifyAndGap(unittest.TestCase):
    def test_verify_tier2_agrees_with_ideal_and_writes_record(self):
        from limen.study import verify_tier2

        with tempfile.TemporaryDirectory() as d:
            ctx = _cpu_ctx(d)
            rec = verify_tier2(random_field_ring(6, 1), ctx=ctx, fidelity_target=0.95)
            self.assertTrue(rec["solutions_agree"])
            self.assertTrue(rec["ideal_is_optimal"])
            self.assertEqual(rec["ecc"]["distance"], 3)
            self.assertTrue(pathlib.Path(rec["path"]).exists())
            self.assertIn("study", str(rec["path"]))

    def test_verify_tier2_distance5_runs_on_stabilizer(self):
        # fidelity 0.99 -> distance 5 -> 37-qubit syndrome circuits, which
        # the pure-Python statevector (24-qubit limit) cannot execute.
        from limen.study import verify_tier2

        rec = verify_tier2(random_field_ring(4, 2), ctx=_cpu_ctx(), fidelity_target=0.99)
        self.assertEqual(rec["ecc"]["distance"], 5)
        self.assertTrue(rec["solutions_agree"])

    def test_gap_simulated_is_kept_out_of_results_root(self):
        from limen.study import gap_offline_demo

        with tempfile.TemporaryDirectory() as d:
            ctx = _cpu_ctx(d)
            rec = gap_offline_demo(random_field_ring(6, 3), "ibm_fez", ctx=ctx, shots=400)
            self.assertTrue(rec["simulated_hardware"])
            self.assertGreaterEqual(rec["measured_success_deficit"], 0.0)
            self.assertEqual(rec["plan"]["backend"]["name"], "ibm_fez")
            self.assertEqual(list(pathlib.Path(d).glob("*.json")), [])  # nothing at root
            self.assertTrue(pathlib.Path(rec["path"]).exists())

    def test_hardware_gap_real_record_has_history_shape(self):
        from limen.router.history import scan_results
        from limen.study import hardware_gap, hardware_plan, simulated_hardware_counts

        qubo = random_field_ring(6, 4)
        with tempfile.TemporaryDirectory() as d:
            ctx = _cpu_ctx(d)
            plan = hardware_plan(qubo, "ibm_fez")
            counts = simulated_hardware_counts(qubo, plan, ctx=ctx, shots=400)
            rec = hardware_gap(qubo, plan, counts, ctx=ctx, job_id="test-job", simulated=False)
            history = scan_results(d)  # the existing scanner must pick it up
            self.assertIn("ibm_fez", history)
            self.assertEqual(history["ibm_fez"].logical_errors, [rec["measured_success_deficit"]])


@unittest.skipUnless(HAS_CUTTING, "qiskit-addon-cutting required")
class TestCuttingValidity(unittest.TestCase):
    def test_marginals_match_exact_within_sampling_noise(self):
        from limen.study import cutting_validity

        rec = cutting_validity(ctx=_cpu_ctx(), n_vars=6, max_sub=3, shots=600, seed=0)
        self.assertFalse(rec["degenerate_symmetric"])
        self.assertGreaterEqual(rec["num_cuts"], 1)
        # QPD estimator noise at this budget is ~0.1; a wrong reconstruction
        # (the old whole-register parity) gives errors of 0.3-0.9 here.
        self.assertLess(rec["mean_abs_marginal_error"], 0.2)

    def test_maxcut_is_flagged_degenerate(self):
        from limen.study import cutting_validity

        rec = cutting_validity(ctx=_cpu_ctx(), n_vars=4, max_sub=2, shots=200, seed=0, problem="maxcut")
        self.assertTrue(rec["degenerate_symmetric"])


if __name__ == "__main__":
    unittest.main()



class TestDistributionGap(unittest.TestCase):
    def test_identical_distributions(self):
        from limen.study._common import distribution_gap

        p = {"00": 0.5, "11": 0.5}
        g = distribution_gap(p, dict(p), energy_of=lambda b: -1.0 if b == "11" else 0.0)
        self.assertAlmostEqual(g["distribution_fidelity"], 1.0)
        self.assertAlmostEqual(g["total_variation_distance"], 0.0)
        self.assertAlmostEqual(g["expected_energy_gap"], 0.0)
        self.assertAlmostEqual(g["expected_energy_ideal"], -0.5)

    def test_disjoint_distributions(self):
        from limen.study._common import distribution_gap

        g = distribution_gap({"00": 1.0}, {"11": 1.0})
        self.assertAlmostEqual(g["distribution_fidelity"], 0.0)
        self.assertAlmostEqual(g["total_variation_distance"], 1.0)
        self.assertNotIn("expected_energy_gap", g)

    def test_noise_moves_energy_up(self):
        from limen.study._common import distribution_gap

        ideal = {"11": 0.9, "00": 0.1}
        noisy = {"11": 0.6, "00": 0.2, "01": 0.1, "10": 0.1}
        g = distribution_gap(ideal, noisy, energy_of=lambda b: -2.0 if b == "11" else 0.0)
        self.assertGreater(g["expected_energy_gap"], 0.0)
        self.assertLess(g["distribution_fidelity"], 1.0)
        self.assertGreater(g["total_variation_distance"], 0.0)

    def test_local_search_matches_brute_force_small(self):
        from limen.qubo_spectrum import qubo_energy_spectrum
        from limen.study._common import local_search_energy

        qubo = build_max_cut_qubo(ring_edges(8))
        order = sorted({v for pair in qubo for v in pair})
        best = local_search_energy(qubo, order, seed=1, restarts=30)
        self.assertAlmostEqual(best, qubo_energy_spectrum(qubo).best_energy)
