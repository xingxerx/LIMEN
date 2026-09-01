"""Tests for limen.router.calibration: seeding BackendProfile.physical_error_rate
from cached live-calibration snapshots, and route() preferring that measured
value over RouteRequest's hardcoded 1e-3 default."""

import json
import pathlib
import tempfile
import unittest

import math

from limen.router import DEFAULT_FLEET, RouteRequest, Tier, apply_calibration, route
from limen.router.calibration import (
    fetch_backend_calibration,
    scan_calibration,
    sign_calibration_record,
    verify_calibration_record,
)
try:
    from limen.security.pqc import generate_signing_key
    _PQC_AVAILABLE = True
except ImportError:
    _PQC_AVAILABLE = False


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
            RouteRequest(self._star_qubo(16), fidelity_target=0.9, credit_budget=10.0)
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
            RouteRequest(self._star_qubo(16), fidelity_target=0.9, credit_budget=10.0),
            fleet=fleet,
        )
        self.assertEqual(plan.tier, Tier.HW_CERTIFIED)
        self.assertEqual(plan.backend.name, "ibm_kingston")
        self.assertAlmostEqual(plan.pipeline_kwargs["physical_error_rate"], 0.0614)
        self.assertTrue(any("calibration" in note for note in plan.notes))


@unittest.skipUnless(_PQC_AVAILABLE, "cryptography>=48 (ML-DSA / FIPS 204) not installed")
class TestPqcSigning(unittest.TestCase):
    """sign_calibration_record/verify_calibration_record are purely
    additive -- fetch_backend_calibration/scan_calibration/apply_calibration
    (tested above) never require a keypair."""

    def test_valid_signature_verifies(self):
        key = generate_signing_key()
        record = {"backend": "ibm_kingston", "physical_error_rate": 0.0614}
        sig = sign_calibration_record(record, key)
        self.assertTrue(verify_calibration_record(record, sig, key.public_key()))

    def test_tampered_record_fails_verification(self):
        key = generate_signing_key()
        record = {"backend": "ibm_kingston", "physical_error_rate": 0.0614}
        sig = sign_calibration_record(record, key)
        tampered = {"backend": "ibm_kingston", "physical_error_rate": 1e-6}
        self.assertFalse(verify_calibration_record(tampered, sig, key.public_key()))

    def test_signature_survives_json_round_trip(self):
        # A record written to disk, reloaded, and re-verified -- the real
        # usage shape for a results/calibration_*.json snapshot.
        key = generate_signing_key()
        record = {"backend": "ibm_fez", "physical_error_rate": 0.0229}
        sig = sign_calibration_record(record, key)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "calibration_ibm_fez_test.json"
            path.write_text(json.dumps(record))
            reloaded = json.loads(path.read_text())
        self.assertTrue(verify_calibration_record(reloaded, sig, key.public_key()))


class _FakeGate:
    def __init__(self, gate: str, qubits: tuple[int, ...]) -> None:
        self.gate = gate
        self.qubits = qubits


class _FakeProperties:
    """Duck-types the subset of qiskit BackendProperties fetch_backend_calibration
    reads: gates/gate_error/gate_length for two-qubit gates, and per-qubit
    readout_error/t1/t2."""

    def __init__(self, num_qubits: int, t1: float, t2: float) -> None:
        self.gates = [_FakeGate("ecr", (i, i + 1)) for i in range(num_qubits - 1)]
        self._t1 = t1
        self._t2 = t2

    def gate_error(self, gate: str, qubits) -> float:
        return 0.01

    def gate_length(self, gate: str, qubits) -> float:
        return 200e-9  # 200ns, a typical two-qubit gate duration

    def readout_error(self, qubit: int) -> float:
        return 0.02

    def t1(self, qubit: int) -> float:
        return self._t1

    def t2(self, qubit: int) -> float:
        return self._t2


class _FakeBackend:
    def __init__(self, num_qubits: int, t1: float, t2: float) -> None:
        self.num_qubits = num_qubits
        self._props = _FakeProperties(num_qubits, t1, t2)

    def properties(self):
        return self._props


class _FakeService:
    def __init__(self, backend: _FakeBackend) -> None:
        self._backend = backend

    def backend(self, name: str) -> _FakeBackend:
        return self._backend


class TestFetchBackendCalibrationDecoherence(unittest.TestCase):
    """T1/T2-weighted decoherence estimate: the piece of physical_error_rate
    that scales with expected circuit depth rather than being a fixed
    per-gate constant (see calibration.py module docstring)."""

    T1 = 100e-6  # 100us, a typical IBM T1

    def test_decoherence_prob_matches_exponential_model(self):
        service = _FakeService(_FakeBackend(3, t1=self.T1, t2=self.T1))
        record = fetch_backend_calibration(
            service, "fake_backend", expected_two_qubit_depth=10
        )
        expected_duration = 10 * 200e-9
        expected_decoherence = 1.0 - math.exp(-expected_duration / self.T1)
        self.assertAlmostEqual(record["decoherence_prob"], expected_decoherence)
        self.assertAlmostEqual(record["avg_t1"], self.T1)
        self.assertAlmostEqual(record["avg_two_qubit_gate_length"], 200e-9)

    def test_deeper_circuit_increases_physical_error_rate(self):
        service = _FakeService(_FakeBackend(3, t1=self.T1, t2=self.T1))
        shallow = fetch_backend_calibration(
            service, "fake_backend", expected_two_qubit_depth=1
        )
        deep = fetch_backend_calibration(
            service, "fake_backend", expected_two_qubit_depth=1000
        )
        self.assertLess(
            shallow["physical_error_rate"], deep["physical_error_rate"]
        )

    def test_defaults_to_depth_one(self):
        service = _FakeService(_FakeBackend(3, t1=self.T1, t2=self.T1))
        record = fetch_backend_calibration(service, "fake_backend")
        self.assertEqual(record["expected_two_qubit_depth"], 1)


if __name__ == "__main__":
    unittest.main()
