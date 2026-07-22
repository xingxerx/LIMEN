"""Tests for limen.router.memory: the persistent SQLite router memory —
trend-aware backend stats, the transpile cache, and the hash-chained
append-only certificate ledger."""

import json
import pathlib
import sqlite3
import tempfile
import unittest

from limen.router import DEFAULT_FLEET, RouteRequest, Tier, informed_fleet, route
from limen.router.memory import (
    METRIC_LOGICAL_ERROR,
    METRIC_PHYSICAL_ERROR_RATE,
    METRIC_QUEUE_SECONDS,
    METRIC_SECONDS_PER_SHOT,
    RouterMemory,
    transpile_cache_key,
)

try:
    from limen.security.pqc import generate_signing_key
    _PQC_AVAILABLE = True
except ImportError:
    _PQC_AVAILABLE = False

_DAY = 86_400.0
# A fixed "now" keeps every recency weight and trend projection in these
# tests deterministic.
_NOW = 1_800_000_000.0


class MemoryTestCase(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = pathlib.Path(self._tmp.name)
        self.memory = RouterMemory(self.dir / "memory.sqlite3")
        self.addCleanup(self.memory.close)


class TestSamplesAndStats(MemoryTestCase):

    def test_no_samples_yields_none(self):
        self.assertIsNone(self.memory.stats("ibm_kingston", METRIC_QUEUE_SECONDS))

    def test_unknown_metric_rejected(self):
        with self.assertRaises(ValueError):
            self.memory.record_sample("ibm_kingston", "vibes", 1.0)

    def test_recency_weighting_pulls_toward_fresh_samples(self):
        # An old cheap sample and a fresh expensive one: the flat mean sits
        # midway, the weighted mean sits near the fresh value.
        self.memory.record_sample(
            "ibm_fez", METRIC_SECONDS_PER_SHOT, 0.001, observed_at=_NOW - 70 * _DAY
        )
        self.memory.record_sample(
            "ibm_fez", METRIC_SECONDS_PER_SHOT, 0.009, observed_at=_NOW - 1 * _DAY
        )
        stats = self.memory.stats(
            "ibm_fez", METRIC_SECONDS_PER_SHOT, half_life_days=7.0, now=_NOW
        )
        self.assertEqual(stats.n, 2)
        self.assertAlmostEqual(stats.mean, 0.005)
        self.assertGreater(stats.weighted_mean, 0.008)
        self.assertEqual(stats.latest, 0.009)

    def test_trend_slope_sign(self):
        for days_ago, value in ((10, 100.0), (5, 200.0), (0, 300.0)):
            self.memory.record_sample(
                "ibm_fez", METRIC_QUEUE_SECONDS, value, observed_at=_NOW - days_ago * _DAY
            )
        stats = self.memory.stats("ibm_fez", METRIC_QUEUE_SECONDS, now=_NOW)
        self.assertAlmostEqual(stats.slope_per_day, 20.0)
        self.assertAlmostEqual(stats.projection(_NOW), 300.0)

    def test_single_sample_has_no_trend(self):
        self.memory.record_sample("ibm_fez", METRIC_QUEUE_SECONDS, 60.0, observed_at=_NOW)
        stats = self.memory.stats("ibm_fez", METRIC_QUEUE_SECONDS, now=_NOW)
        self.assertIsNone(stats.slope_per_day)
        self.assertIsNone(stats.projection(_NOW))
        self.assertEqual(stats.conservative_estimate(_NOW), stats.weighted_mean)

    def test_conservative_estimate_bumps_on_rising_trend_only(self):
        # Rising (worsening) series: estimate >= weighted mean.
        for days_ago, value in ((10, 100.0), (0, 300.0)):
            self.memory.record_sample(
                "ibm_fez", METRIC_QUEUE_SECONDS, value, observed_at=_NOW - days_ago * _DAY
            )
        rising = self.memory.stats("ibm_fez", METRIC_QUEUE_SECONDS, now=_NOW)
        self.assertGreaterEqual(rising.conservative_estimate(_NOW), rising.weighted_mean)
        self.assertAlmostEqual(rising.conservative_estimate(_NOW), 300.0)

        # Falling (improving) series: never extrapolated below the weighted mean.
        for days_ago, value in ((10, 300.0), (0, 100.0)):
            self.memory.record_sample(
                "ibm_marrakesh", METRIC_QUEUE_SECONDS, value,
                observed_at=_NOW - days_ago * _DAY,
            )
        falling = self.memory.stats("ibm_marrakesh", METRIC_QUEUE_SECONDS, now=_NOW)
        self.assertEqual(falling.conservative_estimate(_NOW), falling.weighted_mean)

    def test_record_route_outcome_writes_each_given_metric(self):
        self.memory.record_route_outcome(
            "ibm_kingston",
            seconds_per_shot=0.002,
            queue_seconds=45.0,
            logical_error=0.06,
            observed_at=_NOW,
        )
        for metric in (METRIC_SECONDS_PER_SHOT, METRIC_QUEUE_SECONDS, METRIC_LOGICAL_ERROR):
            self.assertIsNotNone(self.memory.stats("ibm_kingston", metric, now=_NOW))
        self.assertIsNone(
            self.memory.stats("ibm_kingston", METRIC_PHYSICAL_ERROR_RATE, now=_NOW)
        )

    def test_memory_survives_reopen(self):
        self.memory.record_sample("ibm_fez", METRIC_QUEUE_SECONDS, 60.0, observed_at=_NOW)
        self.memory.close()
        with RouterMemory(self.dir / "memory.sqlite3") as reopened:
            stats = reopened.stats("ibm_fez", METRIC_QUEUE_SECONDS, now=_NOW)
            self.assertEqual(stats.n, 1)


class TestApplyMemory(MemoryTestCase):

    def test_backend_without_samples_unchanged(self):
        updated = self.memory.apply_memory(DEFAULT_FLEET, now=_NOW)
        self.assertEqual(updated, DEFAULT_FLEET)

    def test_sampled_fields_overridden_others_kept(self):
        self.memory.record_route_outcome(
            "ibm_kingston", queue_seconds=120.0, logical_error=0.0614, observed_at=_NOW
        )
        updated = {p.name: p for p in self.memory.apply_memory(DEFAULT_FLEET, now=_NOW)}
        kingston = updated["ibm_kingston"]
        self.assertAlmostEqual(kingston.avg_queue_seconds, 120.0)
        self.assertAlmostEqual(kingston.measured_logical_error, 0.0614)
        # No seconds-per-shot samples: the hardcoded cost stays.
        self.assertEqual(kingston.cost_per_shot, 0.002)
        self.assertEqual(updated["ibm_fez"], dict(
            (p.name, p) for p in DEFAULT_FLEET)["ibm_fez"])

    def test_route_prefers_backend_memory_showed_cheaper(self):
        # ibm_fez and ibm_kingston tie on the hardcoded cost, so Tier 1
        # falls through to the name tiebreak (fez). Memory that has seen
        # kingston run cheaper flips the choice — the router is no longer
        # stateless.
        request = RouteRequest(
            qubo={("a", "a"): 1.0, ("a", "b"): -2.0, ("b", "b"): 1.0},
            fidelity_target=0.9,
            credit_budget=10.0,
            force_tier=Tier.HW_STANDARD,
        )
        baseline = route(request, DEFAULT_FLEET)
        self.assertEqual(baseline.backend.name, "ibm_fez")

        self.memory.record_sample(
            "ibm_kingston", METRIC_SECONDS_PER_SHOT, 0.0005, observed_at=_NOW
        )
        informed = route(request, self.memory.apply_memory(DEFAULT_FLEET, now=_NOW))
        self.assertEqual(informed.backend.name, "ibm_kingston")


class TestIngestResults(MemoryTestCase):

    def _write(self, name: str, doc: dict) -> pathlib.Path:
        path = self.dir / name
        path.write_text(json.dumps(doc))
        return path

    def test_ingests_recognized_cert_shapes(self):
        self._write("tsp_eil51_run.json", {
            "qpu_run": {"backend": "ibm_kingston", "shots": 4096, "elapsed_seconds": 8.192},
        })
        self._write("router_tier2_fez.json", {
            "plan": {"backend": {"name": "ibm_fez"}},
            "measured_success_deficit": 0.05,
            "timestamps": {
                "created": "2026-07-01T10:00:00Z",
                "running": "2026-07-01T10:02:00Z",
            },
        })
        self._write("calibration_ibm_fez_1.json", {
            "backend": "ibm_fez",
            "generated_at": "2026-07-10T00:00:00+00:00",
            "physical_error_rate": 0.0229,
        })
        added = self.memory.ingest_results(self.dir)
        self.assertEqual(added, 4)

        kingston = self.memory.stats("ibm_kingston", METRIC_SECONDS_PER_SHOT, now=_NOW)
        self.assertAlmostEqual(kingston.latest, 8.192 / 4096)
        fez_queue = self.memory.stats("ibm_fez", METRIC_QUEUE_SECONDS, now=_NOW)
        self.assertAlmostEqual(fez_queue.latest, 120.0)
        fez_cal = self.memory.stats("ibm_fez", METRIC_PHYSICAL_ERROR_RATE, now=_NOW)
        self.assertAlmostEqual(fez_cal.latest, 0.0229)
        # Calibration observed_at comes from generated_at, not file mtime.
        self.assertAlmostEqual(fez_cal.latest_at, 1_783_641_600.0)

    def test_reingest_is_idempotent(self):
        self._write("tsp_eil51_run.json", {
            "qpu_run": {"backend": "ibm_kingston", "shots": 4096, "elapsed_seconds": 8.192},
        })
        self.assertEqual(self.memory.ingest_results(self.dir), 1)
        self.assertEqual(self.memory.ingest_results(self.dir), 0)
        stats = self.memory.stats("ibm_kingston", METRIC_SECONDS_PER_SHOT, now=_NOW)
        self.assertEqual(stats.n, 1)

    def test_changed_file_replaces_its_old_samples(self):
        path = self._write("tsp_eil51_run.json", {
            "qpu_run": {"backend": "ibm_kingston", "shots": 4096, "elapsed_seconds": 8.192},
        })
        self.memory.ingest_results(self.dir)
        path.write_text(json.dumps({
            "qpu_run": {"backend": "ibm_kingston", "shots": 4096, "elapsed_seconds": 16.384},
        }))
        # Force a different mtime even on coarse-timestamp filesystems.
        import os
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 10))
        self.memory.ingest_results(self.dir)
        stats = self.memory.stats("ibm_kingston", METRIC_SECONDS_PER_SHOT, now=_NOW)
        self.assertEqual(stats.n, 1)
        self.assertAlmostEqual(stats.latest, 16.384 / 4096)

    def test_unrecognized_and_malformed_files_are_skipped(self):
        self._write("fleet_certificate.json", {"fleet": ["ibm_kingston"]})
        (self.dir / "broken.json").write_text("{not json")
        self.assertEqual(self.memory.ingest_results(self.dir), 0)

    def test_informed_fleet_applies_memory_last(self):
        self._write("calibration_ibm_fez_1.json", {
            "backend": "ibm_fez",
            "generated_at": "2026-07-10T00:00:00+00:00",
            "physical_error_rate": 0.0229,
        })
        self.memory.record_sample(
            "ibm_fez", METRIC_PHYSICAL_ERROR_RATE, 0.05,
            observed_at=1_783_800_000.0,
        )
        fleet = informed_fleet(self.dir, DEFAULT_FLEET, memory=self.memory)
        fez = {p.name: p for p in fleet}["ibm_fez"]
        # The scan says 0.0229; the memory ledger (which now also ingested
        # that snapshot, plus holds a fresher direct sample) wins.
        self.assertGreater(fez.physical_error_rate, 0.0229)


class TestTranspileCache(MemoryTestCase):

    def test_key_is_stable_and_length_prefixed(self):
        self.assertEqual(
            transpile_cache_key("qasm", "ibm_fez", "1"),
            transpile_cache_key("qasm", "ibm_fez", "1"),
        )
        self.assertNotEqual(
            transpile_cache_key("ab", "c"), transpile_cache_key("a", "bc")
        )

    def test_roundtrip_and_miss(self):
        key = transpile_cache_key(b"circuit-bytes", "ibm_kingston", "opt1")
        self.assertIsNone(self.memory.transpile_cache_get(key))
        self.memory.transpile_cache_put(key, b"qpy-payload", backend="ibm_kingston")
        self.assertEqual(self.memory.transpile_cache_get(key), b"qpy-payload")

    def test_put_overwrites(self):
        key = transpile_cache_key("c", "b", "1")
        self.memory.transpile_cache_put(key, b"old")
        self.memory.transpile_cache_put(key, b"new")
        self.assertEqual(self.memory.transpile_cache_get(key), b"new")

    def test_lru_eviction(self):
        keys = [transpile_cache_key("circuit", str(i)) for i in range(4)]
        for key in keys:
            self.memory.transpile_cache_put(key, b"payload", max_entries=10)
        self.memory.transpile_cache_get(keys[0])  # refresh: most recently used
        evicted = self.memory.transpile_cache_evict(max_entries=2)
        self.assertEqual(evicted, 2)
        self.assertEqual(self.memory.transpile_cache_get(keys[0]), b"payload")
        self.assertIsNone(self.memory.transpile_cache_get(keys[1]))
        self.assertIsNone(self.memory.transpile_cache_get(keys[2]))
        self.assertEqual(self.memory.transpile_cache_get(keys[3]), b"payload")


class TestCertificateLedger(MemoryTestCase):

    CERT = {"solution": {"a": 1, "b": 0}, "energy": -2.0, "is_optimal": True}

    def test_append_and_read_back(self):
        entry = self.memory.append_certificate(self.CERT, backend="ibm_kingston")
        self.assertEqual(entry.seq, 1)
        stored = list(self.memory.certificates())
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].payload, self.CERT)
        self.assertEqual(stored[0].backend, "ibm_kingston")
        self.assertIsNone(stored[0].signature)

    def test_chain_links_and_verifies(self):
        first = self.memory.append_certificate({"energy": -1.0})
        second = self.memory.append_certificate({"energy": -2.0})
        self.assertNotEqual(first.chain_sha256, second.chain_sha256)
        self.assertTrue(self.memory.verify_ledger())

    def test_backend_filter(self):
        self.memory.append_certificate({"energy": -1.0}, backend="ibm_fez")
        self.memory.append_certificate({"energy": -2.0}, backend="ibm_kingston")
        only_fez = list(self.memory.certificates(backend="ibm_fez"))
        self.assertEqual([e.payload["energy"] for e in only_fez], [-1.0])

    def test_sql_update_and_delete_are_blocked(self):
        self.memory.append_certificate(self.CERT)
        raw = sqlite3.connect(self.memory.path)
        self.addCleanup(raw.close)
        with self.assertRaises(sqlite3.IntegrityError):
            raw.execute("UPDATE certificate_ledger SET payload = '{}' WHERE seq = 1")
        with self.assertRaises(sqlite3.IntegrityError):
            raw.execute("DELETE FROM certificate_ledger WHERE seq = 1")

    def test_verify_detects_tampering_even_past_the_triggers(self):
        self.memory.append_certificate({"energy": -1.0})
        self.memory.append_certificate({"energy": -2.0})
        raw = sqlite3.connect(self.memory.path)
        self.addCleanup(raw.close)
        raw.execute("DROP TRIGGER certificate_ledger_no_update")
        raw.execute(
            "UPDATE certificate_ledger SET payload = ? WHERE seq = 1",
            (json.dumps({"energy": -999.0}),),
        )
        raw.commit()
        self.assertFalse(self.memory.verify_ledger())

    @unittest.skipUnless(_PQC_AVAILABLE, "cryptography>=48 (ML-DSA / FIPS 204) not installed")
    def test_signed_entry_verifies_and_binds_to_chain(self):
        key = generate_signing_key()
        self.memory.append_certificate({"energy": -1.0})
        entry = self.memory.append_certificate(
            self.CERT, backend="ibm_kingston", private_key=key
        )
        self.assertIsNotNone(entry.signature)
        self.assertTrue(
            self.memory.verify_certificate_signature(entry, key.public_key())
        )
        # A signature over the same payload at a different chain position
        # must not verify: the chain head is part of the signed content.
        import dataclasses
        moved = dataclasses.replace(entry, chain_sha256="0" * 64)
        self.assertFalse(
            self.memory.verify_certificate_signature(moved, key.public_key())
        )

    @unittest.skipUnless(_PQC_AVAILABLE, "cryptography>=48 (ML-DSA / FIPS 204) not installed")
    def test_unsigned_entry_reports_false_not_error(self):
        key = generate_signing_key()
        entry = self.memory.append_certificate(self.CERT)
        self.assertFalse(
            self.memory.verify_certificate_signature(entry, key.public_key())
        )


if __name__ == "__main__":
    unittest.main()
