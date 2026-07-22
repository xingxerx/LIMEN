# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Persistent router memory: a SQLite ledger that makes routing stateful.

limen.router.history and limen.router.calibration already close the loop
from past runs to the next route() call, but both are stateless rescans:
every caller re-reads all of results/ and folds flat means into the fleet.
A flat mean has no notion of *when* a sample was observed — a queue-time
sample from six weeks ago weighs the same as one from this morning, and a
backend that is steadily drifting worse looks identical to one that is
stable. This module is the stateful successor (the "run-history cost
model" that budget_router's docstring promised):

    - **Backend sample ledger** — every timing/error observation is a row
      (backend, metric, value, observed_at). :meth:`RouterMemory.stats`
      computes recency-weighted estimates (exponential half-life decay)
      plus a least-squares trend slope, and :meth:`RouterMemory.apply_memory`
      folds them into a fleet the same way apply_history/apply_calibration
      do — except the estimate is trend-aware: when the trend says a
      metric is rising (worse), the estimate is the *larger* of the
      weighted mean and the trend's projection at now. Never smaller: an
      improving trend must be earned by fresh samples, not extrapolated.
      This is the same conservative-envelope stance as
      ``predicted_logical_error_bound = max(model, prior)``.
    - **Transpile cache** — content-addressed BLOB store for transpiled
      circuit payloads (e.g. QPY bytes), keyed by a caller-built hash of
      (circuit, backend, options). Transpiling for a 156q backend costs
      seconds; re-running the same QUBO shape against the same backend
      should not pay it twice. Payloads are opaque bytes — this module
      never imports qiskit.
    - **Append-only certificate ledger** — each appended certificate is
      hash-chained to its predecessor (``chain = sha256(prev_chain +
      payload_sha256)``), and SQL triggers abort any UPDATE/DELETE on the
      table. A run without a certificate happened; a run whose
      certificate is chained into this ledger is *witnessed*, and the
      whole history can be re-verified with :meth:`RouterMemory.verify_ledger`.
      Callers who hold an ML-DSA-65 key (limen.security.pqc) can
      additionally sign each entry's chain head — signing the chain binds
      the signature to the entire prior history, not just one payload.
      Purely opt-in, like every other pqc hook in this package.

All timestamps are Unix epoch seconds, UTC. The store is a single SQLite
file in WAL mode (crash-safe, same "persist before you dispatch" posture
as limen.router.job_state); the schema is created on first open.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import sqlite3
import time
from typing import Any, Iterator

from limen.router.budget_router import BackendProfile
from limen.router.history import _SCANNERS, BackendHistory, _parse_timestamp

DEFAULT_FILENAME = "router_memory.sqlite3"
DEFAULT_HALF_LIFE_DAYS = 7.0
DEFAULT_TRANSPILE_CACHE_ENTRIES = 256

METRIC_SECONDS_PER_SHOT = "seconds_per_shot"
METRIC_QUEUE_SECONDS = "queue_seconds"
METRIC_LOGICAL_ERROR = "logical_error"
METRIC_PHYSICAL_ERROR_RATE = "physical_error_rate"

_METRICS = frozenset(
    {
        METRIC_SECONDS_PER_SHOT,
        METRIC_QUEUE_SECONDS,
        METRIC_LOGICAL_ERROR,
        METRIC_PHYSICAL_ERROR_RATE,
    }
)

_SECONDS_PER_DAY = 86_400.0

# Chain value "before" the first ledger entry. A constant (rather than the
# first payload's own hash) so an empty ledger and a one-entry ledger are
# distinguishable, and so verify_ledger has a fixed starting point.
_GENESIS = hashlib.sha256(b"limen-certificate-ledger-genesis").hexdigest()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS backend_samples (
    id          INTEGER PRIMARY KEY,
    backend     TEXT NOT NULL,
    metric      TEXT NOT NULL,
    value       REAL NOT NULL,
    observed_at REAL NOT NULL,
    source      TEXT
);
CREATE INDEX IF NOT EXISTS idx_backend_samples
    ON backend_samples (backend, metric, observed_at);

CREATE TABLE IF NOT EXISTS ingested_files (
    path  TEXT PRIMARY KEY,
    mtime REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS transpile_cache (
    cache_key    TEXT PRIMARY KEY,
    backend      TEXT NOT NULL,
    payload      BLOB NOT NULL,
    created_at   REAL NOT NULL,
    last_used_at REAL NOT NULL,
    hits         INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS certificate_ledger (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    recorded_at    REAL NOT NULL,
    backend        TEXT,
    payload        TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    chain_sha256   TEXT NOT NULL,
    signature      TEXT
);
CREATE TRIGGER IF NOT EXISTS certificate_ledger_no_update
    BEFORE UPDATE ON certificate_ledger
BEGIN
    SELECT RAISE(ABORT, 'certificate ledger is append-only');
END;
CREATE TRIGGER IF NOT EXISTS certificate_ledger_no_delete
    BEFORE DELETE ON certificate_ledger
BEGIN
    SELECT RAISE(ABORT, 'certificate ledger is append-only');
END;
"""


def _canonical_json(obj: dict[str, Any]) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace.

    Same encoding as limen.security.pqc.canonical_json_bytes, reimplemented
    here so the ledger's hash chain never requires the cryptography extra.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def transpile_cache_key(*parts: str | bytes) -> str:
    """Build a content-addressed cache key from the things that determine a
    transpilation output: typically (circuit serialization, backend name,
    optimization level, and — if the caller wants calibration-epoch
    invalidation — the calibration snapshot's ``generated_at``).

    Parts are length-prefixed before hashing so ``("ab", "c")`` and
    ``("a", "bc")`` cannot collide.
    """
    h = hashlib.sha256()
    for part in parts:
        raw = part.encode("utf-8") if isinstance(part, str) else part
        h.update(str(len(raw)).encode("ascii"))
        h.update(b":")
        h.update(raw)
    return h.hexdigest()


@dataclasses.dataclass(frozen=True)
class MetricStats:
    """Trend-aware summary of one (backend, metric) sample series.

    ``weighted_mean`` uses exponential recency decay (half-life in days);
    ``slope_per_day``/``trend_intercept`` are the ordinary-least-squares
    line over (epoch-days, value), or None when fewer than two distinct
    observation times exist.
    """

    backend: str
    metric: str
    n: int
    mean: float
    weighted_mean: float
    latest: float
    latest_at: float
    slope_per_day: float | None
    trend_intercept: float | None

    def projection(self, at: float) -> float | None:
        """Trend-line value at epoch-seconds *at*, clamped non-negative.
        None when no trend line exists."""
        if self.slope_per_day is None or self.trend_intercept is None:
            return None
        return max(0.0, self.trend_intercept + self.slope_per_day * (at / _SECONDS_PER_DAY))

    def conservative_estimate(self, at: float) -> float:
        """The recency-weighted mean, bumped up to the trend projection when
        the trend is rising. Every metric in this module is
        lower-is-better (cost, queue time, error rates), so "rising" means
        "getting worse" and the envelope only ever widens — a falling
        trend is never extrapolated below the weighted mean."""
        estimate = self.weighted_mean
        if self.slope_per_day is not None and self.slope_per_day > 0.0:
            projected = self.projection(at)
            if projected is not None:
                estimate = max(estimate, projected)
        return estimate


@dataclasses.dataclass(frozen=True)
class LedgerEntry:
    """One witnessed certificate in the append-only ledger."""

    seq: int
    recorded_at: float
    backend: str | None
    payload: dict[str, Any]
    payload_sha256: str
    chain_sha256: str
    signature: str | None

    def signed_content(self) -> dict[str, Any]:
        """The dict an ML-DSA signature covers: the payload hash plus the
        chain head, so the signature attests to the entire prior history."""
        return {
            "payload_sha256": self.payload_sha256,
            "chain_sha256": self.chain_sha256,
        }


class RouterMemory:
    """SQLite-backed persistent memory for the budget router.

    Usable as a context manager; :meth:`close` is idempotent. All write
    methods commit before returning, so a crash between calls never loses
    an acknowledged record.
    """

    def __init__(self, path: pathlib.Path | str) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @classmethod
    def in_results_dir(cls, results_dir: pathlib.Path | str) -> "RouterMemory":
        """Open (creating if needed) the memory ledger that lives alongside
        the certs in *results_dir*."""
        return cls(pathlib.Path(results_dir) / DEFAULT_FILENAME)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RouterMemory":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Backend sample ledger + trend-aware stats
    # ------------------------------------------------------------------

    def record_sample(
        self,
        backend: str,
        metric: str,
        value: float,
        *,
        observed_at: float | None = None,
        source: str | None = None,
    ) -> None:
        """Append one observation for (backend, metric)."""
        if metric not in _METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}. Choose from: " + ", ".join(sorted(_METRICS))
            )
        with self._conn:
            self._conn.execute(
                "INSERT INTO backend_samples (backend, metric, value, observed_at, source)"
                " VALUES (?, ?, ?, ?, ?)",
                (backend, metric, float(value),
                 time.time() if observed_at is None else float(observed_at), source),
            )

    def record_route_outcome(
        self,
        backend: str,
        *,
        seconds_per_shot: float | None = None,
        queue_seconds: float | None = None,
        logical_error: float | None = None,
        physical_error_rate: float | None = None,
        observed_at: float | None = None,
        source: str | None = None,
    ) -> None:
        """Record whichever measurements one finished run produced, in one
        call — the morning-weighing hook a dispatcher calls after the
        certificate lands."""
        for metric, value in (
            (METRIC_SECONDS_PER_SHOT, seconds_per_shot),
            (METRIC_QUEUE_SECONDS, queue_seconds),
            (METRIC_LOGICAL_ERROR, logical_error),
            (METRIC_PHYSICAL_ERROR_RATE, physical_error_rate),
        ):
            if value is not None:
                self.record_sample(
                    backend, metric, value, observed_at=observed_at, source=source
                )

    def stats(
        self,
        backend: str,
        metric: str,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        now: float | None = None,
    ) -> MetricStats | None:
        """Trend-aware summary of every recorded sample for (backend,
        metric), or None when no samples exist."""
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        rows = self._conn.execute(
            "SELECT value, observed_at FROM backend_samples"
            " WHERE backend = ? AND metric = ? ORDER BY observed_at, id",
            (backend, metric),
        ).fetchall()
        if not rows:
            return None
        now = time.time() if now is None else now
        values = [v for v, _ in rows]
        times = [t for _, t in rows]

        weights = [0.5 ** (max(0.0, now - t) / (half_life_days * _SECONDS_PER_DAY)) for t in times]
        total_weight = sum(weights)
        weighted_mean = (
            sum(w * v for w, v in zip(weights, values)) / total_weight
            if total_weight > 0
            else sum(values) / len(values)
        )

        slope: float | None = None
        intercept: float | None = None
        if len(rows) >= 2 and max(times) > min(times):
            xs = [t / _SECONDS_PER_DAY for t in times]
            x_mean = sum(xs) / len(xs)
            y_mean = sum(values) / len(values)
            sxx = sum((x - x_mean) ** 2 for x in xs)
            sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
            slope = sxy / sxx
            intercept = y_mean - slope * x_mean

        return MetricStats(
            backend=backend,
            metric=metric,
            n=len(rows),
            mean=sum(values) / len(values),
            weighted_mean=weighted_mean,
            latest=values[-1],
            latest_at=times[-1],
            slope_per_day=slope,
            trend_intercept=intercept,
        )

    def apply_memory(
        self,
        fleet: tuple[BackendProfile, ...],
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        now: float | None = None,
    ) -> tuple[BackendProfile, ...]:
        """Fold recorded samples into a fleet, field-for-field the same
        mapping as apply_history + apply_calibration — but with recency
        weighting and the conservative trend bump (see
        :meth:`MetricStats.conservative_estimate`). A backend with no
        samples for a metric keeps that field unchanged."""
        now = time.time() if now is None else now
        field_metric = (
            ("cost_per_shot", METRIC_SECONDS_PER_SHOT),
            ("avg_queue_seconds", METRIC_QUEUE_SECONDS),
            ("measured_logical_error", METRIC_LOGICAL_ERROR),
            ("physical_error_rate", METRIC_PHYSICAL_ERROR_RATE),
        )
        updated: list[BackendProfile] = []
        for profile in fleet:
            changes: dict[str, float] = {}
            for field_name, metric in field_metric:
                entry = self.stats(
                    profile.name, metric, half_life_days=half_life_days, now=now
                )
                if entry is not None:
                    changes[field_name] = entry.conservative_estimate(now)
            updated.append(
                dataclasses.replace(profile, **changes) if changes else profile
            )
        return tuple(updated)

    def ingest_results(self, results_dir: pathlib.Path | str) -> int:
        """Backfill the sample ledger from results/ certs, incrementally.

        Recognizes the same cert shapes as limen.router.history's scanners
        plus calibration_*.json snapshots. Each file is ingested once (a
        changed mtime re-ingests it — matching how re-fetched calibration
        snapshots and re-written certs behave in results/). Returns the
        number of samples added. Unrecognized files are recorded as
        ingested-with-zero-samples so they are not re-parsed every call.
        """
        added = 0
        for path in sorted(pathlib.Path(results_dir).glob("*.json")):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            row = self._conn.execute(
                "SELECT mtime FROM ingested_files WHERE path = ?", (str(path),)
            ).fetchone()
            if row is not None and row[0] == mtime:
                continue
            try:
                doc = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                doc = None
            samples: list[tuple[str, str, float, float]] = []
            if isinstance(doc, dict):
                samples = self._extract_samples(doc, default_observed_at=mtime)
            with self._conn:
                if row is not None:
                    # Re-ingest of a changed file: drop its old samples so a
                    # rewritten cert doesn't double-count.
                    self._conn.execute(
                        "DELETE FROM backend_samples WHERE source = ?", (str(path),)
                    )
                self._conn.executemany(
                    "INSERT INTO backend_samples"
                    " (backend, metric, value, observed_at, source)"
                    " VALUES (?, ?, ?, ?, ?)",
                    [
                        (backend, metric, value, observed_at, str(path))
                        for backend, metric, value, observed_at in samples
                    ],
                )
                self._conn.execute(
                    "INSERT INTO ingested_files (path, mtime) VALUES (?, ?)"
                    " ON CONFLICT(path) DO UPDATE SET mtime = excluded.mtime",
                    (str(path), mtime),
                )
            added += len(samples)
        return added

    @staticmethod
    def _extract_samples(
        doc: dict[str, Any], *, default_observed_at: float
    ) -> list[tuple[str, str, float, float]]:
        """(backend, metric, value, observed_at) samples in one results doc."""
        samples: list[tuple[str, str, float, float]] = []

        # Calibration snapshot shape (limen.router.calibration).
        backend = doc.get("backend")
        rate = doc.get("physical_error_rate")
        if isinstance(backend, str) and rate is not None and "generated_at" in doc:
            generated = _parse_timestamp(doc.get("generated_at"))
            observed_at = (
                generated.timestamp() if generated is not None else default_observed_at
            )
            samples.append((backend, METRIC_PHYSICAL_ERROR_RATE, float(rate), observed_at))
            return samples

        # Cert shapes recognized by limen.router.history's scanners.
        history: dict[str, BackendHistory] = {}
        for scanner in _SCANNERS:
            if scanner(doc, history):
                break
        for name, entry in history.items():
            for value in entry.seconds_per_shot:
                samples.append((name, METRIC_SECONDS_PER_SHOT, value, default_observed_at))
            for value in entry.queue_seconds:
                samples.append((name, METRIC_QUEUE_SECONDS, value, default_observed_at))
            for value in entry.logical_errors:
                samples.append((name, METRIC_LOGICAL_ERROR, value, default_observed_at))
        return samples

    # ------------------------------------------------------------------
    # Transpile cache
    # ------------------------------------------------------------------

    def transpile_cache_get(self, cache_key: str) -> bytes | None:
        """Return the cached payload for *cache_key*, or None on a miss.
        A hit refreshes the entry's LRU position."""
        row = self._conn.execute(
            "SELECT payload FROM transpile_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        with self._conn:
            self._conn.execute(
                "UPDATE transpile_cache SET last_used_at = ?, hits = hits + 1"
                " WHERE cache_key = ?",
                (time.time(), cache_key),
            )
        return row[0]

    def transpile_cache_put(
        self,
        cache_key: str,
        payload: bytes,
        *,
        backend: str = "",
        max_entries: int = DEFAULT_TRANSPILE_CACHE_ENTRIES,
    ) -> None:
        """Store (or overwrite) a transpiled payload, then evict the
        least-recently-used entries beyond *max_entries*."""
        now = time.time()
        with self._conn:
            self._conn.execute(
                "INSERT INTO transpile_cache"
                " (cache_key, backend, payload, created_at, last_used_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(cache_key) DO UPDATE SET"
                "  payload = excluded.payload, backend = excluded.backend,"
                "  last_used_at = excluded.last_used_at",
                (cache_key, backend, payload, now, now),
            )
        self.transpile_cache_evict(max_entries)

    def transpile_cache_evict(self, max_entries: int) -> int:
        """Drop least-recently-used cache entries beyond *max_entries*;
        returns how many were evicted."""
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative")
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM transpile_cache WHERE cache_key IN ("
                " SELECT cache_key FROM transpile_cache"
                " ORDER BY last_used_at DESC, cache_key"
                " LIMIT -1 OFFSET ?)",
                (max_entries,),
            )
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Append-only certificate ledger
    # ------------------------------------------------------------------

    def append_certificate(
        self,
        payload: dict[str, Any],
        *,
        backend: str | None = None,
        private_key: Any = None,
        recorded_at: float | None = None,
    ) -> LedgerEntry:
        """Chain *payload* (e.g. ``EndToEndCertificate.to_dict()``) onto the
        ledger. With *private_key* (an ML-DSA-65 key from
        limen.security.pqc — requires the ``pqc`` extra), the entry's
        chain head is signed, binding the signature to every prior entry."""
        text = _canonical_json(payload)
        payload_hash = _sha256_hex(text.encode("utf-8"))
        row = self._conn.execute(
            "SELECT chain_sha256 FROM certificate_ledger ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        prev_chain = row[0] if row is not None else _GENESIS
        chain = _sha256_hex((prev_chain + payload_hash).encode("ascii"))

        signature: str | None = None
        if private_key is not None:
            from limen.security.pqc import sign_json

            signature = sign_json(
                private_key, {"payload_sha256": payload_hash, "chain_sha256": chain}
            )

        recorded_at = time.time() if recorded_at is None else recorded_at
        with self._conn:
            cursor = self._conn.execute(
                "INSERT INTO certificate_ledger"
                " (recorded_at, backend, payload, payload_sha256, chain_sha256, signature)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (recorded_at, backend, text, payload_hash, chain, signature),
            )
        return LedgerEntry(
            seq=int(cursor.lastrowid or 0),
            recorded_at=recorded_at,
            backend=backend,
            payload=payload,
            payload_sha256=payload_hash,
            chain_sha256=chain,
            signature=signature,
        )

    def certificates(self, backend: str | None = None) -> Iterator[LedgerEntry]:
        """Iterate ledger entries in append order, optionally filtered by
        backend."""
        query = (
            "SELECT seq, recorded_at, backend, payload, payload_sha256,"
            " chain_sha256, signature FROM certificate_ledger"
        )
        params: tuple[Any, ...] = ()
        if backend is not None:
            query += " WHERE backend = ?"
            params = (backend,)
        query += " ORDER BY seq"
        for seq, recorded_at, entry_backend, text, payload_hash, chain, sig in (
            self._conn.execute(query, params)
        ):
            yield LedgerEntry(
                seq=seq,
                recorded_at=recorded_at,
                backend=entry_backend,
                payload=json.loads(text),
                payload_sha256=payload_hash,
                chain_sha256=chain,
                signature=sig,
            )

    def verify_ledger(self) -> bool:
        """Recompute the whole hash chain from genesis. False means some
        stored payload or chain value no longer matches what was appended
        (tampering, corruption, or truncation-and-regrowth)."""
        prev_chain = _GENESIS
        for entry in self.certificates():
            expected_payload_hash = _sha256_hex(
                _canonical_json(entry.payload).encode("utf-8")
            )
            if entry.payload_sha256 != expected_payload_hash:
                return False
            if entry.chain_sha256 != _sha256_hex(
                (prev_chain + entry.payload_sha256).encode("ascii")
            ):
                return False
            prev_chain = entry.chain_sha256
        return True

    def verify_certificate_signature(self, entry: LedgerEntry, public_key: Any) -> bool:
        """Verify an entry's ML-DSA-65 signature over its chain head.
        Returns False for unsigned entries — an absent signature is one
        outcome, not an error path. Requires the ``pqc`` extra."""
        if entry.signature is None:
            return False
        from limen.security.pqc import verify_json

        return verify_json(public_key, entry.signed_content(), entry.signature)
