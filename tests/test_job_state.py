"""Tests for limen.router.job_state: local persistence of a long-running
QPU job's lifecycle, and the transient-error-only retry helper."""

import pathlib
import tempfile
import unittest

from limen.router.job_state import (
    JobState,
    JobStatus,
    cert_path,
    load_state,
    retry_transient,
    save_state,
    state_path,
)


class TestJobStateRoundTrip(unittest.TestCase):

    def test_save_and_load_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = pathlib.Path(tmp)
            state = JobState(
                job_id="abc123",
                status=JobStatus.SUBMITTED,
                plan={"tier": 2},
                submitted_at="2026-07-06 20:00:00 UTC",
            )
            save_state(results_dir, state)
            loaded = load_state(results_dir, "abc123")
            self.assertEqual(loaded, state)

    def test_load_missing_state_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_state(pathlib.Path(tmp), "nope"))

    def test_state_and_cert_paths_are_distinct_and_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = pathlib.Path(tmp)
            self.assertNotEqual(
                state_path(results_dir, "abc123"), cert_path(results_dir, "abc123")
            )
            self.assertIn("abc123", state_path(results_dir, "abc123").name)

    def test_status_transition_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = pathlib.Path(tmp)
            state = JobState(
                job_id="abc123",
                status=JobStatus.SUBMITTED,
                plan={},
                submitted_at="2026-07-06 20:00:00 UTC",
            )
            save_state(results_dir, state)
            state.status = JobStatus.RUNNING
            state.last_polled_at = "2026-07-06 20:05:00 UTC"
            save_state(results_dir, state)
            loaded = load_state(results_dir, "abc123")
            self.assertEqual(loaded.status, JobStatus.RUNNING)
            self.assertEqual(loaded.last_polled_at, "2026-07-06T20:05:00+00:00")

    def test_error_message_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            results_dir = pathlib.Path(tmp)
            state = JobState(
                job_id="abc123",
                status=JobStatus.TIMED_OUT,
                plan={},
                submitted_at="2026-07-06 20:00:00 UTC",
                error="polling ceiling exceeded",
            )
            save_state(results_dir, state)
            loaded = load_state(results_dir, "abc123")
            self.assertEqual(loaded.error, "polling ceiling exceeded")


class TestRetryTransient(unittest.TestCase):

    def test_succeeds_first_try_without_retry(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        result = retry_transient(fn, attempts=3, base_delay=0.0, retryable=(ValueError,))
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 1)

    def test_retries_only_retryable_exception_then_succeeds(self):
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("transient")
            return "ok"

        result = retry_transient(fn, attempts=3, base_delay=0.0, retryable=(ValueError,))
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 3)

    def test_exhausts_attempts_and_raises(self):
        def fn():
            raise ValueError("always fails")

        with self.assertRaises(ValueError):
            retry_transient(fn, attempts=2, base_delay=0.0, retryable=(ValueError,))

    def test_non_retryable_exception_surfaces_immediately(self):
        calls = []

        def fn():
            calls.append(1)
            raise KeyError("not transient")

        with self.assertRaises(KeyError):
            retry_transient(fn, attempts=3, base_delay=0.0, retryable=(ValueError,))
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
