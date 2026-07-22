"""limend spool loop, exercised end-to-end with offline (simulator) plans
so no QPU credentials or network access are needed."""

import json
import pathlib
import tempfile
import unittest

from limen.limend.daemon import run_forever
from limen.limend.spool import CERTS, DONE, FAILED, PENDING, ensure_spool_dirs


def star_maxcut_json(n_leaves: int) -> list:
    pairs = []
    for i in range(n_leaves):
        leaf = f"leaf{i}"
        pairs.append([["hub", leaf], 2.0])
        pairs.append([[leaf, leaf], -1.0])
    pairs.append([["hub", "hub"], -1.0])
    return pairs


class TestLimendSpoolLoop(unittest.TestCase):
    def test_success_writes_cert_and_moves_to_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            spool = pathlib.Path(tmp) / "spool"
            dirs = ensure_spool_dirs(spool)
            request = {
                "job_id": "job-abc",
                "qubo": star_maxcut_json(3),
                "fidelity_target": 0.9,
                "credit_budget": 0.0,
                "offline": True,
            }
            (dirs[PENDING] / "job-abc.json").write_text(json.dumps(request))

            run_forever(spool, once=True)

            self.assertFalse((dirs[PENDING] / "job-abc.json").exists())
            self.assertTrue((dirs[DONE] / "job-abc.json").exists())
            cert_path = dirs[CERTS] / "job-abc.json"
            self.assertTrue(cert_path.exists())
            cert = json.loads(cert_path.read_text())
            self.assertEqual(cert["job_id"], "job-abc")
            self.assertIn("backend", cert)
            self.assertIn("tier", cert)
            self.assertIsInstance(cert["fidelity_estimate"], float)
            self.assertEqual(cert["lamport"], 1)

    def test_bad_request_lands_in_failed_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            spool = pathlib.Path(tmp) / "spool"
            dirs = ensure_spool_dirs(spool)
            (dirs[PENDING] / "job-bad.json").write_text("{not valid json")

            run_forever(spool, once=True)

            self.assertFalse((dirs[PENDING] / "job-bad.json").exists())
            failed_path = dirs[FAILED] / "job-bad.json"
            self.assertTrue(failed_path.exists())
            record = json.loads(failed_path.read_text())
            self.assertIn("error", record)

    def test_lamport_increments_per_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            spool = pathlib.Path(tmp) / "spool"
            dirs = ensure_spool_dirs(spool)
            for job_id in ("job-1", "job-2"):
                request = {
                    "job_id": job_id,
                    "qubo": star_maxcut_json(2),
                    "fidelity_target": 0.9,
                    "credit_budget": 0.0,
                    "offline": True,
                }
                (dirs[PENDING] / f"{job_id}.json").write_text(json.dumps(request))

            run_forever(spool, once=True)

            lamports = sorted(
                json.loads((dirs[CERTS] / f"{j}.json").read_text())["lamport"]
                for j in ("job-1", "job-2")
            )
            self.assertEqual(lamports, [1, 2])


if __name__ == "__main__":
    unittest.main()
