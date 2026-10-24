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
            self.assertIsInstance(cert["success_probability"], float)
            self.assertEqual(cert["fidelity_estimate"], cert["success_probability"])
            self.assertEqual(cert["lamport"], 1)

    def test_below_target_job_stays_done_and_records_shortfall(self):
        """3-leaf star Max-Cut, fidelity_target 0.9, credit_budget 0,
        offline Tier 0 sim: the job finishes done even when the
        measured success probability misses the named fidelity target.
        """
        with tempfile.TemporaryDirectory() as tmp:
            spool = pathlib.Path(tmp) / "spool"
            dirs = ensure_spool_dirs(spool)
            request = {
                "job_id": "job-below",
                "qubo": star_maxcut_json(3),
                "fidelity_target": 0.9,
                "credit_budget": 0.0,
                "offline": True,
            }
            (dirs[PENDING] / "job-below.json").write_text(json.dumps(request))

            with self.assertLogs("limen.limend", level="WARNING") as cm:
                run_forever(spool, once=True)

            self.assertFalse((dirs[PENDING] / "job-below.json").exists())
            self.assertTrue((dirs[DONE] / "job-below.json").exists())
            self.assertFalse((dirs[FAILED] / "job-below.json").exists())
            cert = json.loads((dirs[CERTS] / "job-below.json").read_text())
            self.assertEqual(cert["job_id"], "job-below")
            self.assertIsInstance(cert["success_probability"], float)
            self.assertEqual(cert["fidelity_estimate"], cert["success_probability"])
            self.assertEqual(cert["fidelity_target"], 0.9)
            self.assertFalse(cert["target_met"])
            self.assertLess(cert["success_probability"], cert["fidelity_target"])
            self.assertIn("note", cert)
            self.assertIn("Success probability", cert["note"])
            self.assertIn("fidelity target", cert["note"])
            self.assertIn("0.90", cert["note"])
            warning_text = "\n".join(cm.output)
            self.assertIn("Success probability", warning_text)
            self.assertIn("fidelity target", warning_text)
            self.assertIn("0.90", warning_text)

    def test_met_target_job_has_no_shortfall_note(self):
        """Same offline star, with a fidelity target the simulator
        can meet, so target_met is true and no shortfall note/warning.
        """
        with tempfile.TemporaryDirectory() as tmp:
            spool = pathlib.Path(tmp) / "spool"
            dirs = ensure_spool_dirs(spool)
            request = {
                "job_id": "job-met",
                "qubo": star_maxcut_json(3),
                "fidelity_target": 0.0,
                "credit_budget": 0.0,
                "offline": True,
            }
            (dirs[PENDING] / "job-met.json").write_text(json.dumps(request))

            with self.assertLogs("limen.limend", level="INFO") as cm:
                run_forever(spool, once=True)

            self.assertTrue((dirs[DONE] / "job-met.json").exists())
            self.assertFalse((dirs[FAILED] / "job-met.json").exists())
            cert = json.loads((dirs[CERTS] / "job-met.json").read_text())
            self.assertGreaterEqual(cert["success_probability"], cert["fidelity_target"])
            self.assertTrue(cert["target_met"])
            self.assertEqual(cert["fidelity_estimate"], cert["success_probability"])
            self.assertNotIn("note", cert)
            log_text = "\n".join(cm.output)
            self.assertIn("success_probability=", log_text)
            self.assertNotIn("fidelity=", log_text)
            self.assertFalse(any("WARNING" in line for line in cm.output))

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
