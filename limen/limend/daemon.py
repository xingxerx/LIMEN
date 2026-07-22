# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""``limend`` -- the daemon that turns spooled QUBO requests into
DUCTEI-ready certificate summaries.

Watches ``spool/pending/`` for ``{job_id}.json`` request files, routes
and executes each one through :func:`limen.pipeline.run_route_request`
(called as a black box -- this module never touches routing or pipeline
internals), and writes the result to ``spool/certs/`` (success) or
``spool/failed/`` (anything that went wrong, with the reason attached).
See spool.py for the full directory contract.

The daemon owns exactly one thing routing doesn't: the Lamport counter
attached to each certificate summary. It's a plain in-memory counter,
one per daemon instance, incremented once per envelope produced --
DUCTEI's causal gate keys on ``(job_id, scopes)`` (see
ductei-core/src/gate.rs), so a counter that resets on restart never
collides with a prior run's accepted state.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import uuid
from typing import Any

from limen.limend.spool import CERTS, DONE, FAILED, PENDING, ensure_spool_dirs

logger = logging.getLogger("limen.limend")

QPU_TOKEN_ENV = "LIMEN_QPU_TOKEN"
QPU_INSTANCE_ENV = "LIMEN_QPU_INSTANCE"


def _qubo_from_json(raw: Any) -> dict[tuple[str, str], float]:
    """Inverse of the ``[[i, j], w]`` list encoding used elsewhere in the
    repo for round-tripping a ``dict[tuple, float]`` through JSON (see
    ``limen.core.compiler.PhysicalEncoding.to_dict``/``from_dict``).
    """
    if not isinstance(raw, list):
        raise ValueError(
            "qubo must be a list of [[var_a, var_b], weight] entries, "
            f"got {type(raw).__name__}"
        )
    qubo: dict[tuple[str, str], float] = {}
    for entry in raw:
        (a, b), w = entry
        qubo[(str(a), str(b))] = float(w)
    return qubo


def _request_from_json(data: dict[str, Any]) -> Any:
    from limen.router import RouteRequest, Tier

    kwargs: dict[str, Any] = {
        "qubo": _qubo_from_json(data["qubo"]),
        "fidelity_target": float(data["fidelity_target"]),
        "credit_budget": float(data["credit_budget"]),
    }
    if "force_tier" in data and data["force_tier"] is not None:
        kwargs["force_tier"] = Tier(int(data["force_tier"]))
    if "physical_error_rate" in data and data["physical_error_rate"] is not None:
        kwargs["physical_error_rate"] = float(data["physical_error_rate"])
    if "offline" in data and data["offline"] is not None:
        kwargs["offline"] = bool(data["offline"])
    return RouteRequest(**kwargs)


def _current_rss_mb() -> float | None:
    """Best-effort resident set size in MB; ``None`` where unavailable
    (e.g. no ``resource`` module on Windows dev boxes -- the memory
    ceiling simply becomes a no-op there, production is Linux).
    """
    try:
        import resource
        import sys

        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return ru_maxrss / 1024 if sys.platform != "darwin" else ru_maxrss / (1024 * 1024)
    except Exception:
        return None


def _write_failed(dirs: dict[str, Any], job_id: str, request_payload: Any, error: BaseException) -> None:
    record = {
        "job_id": job_id,
        "request": request_payload,
        "error": f"{type(error).__name__}: {error}",
        "failed_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    path = dirs[FAILED] / f"{job_id}.json"
    path.write_text(json.dumps(record, indent=2))
    logger.error("job %s failed: %s", job_id, record["error"])


def process_one(
    path: Any,
    dirs: dict[str, Any],
    lamport: int,
    *,
    results_dir: Any,
    qpu_token: str | None,
    qpu_instance: str | None,
    memory: Any,
) -> bool:
    """Process a single pending request file. Returns True on success."""
    from limen.pipeline import run_route_request

    raw_text = path.read_text()
    job_id = path.stem
    data: dict[str, Any] | None = None
    try:
        data = json.loads(raw_text)
        job_id = str(data.get("job_id") or job_id or uuid.uuid4())
        request = _request_from_json(data)
    except Exception as exc:
        _write_failed(dirs, job_id, data if data is not None else raw_text, exc)
        path.unlink(missing_ok=True)
        return False

    try:
        cert = run_route_request(
            request,
            results_dir=results_dir,
            emit_report=True,
            qpu_token=qpu_token,
            qpu_instance=qpu_instance,
            memory=memory,
        )
        report = cert.metadata.get("route_report")
        if not report:
            raise RuntimeError(
                "route_report missing from certificate metadata -- cannot "
                "build a CertSummary without a backend/tier"
            )
        cert_summary = {
            "job_id": job_id,
            "backend": report["backend"],
            "tier": int(report["tier"]),
            "fidelity_estimate": cert.success_probability,
            "lamport": lamport,
        }
        (dirs[CERTS] / f"{job_id}.json").write_text(json.dumps(cert_summary, indent=2))
        path.rename(dirs[DONE] / f"{job_id}.json")
        logger.info("job %s done: backend=%s tier=%s fidelity=%.4f", job_id,
                     cert_summary["backend"], cert_summary["tier"], cert_summary["fidelity_estimate"])
        return True
    except Exception as exc:
        _write_failed(dirs, job_id, data, exc)
        path.unlink(missing_ok=True)
        return False


def run_forever(
    spool_dir: Any,
    *,
    poll_interval: float = 1.0,
    results_dir: Any = None,
    memory: Any = None,
    memory_ceiling_mb: float | None = None,
    once: bool = False,
) -> None:
    """Watch ``spool_dir`` and process requests until *once* or the
    memory ceiling is hit.

    Returning (rather than looping forever in-process) on a ceiling
    breach is deliberate: this function is meant to run under a
    restart-on-crash supervisor (systemd ``Restart=always`` or
    equivalent -- see scripts/limend.service), so a clean exit here is
    a preemptive restart, not a crash.
    """
    dirs = ensure_spool_dirs(spool_dir)
    qpu_token = os.environ.get(QPU_TOKEN_ENV)
    qpu_instance = os.environ.get(QPU_INSTANCE_ENV)
    lamport = 0

    while True:
        pending_files = sorted(dirs[PENDING].glob("*.json"))
        for f in pending_files:
            lamport += 1
            with contextlib.suppress(FileNotFoundError):
                process_one(
                    f, dirs, lamport,
                    results_dir=results_dir,
                    qpu_token=qpu_token,
                    qpu_instance=qpu_instance,
                    memory=memory,
                )

        if once:
            return

        if memory_ceiling_mb is not None:
            rss = _current_rss_mb()
            if rss is not None and rss >= memory_ceiling_mb:
                logger.warning(
                    "memory ceiling reached (%.1fMB >= %.1fMB); exiting for "
                    "supervisor restart", rss, memory_ceiling_mb,
                )
                return

        time.sleep(poll_interval)
