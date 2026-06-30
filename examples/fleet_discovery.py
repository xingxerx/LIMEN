# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Fleet discovery for LIMEN.

Queries the IBM Quantum Platform (and, optionally, AWS Braket) for the
operational, non-simulator backends actually reachable with the caller's
credentials, then cross-references results/*.json for real job IDs to
determine which backends LIMEN has *actually* run on (as opposed to ones
that are merely reachable). Writes results/fleet_certificate.json.

A backend is marked "validated": true only if a job_id tied to that backend
name is found on disk in results/ — never from assertion.

Required environment variables (or pass as flags):
    IBM_QUANTUM_TOKEN  — IBM Quantum Platform API token
    IBM_QUANTUM_CRN    — IBM Quantum instance CRN

Usage::

    python examples/fleet_discovery.py
    python examples/fleet_discovery.py --ibm-token <token> --ibm-crn <crn>
    python examples/fleet_discovery.py --braket
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass

_RESULTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "results"
_CERT_PATH = _RESULTS_DIR / "fleet_certificate.json"

# Braket QPU device ARNs LIMEN knows how to target. These are not queried
# unless --braket is passed (querying requires AWS credentials with Braket
# access); without it they're listed as "not queried" rather than guessed.
_BRAKET_DEVICES = {
    "QuEra Aquila": "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
    "Rigetti Ankaa-3": "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
}


def discover_ibm_backends(token: str, crn: str) -> list[dict[str, Any]]:
    """Query the live, operational, non-simulator IBM backend list.

    Args:
        token: IBM Quantum Platform API token.
        crn: Service instance CRN.

    Returns:
        List of dicts with keys ``name``, ``num_qubits``, ``modality``.
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore[import]
    except ModuleNotFoundError as exc:
        print(
            "ERROR: qiskit-ibm-runtime not installed.\n"
            "Install with: pip install qiskit-ibm-runtime\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        token=token,
        instance=crn,
    )
    backends = service.backends(operational=True, simulator=False)
    return [
        {
            "name": b.name,
            "num_qubits": b.num_qubits,
            "modality": "superconducting",
        }
        for b in backends
    ]


def discover_braket_devices() -> list[dict[str, Any]]:
    """Query live status for the known Braket QPU devices.

    Requires AWS credentials with Braket access. Devices that error out
    (e.g. no credentials, retired) are reported with their error rather
    than silently dropped.

    Returns:
        List of dicts with keys ``name``, ``arn``, ``status``,
        ``num_qubits`` (``None`` if unavailable), ``modality``.
    """
    try:
        from braket.aws import AwsDevice  # type: ignore[import]
    except ModuleNotFoundError as exc:
        print(
            "ERROR: amazon-braket-sdk not installed.\n"
            "Install with: pip install amazon-braket-sdk\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    results = []
    for name, arn in _BRAKET_DEVICES.items():
        modality = "neutral-atom" if "quera" in arn else "superconducting"
        try:
            device = AwsDevice(arn)
            results.append(
                {
                    "name": name,
                    "arn": arn,
                    "status": device.status,
                    "num_qubits": getattr(device.properties.paradigm, "qubitCount", None),
                    "modality": modality,
                }
            )
        except Exception as exc:  # noqa: BLE001 - report, don't hide, device errors
            results.append(
                {
                    "name": name,
                    "arn": arn,
                    "status": f"ERROR: {exc}",
                    "num_qubits": None,
                    "modality": modality,
                }
            )
    return results


_MODALITY_BY_PROVIDER_PREFIX = {
    "ionq": "trapped-ion",
    "aqt": "trapped-ion",
    "rigetti": "superconducting",
    "iqm": "superconducting",
}


def discover_openquantum_backends(client_id: str, client_secret: str) -> list[dict[str, Any]]:
    """Query live backend classes reachable through Open Quantum's unified credential.

    Open Quantum routes one credential to IonQ, Rigetti, IQM, and AQT
    hardware. A backend class being listed does not mean it can be
    submitted to right now — ``accepting_jobs`` is reported separately
    since IBM/Braket-style "operational" status does not capture that.

    Returns:
        List of dicts with keys ``name`` (short_code), ``num_qubits``,
        ``modality``, ``status``, ``accepting_jobs``.
    """
    try:
        from limen.backends.openquantum import list_backend_classes
    except ImportError as exc:
        print(
            "ERROR: openquantum-sdk not installed.\n"
            "Install with: pip install limen[openquantum]\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    classes = list_backend_classes(client_id, client_secret)
    results = []
    for c in classes:
        prefix = c.short_code.split(":", 1)[0]
        num_qubits_match = re.search(r"(\d+)\s*-?qubit", c.description, re.IGNORECASE)
        results.append(
            {
                "name": c.short_code,
                "num_qubits": int(num_qubits_match.group(1)) if num_qubits_match else None,
                "modality": _MODALITY_BY_PROVIDER_PREFIX.get(prefix, "unknown"),
                "status": c.status,
                "accepting_jobs": c.accepting_jobs,
            }
        )
    return results


def scan_validated_backends(results_dir: pathlib.Path) -> dict[str, list[str]]:
    """Scan results/*.json for real job_id evidence, grouped by backend name.

    A backend only counts as validated if a non-null ``job_id`` appears
    in the same result file as a ``backend`` field naming it.

    Returns:
        Dict mapping backend name to the list of job IDs found for it.
    """
    evidence: dict[str, list[str]] = {}
    if not results_dir.is_dir():
        return evidence

    for path in results_dir.glob("*.json"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        backend_names = set(re.findall(r'"backend"\s*:\s*"([^"]+)"', text))
        backend_names.discard("simulator-only")
        backend_names = {b for b in backend_names if "simulator" not in b and "aer" not in b}
        if not backend_names:
            continue
        job_ids = re.findall(r'"job_id"\s*:\s*"([^"]+)"', text)
        if not job_ids:
            continue
        for name in backend_names:
            evidence.setdefault(name, []).extend(job_ids)

    for name in evidence:
        evidence[name] = sorted(set(evidence[name]))
    return evidence


def build_certificate(
    ibm_backends: list[dict[str, Any]],
    braket_devices: list[dict[str, Any]],
    openquantum_backends: list[dict[str, Any]],
    evidence: dict[str, list[str]],
) -> dict[str, Any]:
    """Assemble the fleet certificate from discovery + on-disk job evidence."""
    nodes: list[dict[str, Any]] = []

    for b in ibm_backends:
        job_ids = evidence.get(b["name"], [])
        nodes.append(
            {
                "backend": b["name"],
                "provider": "IBM Quantum Platform",
                "modality": b["modality"],
                "num_qubits": b["num_qubits"],
                "validated": len(job_ids) > 0,
                "job_ids": job_ids,
            }
        )

    for d in braket_devices:
        job_ids = evidence.get(d["name"], [])
        nodes.append(
            {
                "backend": d["name"],
                "provider": "AWS Braket",
                "modality": d["modality"],
                "num_qubits": d["num_qubits"],
                "status": d["status"],
                "validated": len(job_ids) > 0,
                "job_ids": job_ids,
            }
        )

    for o in openquantum_backends:
        job_ids = evidence.get(o["name"], [])
        nodes.append(
            {
                "backend": o["name"],
                "provider": "Open Quantum",
                "modality": o["modality"],
                "num_qubits": o["num_qubits"],
                "status": o["status"],
                "accepting_jobs": o["accepting_jobs"],
                "validated": len(job_ids) > 0,
                "job_ids": job_ids,
            }
        )

    total_qubits = sum(n["num_qubits"] for n in nodes if isinstance(n.get("num_qubits"), int))
    validated_count = sum(1 for n in nodes if n["validated"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "live_query",
        "nodes": nodes,
        "summary": {
            "total_nodes": len(nodes),
            "validated_nodes": validated_count,
            "total_physical_qubits": total_qubits,
        },
    }


def print_report(cert: dict[str, Any]) -> None:
    W = 63
    print(f"── LIMEN Fleet Certificate {'─' * (W - 26)}")
    print(f"  Generated : {cert['generated_at']}")
    print()
    for n in cert["nodes"]:
        mark = "VALIDATED" if n["validated"] else "unvalidated"
        qubits = n["num_qubits"] if n["num_qubits"] is not None else "?"
        print(f"  [{mark:11}] {n['backend']:20} {qubits:>4} qubits  ({n['provider']})")
        if n["job_ids"]:
            print(f"               job_ids: {', '.join(n['job_ids'])}")
    print()
    s = cert["summary"]
    print(f"  Total nodes            : {s['total_nodes']}")
    print(f"  Validated on real jobs : {s['validated_nodes']}")
    print(f"  Total physical qubits  : {s['total_physical_qubits']}")
    print(f"── End {'─' * (W - 6)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ibm-token", default=os.environ.get("IBM_QUANTUM_TOKEN"))
    parser.add_argument("--ibm-crn", default=os.environ.get("IBM_QUANTUM_CRN"))
    parser.add_argument(
        "--braket", action="store_true", help="Also query AWS Braket device status."
    )
    parser.add_argument(
        "--openquantum", action="store_true",
        help="Also query Open Quantum (IonQ/Rigetti/IQM/AQT) backend status.",
    )
    parser.add_argument(
        "--openquantum-client-id", default=os.environ.get("OPENQUANTUM_CLIENT_ID")
    )
    parser.add_argument(
        "--openquantum-client-secret", default=os.environ.get("OPENQUANTUM_CLIENT_SECRET")
    )
    args = parser.parse_args()

    if not args.ibm_token or not args.ibm_crn:
        print(
            "ERROR: IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN are required "
            "(env vars or --ibm-token/--ibm-crn).",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Querying IBM Quantum Platform for operational backends ...")
    ibm_backends = discover_ibm_backends(args.ibm_token, args.ibm_crn)

    braket_devices: list[dict[str, Any]] = []
    if args.braket:
        print("Querying AWS Braket for QPU device status ...")
        braket_devices = discover_braket_devices()

    openquantum_backends: list[dict[str, Any]] = []
    if args.openquantum:
        if not args.openquantum_client_id or not args.openquantum_client_secret:
            print(
                "ERROR: --openquantum requires OPENQUANTUM_CLIENT_ID and "
                "OPENQUANTUM_CLIENT_SECRET (env vars or --openquantum-client-id/secret).",
                file=sys.stderr,
            )
            sys.exit(1)
        print("Querying Open Quantum for backend class status ...")
        openquantum_backends = discover_openquantum_backends(
            args.openquantum_client_id, args.openquantum_client_secret
        )

    print("Scanning results/ for real job-ID evidence ...")
    evidence = scan_validated_backends(_RESULTS_DIR)

    cert = build_certificate(ibm_backends, braket_devices, openquantum_backends, evidence)

    _RESULTS_DIR.mkdir(exist_ok=True)
    _CERT_PATH.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")

    print()
    print_report(cert)
    print()
    print(f"Wrote {_CERT_PATH}")


if __name__ == "__main__":
    main()
