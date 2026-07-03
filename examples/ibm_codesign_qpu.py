# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Closed-loop Stackelberg co-design experiment against a real IBM QPU.

Where examples/ibm_qpu_demo.py runs one open-loop shot, this script closes
the loop: each co-design iteration runs the current encoding's QAOA circuit
on real hardware (ibm_kingston), measures the hardware-noise fraction (total
variation distance between measured and ideal distributions), and feeds it
back into the κ scoring via the chain_break_fraction_fn callback — the IBM
analog of dwave_chain_break_fn.

The Stackelberg leader's chain-strength moves reach the physical circuit
through the cost-Hamiltonian scale (cost_scale = chain_strength / base),
the gate-model analog of annealer penalty strength. The question this
experiment answers: does the loop improve the QPU optimal-shot rate?

Required environment variables (or .env in the project root):
    IBM_QUANTUM_TOKEN  — IBM Quantum Platform API token
    IBM_QUANTUM_CRN    — IBM Quantum instance CRN

Usage::

    python examples/ibm_codesign_qpu.py
"""

import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Windows consoles often default to cp1252, which cannot print κ/β/─.
_reconfigure = getattr(sys.stdout, "reconfigure", None)
if _reconfigure and (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
    _reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv  # type: ignore[import]
    load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.backends.qiskit_backend import ibm_noise_fn, run_qiskit_qpu
from limen.codesign.solver import run_codesign

QUBO: dict[tuple[str, str], float] = {
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x0", "x1"):  2.0,
}

_OPTIMAL_ENERGY = -1.0
_BACKEND_NAME = "ibm_kingston"
_SHOTS = 1000
_MAX_ITERATIONS = 5


def _optimal_rate(energies: list[float]) -> float:
    """Fraction of shots at the optimal energy, as a percentage."""
    if not energies:
        return 0.0
    return (
        sum(1 for e in energies if abs(e - _OPTIMAL_ENERGY) < 1e-9)
        / len(energies)
        * 100.0
    )


def main() -> None:
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")
    if not token or not crn:
        print(
            "ERROR: IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN must be set.",
            file=sys.stderr,
        )
        sys.exit(1)

    W = 65

    # 1. Compile the QUBO.
    graph = from_qubo_dict(QUBO)
    encoding = compile_lexicographic(graph, default_hardware_graph(2))
    base_cs = encoding.chain_strength

    # 2. Baseline open-loop QPU run (cost_scale = 1.0).
    print(f"[1/3] Baseline QPU run on {_BACKEND_NAME} ...")
    baseline = run_qiskit_qpu(
        encoding, token=token, crn=crn,
        backend_name=_BACKEND_NAME, shots=_SHOTS,
    )
    baseline_rate = _optimal_rate(baseline.energies)
    print(f"      optimal rate {baseline_rate:.1f}%  "
          f"(job {baseline.metadata['job_id']})")

    # 3. Closed-loop co-design with real hardware-noise feedback.
    print(f"[2/3] Stackelberg co-design loop "
          f"({_MAX_ITERATIONS} iterations max, 1 QPU job each) ...")
    noise_fn = ibm_noise_fn(
        token=token, crn=crn,
        backend_name=_BACKEND_NAME, shots=_SHOTS,
        base_chain_strength=base_cs,
    )
    cd = run_codesign(
        encoding,
        target_kappa=0.85,
        max_iterations=_MAX_ITERATIONS,
        runs_per_iteration=500,
        seed=42,
        chain_break_fraction_fn=noise_fn,
    )
    qpu_history = noise_fn.history  # type: ignore[attr-defined]

    # 4. Final QPU run with the optimised encoding.
    final_scale = cd.encoding.chain_strength / base_cs
    print(f"[3/3] Final QPU run (cost_scale {final_scale:.4f}) ...")
    final = run_qiskit_qpu(
        cd.encoding, token=token, crn=crn,
        backend_name=_BACKEND_NAME, shots=_SHOTS,
        cost_scale=final_scale,
    )
    final_rate = _optimal_rate(final.energies)

    # 5. Persist raw results first — QPU time is too expensive to lose to a
    # report-formatting failure.
    out_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"codesign_qpu_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(
        json.dumps(
            {
                "backend": _BACKEND_NAME,
                "shots": _SHOTS,
                "qubo": [[list(k), v] for k, v in QUBO.items()],
                "baseline": {
                    "optimal_rate_pct": baseline_rate,
                    "counts": baseline.metadata["counts"],
                    "job_id": baseline.metadata["job_id"],
                },
                "codesign": {
                    "converged": cd.converged,
                    "kappa": cd.kappa,
                    "kappa_std": cd.kappa_std,
                    "iterations": cd.iterations,
                    "chain_strength_history": cd.chain_strength_history,
                    "confidence_history": cd.confidence_history,
                    "solver_backend": cd.metadata["solver_backend"],
                    "qpu_history": qpu_history,
                },
                "final": {
                    "cost_scale": final_scale,
                    "optimal_rate_pct": final_rate,
                    "counts": final.metadata["counts"],
                    "job_id": final.metadata["job_id"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Raw results saved to {out_path}")

    # 6. Report.
    print()
    print(f"── LIMEN Closed-Loop QPU Co-Design {'─' * (W - 35)}")
    print(f"  Problem  : trivial 2-var QUBO, optimal energy {_OPTIMAL_ENERGY}")
    print(f"  Backend  : {_BACKEND_NAME} (real hardware), {_SHOTS} shots/job")
    print(f"  Feedback : TVD(measured, ideal) → κ cbf term")
    print(f"  Solver   : {cd.metadata['solver_backend']} "
          f"(limen_core fallback chain)")
    print()
    print("  Iter  chain_str  cost_scale  confidence  noise(TVD)  QPU opt %")
    for i, h in enumerate(qpu_history):
        conf = cd.confidence_history[i] if i < len(cd.confidence_history) else float("nan")
        print(
            f"  {i:>4}  {h['chain_strength']:>9.4f}  {h['cost_scale']:>10.4f}"
            f"  {conf:>10.4f}  {h['tvd']:>10.4f}"
            f"  {h['qpu_optimal_rate'] * 100:>8.1f}"
        )
    print()
    print(f"  Converged      : {cd.converged}  (κ = {cd.kappa:.4f}, "
          f"target 0.85, κ std {cd.kappa_std:.4f})")
    print(f"  Chain strength : {base_cs:.4f} → {cd.encoding.chain_strength:.4f}")
    print()
    print("  QPU optimal-shot rate")
    print(f"    Baseline (open loop)  : {baseline_rate:.1f}%")
    print(f"    Final (co-designed)   : {final_rate:.1f}%")
    print(f"    Improvement           : {final_rate - baseline_rate:+.1f} pp")
    print(f"── End {'─' * (W - 7)}")


if __name__ == "__main__":
    main()
