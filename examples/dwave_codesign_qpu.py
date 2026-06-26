# Copyright 2026 LIMEN Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Closed-loop Stackelberg co-design experiment against a real D-Wave QPU.

The D-Wave analog of examples/ibm_codesign_qpu.py. Where that script feeds
IBM hardware noise (TVD between measured and ideal distributions) back into
the kappa scoring, this script feeds the real chain-break fraction measured
on a live D-Wave annealer back into the same loop via the
chain_break_fraction_fn callback (dwave_chain_break_fn) — this is the path
that dwave_chain_break_fn's docstring always described but that, before this
script existed, had never actually been wired up against live hardware.

The Stackelberg leader's chain-strength moves reach the physical sampler
directly: PhysicalEncoding.chain_strength is the literal chain_strength
argument passed to D-Wave's sampler.sample(). The question this experiment
answers: does the loop reduce the QPU's chain-break fraction and improve its
optimal-shot rate?

Required environment variables (or .env in the project root):
    DWAVE_API_TOKEN     — D-Wave Leap API token
    DWAVE_API_ENDPOINT  — D-Wave Leap API endpoint URL (optional; falls back
                           to the Ocean SDK's default configuration if unset)

Usage::

    python examples/dwave_codesign_qpu.py
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
from limen.backends.dwave import dwave_chain_break_fn, run_dwave
from limen.codesign.solver import run_codesign

QUBO: dict[tuple[str, str], float] = {
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x0", "x1"):  2.0,
}

_OPTIMAL_ENERGY = -1.0
_NUM_READS = 1000
_MAX_ITERATIONS = 5


def _optimal_rate(energies: list[float]) -> float:
    """Fraction of reads at the optimal energy, as a percentage."""
    if not energies:
        return 0.0
    return (
        sum(1 for e in energies if abs(e - _OPTIMAL_ENERGY) < 1e-9)
        / len(energies)
        * 100.0
    )


def main() -> None:
    token = os.environ.get("DWAVE_API_TOKEN")
    endpoint = os.environ.get("DWAVE_API_ENDPOINT")
    if not token:
        print("ERROR: DWAVE_API_TOKEN must be set.", file=sys.stderr)
        sys.exit(1)

    W = 65

    # 1. Compile the QUBO.
    graph = from_qubo_dict(QUBO)
    encoding = compile_lexicographic(graph, default_hardware_graph(2))
    base_cs = encoding.chain_strength

    # 2. Baseline open-loop QPU run.
    print("[1/3] Baseline QPU run on a real D-Wave annealer ...")
    baseline = run_dwave(
        encoding, num_reads=_NUM_READS,
        use_qpu=True, qpu_endpoint=endpoint, qpu_token=token,
    )
    baseline_rate = _optimal_rate(baseline.energies)
    print(f"      optimal rate {baseline_rate:.1f}%  "
          f"chain-break fraction {baseline.chain_break_fraction:.4f}")

    # 3. Closed-loop co-design with real chain-break-fraction feedback.
    print(f"[2/3] Stackelberg co-design loop "
          f"({_MAX_ITERATIONS} iterations max, 1 QPU submission each) ...")
    cbf_fn = dwave_chain_break_fn(
        num_reads=500, use_qpu=True,
        qpu_endpoint=endpoint, qpu_token=token,
    )
    cd = run_codesign(
        encoding,
        target_kappa=0.85,
        max_iterations=_MAX_ITERATIONS,
        runs_per_iteration=500,
        seed=42,
        chain_break_fraction_fn=cbf_fn,
    )
    qpu_history = cbf_fn.history  # type: ignore[attr-defined]

    # 4. Final QPU run with the optimised encoding.
    print(f"[3/3] Final QPU run (chain strength {cd.encoding.chain_strength:.4f}) ...")
    final = run_dwave(
        cd.encoding, num_reads=_NUM_READS,
        use_qpu=True, qpu_endpoint=endpoint, qpu_token=token,
    )
    final_rate = _optimal_rate(final.energies)

    # 5. Persist raw results first — QPU time is too expensive to lose to a
    # report-formatting failure.
    out_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"codesign_dwave_qpu_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(
        json.dumps(
            {
                "num_reads": _NUM_READS,
                "qubo": [[list(k), v] for k, v in QUBO.items()],
                "baseline": {
                    "optimal_rate_pct": baseline_rate,
                    "chain_break_fraction": baseline.chain_break_fraction,
                    "timing": baseline.timing,
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
                    "chain_strength": cd.encoding.chain_strength,
                    "optimal_rate_pct": final_rate,
                    "chain_break_fraction": final.chain_break_fraction,
                    "timing": final.timing,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Raw results saved to {out_path}")

    # 6. Report.
    print()
    print(f"── LIMEN Closed-Loop D-Wave QPU Co-Design {'─' * (W - 42)}")
    print(f"  Problem  : trivial 2-var QUBO, optimal energy {_OPTIMAL_ENERGY}")
    print(f"  Backend  : real D-Wave QPU, {_NUM_READS} reads/job")
    print(f"  Feedback : measured chain-break fraction → κ cbf term")
    print(f"  Solver   : {cd.metadata['solver_backend']} "
          f"(limen_core fallback chain)")
    print()
    print("  Iter  chain_str  confidence  chain-break  best_energy")
    for i, h in enumerate(qpu_history):
        conf = cd.confidence_history[i] if i < len(cd.confidence_history) else float("nan")
        print(
            f"  {i:>4}  {h['chain_strength']:>9.4f}"
            f"  {conf:>10.4f}  {h['chain_break_fraction']:>11.4f}"
            f"  {h['best_energy']:>11.4f}"
        )
    print()
    print(f"  Converged      : {cd.converged}  (κ = {cd.kappa:.4f}, "
          f"target 0.85, κ std {cd.kappa_std:.4f})")
    print(f"  Chain strength : {base_cs:.4f} → {cd.encoding.chain_strength:.4f}")
    print()
    print("  QPU optimal-shot rate")
    print(f"    Baseline (open loop)  : {baseline_rate:.1f}%  "
          f"(chain-break {baseline.chain_break_fraction:.4f})")
    print(f"    Final (co-designed)   : {final_rate:.1f}%  "
          f"(chain-break {final.chain_break_fraction:.4f})")
    print(f"    Improvement           : {final_rate - baseline_rate:+.1f} pp")
    print(f"── End {'─' * (W - 7)}")


if __name__ == "__main__":
    main()
