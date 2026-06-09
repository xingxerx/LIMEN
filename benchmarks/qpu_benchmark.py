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
"""LIMEN scaling benchmark: simulator vs real IBM QPU.

Runs a ladder of progressively harder QUBOs (trivial 2-var, then ring
Max-Cut at n = 3, 4, 6, 8, 10, 12) through the full LIMEN pipeline and
records, for each problem:

  - validator confidence (probabilistic validator, 1000 runs)
  - calibration margin κ from the Stackelberg co-design loop (simulation mode)
  - ideal QAOA optimal-shot rate (noiseless statevector, β=γ=0.1, p=1)
  - real QPU optimal-shot rate (ibm_kingston via SamplerV2)

All QPU circuits are batched into a single SamplerV2 job to minimise
queue overhead. Results are written as a Markdown table to
benchmarks/RESULTS.md and as raw JSON to results/.

If IBM_QUANTUM_TOKEN / IBM_QUANTUM_CRN are not set the QPU column is
skipped and the benchmark runs simulator-only.

Usage::

    python benchmarks/qpu_benchmark.py
"""

import json
import os
import pathlib
import sys
import time
from itertools import product as iproduct
from typing import Any

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
from limen.backends.qiskit_backend import (
    _bits_to_assignment,
    _build_qaoa_ansatz,
    _ideal_distribution,
    _qubo_energy,
)
from limen.codesign.solver import run_codesign
from limen.validator.validator import validate

_BACKEND_NAME = "ibm_kingston"
_SHOTS = 1000
_REPS = 1

# ---------------------------------------------------------------------------
# Problem ladder
# ---------------------------------------------------------------------------

TRIVIAL_QUBO: dict[tuple[str, str], float] = {
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x0", "x1"):  2.0,
}


def ring_max_cut_qubo(n: int) -> dict[tuple[str, str], float]:
    """Max-Cut QUBO on the cycle graph C_n with variables x00..x{n-1}."""
    qubo: dict[tuple[str, str], float] = {}
    names = [f"x{i:02d}" for i in range(n)]
    for i in range(n):
        u, v = names[i], names[(i + 1) % n]
        key = (min(u, v), max(u, v))
        qubo[key] = qubo.get(key, 0.0) + 1.0
        qubo[(u, u)] = qubo.get((u, u), 0.0) - 1.0
        qubo[(v, v)] = qubo.get((v, v), 0.0) - 1.0
    return qubo


PROBLEMS: list[tuple[str, int, dict[tuple[str, str], float]]] = [
    ("trivial-2", 2, TRIVIAL_QUBO),
    ("ring-3", 3, ring_max_cut_qubo(3)),
    ("ring-4", 4, ring_max_cut_qubo(4)),
    ("ring-6", 6, ring_max_cut_qubo(6)),
    ("ring-8", 8, ring_max_cut_qubo(8)),
    ("ring-10", 10, ring_max_cut_qubo(10)),
    ("ring-12", 12, ring_max_cut_qubo(12)),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def brute_force_optimum(qubo: dict[tuple[str, str], float]) -> float:
    """Exact minimum QUBO energy by enumeration (n ≤ ~16)."""
    variables = sorted({name for pair in qubo for name in pair})
    return min(
        _qubo_energy(qubo, dict(zip(variables, bits)))
        for bits in iproduct((0, 1), repeat=len(variables))
    )


def ideal_optimal_rate(
    qubo: dict[tuple[str, str], float], optimal_energy: float
) -> float:
    """Noiseless QAOA optimal-shot probability via statevector."""
    variables = sorted({name for pair in qubo for name in pair})
    ansatz = _build_qaoa_ansatz(qubo, variables, _REPS, 1.0)
    ideal = _ideal_distribution(ansatz, [0.1] * ansatz.num_parameters)
    rate = 0.0
    for bs, p in ideal.items():
        energy = _qubo_energy(qubo, _bits_to_assignment(bs, variables))
        if abs(energy - optimal_energy) < 1e-9:
            rate += p
    return rate


def qpu_optimal_rates(
    rows: list[dict[str, Any]], token: str, crn: str
) -> str:
    """Run all problems as one batched SamplerV2 job; fill rows in place.

    Returns the job id.
    """
    from qiskit_ibm_runtime import (  # type: ignore[import]
        QiskitRuntimeService,
        SamplerV2,
    )
    from qiskit.transpiler.preset_passmanagers import (  # type: ignore[import]
        generate_preset_pass_manager,
    )

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform", token=token, instance=crn
    )
    backend = service.backend(_BACKEND_NAME)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)

    pubs = []
    for row in rows:
        qubo = row["_qubo"]
        variables = sorted({name for pair in qubo for name in pair})
        ansatz = _build_qaoa_ansatz(qubo, variables, _REPS, 1.0)
        ansatz.measure_all()
        transpiled = pm.run(ansatz)
        row["circuit_depth"] = transpiled.depth()
        pubs.append((transpiled, [0.1] * ansatz.num_parameters))

    print(f"  Submitting 1 batched job ({len(pubs)} circuits, "
          f"{_SHOTS} shots each) to {_BACKEND_NAME} ...")
    sampler = SamplerV2(mode=backend)
    job = sampler.run(pubs, shots=_SHOTS)
    print(f"  Job id: {job.job_id()} — waiting for results ...")
    results = job.result()

    for row, pub_result in zip(rows, results):
        qubo = row["_qubo"]
        variables = sorted({name for pair in qubo for name in pair})
        counts: dict[str, int] = pub_result.data.meas.get_counts()
        total = sum(counts.values())
        hit = sum(
            cnt
            for bs, cnt in counts.items()
            if abs(
                _qubo_energy(qubo, _bits_to_assignment(bs, variables))
                - row["optimal_energy"]
            ) < 1e-9
        )
        row["qpu_optimal_rate"] = hit / total if total else 0.0
        row["qpu_counts_top8"] = dict(
            sorted(counts.items(), key=lambda x: -x[1])[:8]
        )
    return job.job_id()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")
    qpu_enabled = bool(token and crn)

    rows: list[dict[str, Any]] = []
    for name, n, qubo in PROBLEMS:
        print(f"[{name}] compiling and validating ...")
        graph = from_qubo_dict(qubo)
        encoding = compile_lexicographic(graph, default_hardware_graph(n))
        vr = validate(encoding, runs=1000, seed=42)

        cd = run_codesign(
            encoding,
            target_kappa=0.85,
            max_iterations=10,
            runs_per_iteration=500,
            seed=42,
        )

        optimal_energy = brute_force_optimum(qubo)
        sim_rate = ideal_optimal_rate(qubo, optimal_energy)

        rows.append(
            {
                "problem": name,
                "n_vars": n,
                "optimal_energy": optimal_energy,
                "confidence": vr.confidence,
                "kappa": cd.kappa,
                "kappa_converged": cd.converged,
                "chain_strength": cd.encoding.chain_strength,
                "sim_optimal_rate": sim_rate,
                "qpu_optimal_rate": None,
                "circuit_depth": None,
                "_qubo": qubo,
            }
        )

    job_id = None
    if token and crn:
        print(f"\nRunning QPU pass on {_BACKEND_NAME} ...")
        job_id = qpu_optimal_rates(rows, token, crn)
    else:
        print("\nIBM credentials not set — skipping QPU pass.")

    # ── Markdown table ────────────────────────────────────────────────
    lines = [
        "# LIMEN scaling benchmark — simulator vs IBM QPU",
        "",
        f"- Date: {time.strftime('%Y-%m-%d')}",
        f"- Backend: {_BACKEND_NAME} (Heron R2, 156 qubits)"
        if qpu_enabled else "- Backend: simulator only (no IBM credentials)",
        f"- QAOA: p={_REPS}, fixed β=γ=0.1, {_SHOTS} shots, "
        "optimization_level=1 transpilation",
        "- Validator: 1000 runs, seed 42. "
        "κ from Stackelberg co-design (simulation mode, ≤10 iterations).",
        f"- QPU job id: {job_id}" if job_id else "",
        "",
        "| Problem | Vars | Depth | Confidence | κ | Sim optimal % | QPU optimal % |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        qpu_pct = (
            f"{r['qpu_optimal_rate'] * 100:.1f}"
            if r["qpu_optimal_rate"] is not None
            else "—"
        )
        depth = r["circuit_depth"] if r["circuit_depth"] is not None else "—"
        lines.append(
            f"| {r['problem']} | {r['n_vars']} | {depth} "
            f"| {r['confidence'] * 100:.1f}% | {r['kappa']:.3f} "
            f"| {r['sim_optimal_rate'] * 100:.1f} | {qpu_pct} |"
        )
    table = "\n".join(line for line in lines if line is not None) + "\n"

    out_md = pathlib.Path(__file__).resolve().parent / "RESULTS.md"
    out_md.write_text(table, encoding="utf-8")

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    serialisable = [
        {k: v for k, v in r.items() if k != "_qubo"} for r in rows
    ]
    out_json.write_text(
        json.dumps(
            {
                "backend": _BACKEND_NAME if qpu_enabled else "simulator-only",
                "shots": _SHOTS,
                "reps": _REPS,
                "job_id": job_id,
                "rows": serialisable,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(table)
    print(f"Markdown table : {out_md}")
    print(f"Raw JSON       : {out_json}")


if __name__ == "__main__":
    main()
