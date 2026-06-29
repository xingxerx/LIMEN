# Copyright (C) 2026 Jemone McCubbin / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""COBYLA-tuned QAOA, validated then run in parallel across two real QPUs.

Combines the three best-practice steps discussed for using the free-tier
IBM Quantum credits efficiently while ibm_marrakesh's ~1900-job backlog
drains:

  1. Tune QAOA beta/gamma with COBYLA against the noiseless Aer
     statevector distribution (zero QPU cost).
  2. Gate the real submission on a minimum ideal optimal-shot rate —
     never burn a queue slot on parameters that would have produced
     noise-dominated results even without hardware errors.
  3. Submit the validated circuit to two independently-queued, already
     -validated backends (ibm_kingston, ibm_fez) in parallel rather than
     sequentially, since their queues are independent.

Note on "multi-backend": this runs the SAME circuit on two physical
QPUs in parallel for throughput and cross-hardware noise comparison.
It does not entangle qubits across chips -- that is not physically
possible with two separate QPUs. Genuine distributed *compilation*
(splitting one logical problem across peer LIMEN nodes) is a separate,
existing feature -- see examples/distributed_two_node.py.

Usage::

    python examples/cobyla_multi_backend.py
"""

import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product as iproduct
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

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
    QiskitResult,
    _bits_to_assignment,
    _build_qaoa_ansatz,
    _ideal_distribution,
    _qubo_energy,
    run_qiskit_qpu,
)

# ---------------------------------------------------------------------------
# Problem: ring-8 Max-Cut (8 qubits, well within Aer statevector limits and
# small enough to queue quickly on both backends).
# ---------------------------------------------------------------------------

_N = 8
_REPS = 2
_SHOTS = 1000
_MIN_IDEAL_OPTIMAL_RATE = 0.01  # abort submission below this (pre-tuning baseline is ~0.4%)
_BACKENDS = ["ibm_kingston", "ibm_fez"]


def ring_max_cut_qubo(n: int) -> dict[tuple[str, str], float]:
    qubo: dict[tuple[str, str], float] = {}
    names = [f"x{i:02d}" for i in range(n)]
    for i in range(n):
        u, v = names[i], names[(i + 1) % n]
        key = (min(u, v), max(u, v))
        qubo[key] = qubo.get(key, 0.0) + 1.0
        qubo[(u, u)] = qubo.get((u, u), 0.0) - 1.0
        qubo[(v, v)] = qubo.get((v, v), 0.0) - 1.0
    return qubo


def brute_force_optimum(qubo: dict[tuple[str, str], float]) -> float:
    variables = sorted({name for pair in qubo for name in pair})
    return min(
        _qubo_energy(qubo, dict(zip(variables, bits)))
        for bits in iproduct((0, 1), repeat=len(variables))
    )


def ideal_optimal_rate(
    qubo: dict[tuple[str, str], float],
    variables: list[str],
    params: list[float],
    optimal_energy: float,
) -> float:
    """Noiseless optimal-shot probability for a given QAOA parameter vector."""
    ansatz = _build_qaoa_ansatz(qubo, variables, _REPS, 1.0)
    dist = _ideal_distribution(ansatz, params)
    if not dist:
        return 0.0
    rate = 0.0
    for bs, p in dist.items():
        energy = _qubo_energy(qubo, _bits_to_assignment(bs, variables))
        if abs(energy - optimal_energy) < 1e-9:
            rate += p
    return rate


# ---------------------------------------------------------------------------
# Step 1: COBYLA tuning against the noiseless Aer statevector.
# ---------------------------------------------------------------------------

def tune_with_cobyla(
    qubo: dict[tuple[str, str], float],
    variables: list[str],
    optimal_energy: float,
) -> tuple[list[float], float, float]:
    """Return (tuned_params, baseline_rate, tuned_rate)."""
    from scipy.optimize import minimize

    n_params = 2 * _REPS  # one (gamma, beta) pair per layer
    x0 = [0.1] * n_params

    def objective(x: list[float]) -> float:
        rate = ideal_optimal_rate(qubo, variables, list(x), optimal_energy)
        return -rate  # COBYLA minimizes; we want to maximize optimal rate

    baseline_rate = ideal_optimal_rate(qubo, variables, x0, optimal_energy)

    print(f"  Baseline (beta=gamma=0.1) ideal optimal rate: {baseline_rate * 100:.2f}%")
    print("  Running COBYLA against the noiseless Aer statevector ...")
    result = minimize(objective, x0, method="COBYLA", options={"maxiter": 200, "rhobeg": 0.3})
    tuned_params = list(result.x)
    tuned_rate = ideal_optimal_rate(qubo, variables, tuned_params, optimal_energy)
    print(f"  Tuned ideal optimal rate: {tuned_rate * 100:.2f}%  (params={tuned_params})")
    return tuned_params, baseline_rate, tuned_rate


# ---------------------------------------------------------------------------
# Step 2: pre-submission validation gate.
# ---------------------------------------------------------------------------

def validate_before_submit(tuned_rate: float, threshold: float) -> None:
    if tuned_rate < threshold:
        raise RuntimeError(
            f"Refusing to submit: tuned ideal optimal rate {tuned_rate * 100:.2f}% "
            f"is below the {threshold * 100:.2f}% minimum. Re-tune or pick a smaller "
            "problem before spending a real QPU job."
        )
    print(f"  Validation gate passed ({tuned_rate * 100:.2f}% >= {threshold * 100:.2f}%) "
          "-- proceeding to real hardware.")


# ---------------------------------------------------------------------------
# Step 3: parallel submission to two independently-validated backends.
# ---------------------------------------------------------------------------

def submit_to_backend(
    backend_name: str,
    encoding: Any,
    token: str,
    crn: str,
    tuned_params: list[float],
) -> dict[str, Any]:
    """Run the tuned QAOA ansatz on one backend; tuned_params override the fixed 0.1/0.1."""
    qubo = encoding.qubo
    variables: list[str] = sorted({name for pair in qubo for name in pair})

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2  # type: ignore[import]
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # type: ignore[import]

    ansatz = _build_qaoa_ansatz(qubo, variables, _REPS, 1.0)
    measured = ansatz.copy()
    measured.measure_all()

    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=crn)
    backend = service.backend(backend_name)
    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    transpiled = pm.run(measured)

    t0 = time.time()
    sampler = SamplerV2(mode=backend)
    job = sampler.run([(transpiled, tuned_params)], shots=_SHOTS)
    print(f"  [{backend_name}] job id: {job.job_id()} -- waiting ...")
    pub_result = job.result()[0]
    elapsed = time.time() - t0

    counts: dict[str, int] = pub_result.data.meas.get_counts()
    total = sum(counts.values())
    optimal_energy = brute_force_optimum(qubo)
    hit = sum(
        cnt for bs, cnt in counts.items()
        if abs(_qubo_energy(qubo, _bits_to_assignment(bs, variables)) - optimal_energy) < 1e-9
    )

    return {
        "backend": backend_name,
        "job_id": job.job_id(),
        "circuit_depth": transpiled.depth(),
        "shots": _SHOTS,
        "elapsed_seconds": elapsed,
        "qpu_optimal_rate": hit / total if total else 0.0,
        "counts_top8": dict(sorted(counts.items(), key=lambda x: -x[1])[:8]),
    }


def main() -> None:
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    crn = os.environ.get("IBM_QUANTUM_CRN")
    if not token or not crn:
        print("ERROR: IBM_QUANTUM_TOKEN / IBM_QUANTUM_CRN not set.", file=sys.stderr)
        sys.exit(1)

    qubo = ring_max_cut_qubo(_N)
    variables = sorted({name for pair in qubo for name in pair})
    optimal_energy = brute_force_optimum(qubo)

    print(f"Problem: ring-{_N} Max-Cut, optimal energy = {optimal_energy}")
    print()
    print("Step 1/3: COBYLA tuning against Aer statevector ...")
    tuned_params, baseline_rate, tuned_rate = tune_with_cobyla(qubo, variables, optimal_energy)

    print()
    print("Step 2/3: pre-submission validation gate ...")
    validate_before_submit(tuned_rate, _MIN_IDEAL_OPTIMAL_RATE)

    print()
    print(f"Step 3/3: parallel submission to {_BACKENDS} ...")
    graph = from_qubo_dict(qubo)
    encoding = compile_lexicographic(graph, default_hardware_graph(_N))

    with ThreadPoolExecutor(max_workers=len(_BACKENDS)) as pool:
        futures = [
            pool.submit(submit_to_backend, b, encoding, token, crn, tuned_params)
            for b in _BACKENDS
        ]
        backend_results = [f.result() for f in futures]

    print()
    print("── Results ──────────────────────────────────────────────")
    for r in backend_results:
        print(f"  {r['backend']:>14}: optimal rate {r['qpu_optimal_rate']*100:5.2f}%  "
              f"depth={r['circuit_depth']}  job={r['job_id']}  ({r['elapsed_seconds']:.1f}s)")

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"cobyla_multi_backend_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(
        json.dumps(
            {
                "problem": f"ring-{_N}-maxcut",
                "optimal_energy": optimal_energy,
                "reps": _REPS,
                "baseline_ideal_optimal_rate": baseline_rate,
                "tuned_ideal_optimal_rate": tuned_rate,
                "tuned_params": tuned_params,
                "validation_threshold": _MIN_IDEAL_OPTIMAL_RATE,
                "backend_results": backend_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    main()
