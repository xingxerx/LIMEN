# Copyright (C) 2026 xingxerx
#
# Licensed under the Elastic License 2.0. See the LICENSE file in the
# repository root for the full terms.

"""Shared plumbing for the tier-0 study runners: where results go, which
local simulator is the exact baseline for a given width, and timing."""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from limen.router import (
    LOCAL_GPU_BACKEND_NAME,
    gpu_qubit_cap,
    local_gpu_enabled,
    query_free_vram_mib,
)

# Pure-Python statevector (limen.gates.simulator) enumerates 2**n outcomes.
PY_STATEVECTOR_MAX_QUBITS = 16
# CPU Aer statevector, double precision: 2**26 * 16 B = 1 GiB.
CPU_AER_MAX_QUBITS = 26


@dataclass(frozen=True)
class StudyContext:
    """Resolved environment for one study run."""

    results_dir: pathlib.Path | None
    local_gpu: bool
    free_vram_mib: int | None
    gpu_cap: int

    @property
    def exact_max_qubits(self) -> int:
        """Widest problem that still has an exact local baseline here."""
        return self.gpu_cap if self.local_gpu else CPU_AER_MAX_QUBITS

    def summary(self) -> dict[str, Any]:
        return {
            "results_dir": str(self.results_dir) if self.results_dir else None,
            "local_gpu": self.local_gpu,
            "free_vram_mib": self.free_vram_mib,
            "gpu_cap": self.gpu_cap,
            "exact_max_qubits": self.exact_max_qubits,
        }


def context(
    results_dir: pathlib.Path | str | None = None, local_gpu: bool | None = None
) -> StudyContext:
    if local_gpu is None:
        local_gpu = local_gpu_enabled()
    free = query_free_vram_mib() if local_gpu else None
    cap = gpu_qubit_cap(free) if local_gpu else 0
    if local_gpu and cap <= 0:
        local_gpu = False
    return StudyContext(
        results_dir=pathlib.Path(results_dir) if results_dir is not None else None,
        local_gpu=local_gpu,
        free_vram_mib=free,
        gpu_cap=cap,
    )


def baseline_backend(n_qubits: int, ctx: StudyContext) -> tuple[str, str]:
    """(backend, qpu_backend_name) run_pipeline kwargs for an exact baseline.

    Pure-Python statevector up to 16 qubits when no GPU tier, Aer on the
    GPU tier when enabled and the problem fits its live cap, CPU Aer
    otherwise up to 26 qubits.
    """
    if n_qubits <= PY_STATEVECTOR_MAX_QUBITS and not ctx.local_gpu:
        return "statevector", "aer_simulator"
    if ctx.local_gpu and n_qubits <= ctx.gpu_cap:
        return "aer", LOCAL_GPU_BACKEND_NAME
    if n_qubits > CPU_AER_MAX_QUBITS:
        raise ValueError(
            f"{n_qubits} qubits has no exact local baseline here (GPU tier "
            f"{'cap ' + str(ctx.gpu_cap) + 'q' if ctx.local_gpu else 'off'}, "
            f"CPU Aer max {CPU_AER_MAX_QUBITS}q)"
        )
    return "aer", "aer_simulator"


def exact_prob_fn(n_qubits: int, ctx: StudyContext, shots: int = 4000) -> Any:
    """Callable CircuitIR -> qubit-0-first probability dict (exact when it fits)."""
    from limen.pipeline import _AER_EXACT_MAX_QUBITS, _aer_exact_probabilities, _aer_probabilities

    backend, name = baseline_backend(n_qubits, ctx)
    if backend == "statevector":
        from limen.gates.simulator import probabilities

        return probabilities
    if n_qubits <= _AER_EXACT_MAX_QUBITS:
        return lambda circuit: _aer_exact_probabilities(circuit, name)
    return lambda circuit: _aer_probabilities(circuit, name, shots)


def z_marginals(dist: dict[str, float], n: int) -> dict[int, float]:
    """<Z_i> per qubit from a qubit-0-first probability dict."""
    out = {i: 0.0 for i in range(n)}
    for bits, p in dist.items():
        for i, b in enumerate(bits):
            out[i] += p * (1.0 - 2.0 * int(b))
    return out


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H%M%SZ", time.gmtime())


def jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(asdict(obj))
    if hasattr(obj, "to_dict"):
        return jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, pathlib.Path):
        return str(obj)
    return obj


def write_record(
    ctx: StudyContext, subdir: str, stem: str, record: dict[str, Any]
) -> pathlib.Path | None:
    """Write *record* to ``results_dir/study/<subdir>/<stem>-<ts>.json``.

    Study output lives under ``study/`` so limen.router.history.scan_results
    (which globs only ``results_dir/*.json``) never ingests a simulated
    number as a hardware measurement. gap.py's real-hardware record is the
    deliberate exception and writes to the results root itself.
    """
    if ctx.results_dir is None:
        return None
    out_dir = ctx.results_dir / "study" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}-{timestamp()}.json"
    path.write_text(json.dumps(jsonable(record), indent=2))
    return path


class Timer:
    seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.seconds = round(time.perf_counter() - self._t0, 3)


def local_search_energy(
    qubo: dict[tuple[str, str], float], order: list[str], seed: int = 0, restarts: int = 20
) -> float:
    """Cheap classical reference: best-of-restarts 1-flip local search.

    Used where no exact optimum exists (brute force stops at 20 vars): a
    lower bound on how good a solution should be, not the optimum itself.
    Shared by cutting_study (scaling baseline) and bench (ratio reference
    for wide Max-Cut instances).
    """
    import random

    from limen.cutting.qubo_bridge import classical_energy

    rng = random.Random(seed)
    best = float("inf")
    for _ in range(restarts):
        x = {v: rng.randint(0, 1) for v in order}
        e = classical_energy(qubo, x)
        improved = True
        while improved:
            improved = False
            for v in order:
                x[v] ^= 1
                e2 = classical_energy(qubo, x)
                if e2 < e - 1e-12:
                    e = e2
                    improved = True
                else:
                    x[v] ^= 1
        best = min(best, e)
    return best


def distribution_gap(
    ideal: dict[str, float],
    measured: dict[str, float],
    energy_of: Any = None,
) -> dict[str, float]:
    """Distribution-level ideal-vs-measured metrics (both qubit-0-first probs).

    ``success_probability`` (mass on the single optimal bitstring) falls
    like 2**-n for p=1 QAOA, so past ~16 qubits a success deficit is pure
    sampling noise (seen: 0.0005 vs 0.001 at 20 qubits). These compare the
    whole distribution instead:

    * ``distribution_fidelity``: classical (Bhattacharyya) fidelity
      (sum sqrt(p*q))**2, 1.0 = identical.
    * ``total_variation_distance``: 0.5 * sum |p - q|, 0.0 = identical.
    * with *energy_of* (bitstring -> energy): ``expected_energy_ideal``,
      ``expected_energy_measured`` and their difference, the number a
      QUBO user actually loses to hardware noise.
    """
    import math

    keys = set(ideal) | set(measured)
    fid = sum(math.sqrt(ideal.get(k, 0.0) * measured.get(k, 0.0)) for k in keys) ** 2
    tvd = 0.5 * sum(abs(ideal.get(k, 0.0) - measured.get(k, 0.0)) for k in keys)
    out: dict[str, float] = {"distribution_fidelity": fid, "total_variation_distance": tvd}
    if energy_of is not None:
        e_ideal = sum(p * energy_of(k) for k, p in ideal.items())
        e_meas = sum(p * energy_of(k) for k, p in measured.items())
        out.update(
            expected_energy_ideal=e_ideal,
            expected_energy_measured=e_meas,
            expected_energy_gap=e_meas - e_ideal,
        )
    return out
