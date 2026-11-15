# Copyright (C) 2026 xingxerx
#
# Licensed under the Elastic License 2.0. See the LICENSE file in the
# repository root for the full terms.

"""CLI: ``python -m limen.study <gap|cut|bench|verify|all> [options]``.

Sizes default to a quick pass that finishes on a CPU in minutes. Pass
``--gpu`` (or set LIMEN_LOCAL_GPU=1) and larger ``--n`` values to use the
tier-0 cap. ``gap`` runs with a simulated noisy backend unless ``--live``
plus IBM credentials are given, in which case it spends one job.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

from limen.study import (
    context,
    cutting_scaling,
    cutting_validity,
    gap_live,
    gap_offline_demo,
    run_bench,
    verify_demo,
)
from limen.study._common import jsonable


def _print(record: dict[str, Any], keys: tuple[str, ...]) -> None:
    for k in keys:
        if k in record:
            v = record[k]
            print(f"  {k}: {json.dumps(jsonable(v)) if isinstance(v, (dict, list)) else v}")
    if record.get("path"):
        print(f"  written: {record['path']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m limen.study", description=__doc__)
    p.add_argument("what", choices=["gap", "cut", "bench", "verify", "all"])
    p.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results"))
    p.add_argument("--gpu", action="store_true", help="force the local GPU tier on (else LIMEN_LOCAL_GPU)")
    p.add_argument("--n", type=int, default=None, help="problem width for gap/cut-validity/verify (vars)")
    p.add_argument("--n-big", type=int, default=None, help="width for cut-scaling (default 2x cap or 40)")
    p.add_argument("--max-sub", type=int, default=None, help="cutting piece size (default: GPU cap or 20)")
    p.add_argument("--shots", type=int, default=4000)
    p.add_argument("--backend", default="ibm_fez", help="hardware backend name for gap")
    p.add_argument("--live", action="store_true", help="gap: submit a real job (IBM_QUANTUM_TOKEN/CRN)")
    p.add_argument("--tsp", default="3,4", help="bench: comma list of TSP city counts")
    p.add_argument("--maxcut", default="12,20", help="bench: comma list of Max-Cut node counts")
    p.add_argument("--json", action="store_true", help="dump full records as JSON to stdout")
    a = p.parse_args(argv)

    ctx = context(a.results_dir, local_gpu=True if a.gpu else None)
    print(f"context: {json.dumps(ctx.summary())}")
    out: dict[str, Any] = {}

    if a.what in ("gap", "all"):
        from limen.frontends.maxcut import build_max_cut_qubo, random_regular_edges

        n = a.n or 10
        qubo = build_max_cut_qubo(random_regular_edges(n, 3, 0))
        if a.live:
            token, crn = os.environ.get("IBM_QUANTUM_TOKEN"), os.environ.get("IBM_QUANTUM_CRN")
            if not (token and crn):
                print("gap --live needs IBM_QUANTUM_TOKEN and IBM_QUANTUM_CRN", file=sys.stderr)
                return 2
            from limen.router import RouterMemory

            mem = RouterMemory.in_results_dir(a.results_dir)
            try:
                rec = gap_live(qubo, a.backend, ctx=ctx, token=token, crn=crn, memory=mem)
            finally:
                mem.close()
        else:
            rec = gap_offline_demo(qubo, a.backend, ctx=ctx)
        print(f"gap ({'live' if a.live else 'simulated hardware'}, {n} vars, {a.backend}):")
        _print(rec, ("expected_energy_gap", "distribution_fidelity", "total_variation_distance", "distribution_note",
                     "measured_success_deficit", "predicted_logical_error_bound", "two_sigma_sampling_noise",
                     "measured_within_prediction", "solutions_agree", "success_probability_note",
                     "baseline_backend"))
        out["gap"] = rec

    if a.what in ("cut", "all"):
        n = a.n or 12
        max_sub = a.max_sub or max(2, n // 2)
        rec = cutting_validity(ctx=ctx, n_vars=n, max_sub=max_sub, shots=a.shots)
        print(f"cutting validity ({n} vars into <= {max_sub}q pieces):")
        _print(rec, ("num_partitions", "num_cuts", "max_abs_marginal_error", "mean_abs_marginal_error",
                     "marginal_sign_agreement", "exact_seconds", "cutting_seconds", "dispatch"))
        out["cut_validity"] = rec
        # Cutting reconstructs one marginal per variable, each a full QPD
        # dispatch, so cost grows ~n * 4**cuts. Default to just over the
        # piece size; raise --n-big deliberately.
        big_sub = a.max_sub or (ctx.gpu_cap if ctx.local_gpu else 20)
        big = a.n_big or (big_sub + 4)
        rec2 = cutting_scaling(ctx=ctx, n_vars=big, max_sub=big_sub, shots=min(a.shots, 1000))
        print(f"cutting scaling ({big} vars into <= {big_sub}q pieces):")
        _print(rec2, ("num_partitions", "num_cuts", "seconds", "decoded_cut_edges", "local_search_cut_edges"))
        out["cut_scaling"] = rec2

    if a.what in ("bench", "all"):
        tsp = tuple(int(x) for x in a.tsp.split(",") if x)
        mc = tuple(int(x) for x in a.maxcut.split(",") if x)
        print("bench:")

        def _row(row: dict[str, Any]) -> None:
            tail = f"ERROR {row['error']}" if row.get("error") else f"ratio={row['approx_ratio']}"
            print(f"  {row['problem']:<28} {row['n_vars']:>3}q  {str(row['backend']):<14} "
                  f"{row['seconds']:>7.1f}s  {tail}", flush=True)

        rec = run_bench(ctx=ctx, tsp_cities=tsp, maxcut_nodes=mc, on_row=_row)
        if rec.get("path"):
            print(f"  written: {rec['path']}")
        out["bench"] = rec

    if a.what in ("verify", "all"):
        n = a.n or 12
        rec = verify_demo(n, ctx=ctx)
        print(f"verify tier 2 vs ideal ({n} vars):")
        _print(rec, ("solutions_agree", "tier2_is_optimal", "ideal_is_optimal", "tier2_success_probability",
                     "ideal_success_probability", "ecc", "tier2_seconds"))
        out["verify"] = rec

    if a.json:
        print(json.dumps(jsonable(out), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
