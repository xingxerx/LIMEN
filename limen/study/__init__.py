# Copyright (C) 2026 xingxerx
#
# Licensed under the Elastic License 2.0. See the LICENSE file in the
# repository root for the full terms.

"""What the local GPU tier is for: the four measurements that turn free
30-qubit statevectors into evidence about the rest of LIMEN.

* :mod:`limen.study.gap`      ideal vs. hardware success probability, fed
  back into the router's history and ledger (the fidelity signal).
* :mod:`limen.study.cutting_study`  circuit cutting at tier-0 partition
  sizes, with the cutting error measured against the exact uncut answer.
* :mod:`limen.study.bench`    recognisable TSP / Max-Cut instances through
  run_route_request on tier 0, scored against known optima.
* :mod:`limen.study.verify`   Tier 2 (ECC) certificates checked against
  the exact tier-0 answer.

Run everything with ``python -m limen.study all --results-dir results``.
Each runner reuses the existing pipeline, cutting, frontend and router
code; the only new machinery is the comparison and the record.
"""

from limen.study._common import StudyContext, context
from limen.study.bench import bench_maxcut, bench_tsp, run_bench
from limen.study.cutting_study import cutting_scaling, cutting_validity
from limen.study.gap import gap_live, gap_offline_demo, hardware_gap, hardware_plan, simulated_hardware_counts
from limen.study.verify import verify_demo, verify_tier2

__all__ = [
    "StudyContext",
    "context",
    "hardware_gap",
    "hardware_plan",
    "simulated_hardware_counts",
    "gap_offline_demo",
    "gap_live",
    "cutting_validity",
    "cutting_scaling",
    "bench_tsp",
    "bench_maxcut",
    "run_bench",
    "verify_tier2",
    "verify_demo",
]
