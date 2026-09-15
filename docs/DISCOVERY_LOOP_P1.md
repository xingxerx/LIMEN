# Discovery Loop P1 — Problem schema + TSP encoder

One-liner: a typed `Problem` (variables, constraints, objective,
`known_optimum`) plus one reference encoder that emits a QUBO
`RouteRequest` the existing pipeline already accepts.

## Acceptance

1. Encode a TSP instance (eil51 coordinate prefix) into a QUBO.
2. Route via `run_route_request` on the simulator fleet (`credit_budget=0`,
   no QPU spend); `MEMORY_AUTO` + temp `results_dir` is fine.
3. Score the certificate solution: `approx_ratio == 1.000` against
   `Problem.known_optimum` (classical tour length).

## Why not full eil51?

TSPLIB eil51 is 51 cities (classical optimum 426). The standard one-hot
tour encoding needs `51^2 = 2601` binary variables — far beyond the
statevector / default QAOA smoke path. P1 therefore uses an **eil51
prefix** (first N cities, same EUC_2D coords as
`benchmarks/tsp_eil51_benchmark.py`).

Default smoke size is **N=2** (4 QUBO variables): QAOA p=1 on the
simulator recovers a feasible optimal tour and `approx_ratio 1.000`.
Larger prefixes keep the same encoder; N=4 matches the historical QPU
sub-problem size and is checked classically (brute-force ground state
equals the known tour optimum) because default QAOA does not reliably
return feasible tours at 16 variables.

## Layout

| Piece | Path |
|-------|------|
| `Problem` schema | `limen/formulation/problem.py` |
| TSP encoder + eil51 fixture | `limen/frontends/tsp.py` |
| Tests | `tests/test_discovery_loop_p1.py` |

## Reuse

QUBO construction matches `tsp_qubo` in the eil51 benchmark and the VRP
frontend one-hot tour (`x_{city}_{position}`). No second QUBO stack.
`approx_ratio = tour_length / known_optimum` (same meaning as the eil51
result JSON `approximation_ratio` / analyze script).

## Out of scope

QPU spend, merge, QAP encoder (TSP preferred while eil51 data exists).
