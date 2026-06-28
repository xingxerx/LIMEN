# TSP eil51 LIMEN Benchmark

- **Date**: 2026-06-27
- **Backend**: ibm_fez
- **Sub-problem**: first 4 cities of eil51 (16 QUBO variables)
- **Full eil51 reference**: 51 cities, classical optimal = 426
- **QAOA**: p=1, β=γ=0.1, 1000 shots

## Compilation

| Metric | Value |
|--------|-------|
| Compiler | lexicographic (1-to-1 embedding) |
| QUBO variables | 16 |
| Chain strength | 3725.5367 |
| Validator confidence | 47.1% |
| Target κ | 0.62 |
| Stackelberg κ | 0.6220 (converged=True, iters=7) |

## Compilation Certificate

| Metric | Value |
|--------|-------|
| n_sites | 16 |
| L1 bound (||H_target - H_compiled||_op ≤) | 0.00e+00 |
| Exact operator norm | 0.00e+00 |
| Max linear error | 0.00e+00 |
| Max quadratic error | 0.00e+00 |
| Natively realizable | True |

## QPU Results

| Metric | Value |
|--------|-------|
| Circuit depth | 780 |
| Best QUBO energy | -5938.0000 |
| Best tour | None |
| Best tour length | N/A (infeasible) |
| Classical sub-problem optimal | 102 |
| Approximation ratio | N/A (infeasible) |
| Feasible sample rate | 0.0% |

## Notes

- eil51 sub-problem: 4 cities, 16 QUBO variables
- Gate model (IBM ibm_kingston): all Ising terms natively realizable
- Lexicographic 1-to-1 embedding: compiled == target up to index relabelling

*Raw JSON: `tsp_eil51_20260627_004217.json`*
