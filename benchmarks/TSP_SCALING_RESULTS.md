# TSP Scaling Study — LIMEN × ibm_kingston

- **Date**: 2026-06-26
- **Shots per run**: 1000
- **Backend**: Aer simulator
- **QPU ceiling**: 12 cities (144 qubits / 156 available)

| Cities | Qubits | Within QPU | Classical Opt | Feasible% | Best QPU | Approx Ratio | Found Optimal | Compile(s) |
|--------|--------|------------|---------------|-----------|----------|--------------|---------------|------------|
| 4 | 16 | ✓ | 102 (br) | 0.1% | 102 | 1.0 | ✓ | 16.3 |
| 5 | 25 | ✓ | 106 (br) | — | — | — | sim_skipped | 0.15 |
| 6 | 36 | ✓ | 113 (br) | — | — | — | sim_skipped | 0.23 |
| 7 | 49 | ✓ | 135 (br) | — | — | — | sim_skipped | 0.35 |
| 8 | 64 | ✓ | 138 (br) | — | — | — | sim_skipped | 0.7 |

*Raw JSON: `tsp_scaling_20260626_215147.json`*
