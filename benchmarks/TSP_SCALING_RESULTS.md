# TSP Scaling Study — LIMEN × ibm_kingston

- **Date**: 2026-06-12
- **Shots per run**: 2000
- **Backend**: ibm_kingston (QPU)
- **QPU ceiling**: 12 cities (144 qubits / 156 available)

| Cities | Qubits | Within QPU | Classical Opt | Feasible% | Best QPU | Approx Ratio | Found Optimal | Compile(s) |
|--------|--------|------------|---------------|-----------|----------|--------------|---------------|------------|
| 4 | 16 | ✓ | 102 (br) | 0.1% | 118 | 1.1569 | ✗ | 25.49 |
| 5 | 25 | ✓ | 106 (br) | 0.0% | none | N/A | ✗ | 0.08 |
| 6 | 36 | ✓ | 113 (br) | 0.0% | none | N/A | ✗ | 0.13 |
| 7 | 49 | ✓ | 135 (br) | 0.0% | none | N/A | ✗ | 0.2 |
| 8 | 64 | ✓ | 138 (br) | 0.0% | none | N/A | ✗ | 0.35 |

*Raw JSON: `tsp_scaling_20260612_174936.json`*
