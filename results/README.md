# Run Telemetry & Execution Results

This directory contains execution logs, compiler metrics, and device telemetry output from benchmark runs in JSON format.

## File Naming Conventions

All result files use the pattern:
```
<type>_<YYYYMMDD>_<HHMMSS>.json
```

Where the `<type>` prefix categorizes the run:

*   **`benchmark_*`** — Standard general benchmark runs comparing compiler execution across QPUs and simulators.
*   **`codesign_qpu_*`** — Run metrics from the Stackelberg co-design loop running on actual physical hardware.
*   **`tsp_eil51_*`** — Results from running the TSPLIB `eil51` 51-city Traveling Salesperson Problem benchmark.
*   **`tsp_scaling_*`** — Results from the TSP scaling study measuring compilation time, qubit usage, and simulation limits across city sizes.

## Telemetry Schema

Each telemetry file records:
1.  **Metadata**: Date/time of execution, target device/backend, shots, and physical limits.
2.  **Compilation Metrics**: Number of cities, variables, compiled qubits, and compile duration.
3.  **Solver Quality**: Success metrics, confidence values, optimal flags, and calibration factors.
4.  **Telemetry Loops**: Loop history data points showing convergence parameters like Chain-Break Fraction (CBF) and learning rate.
