# Changelog

All notable changes to LIMEN are documented in this file. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.8.3] - 2026-07-07

- Budget router (`limen.router`): deterministic fidelity-tier planning for
  QUBO runs — picks a tier, backend, cutting strategy, ECC allocation, and
  shot count against an explicit credit budget before anything is submitted
  (`route()`, `RouteRequest`/`RoutePlan`, `DEFAULT_FLEET`).
- Router cost-model seeding from real run history
  (`limen.router.history`): `scan_results()`/`apply_history()` fold cached
  `results/*.json` run records into the fleet's backend profiles offline.
- Router calibration seeding from live hardware
  (`limen.router.calibration`): `fetch_backend_calibration()` queries
  gate/readout error from IBM Runtime; `scan_calibration()`/
  `apply_calibration()` fold cached `results/calibration_*.json` snapshots
  in offline. `route()` now prefers a backend's calibrated
  `physical_error_rate` over the request's hardcoded default — the first
  real ibm_kingston snapshot (2.586e-2) is 25x closer to measured Tier 2
  behavior than the old 1e-3 guess.
- Crash-resilient QPU job lifecycle (`limen.router.job_state` +
  `pipeline.submit_qpu_job()`): submission is decoupled from
  result-waiting; job state (SUBMITTED/QUEUED/RUNNING/DONE/ERROR/
  CANCELLED/TIMED_OUT) is persisted to disk at every step so a closed
  terminal or mid-poll crash can't strand a completed job. Includes a
  transient-error-only retry helper; errored/cancelled jobs are never
  auto-resubmitted. First hardware validation: ibm_kingston Tier 2 run,
  job `d965qgotcv6s73djc1l0`.
- Azure Quantum backend adapter for Atom Computing gate-model
  neutral-atom devices (`limen.backends.azure_atom`). Status: DORMANT —
  import-clean and unit-tested, but never exercised against a live Azure
  Quantum workspace; not in `DEFAULT_FLEET` until hardware-validated.
- Probabilistic validator refactor with a Rust-extension parity guard
  test (`tests/test_rust_exports.py`), catching stale-wheel drift between
  `limen_core` and the Python fallbacks.

## [0.8.2] - 2026-07-01

- Adaptive ECC patch-budget allocation (`limen.ecc.budget`): wires the
  Rust `qubo_criticality`/`select_patches` primitives (added in 0.8.1) into
  a Python-callable `allocate_ecc_budget()`, closing the gap where they
  were implemented and exposed but never called from any pipeline.
- Rust `limen_core` extension: moved more compute-hot loops (statevector
  simulation, circuit cutting reconstruction, QUBO solving via Qiskit
  exact/QAOA/VQE) into native code.
- Fixed analog interface dependency gaps and stale README status claims.
- Real-hardware circuit cutting (`limen_core::cutting`) and QPU
  job-fetching tools; validated TSP QAOA pipeline on real hardware.

## [0.8.1] - 2026-06-29

- QEC surface-code module: criticality-ranked patch budgeting
  (`src/ecc/selector.rs`, `src/scoring.rs::qubo_criticality`) and circuit
  remapping utilities.
- Open Quantum backend integration for cross-vendor hardware submission;
  fixed an OpenQASM2 `u`-gate bug and validated the Rigetti node.
- NYC VRP demo script with fleet/result certificates.

## [0.8.0] - 2026-06-27

- Logical-qubit layer above physical compilation: distance-3 rotated
  surface code, circuit-level round-trip, and end-to-end QUBO-to-
  certification pipeline.
- Multi-node coordination layer (`limen.distributed`) with distributed
  compilation support in `run_pipeline`.
- Neutral-atom backend: geometric embeddability checks and compilation
  certification.
- Closed-loop Stackelberg co-design for D-Wave QPUs with chain-break
  fraction feedback; chain-based minor-embedding.
- VRP frontend with depot-duplication logic and route decoding.
- Live fleet discovery tool and multi-backend TSP benchmark.

## [0.7.0] - 2026-06-25

- Quantum communication module: teleportation and BB84 QKD protocols,
  quantum channel primitives, neutral-atom backend infrastructure.
- Quantum teleportation circuit execution and fidelity estimation on
  real IBM QPUs; cross-layout fidelity benchmark on `ibm_kingston`.
- Parity-based LHZ compilation for the neutral-atom backend; documented
  universality theorems.

## [0.6.0] - 2026-06-22

- LHZ parity encoder and QuEra/IBMQ calibration loaders.
- Qiskit backend and TSP eil51 benchmark pipeline with a
  `CompilationCertificate`.
- D-Wave backend and neutral-atom delta-model support (Phase 3
  completion tests); live IBM calibration and Qiskit error mitigation.

## [0.5.0] - 2026-06-10

- Closed the co-design loop on real IBM QPU hardware; added a
  pure-Python solver fallback.
- Neutral-atom geometric embeddability check; expanded LHZ certification
  tests; `exact_ising_norm` in Rust.

## [0.4.0] - 2026-06-10

- Phase 3 analog compilation certificates and BEC backend.

## [0.3.1] - 2026-06-09

- IBM QPU demo comparing `ibm_kingston` to the local exact simulator.

## [0.3.0] - 2026-06-05

- Phase 3 foundations: analog substrates, Hamiltonian IR, stub backends,
  and exact verification.
- `HardwareDeltaModel` calibration layer with Rust delta-correction
  functions.

## [0.2.0] - 2026-06-05

- Phase 2 co-design loop: Rust/PyO3 Stackelberg solver plus Python
  co-design layer, with oscillation-penalised learning-rate stability
  tracking (kappa).
- Qiskit and D-Wave backend adapters; end-to-end Max-Cut QUBO demo.

## [0.1.0-alpha] - 2026-06-04

- Initial project scaffold: `LogicalGraph` IR, lexicographic QUBO
  compiler, PyQUBO frontend adapter, and a probabilistic QUBO validator.
