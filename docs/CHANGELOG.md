# Changelog

All notable changes to LIMEN are documented in this file. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

- Persistent router memory (`limen.router.memory`): a SQLite ledger
  (`RouterMemory`) that makes the budget router stateful across runs.
  Three parts: (1) a backend sample ledger with trend-aware stats —
  recency-weighted (exponential half-life) estimates plus a
  least-squares drift slope, folded into a fleet by `apply_memory()`
  under the conservative-envelope rule (a rising/worsening trend bumps
  the estimate up to its projection at now; an improving trend is never
  extrapolated); (2) a content-addressed transpile cache
  (`transpile_cache_key`/`_get`/`_put`, LRU-evicted, payloads are opaque
  bytes — no qiskit dependency); (3) an append-only certificate ledger:
  each entry hash-chained to its predecessor, UPDATE/DELETE blocked by
  SQL triggers, the whole chain re-verifiable via `verify_ledger()`, and
  optionally ML-DSA-65-signed over the chain head (binding each
  signature to the entire prior history — `limen.security.pqc`, opt-in
  as ever). `ingest_results()` backfills the sample ledger from existing
  `results/` certs and calibration snapshots incrementally
  (mtime-deduplicated), and `informed_fleet()` gained an optional
  `memory=` argument that applies the ledger's estimates last, after the
  flat history/calibration scans.
- Router/pipeline hot-path cleanup: `informed_fleet()` no longer runs the
  legacy flat `scan_results`/`scan_calibration` rescan when a `memory=`
  ledger is supplied (that rescan is a strict subset of what
  `RouterMemory.ingest_results` already covers, so it was pure redundant
  I/O once the ledger is warm — ~45% of the call's cost on a 500-cert
  results_dir in local profiling). `run_route_request()`'s
  `memory=True`/path shapes now close the sqlite3 connection they open
  before returning instead of leaving it for GC; pass a long-lived
  `RouterMemory` instance for high-frequency looped calls to avoid the
  ~1ms/call reconnect cost entirely.
- Known gap, not yet fixed: `RouterMemory`'s certificate ledger has no
  retention/compaction policy — it is append-only by design (see
  `limen/router/memory.py` module docstring) and grows on disk without
  bound. Fine at current volume; needs a decision before production
  scale.

## [0.8.4] - 2026-07-21

- Guarded ML-DSA import in `limen.security.pqc`: importing the module
  without `cryptography>=48` installed now raises an `ImportError` with
  the install hint (`pip install limen-compiler[pqc]`) instead of a raw
  `ModuleNotFoundError`/`ImportError` from deep inside `cryptography`.
  `tests/test_pqc.py` skips cleanly when the dependency is absent,
  matching every other optional-extra test module.
- Read the classical register by name instead of assuming `"c"`
  (`limen.pipeline._get_counts_from_pub_result`): QPU/Aer results no
  longer fail when the circuit's classical register is named e.g.
  `"meas"`; reused from `limen/communication/channel.py` and the Tier 2
  Kingston example.
- Renamed all `pip install limen[...]` references (README, docs, error
  messages) to `pip install limen-compiler[...]` to match the published
  PyPI package name.
- Tier 2 QPU-path integration smoke test for `run_route_request()`.

## [0.8.3] - 2026-07-07

Note: the entries below sat under "Unreleased" when v0.8.3 was tagged,
but the tagged code contained all of them — they shipped in 0.8.3.

- `informed_fleet()` (`limen.router`): one call that folds run history
  (`apply_history`/`scan_results`) and calibration snapshots
  (`apply_calibration`/`scan_calibration`) into a fleet, replacing the
  hand-wired two-step chain in every caller. The
  `router_tier2_kingston(_fetch)` examples now route against it, so
  submissions and plan rebuilds use measured `physical_error_rate` and
  run-history priors instead of raw `DEFAULT_FLEET`.
- Measured logical-error prior in the certificate: `route()` forwards a
  backend's history-derived `measured_logical_error` into
  `pipeline_kwargs`; `run_pipeline()` records it as
  `EndToEndCertificate.measured_logical_error_prior` and sets
  `predicted_logical_error_bound = max(model, prior)` — a conservative
  envelope, never an average. `aggregate_logical_error_rate` stays the
  surface-code model's own prediction. The Tier 2 fetch example's
  within-prediction check now compares against the bound.
- T1/T2 decoherence term in `fetch_backend_calibration()`: the
  `physical_error_rate` proxy now averages two-qubit gate error, readout
  error, and an exponential T1-relaxation estimate scaled by
  `expected_two_qubit_depth`; snapshots also record `avg_t1`/`avg_t2`/
  `avg_two_qubit_gate_length`. Unvalidated against a measured deficit
  until the next calibrated hardware run lands.
- `run_pipeline_from_plan()` and `run_route_request()`
  (`limen.pipeline`): RoutePlan and RouteRequest execution entry points.
  `run_route_request()` is the zero-manual-steps path — QUBO + budget in,
  certified answer out: builds the informed fleet, routes, and executes;
  IBM QPU plans go through the decoupled submit -> poll -> certify chain
  with job state persisted to `results_dir` at every poll.
- Cut-circuit bridge (`limen.cutting.qubo_bridge`, `.certificate`,
  `.local_dispatch`): `RoutePlan.use_cutting` plans (problem too large for
  any single backend) now dispatch through `run_cut_route_request()`
  instead of raising `NotImplementedError`. Decomposes the QUBO into Ising
  terms, reconstructs per-qubit `<Z_i>` marginals via `limen.cutting`,
  decodes a solution bitstring by threshold rounding, and certifies it
  with the same ECC term `run_pipeline()` uses. Returns a
  `CuttingCertificate`, not `EndToEndCertificate` (`is_optimal` is always
  `None`; `reconstructed_expected_energy` is an explicitly documented
  mean-field approximation) since circuit cutting reconstructs
  expectation values, not a brute-force-verified optimal bitstring.
- gRPC peer auto-discovery: `run_route_request()` falls back to
  `NodeConfig.from_env().known_peers` (`LIMEN_KNOWN_PEERS`) when
  `server_addresses` is omitted and `LIMEN_NODE_ID` is set.
  `RoutePlan.server_addresses` now records the decision so a plan is
  self-describing for async re-execution.
- Co-design history loop (`limen.codesign.solver.codesign_from_history`):
  seeds a fresh `run_codesign()` run from the best prior chain-strength
  found in `results/` for a given backend. `CoDesignResult` gained
  `to_dict()`/`from_dict()`. Deliberately standalone rather than wired
  into `run_route_request()`'s D-Wave dispatch, since that path compiles
  against a complete hardware graph where `chain_strength` is provably
  inert.
- Substrate-aware routing (`limen.router.problem_profile`): a
  `frustration_index` heuristic and `ProblemProfile` signal, plus
  `BackendProfile.substrate_affinity`, used strictly as a tiebreaker in
  `_select_backend()` after all existing cost/capacity/validation
  filtering — regression-tested to never override those criteria.
- IBM fleet calibration extended to `ibm_fez` and `ibm_marrakesh` (in
  addition to `ibm_kingston`) via `fetch_backend_calibration()`, hardened
  against qubits with missing T1/T2 data.
- `.github/workflows/ci.yml`: cargo test + pytest (py3.11/3.12) matrix,
  building the `limen_core` extension via maturin and installing every
  optional backend extra so `importorskip`-guarded tests run instead of
  skipping.
- Stoer-Wagner min-cut graph partitioning (`limen.distributed.partition`
  + `src/graph_partition.rs`): recursive min-cut bisection replaces
  lexicographic variable chunking, keeping heavily-coupled variables in
  the same partition instead of splitting them by alphabetical name order.
- Second calibrated Tier 2 hardware run submitted on ibm_kingston (job
  `d96mijgtcv6s73djv5a0`), the first routed with a calibration-seeded
  `physical_error_rate` (2.586e-2 vs the prior run's 1e-3 guess).
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
