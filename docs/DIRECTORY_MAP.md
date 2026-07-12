# LIMEN Directory Map & Workspace Guide

This document maps out the structure of the LIMEN project, providing a label and description for every directory and notable file.

---

## 📂 Root Directory
* **`.env`** — Local configuration for environment variables (e.g., API tokens for hardware backends like Qiskit/D-Wave).
* **`.gitignore`** — Directives for files/folders to be excluded from git version control (e.g., build assets, virtual environments).
* **`Cargo.lock`** — Dependency lockfile for the Rust compilation module (`limen_core`).
* **`Cargo.toml`** — Package manifest and dependency configuration for the Rust compilation module.
* **`LICENSE`** — Full text of the Elastic License 2.0 (ELv2) governing this source-available project (changed from Apache 2.0 at v0.3.1).
* **`pyproject.toml`** — Python package configuration defining dependencies, metadata, build tools (Maturin), and pytest options.
* **`pyrightconfig.json`** — Workspace settings for the Pyright static type checker.
* **`README.md`** — Main repository landing page, covering features, architecture, usage, and roadmap.

---

## 📂 `benchmarks/` — Performance & Scaling Studies
Contains benchmark execution scripts and markdown result files.
* **`RESULTS.md`** — Overview of benchmark runs executed on actual physical hardware (e.g., IBM QPU `ibm_kingston`).
* **`TSP_EIL51_RESULTS.md`** — Compilation and runtime performance results for the standard TSPLIB `eil51` instance.
* **`TSP_SCALING_RESULTS.md`** — Scalability benchmarks mapping city size to compiler performance and simulator constraints.
* **`qpu_benchmark.py`** — Runs comparison studies targeting hardware QPUs vs classical simulators.
* **`tsp_eil51_benchmark.py`** — Executes compilation and verification loops for the 51-city TSP instance.
* **`tsp_scaling_study.py`** — Measures compilation overhead, scaling behaviors, and verification ceilings across multi-city TSP.

---

## 📂 `docs/` — General Documentation
High-level overview documentation for LIMEN.
* **`ACCEPTABLE_USE.md`** — Use restrictions supplementing the ELv2 license (prohibits weapons development, mass surveillance, unauthorized cryptanalysis, attacks on critical infrastructure).
* **`APPLY_NOTES.md`** — Notes detailing implementation steps, drop-in file descriptions, and validation routines.
* **`architecture.md`** — High-level system architecture overview detailing frontends, compiler passes, and backends.
* **`CHANGELOG.md`** — Release history in Keep-a-Changelog format.
* **`CONTRIBUTING.md`** — Developer guidelines detailing coding standards, PR workflows, and setup instructions.
* **`DIRECTORY_MAP.md`** *(This file)* — High-level map labeling all folders and files in the workspace.
* **`ROADMAP.md`** — Tracks the six-phase feedback loops: co-design history loop (Phase 2 × Phase 6), router-peer auto-discovery (Phase 4 × Phase 6), automatic substrate selection (Phase 3), and the cut-circuit → certificate bridge (Phase 4 × Phase 5) are all shipped; documents what remains explicitly out of scope.

---

## 📂 `examples/` — Demonstrations & Usage Guides
Practical entry-point examples showcasing how to use LIMEN's components.
* **`analog_demo.py`** — Demonstrates the compilation pipeline from LogicalGraph to physical neutral-atom and photonic topologies.
* **`codesign_demo.py`** — Showcases the joint hardware-software Stackelberg co-design loop for optimizing embeddings and penalty margins.
* **`communication_demo.py`** — Showcases state teleportation and QKD (BB84) key exchange protocols.
* **`dwave_codesign_qpu.py`** — D-Wave analog of `ibm_codesign_qpu.py`: closes the Stackelberg co-design loop against a real D-Wave QPU, feeding the measured chain-break fraction back into the κ scoring via `dwave_chain_break_fn`.
* **`ibm_codesign_qpu.py`** — Demonstrates physical co-design mapping running on IBM gate-model hardware.
* **`ibm_qpu_demo.py`** — Standard circuit compilation and execution walkthrough using the IBM Qiskit backend.
* **`max_cut.py`** — Compiles a classical Max-Cut graph problem into a Hamiltonian suitable for physical hardware execution.
* **`analyze_tsp_qpu_results.py`** — Post-processes fetched TSP QPU job counts into tour-quality metrics.
* **`cobyla_multi_backend.py`** — COBYLA-driven QAOA parameter optimisation compared across backends.
* **`teleport_qpu.py`** — Teleportation circuit submission and cross-layout fidelity comparison on real IBM hardware.
* **`cut_circuit_ibm_kingston.py`** / **`_resume_cut_circuit_kingston.py`** — Real-hardware circuit-cutting run on `ibm_kingston` and its crash-resume companion.
* **`cutting_smoke_test.py`** — End-to-end circuit-cutting validation against a plain AerSimulator run.
* **`distributed_two_node.py`** — Two-node gRPC coordination walkthrough (register, heartbeat, calibration sync, distributed compile).
* **`fetch_backend_calibration.py`** — Fetches live gate/readout calibration from IBM Runtime into `results/calibration_*.json` for the budget router.
* **`fetch_qpu_job.py`** / **`fetch_openquantum_job.py`** — Retrieve and persist results for already-submitted IBM / Open Quantum jobs by job id.
* **`fleet_discovery.py`** — Enumerates live QPU backends and qubit counts across configured vendors into `results/fleet_certificate.json`.
* **`router_tier2_kingston.py`** — Budget-router Tier 2 plan + QPU submission (submit-and-exit; persists `{job_id, plan, submitted_at}` without blocking on results).
* **`router_tier2_kingston_fetch.py`** — The sole waiter for router jobs: exponential-backoff polling, on-disk lifecycle state, certification of completed jobs.
* **`vrp_demo.py`** — NYC Vehicle Routing Problem demo through the VRP frontend with fleet/result certificates.


---

## 📂 `limen/` — Core Python Package
The primary Python package namespace containing compiler layers, adapters, and validators.
* **`pipeline.py`** *(new — Phase 5)* — `run_pipeline(qubo, ...)`: end-to-end orchestrator. Converts a QUBO dict to a `LogicalGraph`, compiles to a QAOA `CircuitIR`, runs it on the statevector simulator, verifies optimality against brute force, optionally runs an ECC logical-error certificate, and optionally delegates compilation to peer nodes via the `CompilePartition` gRPC RPC. Returns an `EndToEndCertificate`. The execution step also routes to real backends (`backend="aer"`/`"qpu"`/`"dwave"`), and `submit_qpu_job()` transpiles + submits without blocking on results (paired with `limen.router.job_state` for crash-resilient fetching).


### 📁 `limen/analog/` — Physical Analog Compilation & Layouts
* **`LAYOUT.md`** — Detailed technical description of coordinate mappings, LHZ transformations, and geometry logic.
* **`__init__.py`** — Exports analog structures, certificates, and calibration helpers.
* **`calibration_loader.py`** — Loads and parses hardware parameters to feed calibration models.
* **`certificate.py`** — Classifies physical realizability and generates mathematical proof bounds (Theorem 1).
* **`delta_model.py`** — Manages drift models, estimating calibration margins on physical analog substrates.
* **`hamiltonian.py`** — Defines the substrate-agnostic intermediate representation for Hamiltonian specifications.
* **`lhz.py`** — Provides mappings to transform graphs into triangular Lechner-Hauke-Zoller geometry layouts.
* **📂 `backends/`** — Analog simulators and hardware mapping drivers.
  * **`__init__.py`** — Exposes analog execution routines (BEC, classical simulation, neutral atom, photonics).
  * **`bec.py`** — Adapter for Bose-Einstein Condensate substrates using dem-demler-lukin mapping.
  * **`classical_sim.py`** — Diagonalization-based classical simulation of Ising Hamiltonians up to 20 sites.
  * **`neutral_atom.py`** — Substrate compiler mapping logic onto neutral-atom Rydberg arrays.
  * **`photonic.py`** — Compilation adapter targeting continuous-variable photonic GBS topologies.

### 📁 `limen/backends/` — QPU & Gate-Model Execution Adapters
* **`__init__.py`** — Base hardware interface and driver registrations.
* **`azure_atom.py`** — Azure Quantum adapter for Atom Computing gate-model neutral-atom devices. **DORMANT**: import-clean and unit-tested, never hardware-validated; excluded from the router's `DEFAULT_FLEET`. Blocked because Atom Computing isn't on Azure Quantum's (or AWS Braket's) provider list — see the module docstring; re-verified 2026-07-08.
* **`braket.py`** — QuEra Aquila (analog neutral-atom) adapter via the AWS Braket SDK.
* **`dwave.py`** — Adapter for D-Wave quantum annealers leveraging the Ocean SDK.
* **`neutral_atom.py`** — Gate-model neutral-atom hardware adapter (distinct from the analog `limen/analog/backends/neutral_atom.py`).
* **`openquantum.py`** — Open Quantum platform adapter (validated against the Rigetti `cepheus-1-108q` node).
* **`photonic.py`** — Photonic hardware adapter (distinct from the analog GBS backend).
* **`qiskit_backend.py`** — Adapter for gate-model IBM hardware leveraging Qiskit Runtime.

### 📁 `limen/communication/` — Quantum Communication Primitives (canonical)
* **`__init__.py`** — Exports the full unified symbol set: `QuantumChannel`, `ChannelDeltaModel`, `TeleportResult`, `SiftedKeyResult`, `QKDResult`, `teleport_circuit`, `run_teleport_qpu`, `bb84_circuit`, `sift_and_evaluate`, `estimate_fidelity`.
* **`channel.py`** — Single canonical source for all quantum-channel code. Implements: `ChannelDeltaModel` (coherence/latency model), `QuantumChannel` (Teleportation + BB84 QKD), `teleport_circuit`/`run_teleport_qpu`, `bb84_circuit`/`sift_and_evaluate`, `estimate_fidelity`.

### 📁 `limen/cutting/` — Circuit Cutting *(new — v0.8.2)*
Runs wider-than-any-backend QAOA circuits on real QPUs via quasi-probability decomposition.
* **`__init__.py`** — Exports the cutting pipeline entry points.
* **`partition.py`** — Chooses cut points and splits a `CircuitIR` into sub-circuits.
* **`dispatch.py`** — Submits sub-circuits to a sampler backend and collects per-subcircuit counts.
* **`reconstruct.py`** — Reconstructs the original expectation value from sub-circuit counts (Rust-backed via `limen_core::cutting`).

### 📁 `limen/ecc/` — Error Correction & Certification *(new — Phase 5)*
* **`__init__.py`** — Exports `LogicalErrorCertificate`, surface-code primitives, and roundtrip helpers.
* **`budget.py`** *(new — v0.8.2)* — `allocate_ecc_budget()`: criticality-ranked surface-code patch allocation across QUBO variables under a physical-qubit budget (wires the Rust `qubo_criticality`/`select_patches` primitives).
* **`certificate.py`** — `LogicalErrorCertificate`: computes per-qubit and aggregate logical error rates from the distance-3 surface code analytic formula.
* **`decoder.py`** — `LookupDecoder`: maps a syndrome bit-pattern to the most likely single-qubit correction operator via a pre-computed lookup table.
* **`encoder.py`** — `run_logical_roundtrip`: builds and executes the full syndrome-extraction circuit on the statevector simulator; returns corrected state and correction flags.
* **`surface_code.py`** — `SurfaceCode`: distance-3 rotated surface-code definition — data qubits, X/Z stabilisers, ancilla layout, logical operators.
* **`syndrome.py`** — `build_syndrome_circuit`: constructs a `CircuitIR` that implements stabiliser measurement (Hadamard-sandwiched for X, direct CX for Z) into ancilla qubits.

### 📁 `limen/codesign/` — Hardware-Software Co-Design Engine

* **`__init__.py`** — Exposes codesign solvers and portfolio compilation.
* **`_pyfallback.py`** — Pure-Python fallback implementing the Stackelberg co-design learning loop.
* **`portfolio.py`** — Portfolio selection compiler that ranks and routes workloads across available physical slots.
* **`solver.py`** — Core optimizer computing game-theoretic equilibria for compiler parameters.

### 📁 `limen/distributed/` — Multi-Node Coordination Layer
* **`__init__.py`** — Exports `NodeConfig`, `NodeInfo`, `NodeRegistry`. Requires the `distributed` extra (grpcio).
* **`node.py`** — `NodeInfo`: identity and reachability (`node_id`, `host`, `port`, `device_ids`) of a LIMEN node.
* **`config.py`** — `NodeConfig.from_env()`: reads `LIMEN_NODE_ID`/`LIMEN_NODE_HOST`/`LIMEN_NODE_PORT`/`LIMEN_NODE_DEVICE_IDS`/`LIMEN_KNOWN_PEERS`.
* **`registry.py`** — `NodeRegistry`: peer table with TTL eviction, wrapping the existing `DeltaModelRegistry` and caching remotely-fetched `HardwareDeltaModel`s with a TTL.
* **`marshal.py`** — Conversions between LIMEN dataclasses and protobuf messages, built on the existing `to_dict()`/`from_dict()` convention.
* **`server.py`** — gRPC `CoordinationServicer` (Register/Heartbeat/SyncCalibration/ListPeers) and the `python -m limen.distributed.server` entry point.
* **`client.py`** — `CoordinationClient`: thin wrapper for calling a peer's Coordination service.
* **`partition.py`** — Distributed QUBO partitioning: splits a `LogicalGraph`, dispatches partitions to peers via the `CompilePartition` RPC, and merges the returned encodings into one energetically equivalent encoding.
* **📂 `proto/`** — `coordination.proto` service definition plus its generated `coordination_pb2.py` / `coordination_pb2_grpc.py` (regenerate via `scripts/gen_proto.py`; do not hand-edit).

### 📁 `limen/core/` — Lexicographic Compiler & Logical IR
* **`__init__.py`** — Exports core compilation routines and IR constructs.
* **`compiler.py`** — Translates the LogicalGraph IR to a PhysicalEncoding IR deterministically.
* **`ir.py`** — Defines the LogicalGraph (nodes, edges, quadratic/linear weights) and PhysicalEncoding representations.

### 📁 `limen/gates/` — Gate-Model Intermediate Representation & Execution *(new — Phase 5)*
* **`__init__.py`** — Exports `CircuitIR`, `GateInstr`, `KNOWN_GATES`, `compile_qaoa`, `StatevectorSimulator`, `decompose_unitary_1q`.
* **`ir.py`** — `CircuitIR` + `GateInstr`: typed gate-model circuit IR. `KNOWN_GATES` registry maps gate name → `(arity, param_count)`. Validates qubit indices, arity, and parameter counts at construction time.
* **`qaoa.py`** — `compile_qaoa`: translates a `LogicalGraph` QUBO to a `CircuitIR` using the QAOA ansatz (alternating problem and mixer layers). Includes `qubo_to_ising` conversion and a grid-search parameter optimiser that drives the statevector simulator to maximise the success probability.
* **`qiskit_exec.py`** — `to_qiskit_circuit`: converts a `CircuitIR` to a Qiskit `QuantumCircuit` for optional QPU submission. Requires the `qiskit` extra; import is deferred so that `import limen` succeeds without Qiskit installed.
* **`simulator.py`** — `StatevectorSimulator`: pure-Python exact statevector simulator. Implements all gates in `KNOWN_GATES` as numpy matrix operations. Provides `run(circuit)` → statevector, `sample(circuit, shots)` → measurement counts, `probabilities(circuit)` → probability distribution.
* **`synthesis.py`** — `decompose_unitary_1q`: decomposes an arbitrary 2×2 unitary into a `(rz, ry, rz)` Euler-angle sequence plus a global phase, emitting a sub-list of `GateInstr` objects.

### 📁 `limen/docs/` — Library Architecture & Mathematical Proofs
* **`architecture.md`** — Design documents detailing the lexicographic compiler, IR specifications, and system APIs.
* **`universality_theorem.md`** — Mathematical proofs and construction arguments for restricted quadratic analog mappings.

### 📁 `limen/exceptions/` — Error Definitions
* **`__init__.py`** — Package-wide custom exceptions (e.g., `CompilationError`, `SubstrateError`).

### 📁 `limen/frontends/` — Frontend Parsers
* **`__init__.py`** — Exports standard problem parses.
* **`pyqubo.py`** — Parser to convert PyQUBO model definitions into the LogicalGraph IR.
* **`vrp.py`** — Vehicle Routing Problem frontend: multi-depot-split routing via depot duplication, one-hot QUBO construction (Rust-backed `vrp_qubo_terms`), and route decoding. Use `run_pipeline(backend="dwave")` for VRP-sized instances.

### 📁 `limen/formulation/` — QUBO Auto-Formulation *(new — Phase 7, structured-input tier)*
* **`__init__.py`** — Exports `ConstraintCompiler`, the typed constraint dataclasses, and `default_penalty_weight`.
* **`constraints.py`** — Typed constraint contract: `Equality`, `Inequality` (`<=`/`>=`, binary-slack-encoded), `OneHot`, `AtMostK`, `AtLeastK`, and `AllDifferent` (row one-hot + column at-most-one assignment pattern).
* **`compiler.py`** — `ConstraintCompiler`: merges an objective QUBO with penalty terms expanded from queued constraints into a `LogicalGraph`; `at_most_k=1`/`at_least_k=0` special-case to zero-auxiliary penalties, general-`k` inequalities use a binary-encoded slack variable.
* **`penalty.py`** — `default_penalty_weight`: auto-selects a penalty coefficient that provably dominates the objective's local pull on the constrained variables, so violating a constraint is never profitable.
* Natural-language input is deliberately out of scope here (see `docs/ROADMAP.md`) — an NL layer, if built, translates text into these same typed constraints rather than bypassing them, so an LLM's interpretation never enters the certified pipeline directly.

### 📁 `limen/router/` — Budget Router & Fleet Cost Model *(new — v0.8.3)*
* **`__init__.py`** — Exports `route`, `Tier`, `RouteRequest`/`RoutePlan`, `DEFAULT_FLEET`, and the history/calibration/job-state helpers.
* **`budget_router.py`** — Deterministic fidelity-tier planning: picks tier, backend, cutting strategy, ECC allocation, and shot count for a QUBO against an explicit credit budget before anything is submitted.
* **`calibration.py`** — Seeds backend `physical_error_rate` from live IBM Runtime calibration (`fetch_backend_calibration`) or cached `results/calibration_*.json` snapshots (`scan_calibration`/`apply_calibration`).
* **`history.py`** — Seeds the fleet cost model from finished run certificates in `results/` (`scan_results`/`apply_history`).
* **`job_state.py`** — Crash-resilient QPU job lifecycle: on-disk state machine (SUBMITTED → … → DONE/ERROR/CANCELLED/TIMED_OUT) plus a transient-error-only submission retry helper.

### 📁 `limen/quantum_channel/` — Backward-Compatibility Shims *(refactored — Phase 5)*
All logic has been moved to `limen/communication/channel.py`. These files are thin re-export shims so that existing code importing the old module paths continues to work unchanged.
* **`__init__.py`** — Re-exports `QuantumChannel`, `ChannelDeltaModel`, `QKDResult`, `TeleportResult`, `SiftedKeyResult` from `limen.communication.channel`.
* **`channel_delta.py`** — Re-exports `ChannelDeltaModel`.
* **`qkd.py`** — Re-exports `SiftedKeyResult as QKDResult` (alias preserved for callers that imported the old name).
* **`teleport.py`** — Re-exports `TeleportResult`, `teleport_circuit`, `run_teleport_qpu`.
* **`teleport_analysis.py`** — Re-exports `estimate_fidelity`.

### 📁 `limen/validator/` — Probabilistic Verification Loop
* **`__init__.py`** — Exports verification routines.
* **`validator.py`** — Performs small-instance brute force verification and computes statistical confidence metrics.

---

## 📂 `scripts/` — Developer Tooling
* **`deploy_node.sh`** — Provisions and launches a LIMEN coordination node (gRPC server) on a remote host.
* **`gen_proto.py`** — Regenerates `limen/distributed/proto/coordination_pb2*.py` from `coordination.proto` via `grpc_tools.protoc`.

---

## 📂 `results/` — Run Telemetry & Execution Output
Data directory containing telemetry files and serialized execution runs in JSON format.
* **`README.md`** — Documents file-naming formats and telemetry schemas (e.g., parameter tuning histories, benchmark metrics).

---

## 📂 `src/` — Rust Core Implementation (`limen_core`)
High-performance computational modules written in Rust to accelerate co-design math and Ising simulations.
* **`lib.rs`** — Root of the Rust library crate, exposing Python bindings (via PyO3/Maturin).
* **`delta.rs`** — High-performance routines for calibrating physical drift factors.
* **`frontends.rs`** — VRP one-hot QUBO term construction (`vrp_qubo_terms`).
* **`scoring.rs`** — Computes learning rates, stability-penalized score coefficients, and `qubo_criticality` ranking for ECC budgeting.
* **`stackelberg.rs`** — Implementation of game-theoretic equilibrium solver.
* **`validator.rs`** — Rayon-parallel noisy-run simulation (`simulate_qubo_runs`) and QUBO spectrum enumeration backing the probabilistic validator.
* **📂 `cutting/`** — Circuit-cutting reconstruction kernels.
  * **`mod.rs`** — Module index.
  * **`reconstruct.rs`** — Quasi-probability expectation-value reconstruction from per-subcircuit counts.
* **📂 `ecc/`** — Surface-code kernels.
  * **`mod.rs`** — Module index.
  * **`surface_code.rs`** — Rotated surface-code stabiliser structure.
  * **`decoder.rs`** — Lookup-table construction and logical-failure probability (`build_ecc_lookup_table` / `logical_failure_probability`; makes distance-5 decoding tractable).
  * **`selector.rs`** — Criticality-ranked patch selection (`select_patches`) for ECC budget allocation.
  * **`remapper.rs`** — Circuit remapping utilities for patch layouts.
* **📂 `analog/`** — Rust representation of physical layout structures.
  * **`mod.rs`** — Module index.
  * **`interface.rs`** — Core layout traits.
  * **`neutral_atom.rs`** — Neutral-atom geometry block and coordinate validators.
  * **`photonic.rs`** — Continuous-variable scale calculations.
* **📂 `sim/`** — Rust simulation kernels.
  * **`mod.rs`** — Module index.
  * **`ising_backend.rs`** — Multi-threaded exact diagonalization simulator.
  * **`statevector_backend.rs`** — Exact gate-model statevector simulation (`run_statevector`), the Rust fast path behind `limen.gates.simulator`.
  * **`qudit.rs`** — Multi-level (qudit) simulation support.

---

## 📂 `tests/` — Test Suites
Comprehensive automated testing suite verifying compiler passes, adapters, math solvers, and certificates. **387 tests passing, 3 skipped** (skips are environment-gated on optional SDKs).
* **`test_analog.py`** — Verifies basic analog layout logic and hardware targets.
* **`test_backend_dwave.py`** — Exercises the D-Wave compile and offline verification adapters.
* **`test_backend_qiskit.py`** — Exercises the IBM Qiskit backend compiler pipeline.
* **`test_backends_offline.py`** — Mock/offline validation tests for hardware adapters.
* **`test_bec.py`** — Validates Bose-Einstein Condensate simulator.
* **`test_calibration_loader.py`** — Checks parsing of device calibration parameters.
* **`test_certificate.py`** — Validates Theorem 1 realizability certification.
* **`test_codesign.py`** — Verifies Stackelberg solver correctness against simulated baselines.
* **`test_codesign_cbf.py`** — Validates chain-break fraction feedback loops.
* **`test_compiler_embedding.py`** — Chain-based minor-embedding correctness in the lexicographic compiler.
* **`test_communication.py`** — Verifies unified `QuantumChannel` teleportation fidelity and QKD eavesdropper detection (`limen.communication.channel`).
* **`test_core.py`** — Test suite for basic lexicographic compiler and IR schemas.
* **`test_decoder.py`** *(new — Phase 5)* — Unit tests for `LookupDecoder`: every weight-1 error corrected, empty syndrome is a no-op, unknown syndromes fall back gracefully.
* **`test_delta_model.py`** — Tests drift model regression and calibration scaling bounds.
* **`test_distributed_feedforward.py`** — `TransportFeedforward` RPC round-trip and latency-vs-T2 accounting.
* **`test_distributed_health.py`** — Node health/heartbeat behaviour.
* **`test_distributed_register_retry.py`** — Peer registration retry logic.
* **`test_distributed_registry.py`** — Unit tests for `NodeRegistry` peer add/evict/TTL cache and calibration resolution.
* **`test_distributed_server.py`** — End-to-end tests of the Coordination gRPC service (register/heartbeat/list-peers/sync-calibration) against a real server.
* **`test_distributed_tls.py`** — TLS-secured coordination channel setup.
* **`test_ecc_budget.py`** — `allocate_ecc_budget()`: criticality ranking, budget exhaustion, patch layout validity.
* **`test_ecc_roundtrip.py`** *(new — Phase 5)* — Circuit-level ECC round-trip: all weight-1 X errors corrected by gate-executed syndrome circuit, no-error path is identity, gate-executed syndrome matches analytic calculation.
* **`test_frontend_vrp.py`** — VRP frontend: depot duplication, one-hot QUBO structure, route decoding.
* **`test_gate_exec.py`** *(new — Phase 5)* — `CircuitIR` → Qiskit conversion: Bell state measurement distribution, invalid circuit rejection, hand-written circuit matching.
* **`test_gate_ir.py`** *(new — Phase 5)* — `CircuitIR` validation: rejects unknown gates, wrong arity, wrong param count, out-of-range qubit indices; valid circuit round-trips.
* **`test_gate_simulator.py`** *(new — Phase 5)* — `StatevectorSimulator` correctness: Bell state probabilities and statevector, norm preservation, SWAP gate, U-gate matching Hadamard, X-flip, sample counts deterministic.
* **`test_gate_synthesis.py`** *(new — Phase 5)* — `decompose_unitary_1q`: identity, Pauli-X, Hadamard, arbitrary-angle round-trips, rejection of non-unitary and multi-qubit matrices.
* **`test_geometry.py`** — Exercises layout distance calculations and spatial mappings.
* **`test_job_state.py`** — Router job-state persistence: save/load round-trip, lifecycle transitions, transient-error-only retry.
* **`test_lhz.py`** — Exercises triangular LHZ layout transformations.
* **`test_lhz_certificate.py`** — Tests correctness of LHZ realizability checks.
* **`test_lhz_fallback.py`** *(new — Phase 5)* — Confirms that geometrically frustrated and negative-coupling problems auto-route through the LHZ fallback; natively realizable inputs skip it.
* **`test_logical_certificate.py`** *(new — Phase 5)* — `LogicalErrorCertificate`: field correctness, quadratic scaling with physical error rate, IBM Kingston rate suppression, zero-error edge case.
* **`test_partition.py`** *(new — Phase 4)* — `partition_graph`: balanced split, cross-edge ownership, no interaction dropped or duplicated, invalid partition count rejected; `NamespacedHardwareGraph` collision test; merge correctness vs single-shot compile.
* **`test_phase3_completion.py`** — Aggregated validation tests for Phase 3 functionality.
* **`test_physics_validation.py`** *(new — Phase 5)* — **Physics-first tests**: random QUBO vs brute-force (QAOA never beats exact minimum), energy consistency, 2-var and 3–4-var optimum discovery rates, logical error monotonicity, distance-3 super-linear suppression, weight-1 correction, weight-2 uncorrectable boundary, BB84 eavesdrop QBER detection.
* **`test_pipeline.py`** *(new — Phase 5)* — `run_pipeline` unit tests: optimum energy, unique optimum, single-variable, certificate serialization, ECC certificate composed with/without error rate, distributed compilation absent by default.
* **`test_pipeline_distributed.py`** *(new — Phase 5)* — Live gRPC round-trip: `run_pipeline(server_addresses=[...])` compiles over `CompilePartition` RPC, result is serializable, solution is still certified.
* **`test_pyfallback.py`** — Asserts that python fallbacks align mathematically with Rust routines.
* **`test_qaoa.py`** *(new — Phase 5)* — `qubo_to_ising` mapping (linear and quadratic terms), `compile_qaoa` circuit validity, layer count scaling, Hadamard opening layer, mismatched-params error, bitstring-to-assignment ordering.
* **`test_quantum_channel.py`** — Backward-compatibility test for the old `limen.quantum_channel` import path; all symbols resolve to the unified `limen.communication.channel` implementation.
* **`test_router.py`** — Budget router: deterministic tier/backend/shot planning, budget constraints, fleet profiles.
* **`test_router_calibration.py`** — Calibration seeding: snapshot scan, `apply_calibration`, `route()` preferring calibrated error rates.
* **`test_router_history.py`** — History cost model: `scan_results` over cached certs and `apply_history` profile updates.
* **`test_rust_exports.py`** — Guard test asserting the built `limen_core` wheel exports every symbol the Python fast paths import (catches stale-wheel drift).
* **`test_sim.py`** — Unit tests for the exact-diagonalization (Rust) Ising simulator.
* **`test_surface_code.py`** *(new — Phase 5)* — `SurfaceCode` structure: 9 data qubits, 8 stabilizers, correct weights (2 or 4), logical operators spanning 3 qubits, weight-1 error detectability, no undetected weight-≤2 logical error.
* **`test_syndrome_circuit.py`** *(new — Phase 5)* — `build_syndrome_circuit`: `CircuitIR` validity, qubit count = data + ancilla, X-stabiliser Hadamard sandwich, Z-stabiliser CX-into-ancilla pattern.
