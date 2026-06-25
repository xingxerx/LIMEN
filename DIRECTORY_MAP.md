# LIMEN Directory Map & Workspace Guide

This document maps out the structure of the LIMEN project, providing a label and description for every directory and notable file.

---

## 📂 Root Directory
* **`.env`** — Local configuration for environment variables (e.g., API tokens for hardware backends like Qiskit/D-Wave).
* **`.gitignore`** — Directives for files/folders to be excluded from git version control (e.g., build assets, virtual environments).
* **`Cargo.lock`** — Dependency lockfile for the Rust compilation module (`limen_core`).
* **`Cargo.toml`** — Package manifest and dependency configuration for the Rust compilation module.
* **`CONTRIBUTING.md`** — Developer guidelines detailing coding standards, PR workflows, and setup instructions.
* **`DIRECTORY_MAP.md`** *(This file)* — High-level map labeling all folders and files in the workspace.
* **`LICENSE`** — Full text of the Apache 2.0 license governing this open-source project.
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
* **`APPLY_NOTES.md`** — Notes detailing implementation steps, drop-in file descriptions, and validation routines.
* **`architecture.md`** — High-level system architecture overview detailing frontends, compiler passes, and backends.

---

## 📂 `examples/` — Demonstrations & Usage Guides
Practical entry-point examples showcasing how to use LIMEN's components.
* **`analog_demo.py`** — Demonstrates the compilation pipeline from LogicalGraph to physical neutral-atom and photonic topologies.
* **`codesign_demo.py`** — Showcases the joint hardware-software Stackelberg co-design loop for optimizing embeddings and penalty margins.
* **`communication_demo.py`** — Showcases state teleportation and QKD (BB84) key exchange protocols.
* **`ibm_codesign_qpu.py`** — Demonstrates physical co-design mapping running on IBM gate-model hardware.
* **`ibm_qpu_demo.py`** — Standard circuit compilation and execution walkthrough using the IBM Qiskit backend.
* **`max_cut.py`** — Compiles a classical Max-Cut graph problem into a Hamiltonian suitable for physical hardware execution.


---

## 📂 `limen/` — Core Python Package
The primary Python package namespace containing compiler layers, adapters, and validators.

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
* **`dwave.py`** — Adapter for D-Wave quantum annealers leveraging the Ocean SDK.
* **`qiskit_backend.py`** — Adapter for gate-model IBM hardware leveraging Qiskit Runtime.

### 📁 `limen/communication/` — Quantum Communication Primitives
* **`__init__.py`** — Exports QuantumChannel and protocol result schemas.
* **`channel.py`** — Implements quantum teleportation (Bell measurement + feedforward) and BB84 QKD.

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
* **📂 `proto/`** — `coordination.proto` service definition plus its generated `coordination_pb2.py` / `coordination_pb2_grpc.py` (regenerate via `scripts/gen_proto.py`; do not hand-edit).

### 📁 `limen/core/` — Lexicographic Compiler & Logical IR
* **`__init__.py`** — Exports core compilation routines and IR constructs.
* **`compiler.py`** — Translates the LogicalGraph IR to a PhysicalEncoding IR deterministically.
* **`ir.py`** — Defines the LogicalGraph (nodes, edges, quadratic/linear weights) and PhysicalEncoding representations.

### 📁 `limen/docs/` — Library Architecture & Mathematical Proofs
* **`architecture.md`** — Design documents detailing the lexicographic compiler, IR specifications, and system APIs.
* **`universality_theorem.md`** — Mathematical proofs and construction arguments for restricted quadratic analog mappings.

### 📁 `limen/exceptions/` — Error Definitions
* **`__init__.py`** — Package-wide custom exceptions (e.g., `CompilationError`, `SubstrateError`).

### 📁 `limen/frontends/` — Frontend Parsers
* **`__init__.py`** — Exports standard problem parses.
* **`pyqubo.py`** — Parser to convert PyQUBO model definitions into the LogicalGraph IR.

### 📁 `limen/validator/` — Probabilistic Verification Loop
* **`__init__.py`** — Exports verification routines.
* **`validator.py`** — Performs small-instance brute force verification and computes statistical confidence metrics.

---

## 📂 `scripts/` — Developer Tooling
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
* **`scoring.rs`** — Computes learning rates and stability-penalized score coefficients.
* **`stackelberg.rs`** — Implementation of game-theoretic equilibrium solver.
* **📂 `analog/`** — Rust representation of physical layout structures.
  * **`mod.rs`** — Module index.
  * **`interface.rs`** — Core layout traits.
  * **`neutral_atom.rs`** — Neutral-atom geometry block and coordinate validators.
  * **`photonic.rs`** — Continuous-variable scale calculations.
* **📂 `sim/`** — Rust simulation kernels.
  * **`mod.rs`** — Module index.
  * **`ising_backend.rs`** — Multi-threaded exact diagonalization simulator.

---

## 📂 `tests/` — Test Suites
Comprehensive automated testing suite verifying compiler passes, adapters, math solvers, and certificates.
* **`test_analog.py`** — Verifies basic analog layout logic and hardware targets.
* **`test_backend_dwave.py`** — Exercises the D-Wave compile and offline verification adapters.
* **`test_backend_qiskit.py`** — Exercises the IBM Qiskit backend compiler pipeline.
* **`test_backends_offline.py`** — Mock/offline validation tests for hardware adapters.
* **`test_bec.py`** — Validates Bose-Einstein Condensate simulator.
* **`test_calibration_loader.py`** — Checks parsing of device calibration parameters.
* **`test_certificate.py`** — Validates Theorem 1 realizability certification.
* **`test_codesign.py`** — Verifies Stackelberg solver correctness against simulated baselines.
* **`test_codesign_cbf.py`** — Validates chain-break fraction feedback loops.
* **`test_communication.py`** — Verifies QuantumChannel teleportation fidelity and QKD eavesdropper detection.
* **`test_core.py`** — Test suite for basic lexicographic compiler and IR schemas.

* **`test_delta_model.py`** — Tests drift model regression and calibration scaling bounds.
* **`test_distributed_registry.py`** — Unit tests for `NodeRegistry` peer add/evict/TTL cache and calibration resolution.
* **`test_distributed_server.py`** — End-to-end tests of the Coordination gRPC service (register/heartbeat/list-peers/sync-calibration) against a real server.
* **`test_geometry.py`** — Exercises layout distance calculations and spatial mappings.
* **`test_lhz.py`** — Exercises triangular LHZ layout transformations.
* **`test_lhz_certificate.py`** — Tests correctness of LHZ realizability checks.
* **`test_phase3_completion.py`** — Aggregated validation tests for Phase 3 functionality.
* **`test_pyfallback.py`** — Asserts that python fallbacks align mathematically with Rust routines.
* **`test_sim.py`** — Unit tests for the exact-diagonalization simulator.
