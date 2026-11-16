# LIMEN status and history

The long-form status notes that used to live in the README, kept here so the
README can stay short for a first-time reader. Facts below were true at the
version noted; the README carries the current numbers.

## Status narrative (as of v0.9.x)

Phase 1 complete, Phase 2 complete, Phase 3 substantially implemented, Phase 4
complete, Phase 5 complete, Phase 6 in progress. Hardware validated.

Core IR, compiler, validator, PyQUBO frontend, D-Wave and Qiskit backend
adapters shipped and tested. Stackelberg co-design loop operational with
Rust-backed kappa scoring, stability-penalised learning rate, and portfolio
compilation across backend slots. Chain-break-fraction feedback is wired into
the loop on both hardware paths: `ibm_noise_fn` (real-hardware total variation
distance) drives `examples/ibm_codesign_qpu.py`, validated against
`ibm_kingston`, and `dwave_chain_break_fn` (real chain-break fraction) drives
the equivalent closed loop against a live D-Wave QPU in
`examples/dwave_codesign_qpu.py`. With no callback supplied, chain-break
fraction defaults to 0.0 (simulation mode); this is documented behaviour, not
an inactive feature.

Analog interface layer implemented: classical Ising simulator (exact
diagonalisation, 20 sites or fewer), neutral-atom heuristic backend (van der
Waals layout, Rydberg parameters), photonic GBS backend (Arrazola-Bromley
adjacency encoding), HardwareDeltaModel calibration layer (per-site detuning,
per-pair coupling, and global Rabi-drive correction, each with a pure-Python
fallback alongside the Rust-backed implementation). The neutral-atom geometric
embeddability check (Theorem 2 geometry condition) always runs: numpy when
available, a pure-Python Jacobi eigenvalue fallback otherwise, consistent with
LIMEN's zero-mandatory-dependency core (`dependencies = []` in
`pyproject.toml`). Restricted-class universality results (Theorems 1 to 5) are
documented in `limen/docs/universality_theorem.md`; a fully general
universality theorem for arbitrary (non-diagonal, time-dependent) analog
Hamiltonians remains open research and is not claimed, nor is off-diagonal
(X/Y) or time-dependent Hamiltonian support implemented in any backend.

**Gate-model track:** QUBO to QAOA compilation to exact statevector simulation
to surface-code (d=3) ECC syndrome extraction and logical-error certification,
end-to-end with no external SDK required. `run_pipeline` returns a single
`EndToEndCertificate` with solution, optimality flag, success probability,
per-qubit logical error rate, and ECC roundtrip verification. Distributed
compilation over gRPC (`CompilePartition` RPC) is wired into
`run_pipeline(server_addresses=[...])`. The execution step also routes to real
backends: `backend="aer"` / `"qpu"` (IBM Quantum hardware) and
`backend="dwave"` (real D-Wave QPU or simulated annealer), alongside the
default offline statevector simulator.

**Multi-node coordination:** beyond discovery and calibration sync, nodes can
transport the classical feedforward bits of a teleportation protocol to a peer
over gRPC (`TransportFeedforward` RPC) and measure real round-trip latency
against the qubit's T2 coherence time (`run_distributed_feedforward_teleport`).

**First real hardware validation:** IBM ibm_kingston (Heron R2, 156 qubits),
June 9 2026. Benchmark results in `benchmarks/RESULTS.md`.

**Rust acceleration layer (v0.8.x):** every enumeration- or sampling-shaped hot
loop in the library is Rust-backed via the `limen_core` PyO3 extension, each
behind a `try: from limen.limen_core import ...` fast path with an equivalent
pure-Python fallback: QUBO spectrum enumeration, the validator's noisy-run
simulation (`simulate_qubo_runs`, rayon-parallel), surface-code lookup-table
construction and logical-error certification (`build_ecc_lookup_table` /
`logical_failure_probability`; distance-5 decoding drops from
intractable-in-Python to seconds), the VRP one-hot QUBO builder
(`vrp_qubo_terms`), statevector simulation, kappa scoring, and the analog delta
corrections. The full test suite runs about 8x faster with the extension built;
hardware SDK adapters, the gRPC distributed layer, and pipeline orchestration
deliberately stay Python (I/O-bound glue, no compute of their own). Build with
`maturin develop --release`.

**Circuit cutting (gate-model):** `limen/cutting` plus the Rust
`limen_core::cutting` module split a wide circuit into sub-circuits via
quasi-probability decomposition and reconstruct the original expectation value
from real per-subcircuit sampler counts (`limen/cutting/dispatch.py`,
`limen/cutting/reconstruct.py`, `src/cutting/reconstruct.rs`). Validated
end-to-end against a plain AerSimulator run in `examples/cutting_smoke_test.py`.

**Budget router (v0.8.3):** `limen.router` plans a QUBO run before any credit
is spent: deterministic fidelity-tier selection (backend, cutting strategy, ECC
allocation, shot count) against an explicit budget. Backend profiles are seeded
from real run history (`limen.router.history`) and from live IBM calibration
snapshots (`limen.router.calibration`); the first ibm_kingston calibration
(physical error rate 2.586e-2) is 25x closer to measured Tier 2 behaviour than
the previous hardcoded 1e-3. QPU job submission is decoupled from
result-waiting with an on-disk state machine (`limen.router.job_state`), so a
crashed terminal or a long maintenance-window queue cannot strand a completed
job. Hardware-validated on ibm_kingston (job `d965qgotcv6s73djc1l0`).

## Quantum communication case study

LIMEN's quantum-channel module also runs and certifies real-hardware
protocols, not just optimization circuits. As a cross-layout fidelity case
study, the standard 3-qubit teleportation circuit was submitted to
`ibm_kingston` on two physical qubit mappings that share the state and Alice
qubits but differ in the Bob relay qubit: q[10,11,12] (job
`d8u5qqkbp3hs7385g0mg`) and q[10,11,18] (job `d8u6b6kbp3hs7385gil0`), 1000
shots each. The q[10,11,12] layout achieved fidelity 0.954 versus 0.938 for
q[10,11,18], a 1.6 percentage-point gap, isolating the effect of Bob relay
qubit choice on end-to-end teleportation fidelity while holding the state and
Alice qubits fixed. Full counts, timestamps, and verified job layouts:
`results/teleport_summary.json`.

## Phase history

### Phase 1, Core (complete)
- Logical graph IR schema
- PyQUBO frontend adapter
- Lexicographic compiler (D-Wave backend)
- Probabilistic validator (small-instance classical verification)
- IBM Qiskit backend adapter

### Phase 2, Co-design loop (complete)
- Stackelberg co-design loop (joint penalty plus embedding optimization)
- Robust equilibrium selection with calibration margin
- Portfolio compilation with runtime switching conditions

### Phase 3, Analog interface (complete for the restricted class)
- Hamiltonian IR interface specification
- Classical Ising simulation backend (exact diagonalisation, 20 sites or fewer)
- Neutral-atom heuristic backend (van der Waals layout, Rydberg parameters)
- Neutral-atom LHZ parity-encoding fallback (Theorem 3, automatic for
  non-natively-realizable targets)
- Photonic GBS-inspired backend (Arrazola-Bromley adjacency encoding)
- Restricted-class universality results documented and implemented
  (Theorems 1 to 5, `limen/docs/universality_theorem.md`)

### Phase 4, Multi-node distributed architecture (complete)
- Milestone 1: node identity, registry, and gRPC coordination service
  (discovery, heartbeat, calibration sync)
- Milestone 2: distributed QUBO partitioning across nodes (`CompilePartition`
  RPC, wired into `run_pipeline`)
- Milestone 3: classical feedforward transport over `QuantumChannel`
  (`TransportFeedforward` RPC; `run_distributed_feedforward_teleport`
  measures real round-trip latency and feeds it into `ChannelDeltaModel`)

### Phase 5, Gate-model track and ECC certification (complete)
- QAOA compiler: QUBO to Ising to `CircuitIR` with grid-search parameter
  optimisation
- Pure-Python statevector simulator (exact, no external SDK, about a
  14-qubit ceiling)
- Gate-model synthesis: arbitrary 1-qubit unitary decomposition into
  `(rz, ry, rz)` plus `cx`
- Distance-3 rotated surface-code ECC: stabilisers, syndrome extraction,
  lookup decoder
- Gate-executed ECC round-trip: syndrome circuit run on the statevector
  simulator
- `EndToEndCertificate`: solution, optimality, QAOA success rate, logical
  error rate, ECC metadata
- `run_pipeline` end-to-end orchestrator (local and distributed)
- Physics-validation test suite (random QUBO vs brute-force, error-rate
  sweep, distance scaling, weight-2 uncorrectable boundary, BB84 eavesdrop
  detection)
- QPU backend integration: `backend="aer"` / `"qpu"` (IBM hardware via Qiskit
  Runtime) and `backend="dwave"` wired into `run_pipeline`

### Phase 6, Budget router and fleet cost model (complete except as noted)
- Budget router (`limen.router`): deterministic fidelity-tier planning
- Cost-model seeding from run history (`scan_results()` / `apply_history()`)
- Calibration seeding from live hardware (`fetch_backend_calibration()`,
  `scan_calibration()` / `apply_calibration()`)
- Crash-resilient QPU job lifecycle (`limen.router.job_state`)
- First hardware validation: ibm_kingston Tier 2 run (job
  `d965qgotcv6s73djc1l0`)
- Router-driven multi-backend dispatch: `run_route_request()`
- Cut-circuit bridge: `run_cut_route_request()` returning a
  `CuttingCertificate`
- gRPC peer auto-discovery via `LIMEN_KNOWN_PEERS`
- Substrate-aware routing (`ProblemProfile` / `substrate_affinity`)
- Co-design history loop (`codesign_from_history()`)
- Local GPU statevector tier 0 (`LIMEN_LOCAL_GPU=1`, VRAM-sized at call time)
- Azure Quantum / Atom Computing adapter (`limen.backends.azure_atom`):
  import-clean and unit-tested, never run against a live workspace. Paused;
  Atom Computing is not exposed through the Azure marketplace. Excluded from
  `DEFAULT_FLEET`.
