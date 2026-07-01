# LIMEN

> *Where classical problems cross into physical computation.*

**LIMEN** is a physics aware compiler stack for translating classical optimization problems into native quantum and analog substrates. It sits at the threshold between human readable problem specifications and the physical hardware that computes them  providing a deterministic, reproducible compilation pipeline with measurable confidence in every result.

---

## The Problem

Physics native hardware exists. Quantum annealers, gate-model processors, and neutral-atom arrays are accessible today. But using them requires deep expertise in Hamiltonian formulation, hardware-specific embedding, and penalty coefficient engineering knowledge that lives in physics departments, not software teams.

The result: the same problem formulated by two different engineers produces two different encodings. Results vary between hardware generations. Confidence in outputs is largely taken on faith. And every new hardware vendor requires rebuilding the entire toolchain from scratch.

LIMEN fixes this.

---

## What LIMEN Does

LIMEN provides five things that don't exist together anywhere today:

**1. A deterministic logical graph IR**
A hardware-agnostic intermediate representation for optimization problems. The same problem always compiles the same way. Swap the backend — D-Wave, IBM, classical simulator — without touching your formulation.

**2. A lexicographic compiler**
Single-pass, deterministic compilation from logical IR to physical encoding. No heuristic randomness. No undocumented magic. Reproducible output every time.

**3. Hardware adapters**
Clean interfaces to D-Wave (via Ocean SDK) and IBM Quantum (via Qiskit) out of the box. New backends are first-class citizens: write an adapter, plug it in.

**4. A gate-model pipeline with error-correction certification**
An end-to-end path from QUBO to a certified result: QAOA compilation, exact statevector simulation, classical optimality verification, and a gate-executed surface-code ECC round-trip — all offline, no QPU required. Returns a single `EndToEndCertificate` with solution, optimality flag, logical error rate, and ECC metadata.

**5. A probabilistic validator**
For every compiled problem, LIMEN produces a confidence bound — not a proof of correctness, but a measurable signal. Small-instance classical verification, cross-architecture sampling, and distribution analysis tell you how much to trust the result.

---

## Workspace Organization

To help navigate the files and components of LIMEN, see the [Directory Map](DIRECTORY_MAP.md) which labels and defines the role of every directory, script, and source file in the repository.

---

## Architecture


```
                         [ Domain Problem ]
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │ SEMANTIC FRONTEND             │
                 │ PyQUBO · OpenFermion · PDE    │
                 └──────────────┬────────────────┘
                                │  LogicalGraph IR
             ┌──────────────────┴──────────────────┐
             │  analog / annealing track            │  gate-model track
             ▼                                      ▼
┌────────────────────────────┐        ┌─────────────────────────────┐
│ LEXICOGRAPHIC COMPILER     │        │ QAOA COMPILER               │
│ Deterministic · fixed seed │        │ QUBO → Ising → CircuitIR    │
│ Greedy minor-embedding     │        │ Grid-search parameter opt.  │
└────────────┬───────────────┘        └──────────────┬──────────────┘
             │  PhysicalEncoding                      │  CircuitIR
             ▼                                        ▼
┌────────────────────────────┐        ┌─────────────────────────────┐
│ PROBABILISTIC VALIDATOR    │        │ STATEVECTOR SIMULATOR       │
│ Brute-force ≤ 20 vars      │        │ Exact · Rust-backed         │
│ Confidence bounds          │        │ No sampling noise           │
└────┬───────────────────────┘        └──────────────┬──────────────┘
     │                                               │  probs + solution
     ├── open-loop ──► D-Wave · IBM · Simulator      ▼
     │                                ┌─────────────────────────────┐
     └── co-design ──►                │ SURFACE-CODE ECC            │
         ┌──────────────────────┐     │ d=3 rotated patch           │
         │ STACKELBERG LOOP     │     │ Gate-executed syndrome      │
         │ κ scoring · Rust     │     │ Lookup decoder              │
         │ Adaptive learning    │     └──────────────┬──────────────┘
         └──────────┬───────────┘                    │  LogicalErrorCertificate
                    ▼                                 │
         ┌──────────────────────┐                    │
         │ PORTFOLIO COMPILER   │                    │
         │ Multi-backend rank   │                    │
         │ Runtime switching    │                    │
         └──────────┬───────────┘                    │
                    ▼                                 │
         ┌──────────────────────┐                    │
         │ HAMILTONIAN IR       │◄───────────────────┘
         │ Z-basis Ising        │  EndToEndCertificate
         │ Substrate-agnostic   │  (solution · optimality ·
         └──────┬───────────────┘   logical error rate ·
                │                   distributed compilation)
       ┌────────┴────────┐
       ▼                 ▼
┌─────────────┐    ┌───────────────┐
│ NEUTRAL-ATOM│    │ PHOTONIC (GBS)│
│ Rydberg     │    │ CV optical    │
│ [Thm 2,3 OK]│    │ [Thm 4 OK]    │
└─────────────┘    └───────────────┘
```

---

## Quantum Communication

LIMEN's quantum-channel module also runs and certifies real-hardware protocols, not just optimization circuits. As a cross-layout fidelity case study, the standard 3-qubit teleportation circuit was submitted to `ibm_kingston` on two physical qubit mappings that share the state and Alice qubits but differ in the Bob relay qubit: q[10,11,12] (job `d8u5qqkbp3hs7385g0mg`) and q[10,11,18] (job `d8u6b6kbp3hs7385gil0`), 1000 shots each. The q[10,11,12] layout achieved fidelity 0.954 versus 0.938 for q[10,11,18] — a 1.6 percentage-point gap — isolating the effect of Bob relay qubit choice on end-to-end teleportation fidelity while holding the state and Alice qubits fixed, and demonstrating LIMEN's ability to certify protocol performance sensitivity to physical qubit mapping on real NISQ hardware. Full counts, timestamps, and verified job layouts: [`results/teleport_summary.json`](results/teleport_summary.json).

---

## Multi-Node Coordination

LIMEN instances can discover each other and exchange hardware calibration state over a gRPC `Coordination` service (`limen.distributed`): each node advertises its `node_id` and the device IDs it serves, registers with a static list of known peers on startup, and can pull a peer's `HardwareDeltaModel` on demand via `SyncCalibration`. On top of this substrate, a QUBO can be **compiled across nodes**: the logical graph is partitioned, each partition is dispatched to a peer via the `CompilePartition` RPC, and the returned encodings are merged into a single encoding energetically equivalent to a one-shot local compile. The end-to-end pipeline exposes this directly — pass `server_addresses=[...]` to `run_pipeline` (see [Getting Started](#getting-started)). See [`limen/docs/architecture.md`](limen/docs/architecture.md#multi-node-coordination-layer) for the design and [`limen/distributed/`](limen/distributed/) for the implementation. Requires the `distributed` extra: `pip install limen[distributed]`.

---

## What LIMEN Is Not

LIMEN is not a general-purpose quantum computing framework or a user-facing circuit runner. It does not replace Qiskit or Cirq. Its job is **translation and certification**, not general quantum execution.

LIMEN *does* include a pure-Python statevector simulator (`limen.gates.simulator`), but its purpose is narrow: it provides an exact, dependency-free backend for offline ECC certification and QAOA parameter validation — without requiring a QPU or an SDK installation. It is capped at roughly 14 logical qubits before memory pressure becomes significant. It is not intended for production circuit workloads.

LIMEN does not claim to solve NP-hard problems. It claims to translate them correctly and reproducibly onto hardware that attempts to solve them, and to give you a measurable signal about how much to trust the result.

The analog substrate layer (BEC, photonic, continuous-variable) is defined as an interface not yet fully implemented. The mathematics required for a constructive universality theorem on those substrates does not yet exist. When it does, LIMEN will be ready to receive it.

---

## Roadmap

### Phase 1 — Core ✓
- [x] Logical graph IR schema
- [x] PyQUBO frontend adapter
- [x] Lexicographic compiler (D-Wave backend)
- [x] Probabilistic validator (small-instance classical verification)
- [x] IBM Qiskit backend adapter

### Phase 2 — Co-Design Loop (Complete)
- [x] Stackelberg co-design loop (joint penalty + embedding optimization)
- [x] Robust equilibrium selection with calibration margin
- [x] Portfolio compilation with runtime switching conditions

### Phase 3 — Analog Interface
- [x] Hamiltonian IR interface specification
- [x] Classical Ising simulation backend (exact diagonalisation, ≤ 20 sites)
- [x] Neutral-atom heuristic backend (van der Waals layout, Rydberg parameters)
- [x] Neutral-atom LHZ parity-encoding fallback (Theorem 3 — automatic for non-natively-realizable targets)
- [x] Photonic GBS-inspired backend (Arrazola-Bromley adjacency encoding)
- [x] Restricted-class universality results documented and implemented (Theorems 1-5, `limen/docs/universality_theorem.md`); a fully general universality theorem for arbitrary (non-diagonal, time-dependent) analog Hamiltonians remains open research and is not claimed

### Phase 4 — Multi-Node Distributed Architecture (Complete)
- [x] Milestone 1: node identity, registry, and gRPC coordination service (discovery, heartbeat, calibration sync)
- [x] Milestone 2: distributed QUBO partitioning across nodes (`CompilePartition` RPC, wired into `run_pipeline`)
- [x] Milestone 3: classical feedforward transport over `QuantumChannel` — `TransportFeedforward` RPC carries Alice's Bell-measurement bits to a peer node, which applies Bob's correction and echoes it back; `run_distributed_feedforward_teleport` measures real round-trip latency and feeds it into `ChannelDeltaModel`

### Phase 5 — Gate-Model Track & ECC Certification (Complete)
- [x] QAOA compiler: QUBO → Ising → `CircuitIR` with grid-search parameter optimisation
- [x] Pure-Python statevector simulator (exact, no external SDK, ~14-qubit ceiling)
- [x] Gate-model synthesis: arbitrary 1-qubit unitary decomposition into `(rz, ry, rz)` + `cx`
- [x] Distance-3 rotated surface-code ECC: stabilisers, syndrome extraction, lookup decoder
- [x] Gate-executed ECC round-trip: syndrome circuit run on the statevector simulator
- [x] `EndToEndCertificate`: solution + optimality + QAOA success rate + logical error rate + ECC metadata
- [x] `run_pipeline` end-to-end orchestrator (local and distributed)
- [x] Physics-validation test suite (random QUBO vs brute-force, error-rate sweep, distance scaling, weight-2 uncorrectable boundary, BB84 eavesdrop detection)
- [x] QPU backend integration for pipeline execution step: `backend="aer"`/`"qpu"` (real IBM hardware via Qiskit Runtime) and `backend="dwave"` (real D-Wave QPU or simulated annealer) are all wired into `run_pipeline`
- [x] Classical feedforward transport over `QuantumChannel` (Milestone 3, see Phase 4 above)

---

## Getting Started

```bash
pip install limen
```

```python
from limen import from_qubo_dict, compile_lexicographic, default_hardware_graph, validate

# Define your problem as a QUBO dict
graph = from_qubo_dict({
    ('x0', 'x0'): -1.0,
    ('x1', 'x1'): -1.0,
    ('x0', 'x1'):  2.0,
})

# Compile to a physical encoding
encoding = compile_lexicographic(graph, default_hardware_graph(4))

# Validate confidence
result = validate(encoding, runs=1000)
print(result.confidence)  # 0.0 – 1.0
```

### End-to-end pipeline (gate-model + ECC, optionally distributed)

`run_pipeline` threads a QUBO through the whole stack — QAOA compilation, statevector execution, classical optimality check, and a surface-code logical-error certificate — and returns a single `EndToEndCertificate`:

```python
import limen

cert = limen.run_pipeline(
    {('x0', 'x0'): -1.0, ('x1', 'x1'): -1.0, ('x0', 'x1'): 2.0},
    physical_error_rate=0.01,
)
print(cert.solution, cert.is_optimal, cert.logical_error_rate)

# Compile across peer nodes over the Coordination CompilePartition RPC
# (requires the `distributed` extra and reachable peers):
cert = limen.run_pipeline(
    {('x0', 'x0'): -1.0, ('x1', 'x1'): -1.0, ('x0', 'x1'): 2.0},
    server_addresses=["127.0.0.1:50051"],
    num_partitions=2,
)
print(cert.distributed_compilation)
```

---

## Design Principles

**Determinism first.** The same input always produces the same encoding. Reproducibility is not optional for scientific infrastructure.

**Honest about limits.** LIMEN reports confidence bounds, not correctness proofs. The validation loop is probabilistic by design, not by accident.

**Hardware agnostic formulation, hardware explicit compilation.** The logical IR knows nothing about hardware topology. The compiler knows everything. The boundary between them is hard and intentional.

**Open core.** The compiler, IR, validator, and hardware adapters are source-available under the Elastic License 2.0. Production features (Stackelberg co-design, portfolio compilation, drift-aware margin selection) will be offered under a commercial license for enterprise deployments.

---

## Contributing

LIMEN is early. The most valuable contributions right now are:

- Problem formulations — real optimization problems encoded in the logical IR
- Hardware adapter implementations for new backends
- Validator improvements — better small-instance solvers, additional confidence metrics
- Benchmarking data from real hardware runs

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

Elastic License 2.0 (ELv2) — see [LICENSE](LICENSE) for details.

LIMEN is source-available: anyone can read, fork, and use the code for
research and internal purposes. ELv2 prohibits offering LIMEN as a hosted
or managed service without a commercial license. Use is also subject to
the restrictions in [ACCEPTABLE_USE.md](ACCEPTABLE_USE.md), which
prohibits weapons development, mass surveillance, unauthorized
cryptanalysis, and attacks on critical infrastructure.

As of v0.3.1, LIMEN changed from Apache 2.0 to ELv2. Copies obtained under
Apache 2.0 retain their Apache 2.0 rights; all new versions are licensed
under ELv2.

---

## Status

**v0.8.2 — Phase 1 complete · Phase 2 complete · Phase 3 substantially implemented · Phase 4 complete · Phase 5 complete · Hardware validated.**
Core IR, compiler, validator, PyQUBO frontend, D-Wave and Qiskit backend
adapters shipped and tested. Stackelberg co-design loop operational with
Rust-backed κ scoring, stability-penalised learning rate, and portfolio
compilation across backend slots. Chain-break-fraction feedback is wired
into the loop on both hardware paths: `ibm_noise_fn` (real-hardware total
variation distance) drives `examples/ibm_codesign_qpu.py`, validated against
`ibm_kingston`, and `dwave_chain_break_fn` (real chain-break fraction) now
drives the equivalent closed loop against a live D-Wave QPU in
`examples/dwave_codesign_qpu.py`. With no callback supplied, chain-break
fraction defaults to 0.0 (simulation mode) — this is documented behaviour,
not an inactive feature.
Analog interface layer implemented: classical Ising simulator (exact
diagonalisation, ≤20 sites), neutral-atom heuristic backend (van der Waals
layout, Rydberg parameters), photonic GBS backend (Arrazola-Bromley adjacency
encoding), HardwareDeltaModel calibration layer (per-site detuning, per-pair
coupling, and global Rabi-drive correction, each with a pure-Python fallback
alongside the Rust-backed implementation). The neutral-atom geometric
embeddability check (Theorem 2 geometry condition) likewise always runs —
numpy when available, a pure-Python Jacobi eigenvalue fallback otherwise —
consistent with LIMEN's zero-mandatory-dependency core (`dependencies = []`
in `pyproject.toml`). Restricted-class universality results (Theorems 1-5)
documented in `limen/docs/universality_theorem.md`; a fully general
universality theorem for arbitrary (non-diagonal, time-dependent) analog
Hamiltonians remains open research and is not claimed — nor is off-diagonal
(X/Y) or time-dependent Hamiltonian support yet implemented in any backend.

**Gate-model track fully operational**: QUBO → QAOA compilation → exact
statevector simulation → surface-code (d=3) ECC syndrome extraction and
logical-error certification, end-to-end with no external SDK required.
`run_pipeline` returns a single `EndToEndCertificate` with solution, optimality
flag, success probability, per-qubit logical error rate, and ECC roundtrip
verification. Distributed compilation over gRPC (`CompilePartition` RPC) wired
into `run_pipeline(server_addresses=[...])`. The pipeline's execution step now
also routes to real backends — `backend="aer"`/`"qpu"` (IBM Quantum hardware)
and `backend="dwave"` (real D-Wave QPU or simulated annealer) — alongside the
default offline statevector simulator.

**Multi-node coordination is now end-to-end**: beyond discovery and
calibration sync, nodes can transport the classical feedforward bits of a
teleportation protocol to a peer over gRPC (`TransportFeedforward` RPC) and
measure real round-trip latency against the qubit's T2 coherence time
(`run_distributed_feedforward_teleport`).

**First real hardware validation: IBM ibm_kingston (Heron R2, 156 qubits),
June 9 2026. Benchmark results in benchmarks/RESULTS.md.**

**Rust acceleration layer (v0.8.x):** every enumeration- or sampling-shaped
hot loop in the library is now Rust-backed via the `limen_core` PyO3
extension, each behind a `try: from limen_core import ...` fast path with an
equivalent pure-Python fallback: QUBO spectrum enumeration, the validator's
noisy-run simulation (`simulate_qubo_runs`, rayon-parallel), surface-code
lookup-table construction and logical-error certification
(`build_ecc_lookup_table` / `logical_failure_probability` — distance-5
decoding drops from intractable-in-Python to seconds), the VRP one-hot QUBO
builder (`vrp_qubo_terms`), statevector simulation, κ scoring, and the
analog delta corrections. The full test suite runs ~8x faster with the
extension built; hardware SDK adapters, the gRPC distributed layer, and
pipeline orchestration deliberately stay Python (I/O-bound glue, no compute
of their own). Build with `maturin develop --release`.

**Circuit cutting (gate-model, not yet reflected elsewhere in this README):**
`limen/cutting` + the Rust `limen_core::cutting` module split a wide circuit
into sub-circuits via quasi-probability decomposition and reconstruct the
original expectation value from real per-subcircuit sampler counts
(`limen/cutting/dispatch.py`, `limen/cutting/reconstruct.py`,
`src/cutting/reconstruct.rs`). Validated end-to-end against a plain
AerSimulator run in `examples/cutting_smoke_test.py`.

323 tests passing, 3 skipped (skips are environment-gated: optional SDKs not
installed in this dev environment). Zero regressions.

If you are building in this space and want to collaborate, open an issue.

---

*LIMEN — Latin for threshold. The boundary between two states.*
