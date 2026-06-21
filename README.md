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

LIMEN provides four things that don't exist together anywhere today:

**1. A deterministic logical graph IR**
A hardware agnostic intermediate representation for optimization problems. The same problem always compiles the same way. Swap the backend: D-Wave, IBM, classical simulator. Without touching your formulation.

**2. A lexicographic compiler**
Single pass, deterministic compilation from logical IR to physical encoding. No heuristic randomness. No undocumented magic. Reproducible output every time.

**3. Hardware adapters**
Clean interfaces to D-Wave (via Ocean SDK) and IBM Quantum (via Qiskit) out of the box. New backends are first-class citizens, write an adapter, plug it in.

**4. A probabilistic validator**
For every compiled problem, LIMEN produces a confidence bound not a proof of correctness, but a measurable signal. Small instance classical verification, cross architecture sampling, and distribution analysis tell you how much to trust the result.

---

## Workspace Organization

To help navigate the files and components of LIMEN, see the [Directory Map](DIRECTORY_MAP.md) which labels and defines the role of every directory, script, and source file in the repository.

---

## Architecture


```
                    [ Domain Problem ]
                            │
                            ▼
                ┌─────────────────────────────┐
                │ SEMANTIC FRONTEND           │
                │ PyQUBO · OpenFermion · PDE  │
                └─────────────┬───────────────┘
                              │  LogicalGraph IR
                              ▼
                ┌─────────────────────────────┐
                │ LEXICOGRAPHIC COMPILER      │
                │ Deterministic · Fixed seed  │
                │ Greedy minor-embedding      │
                └─────────────┬───────────────┘
                              │  PhysicalEncoding
                              ▼
                ┌─────────────────────────────┐
                │ PROBABILISTIC VALIDATOR     │
                │ Brute-force small instances │
                │ Confidence bounds           │
                └──────┬──────────────────────┘
                       │
          ┌────────────┴────────────────────────┐
          │  open-loop path                     │  co-design path
          ▼                                     ▼
┌──────────────────┐             ┌───────────────────────────┐
│ PHYSICAL         │             │ STACKELBERG CO-DESIGN LOOP│
│ EXECUTION        │             │ Rust core (limen_core)    │
│ D-Wave · IBM     │             │ κ scoring · stability     │
│ Simulator        │             │ Adaptive learning rate    │
└──────────────────┘             └────────────┬──────────────┘
                                              │  optimised
                                              │  PhysicalEncoding
                                              ▼
                                ┌───────────────────────────┐
                                │ PORTFOLIO COMPILER        │
                                │ Multi-backend ranking     │
                                │ Runtime switching         │
                                └────────────┬──────────────┘
                                             │
                                             ▼
                                ┌───────────────────────────┐
                                │ HAMILTONIAN IR            │
                                │ Z-basis Ising mapping     │
                                │ Substrate-agnostic        │
                                └────────────┬──────────────┘
                                             │
                              ┌──────────────┴──────────────┐
                              ▼                             ▼
                  ┌────────────────────┐      ┌─────────────────────┐
                  │ NEUTRAL-ATOM STUB  │      │   PHOTONIC STUB     │
                  │ Rydberg blockade   │      │   CV optical        │
                  │ [pending theorem]  │      │   [pending theorem] │
                  └────────────────────┘      └─────────────────────┘
```

---

## What LIMEN Is Not

LIMEN is not a quantum computing framework. It does not manage qubits, run circuits, or simulate quantum mechanics. It is a **compiler** — its job is translation, not execution.

LIMEN does not claim to solve NP-hard problems. It claims to translate them correctly and reproducibly onto hardware that attempts to solve them, and to give you a measurable signal about how much to trust the result.

The analog substrate layer (BEC, photonic, continuous-variable) is defined as an interface not yet implemented. The mathematics required for a constructive universality theorem on those substrates does not yet exist. When it does, LIMEN will be ready to receive it.

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
- [ ] Constructive universality theorem for general analog Hamiltonians (pending research — see limen/docs/architecture.md)

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

---

## Design Principles

**Determinism first.** The same input always produces the same encoding. Reproducibility is not optional for scientific infrastructure.

**Honest about limits.** LIMEN reports confidence bounds, not correctness proofs. The validation loop is probabilistic by design, not by accident.

**Hardware agnostic formulation, hardware explicit compilation.** The logical IR knows nothing about hardware topology. The compiler knows everything. The boundary between them is hard and intentional.

**Open core.** The compiler, IR, validator, and hardware adapters are Apache 2.0. Production features (Stackelberg co-design, portfolio compilation, drift-aware margin selection) will be offered under a commercial license for enterprise deployments.

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

Apache 2.0 — see [LICENSE](LICENSE) for details.

The patent grant clause is intentional. Quantum computing is a patent-dense field. LIMEN's Apache 2.0 license protects contributors and users explicitly.

---

## Status

**v0.3.1 — Phase 1 complete · Phase 2 complete · Phase 3 substantially implemented · Hardware validated.**
Core IR, compiler, validator, PyQUBO frontend, D-Wave and Qiskit backend
adapters shipped and tested. Stackelberg co-design loop operational with
Rust-backed κ scoring, stability-penalised learning rate, chain-break fraction
telemetry wired end-to-end, and portfolio compilation across backend slots.
Analog interface layer implemented: classical Ising simulator (exact
diagonalisation, ≤20 sites), neutral-atom heuristic backend (van der Waals
layout, Rydberg parameters), photonic GBS backend (Arrazola-Bromley adjacency
encoding), HardwareDeltaModel calibration layer.
Pure-Python fallback for all Rust-backed paths. Constructive universality
theorem pending research — formal specification in limen/docs/architecture.md.

**First real hardware validation: IBM ibm_kingston (Heron R2, 156 qubits),
June 9 2026. Benchmark results in benchmarks/RESULTS.md.**

54+ tests passing across six test suites.

If you are building in this space and want to collaborate, open an issue.

---

*LIMEN — Latin for threshold. The boundary between two states.*
