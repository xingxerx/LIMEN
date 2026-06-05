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

## Architecture

```
[ Domain Problem ]
        │
        ▼
┌─────────────────────────────┐
│ SEMANTIC FRONTEND           │
│ PyQUBO · OpenFermion · PDE  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ LOGICAL GRAPH IR            │
│ Hardware-agnostic           │
│ Deterministic normal form   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ LEXICOGRAPHIC COMPILER      │
│ Embedding → Penalty solve   │
│ Fixed seed · Single pass    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ PROBABILISTIC VALIDATOR     │
│ Small-instance regression   │
│ Distribution matching       │
│ Confidence bounds           │
└─────────────┬───────────────┘
              │
              ▼
      [ Physical Execution ]
      D-Wave · IBM · Simulator
```

---

## What LIMEN Is Not

LIMEN is not a quantum computing framework. It does not manage qubits, run circuits, or simulate quantum mechanics. It is a **compiler** — its job is translation, not execution.

LIMEN does not claim to solve NP-hard problems. It claims to translate them correctly and reproducibly onto hardware that attempts to solve them, and to give you a measurable signal about how much to trust the result.

The analog substrate layer (BEC, photonic, continuous-variable) is defined as an interface — not yet implemented. The mathematics required for a constructive universality theorem on those substrates does not yet exist. When it does, LIMEN will be ready to receive it.

---

## Roadmap

### Phase 1 — Core ✓
- [x] Logical graph IR schema
- [x] PyQUBO frontend adapter
- [x] Lexicographic compiler (D-Wave backend)
- [x] Probabilistic validator (small-instance classical verification)
- [x] IBM Qiskit backend adapter

### Phase 2 — Co-Design Loop (In Progress)
- [-] Stackelberg co-design loop (joint penalty + embedding optimization)
- [-] Robust equilibrium selection with calibration margin
- [-] Portfolio compilation with runtime switching conditions

### Phase 3 — Analog Interface
- [ ] Hamiltonian IR interface specification (placeholder layer)
- [ ] Constructive universality theorem integration (pending research)
- [ ] Neutral-atom and photonic backend stubs

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

**v0.1.0 — Phase 1 complete.** Core IR, compiler, validator, PyQUBO frontend, D-Wave and Qiskit backend adapters are all shipped and tested. 11 tests passing.

If you are building in this space and want to collaborate, open an issue.

---

*LIMEN — Latin for threshold. The boundary between two states.*
