# LIMEN

> *Where classical problems cross into physical computation.*

**LIMEN** is a router and compiler for optimization problems on quantum hardware. You give it a QUBO, a fidelity target and a credit budget. It picks the cheapest backend that meets the target, runs the problem, and hands back a certificate that says what was run, where, and how much to trust the answer.

```bash
pip install limen-compiler
```

The base install has no third-party dependencies. Everything in the quickstart below runs on it, offline, with no account.

---

## Quickstart (about ten minutes)

### 1. Route a QUBO and get a certified answer

```python
from limen import RouteRequest, run_route_request

# A QUBO is a dict of (variable, variable) -> weight.
qubo = {
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x0", "x1"):  2.0,
}

# A budget of 0 credits keeps the run on the free local simulator.
request = RouteRequest(qubo=qubo, fidelity_target=0.9, credit_budget=0)
cert = run_route_request(request)

print(cert.solution)             # {'x0': 1, 'x1': 0}
print(cert.is_optimal)           # True (checked against brute force)
print(cert.success_probability)  # 0.875
```

### 2. See why the router chose what it chose

```python
cert = run_route_request(request, emit_report=True)
print(cert.metadata["route_report_markdown"])
```

```text
# Route report: statevector (Tier 0)

- **Problem size:** 2 variables, 1000 shots
- **Result:** energy -1, optimal, success probability 87.5%

## Why this backend
- criticality spread 1.000 (threshold 10.25) -> tier 0
- backend statevector (sim, 24q, validated=True), 1000 shots
- ECC: skipped (Tier 0 statevector is noiseless)
```

### 3. Dry-run a hardware plan without spending anything

`offline=True` makes every planning decision against the real hardware fleet (backend, shots, error-correction distance), then executes on the local simulator. Full plan, zero credits.

```python
from limen import RouteRequest, route, run_route_request

dry = RouteRequest(qubo=qubo, fidelity_target=0.9, credit_budget=5, offline=True)

plan = route(dry)
print(plan.backend.name, plan.shots)   # ibm_fez 2500

cert = run_route_request(dry)
print(cert.logical_error_rate)         # 1.793e-05 (distance-3 surface code)
```

### 4. Run on real hardware

Install the extra for your backend, then give the request a real budget and drop `offline`:

```bash
pip install "limen-compiler[ibm]"          # IBM hardware and the local Aer simulator
pip install "limen-compiler[braket]"       # AWS Braket
pip install "limen-compiler[dwave]"        # D-Wave annealers
pip install "limen-compiler[cutting]"      # circuit cutting for oversized problems
pip install "limen-compiler[distributed]"  # gRPC peer mesh
pip install "limen-compiler[pqc]"          # ML-DSA-65 signing
pip install "limen-compiler[all]"          # everything
```

For IBM, set `IBM_QUANTUM_TOKEN` and `IBM_QUANTUM_CRN` (or pass `qpu_token` and `qpu_instance`). Credentials never appear in a route plan or a certificate. Job state is written to `results_dir` at every poll, so a crashed terminal or a long queue cannot strand a finished job.

**Tried it? Tell me what broke.** Open an issue with the QUBO (or its size and shape) and what confused you. Bug reports from real problems are the most useful thing you can send.

---

## How routing works

The router is a pure function of the request and the fleet. The same QUBO with the same request always produces the same plan.

| Tier | Runs on | What you get |
|---|---|---|
| 0 | Local simulator (free, exact). Optional local GPU statevector with `LIMEN_LOCAL_GPU=1`. | Solution checked against brute force |
| 1 | Quantum hardware | Tier 0, plus a surface-code logical-error certificate |
| 2 | Validated hardware only | Tier 1, plus a physical-qubit patch budget spent on the variables that dominate error sensitivity |

- A budget of 0, or a fidelity target below 0.5, stays on Tier 0.
- Between Tier 1 and Tier 2 the router looks at how unevenly error sensitivity is spread across your variables. A flat spread gets plain sampling. A heavy-tailed one gets targeted protection.
- The budget is converted into a shot count before anything is submitted.
- A problem too large for any single backend is split by circuit cutting and returns a `CuttingCertificate`.
- Past runs and live calibration data in `results_dir` are folded into the fleet, so the plan uses measured error rates where they exist.

**Hardware it routes to today**

| Backend | Status |
|---|---|
| IBM `ibm_kingston`, `ibm_fez`, `ibm_marrakesh` (Heron, 156 qubits) | Validated with real jobs |
| IonQ `forte-1` through Open Quantum | Wired, not yet validated |
| D-Wave (Ocean SDK), QuEra Aquila (AWS Braket) | Adapters available through `run_pipeline` |
| Azure Quantum / Atom Computing | Paused. Atom Computing is not offered on a public marketplace. |

---

## What else is in the box

- **Deterministic compiler.** A hardware-agnostic logical graph IR and a single-pass lexicographic compiler. Same input, same encoding, every time.
- **Probabilistic validator.** A confidence bound for every compiled problem, from small-instance classical verification and distribution analysis.
- **Gate-model pipeline.** QUBO to QAOA to execution to a surface-code error certificate in one call (`run_pipeline`), fully offline if you want.
- **Router memory.** A SQLite ledger of past runs with an append-only, hash-chained certificate log.
- **Constraint compiler.** Declare typed constraints (`OneHot`, `AtMostK`, `Equality` and others) and get a validated QUBO, so you do not hand-derive penalty terms.
- **Multi-node compilation.** Nodes discover each other over gRPC and compile one problem across peers.
- **Co-design loop.** Stackelberg joint tuning of penalties and embedding, with portfolio compilation across backends.
- **Analog substrates.** Neutral-atom and photonic backends for a restricted, documented class of Hamiltonians.
- **Rust core.** The hot loops are Rust (PyO3) with pure-Python fallbacks. Prebuilt wheels for Windows, Linux (x86_64 and aarch64) and Apple Silicon.

The lower-level compiler is available directly:

```python
from limen import from_qubo_dict, compile_lexicographic, default_hardware_graph, validate

graph = from_qubo_dict(qubo)
encoding = compile_lexicographic(graph, default_hardware_graph(4))
print(validate(encoding, runs=1000).confidence)  # 0.0 to 1.0
```

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

The budget router sits in front of both tracks and decides which path a request takes. Design notes: [docs/architecture.md](docs/architecture.md). File-by-file map: [docs/DIRECTORY_MAP.md](docs/DIRECTORY_MAP.md).

---

## What LIMEN is not

- **Not a general quantum framework.** It does not replace Qiskit or Cirq. Its job is translation, routing and certification.
- **Not a solver that beats NP-hardness.** It translates problems correctly and reproducibly onto hardware that attempts them, and tells you how much to trust the result.
- **Not a proof of correctness.** Certificates carry confidence bounds and modelled error rates. Optimality is only claimed where a brute-force check was possible.
- **Not a production simulator.** The built-in statevector simulator exists for offline certification and small instances (the router caps it at 24 variables).
- **Not a general analog compiler.** Analog support covers a restricted class (diagonal Ising forms). The general case is open research and is not claimed.

---

## Status

**v0.9.1.** 677 tests passing. Validated on real IBM hardware since June 2026 (first run: `ibm_kingston`, Heron R2, 156 qubits).

- What is built, phase by phase, plus the hardware case studies: [docs/STATUS.md](docs/STATUS.md)
- What is shipped, blocked and out of scope: [docs/ROADMAP.md](docs/ROADMAP.md)
- Release notes: [docs/CHANGELOG.md](docs/CHANGELOG.md)
- Benchmarks: [benchmarks/RESULTS.md](benchmarks/RESULTS.md)

**Open now**

- A routing target for IBM's Nighthawk processors
- Hardware validation for the IonQ path
- Feedback from people outside the project (see the quickstart)

---

## Free and paid tiers

The core of this repository is source-available under the Elastic License 2.0 and free to use, including in production: the IR, compilers, validator, hardware adapters, gate-model pipeline and unsigned certificates, budget router, router memory, circuit cutting, co-design, and portfolio compilation. Two paid tiers check an offline, post-quantum-signed (ML-DSA-65) license key. Nothing phones home.

| Tier | Adds |
|---|---|
| Free | Everything above |
| Pro | Signed certificates (ML-DSA-65 signed ledger entries) |
| Team | Pro, plus multi-node peer registration (`peer_mesh`) and written permission to offer LIMEN as a hosted or managed service, which ELv2 otherwise prohibits |

Key install, verification and how to request a license: [docs/LICENSING.md](docs/LICENSING.md).

---

## Contributing

LIMEN is early. The most valuable contributions right now:

- Real optimization problems run through the quickstart, with notes on what broke
- Hardware adapters for new backends
- Validator improvements: better small-instance solvers, more confidence metrics
- Benchmark data from real hardware runs

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

---

## License

Elastic License 2.0 (ELv2). See [LICENSE](LICENSE).

LIMEN is source-available: anyone can read, fork, and use the code for research and internal purposes. ELv2 prohibits offering LIMEN as a hosted or managed service without a commercial license. Use is also subject to [ACCEPTABLE_USE.md](ACCEPTABLE_USE.md), which prohibits weapons development, mass surveillance, unauthorized cryptanalysis, and attacks on critical infrastructure.

As of v0.3.1, LIMEN changed from Apache 2.0 to ELv2. Copies obtained under Apache 2.0 retain their Apache 2.0 rights. All new versions are licensed under ELv2.

---

*LIMEN: Latin for threshold. The boundary between two states.*
