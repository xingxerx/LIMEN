<!-- Copyright 2026 LIMEN Contributors. Apache 2.0 -->

# LIMEN — Architecture

## Overview

LIMEN is a compiler, not a solver. Its job is to translate a classical optimization problem expressed as a QUBO into a physical encoding suitable for quantum annealing, gate-model, or analog hardware — deterministically, reproducibly, and with a measurable confidence bound on the result. The name is Latin for *threshold*: the boundary between classical problem specification and physical computation.

Determinism is the foundational constraint from which everything else follows. Two researchers running the same LIMEN version on the same problem must produce bit-for-bit identical encodings. This rules out any heuristic randomness in the compilation path. It makes reproducibility across hardware generations possible. It is what separates a scientific instrument from a black box.

The system divides cleanly into two completion states. Phases 1 and 2 — the logical IR, lexicographic compiler, probabilistic validator, Stackelberg co-design loop, and portfolio compiler — are fully implemented and tested. Phase 3, the analog substrate layer, has its interface defined and its stubs in place, but the compilation theorem that would give those stubs real implementations is a research milestone, not an engineering one. That boundary is intentional and documented precisely so that when the theorem arrives, the receiving interface is already there.

---

## The Logical Graph IR

The `LogicalGraph` is a hardware-agnostic normal form for binary optimization problems. It stores variables by name and interactions as weighted pairs. The boundary between the logical IR and the physical compiler is hard and intentional: the IR knows nothing about hardware topology, qubit connectivity, or chain strength. A `LogicalGraph` constructed from a QUBO on a laptop will compile identically on any machine running the same LIMEN version.

Deterministic normal form means two things in practice. First, variables are ordered lexicographically — the IR imposes a canonical ordering so that the same problem expressed with variables in a different input order still produces the same encoding. Second, interactions are deduplicated and canonically represented as `(i, j)` pairs with `i ≤ j`, so no two terms encode the same interaction with different signs through ordering accidents.

JSON serialization is a first-class concern rather than an afterthought. A `LogicalGraph` round-trips through `to_dict` / `from_dict` without loss. This matters for reproducibility across environments: a problem encoded on one machine can be serialized, transmitted, deserialized, and compiled elsewhere with the guarantee that the output is identical.

---

## The Lexicographic Compiler

The lexicographic compiler (`compile_lexicographic`) maps each logical variable to exactly one physical qubit using a greedy 1-to-1 assignment in lexicographic variable order. This is naive by design. It does not perform minor embedding, does not optimize for hardware connectivity, and does not minimize chain length. It is correct by construction for any hardware graph with at least as many nodes as logical variables.

The chain strength is auto-calculated as `1.5 * max(|weights|)` with a floor of `1.0`. This heuristic is deliberately conservative: it over-penalizes chains relative to problem terms, which reduces chain breaks at the cost of compressing the energy gap between near-optimal solutions. It is an acceptable starting point because the Stackelberg co-design loop will adjust it.

What the lexicographic compiler gets wrong on real QPU hardware: it assumes a complete hardware graph (`default_hardware_graph(n)`) which does not match the sparse Pegasus or Zephyr topologies of real annealers. For non-complete topologies, the compiler would need a minor-embedding step before the 1-to-1 assignment. This is the primary engineering gap for production QPU use.

---

## The Probabilistic Validator

The validator (`validate`) measures confidence as the fraction of simulated runs whose energy falls within 5% of the best energy found. It is not a correctness proof. It is an honest signal.

For instances with 20 or fewer variables, the validator also runs a brute-force exhaustive search to find the true classical optimum, giving an absolute reference for the energy gap. Above 20 variables, the brute-force step is skipped and `classical_energy` is `None`. This limit is not a lazy cutoff — 2^20 is approximately 1 million evaluations, which is fast; 2^21 starts to be slow; the cutoff is where the cost exceeds the value.

The validator is self-referential on large instances: it simulates runs by perturbing the best-known solution with bit-flip noise, then measures how often the perturbed runs return close to the known best. This measures the stability of the encoding's energy landscape, not the quality of real hardware sampling. It is the honest approximation available without real QPU access.

---

## The Stackelberg Co-Design Loop

The co-design loop frames penalty coefficient selection as a Stackelberg game. The compiler is the leader: it proposes a chain strength (the penalty coefficient governing how strongly physical qubits in a chain are tied together). The hardware is the follower: it responds with a solution distribution. The equilibrium is the chain strength at which the calibration margin κ is maximised.

The κ formula implemented in `src/scoring.rs` is:

```
κ = 0.5 * confidence + 0.3 * gap_term + 0.2 * cbf_penalty
```

where:

```
gap_term    = min(|E_second - E_best|, 10.0) / 10.0
cbf_penalty = 1.0 - chain_break_fraction
```

The weights reflect the signal hierarchy. Confidence is the primary signal: it directly measures how often the hardware finds good solutions. The energy gap term measures how well-separated the optimum is from the next-best solution — a proxy for problem hardness relative to the current encoding; a large gap means the hardware has an easier time distinguishing the optimum. The chain break fraction penalty is a hardware health signal: broken chains mean qubits in the same logical variable disagree, which corrupts the solution.

**Open item:** `chain_break_fraction` is currently `0.0` everywhere. The value will be non-zero only when responses come from a real QPU that reports per-chain break counts. The D-Wave Ocean SDK includes this in its sample response; the field is wired up in `run_codesign` and will become active when the QPU execution path is exercised.

The adaptive learning rate addresses oscillation. When κ moves up and down between iterations rather than monotonically improving, it indicates the encoding is near a bifurcation point in the penalty landscape where small chain strength changes flip the dominant solution regime. The stability penalty is:

```
stability_penalty = clamp(kappa_std * 5.0, 0.0, 0.9)
effective_lr      = base_lr * (1.0 - stability_penalty)
```

A κ standard deviation of 0.18 (moderate oscillation) reduces the effective learning rate by 90%, forcing the solver to take smaller steps until the landscape stabilises.

---

## Portfolio Compilation

Portfolio compilation (`compile_portfolio`) runs the co-design loop independently for each candidate backend and ranks the results by κ. The output is an ordered list of `(encoding, κ)` pairs, one per backend slot. The `SwitchingCondition` mechanism allows runtime selection between backends based on conditions that are not known at compile time — hardware availability, queue depth, cost constraints.

This is the mechanism for hardware-agnostic runtime selection. The decision of which hardware to use is deferred past compile time, but the compilation for each candidate is done deterministically in advance.

---

### Mathematical Bounds for Constructive Universality

A blanket, unconstrained deterministic universal mapping across generalized physical mediums is impossible due to topological boundaries. As established in Hornischer’s 2025 Fraïssé dynamical system limits proof, there exists no coordinate-free transformation that can map any arbitrary physical Hamiltonian into a universal logical space without encountering non-computable state-space blowups or breaking the adiabatic theorem.

LIMEN bypasses this limitation by explicitly restricting its universality subclass to variational energy landscapes that minimize quadratic forms (Ising/QUBO). By focusing on quadratic forms, LIMEN allows for coordinate-free parameter projection down to localized physical substrates. This restriction transforms a non-computable universal mapping problem into a solvable algebraic rewrite task. The "Universality Layer" in LIMEN is thus not a claim of general-purpose quantum computing universality, but a claim of constructive universality for quadratic energy minimization across diverse physical substrates.

---

## The Hamiltonian IR

The Hamiltonian IR (`HamiltonianIR`) is the output of the LIMEN compiler stack toward analog substrates. It is produced by the standard QUBO → Ising substitution:

```
x_i = (1 + σ_i^z) / 2
```

Substituting into the QUBO objective `Σ_{ij} Q_ij x_i x_j`:

- Linear terms: `Q_ii * x_i → (Q_ii / 2) * Z_i + Q_ii / 2`
- Quadratic terms: `Q_ij * x_i x_j → (Q_ij / 4) * Z_i Z_j + (Q_ij / 4)(Z_i + Z_j) + Q_ij / 4`

The additive constants are discarded; they are irrelevant to finding the minimum. The single-site contributions from quadratic terms are collected as additional Z terms. The result is a sum of Z and ZZ `HamiltonianTerm` objects stored in `HamiltonianIR.terms`.

Only Z and ZZ terms are emitted. This is the Ising Z-basis: the natural language of quantum annealing and the starting point for analog substrate compilation. X and Y operators appear in gate-model circuits and in Rydberg drive terms, but they are the backend's responsibility to introduce — they are not present in the problem Hamiltonian as LIMEN sees it.

`HamiltonianIR` is substrate-agnostic by design. The mapping from Z/ZZ operators to substrate-specific physical interactions — blockade potentials, optical modes, scattering lengths — is the job of the backend adapter, not the IR. This separation is what allows the same IR to target neutral-atom, photonic, or BEC hardware without changes.

---

## The Phase 3 Research Gap

The following is a precise specification of what is missing. The interface is defined in `limen/analog/hamiltonian.py`; what does not exist is the compilation theorem that would give the backend stubs real implementations.

### Neutral-atom backends (Rydberg blockade)

The interaction Hamiltonian for a neutral-atom array is:

```
H = Σ_i (Ω/2) σ_i^x - Σ_i Δ n_i + Σ_{i<j} V(r_ij) n_i n_j
```

where `V(r_ij) = C_6 / r_ij^6` is the van der Waals interaction, `Ω` is the Rabi frequency (global drive), `Δ` is the detuning, and `n_i = (1 - σ_i^z) / 2` is the number operator. A constructive universality theorem for this substrate would need to establish:

- A mapping from arbitrary Z/ZZ Ising coefficients `(h_i, J_ij)` to hardware parameters `(Ω, Δ, {r_ij})` — specifically, a spatial layout algorithm that places atoms such that the resulting van der Waals interactions approximate the target `J_ij` values to within a computable error bound
- A treatment of the blockade constraint: pairs of atoms within the blockade radius `r_b = (C_6 / Ω)^{1/6}` cannot both be in the excited state simultaneously, which restricts the encodable interaction graph
- A compilation certificate: a bound on the approximation error `‖H_target - H_compiled‖` as a function of the number of atoms and the layout precision

### Photonic backends (continuous-variable)

The relevant Hilbert space is the Fock space of optical modes. Available operations split into Gaussian (squeezing `S(r)`, displacement `D(α)`, beamsplitters `BS(θ,φ)`) and non-Gaussian (Kerr interactions `χ (a†a)^2`, photon-number-resolving measurement). A constructive universality theorem would need to establish:

- A mapping from binary QUBO variables to optical modes — the most natural encoding associates a binary variable with the photon parity of a mode, but other encodings are possible and may have better noise properties
- A circuit decomposition expressing the target Ising Hamiltonian evolution `e^{-iHt}` in terms of available Gaussian and non-Gaussian gates, with an explicit gate count as a function of problem size
- A readout protocol that recovers the binary assignment from a homodyne or heterodyne measurement outcome with bounded error probability

This is a non-trivial result because binary variables map awkwardly onto the continuous-variable Hilbert space, and the available non-Gaussian operations on current photonic hardware are limited and noisy.

---

## Multi-Node Coordination Layer

`limen.distributed` is the foundation for running LIMEN across more than one process: node identity, a peer registry, and a gRPC `Coordination` service for discovery and `HardwareDeltaModel` sync. It is deliberately scoped — this is the transport that distributed QUBO partitioning and cross-node classical feedforward transport will dispatch over, not those features themselves.

A node is identified by `NodeInfo` (`node_id`, `host`, `port`, the `device_ids` it serves) and configured from the environment via `NodeConfig.from_env()` (`LIMEN_NODE_ID`, `LIMEN_NODE_HOST`/`PORT`, `LIMEN_NODE_DEVICE_IDS`, `LIMEN_KNOWN_PEERS`). There is no service-discovery infrastructure (no etcd, no consul) — peers are a static list configured per node and exchanged via mutual self-registration at startup. For two nodes that each list the other in `LIMEN_KNOWN_PEERS`, registration becomes symmetric without any merge logic: A's `Register` call against B populates B's registry with A, and B's own startup call against A populates A's registry with B.

`NodeRegistry` wraps the existing single-process `DeltaModelRegistry` (`limen/analog/delta_model.py`) rather than replacing it. Local device lookups behave exactly as they did before this layer existed. A device ID not found locally falls through to a TTL'd cache of models fetched from peers via `SyncCalibration` — the registry itself does no network I/O; callers populate the cache after a round trip through `CoordinationClient`.

Wire messages mirror the project's existing `to_dict()` / `from_dict()` JSON-safe-dict convention rather than introducing a parallel schema (`limen/distributed/marshal.py`): a `HardwareDeltaModelProto` has the same shape as `HardwareDeltaModel.to_dict()`, down to the stringified tuple keys for `coupling_scale_errors`. The one deliberate lossy spot is `metadata`, which is `map<string, string>` on the wire — non-string metadata values are coerced via `str()` and will not round-trip to their original type. No caller needed richer metadata for milestone 1.

Explicitly out of scope for this layer: TLS/auth on the gRPC channel (fine for a LAN/VPN-trusted pair of nodes; flagged as a follow-up once topology is proven), and any external service-discovery system. The static peer list is sufficient to validate the two-node case; it is not meant to scale to large clusters without revisiting node discovery.

---

## PyO3 Bridge

The inner scoring loop (`StackelbergSolver.solve`, `compute`, `compute_stability`) lives in Rust and is exposed to Python via PyO3. The reasons are iteration speed, absence of the GIL during the inner loop, and deterministic floating-point behavior across platforms (Rust's `f64` operations are IEEE 754 strict in the same way across all targets LIMEN supports).

The PyO3 boundary is thin. What crosses from Python to Rust: `Vec<f64>` for confidence, energy, and chain-break histories; `f64` for chain strength and learning rate; `usize` for iteration counts. What stays on the Python side: the `PhysicalEncoding` dataclass, the `LogicalGraph` IR, and all recompilation logic. Rust owns the scoring function and convergence decision; Python owns the compilation and the loop structure. This division means the Rust extension is optional — if `limen_core` is not built, `run_codesign` raises `ImportError` cleanly and the rest of the library continues to function.

---

## Invariants

These are the architectural invariants that must never be violated:

0. **Determinism.** The same input always produces the same encoding. No randomness in the compilation path.
1. **IR isolation.** The `LogicalGraph` IR knows nothing about hardware topology. The compiler knows everything. The boundary between them is hard.
2. **Honest bounds.** Confidence bounds are always reported, never suppressed or rounded up. If a confidence value cannot be computed, it is `None`, not `1.0`.
3. **Optional SDKs.** All hardware SDK dependencies (`dwave-ocean-sdk`, `qiskit`, `pyqubo`) are optional at import time. A bare `import limen` with none of these installed must succeed.
4. **Optional Rust.** The `limen_core` Rust extension is optional at import time. All functionality except the Stackelberg co-design loop is available without it.
5. **License hygiene.** Apache 2.0 throughout. No copyleft dependencies. The patent grant clause is intentional.
6. **Optional distributed deps.** `grpcio`/`grpcio-tools`/`protobuf` are only required by `limen.distributed` (the `distributed` extra). A bare `import limen` with none of these installed must succeed.
