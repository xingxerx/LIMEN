<!-- Copyright 2026 LIMEN Contributors. Apache 2.0 -->

# LIMEN — Architecture

## Overview

LIMEN is a compiler, not a solver. Its job is to translate a classical optimization problem expressed as a QUBO into a physical encoding suitable for quantum annealing, gate-model, or analog hardware — deterministically, reproducibly, and with a measurable confidence bound on the result. The name is Latin for *threshold*: the boundary between classical problem specification and physical computation.

Determinism is the foundational constraint from which everything else follows. Two researchers running the same LIMEN version on the same problem must produce bit-for-bit identical encodings. This rules out any heuristic randomness in the compilation path. It makes reproducibility across hardware generations possible. It is what separates a scientific instrument from a black box.

The system divides cleanly into two completion states. Phases 1 and 2 — the logical IR, lexicographic compiler, probabilistic validator, Stackelberg co-design loop, and portfolio compiler — are fully implemented and tested. Phase 3, the analog substrate layer, is *not* a set of unimplemented stubs: `limen/docs/universality_theorem.md` proves restricted-class compilation theorems (Theorems 2, 4, 5) for the neutral-atom, photonic, and BEC backends, each giving an exact, certified coefficient mapping for diagonal Z/ZZ Ising Hamiltonians, and all three backends are implemented and tested against those proofs. What remains a research milestone, not an engineering one, is the fully general theorem described below — covering arbitrary, non-diagonal, time-dependent analog Hamiltonians (e.g. the raw Rydberg blockade Hamiltonian with its Ω/Δ drive terms, or photonic circuits built from non-Gaussian gates) rather than the restricted diagonal class LIMEN currently compiles. That boundary is intentional and documented precisely so that when the general theorem arrives, the receiving interface is already there.

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

## The Gate-Model Track

The gate-model track is a parallel compilation path that branches from the `LogicalGraph` IR. Where the lexicographic compiler produces a `PhysicalEncoding` for annealing or analog hardware, the gate-model track produces a `CircuitIR` — a typed, validated sequence of quantum gate instructions — and then executes, certifies, and returns a single `EndToEndCertificate`.

### QAOA Compiler (`limen.gates.qaoa`)

`compile_qaoa(graph, layers, grid_size)` implements the Quantum Approximate Optimization Algorithm ansatz:

1. **QUBO → Ising substitution** (`qubo_to_ising`): `x_i = (1 - z_i) / 2` maps each binary variable to a spin. Linear QUBO coefficients become single-site Z fields; quadratic coefficients become ZZ coupling terms.
2. **Circuit construction**: For each QAOA layer, a problem unitary `exp(-i γ H_P)` (implemented as `rz(2γ J_ij)` + `cx` for each ZZ coupling, `rz(2γ h_i)` for each Z field) is followed by a mixer unitary `exp(-i β H_M)` (implemented as `rx(2β)` on each qubit), preceded by a uniform superposition layer (H on every qubit).
3. **Parameter optimisation**: A grid search over `(γ, β)` ∈ `[0, π] × [0, π]` drives the statevector simulator to maximise the probability of the optimal bitstring. The grid resolution is `grid_size × grid_size` (default 20×20).

The output is a `CircuitIR` and a variable-order mapping used to interpret bitstrings back into named assignments.

### Statevector Simulator (`limen.gates.simulator`)

`StatevectorSimulator` is a pure-Python exact simulator with no external SDK dependency. It maintains a 2ⁿ complex statevector and applies gates as numpy matrix operations.

- All gates in `KNOWN_GATES` are supported: `h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`, `u`, `cx`, `cz`, `swap`.
- `run(circuit)` → statevector (exact).
- `probabilities(circuit)` → measurement probability distribution over all 2ⁿ bitstrings.
- `sample(circuit, shots)` → simulated measurement count dict (deterministic with a fixed seed).

Practical ceiling: ~14 logical qubits before memory pressure becomes significant on a laptop. This is intentional — the simulator is a certification tool, not a production execution backend. For QPU execution, `limen.gates.qiskit_exec` converts a `CircuitIR` to a Qiskit `QuantumCircuit` for submission to IBM hardware.

### Gate Synthesis (`limen.gates.synthesis`)

`decompose_unitary_1q(U)` decomposes an arbitrary 2×2 unitary into a `(rz, ry, rz)` Euler-angle sequence plus a global phase. This makes the gate-model IR complete for arbitrary single-qubit operations: any 1-qubit gate can be expressed as a sequence of native `rz`/`ry` instructions already in `KNOWN_GATES`.

---

## Surface-Code ECC & Logical Error Certification

### Surface Code (`limen.ecc.surface_code`)

The `SurfaceCode` class implements the distance-3 rotated surface code:
- **9 data qubits** arranged in a 3×3 patch.
- **8 stabilisers** (4 X-type, 4 Z-type); all boundary stabilisers have weight 2, bulk stabilisers have weight 4.
- **Logical X** and **logical Z** operators each span 3 qubits across the patch.
- Code distance 3: any single-qubit error produces a unique, non-trivial syndrome.

### Syndrome Circuit (`limen.ecc.syndrome`)

`build_syndrome_circuit(code)` constructs a `CircuitIR` that performs stabiliser extraction:
- **Z stabilisers**: one ancilla qubit per stabiliser, CX from each data qubit in the stabiliser support into the ancilla.
- **X stabilisers**: Hadamard on the ancilla before and after the CX sequence (standard Hadamard sandwich).

The circuit runs on the `StatevectorSimulator` and returns syndrome bits as measurement outcomes.

### Lookup Decoder (`limen.ecc.decoder`)

`LookupDecoder` pre-computes a table mapping each weight-1 error pattern (9 single-qubit X errors) to its syndrome, then inverts the table for decoding. Given a syndrome pattern from the simulator output, the decoder returns the most likely single-qubit correction. Unknown syndromes (weight-2 or higher) fall back to no correction — this is the intentional boundary enforcement: a d=3 code corrects all weight-1 errors and is not guaranteed to correct weight-2.

### ECC Round-Trip (`limen.ecc.encoder`)

`run_logical_roundtrip(code, simulator)` performs the full gate-executed ECC verification loop:

1. For each data qubit `i`, inject a single-qubit X error on qubit `i`.
2. Run the syndrome extraction circuit on the simulator.
3. Decode the syndrome using the lookup decoder.
4. Apply the correction and verify that the post-correction state matches the original (no error) state.

Returns a dict of `{qubit_index: corrected}` booleans. A certificate passes `roundtrip_corrects_all_weight1` if all 9 are `True`.

### Logical Error Certificate (`limen.ecc.certificate`)

`LogicalErrorCertificate` computes the analytic per-qubit logical error rate using the leading-order code-distance formula:

```
p_L ≈ C(d, ⌊(d+1)/2⌋) * p^⌈(d+1)/2⌉
```

For d=3: `p_L ≈ 3 p²` (three weight-2 configurations that lead to logical failure). At `p = 0.01`, this gives `p_L ≈ 1.73 × 10⁻³`, a ~5.8× suppression over the physical rate.

The aggregate rate across all `n` logical qubits is `1 - (1 - p_L)^n`.

---

## The `run_pipeline` Orchestrator

`limen.pipeline.run_pipeline(qubo, ...)` is the single entry point for the gate-model track. Its steps:

1. **QUBO → LogicalGraph** (`from_qubo_dict`).
2. **Partitioning** (if `server_addresses` provided): split the `LogicalGraph` into `num_partitions` sub-graphs via `partition_graph`, dispatch each to a peer node's `CompilePartition` gRPC RPC, merge the returned encodings.
3. **QAOA compilation** (`compile_qaoa`): QUBO → QAOA `CircuitIR` with parameter optimisation.
4. **Simulation** (`StatevectorSimulator.probabilities`): exact measurement distribution.
5. **Solution extraction**: most-likely bitstring → named assignment via variable-order mapping.
6. **Optimality check**: brute-force QUBO energy evaluation over all 2ⁿ assignments (skipped above 20 variables).
7. **ECC certification** (if `physical_error_rate` provided): `LogicalErrorCertificate` analytic rate + gate-executed `run_logical_roundtrip`.
8. **Certificate assembly**: `EndToEndCertificate` with solution, is_optimal, energy, QAOA success probability, ECC metadata, and notes.

---

## Portfolio Compilation

Portfolio compilation (`compile_portfolio`) runs the co-design loop independently for each candidate backend and ranks the results by κ. The output is an ordered list of `(encoding, κ)` pairs, one per backend slot. The `SwitchingCondition` mechanism allows runtime selection between backends based on conditions that are not known at compile time — hardware availability, queue depth, cost constraints.

This is the mechanism for hardware-agnostic runtime selection. The decision of which hardware to use is deferred past compile time, but the compilation for each candidate is done deterministically in advance.

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

This section describes the gap that remains *after* the restricted-class theorems (2, 4, 5 in `limen/docs/universality_theorem.md`) were proved and implemented. Those theorems compile arbitrary diagonal Z/ZZ Ising coefficients exactly (with a computable error bound, Theorem 1) onto each substrate's native parameters — that part is done and tested. What follows is a precise specification of the harder, still-open problem: a *general* universality theorem covering the full native Hamiltonian of each substrate (drive terms, non-Gaussian gates, blockade constraints), not just its diagonal Ising sector. The interface for this is defined in `limen/analog/hamiltonian.py`.

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

`limen.distributed` is the foundation for running LIMEN across more than one process: node identity, a peer registry, and a gRPC `Coordination` service for discovery, `HardwareDeltaModel` sync, and distributed QUBO compilation. The `CompilePartition` RPC is fully wired into `run_pipeline(server_addresses=[...])` — passing one or more peer addresses delegates graph partitioning and sub-graph compilation to remote nodes, with the merged result certified locally.

A node is identified by `NodeInfo` (`node_id`, `host`, `port`, the `device_ids` it serves) and configured from the environment via `NodeConfig.from_env()` (`LIMEN_NODE_ID`, `LIMEN_NODE_HOST`/`PORT`, `LIMEN_NODE_DEVICE_IDS`, `LIMEN_KNOWN_PEERS`). There is no service-discovery infrastructure (no etcd, no consul) — peers are a static list configured per node and exchanged via mutual self-registration at startup. For two nodes that each list the other in `LIMEN_KNOWN_PEERS`, registration becomes symmetric without any merge logic: A's `Register` call against B populates B's registry with A, and B's own startup call against A populates A's registry with B.

`NodeRegistry` wraps the existing single-process `DeltaModelRegistry` (`limen/analog/delta_model.py`) rather than replacing it. Local device lookups behave exactly as they did before this layer existed. A device ID not found locally falls through to a TTL'd cache of models fetched from peers via `SyncCalibration` — the registry itself does no network I/O; callers populate the cache after a round trip through `CoordinationClient`.

Wire messages mirror the project's existing `to_dict()` / `from_dict()` JSON-safe-dict convention rather than introducing a parallel schema (`limen/distributed/marshal.py`): a `HardwareDeltaModelProto` has the same shape as `HardwareDeltaModel.to_dict()`, down to the stringified tuple keys for `coupling_scale_errors`. The one deliberate lossy spot is `metadata`, which is `map<string, string>` on the wire — non-string metadata values are coerced via `str()` and will not round-trip to their original type. No caller needed richer metadata for milestone 1.

Explicitly out of scope for this layer: TLS/auth on the gRPC channel (fine for a LAN/VPN-trusted pair of nodes; flagged as a follow-up once topology is proven), and any external service-discovery system. The static peer list is sufficient to validate the two-node case; it is not meant to scale to large clusters without revisiting node discovery.

---

## PyO3 Bridge

The inner scoring loop (`StackelbergSolver.solve`, `compute`, `compute_stability`) lives in Rust and is exposed to Python via PyO3. The reasons are iteration speed, absence of the GIL during the inner loop, and deterministic floating-point behavior across platforms (Rust's `f64` operations are IEEE 754 strict in the same way across all targets LIMEN supports).

The PyO3 boundary is thin. What crosses from Python to Rust: `Vec<f64>` for confidence, energy, and chain-break histories; `f64` for chain strength and learning rate; `usize` for iteration counts. What stays on the Python side: the `PhysicalEncoding` dataclass, the `LogicalGraph` IR, and all recompilation logic. Rust owns the scoring function and convergence decision; Python owns the compilation and the loop structure. This division means the Rust extension is optional — if `limen_core` is not built, `run_codesign` raises `ImportError` cleanly and the rest of the library continues to function.

The same Rust-first / Python-fallback split now covers every enumeration- or sampling-shaped hot loop in the library, not just the scoring function. Each of these is a `try: from limen_core import ... / except ImportError:` fast path in front of a pure-Python reference implementation that produces equivalent output:

- `qubo_energy_spectrum` — the shared O(2^n) brute-force QUBO enumeration behind `validator.brute_force_solve`, `codesign._second_best_energy`, and the Qiskit backend's exact solver.
- `simulate_qubo_runs` — the validator's noisy-run simulation (`validator.simulate_runs`): bit flips from a seeded SplitMix64 stream plus per-run QUBO energy evaluation, parallelised with rayon. Deterministic per seed, but its RNG stream differs from the Python fallback's Mersenne Twister, so exact energies differ between backends (both are valid samples of the same noise model).
- `build_ecc_lookup_table` / `logical_failure_probability` — the 2^n X-error enumerations behind `ecc.decoder.LookupDecoder` and `ecc.certificate.certify_logical_qubit`, bitmask-based and identical to the Python reference down to equal-weight tie-breaking. This moves distance-5 lookup decoding (2^25 patterns) from intractable-in-Python to seconds.
- `vrp_qubo_terms` — the O(n^3) one-hot QUBO term construction behind `frontends.vrp.vrp_qubo`, returning the finished Python dict directly (one cached `PyString` per variable) because at ~300k terms the Python-object boundary, not the arithmetic, is the cost to beat.
- `run_statevector` and the `exact_ising_norm` / delta-correction functions, which predate this list and follow the same pattern.

What deliberately stays Python: hardware SDK adapters (`limen.backends.*`, `limen.gates.qiskit_exec` — thin glue over Qiskit/D-Wave/Braket APIs, no compute of their own), the gRPC distributed layer (`limen.distributed`), pipeline orchestration, and the IR dataclasses. Those are I/O-bound or API-bound; porting them would add build complexity without measurable speedup.

---

## Unified Quantum Channel

All quantum communication code lives in `limen.communication.channel` — the single canonical source. The old `limen.quantum_channel.*` module tree still exists as a set of thin re-export shims so existing code does not break, but contains no logic of its own. Callers should import from `limen.communication` going forward.

The unified module provides:
- `ChannelDeltaModel` — coherence time and latency model for a physical channel.
- `QuantumChannel` — high-level protocol runner for Teleportation and BB84 QKD.
- `teleport_circuit` / `run_teleport_qpu` — Bell-measurement circuit builder and QPU submission.
- `bb84_circuit` / `sift_and_evaluate` — BB84 basis preparation and sifting.
- `estimate_fidelity` — cross-layout fidelity analysis helper.

---

## Invariants

These are the architectural invariants that must never be violated:

0. **Determinism.** The same input always produces the same encoding. No randomness in the compilation path.
1. **IR isolation.** The `LogicalGraph` IR knows nothing about hardware topology. The compiler knows everything. The boundary between them is hard.
2. **Honest bounds.** Confidence bounds are always reported, never suppressed or rounded up. If a confidence value cannot be computed, it is `None`, not `1.0`.
3. **Optional SDKs.** All hardware SDK dependencies (`dwave-ocean-sdk`, `qiskit`, `pyqubo`) are optional at import time. A bare `import limen` with none of these installed must succeed.
4. **Optional Rust.** The `limen_core` Rust extension is optional at import time. All functionality is available without it via pure-Python fallbacks; the extension only changes speed (and, for seeded simulation, the RNG stream), never behavior.
5. **License hygiene.** Apache 2.0 throughout. No copyleft dependencies. The patent grant clause is intentional.
6. **Optional distributed deps.** `grpcio`/`grpcio-tools`/`protobuf` are only required by `limen.distributed` (the `distributed` extra). A bare `import limen` with none of these installed must succeed.
