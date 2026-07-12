# LIMEN Roadmap

> *All currently implemented phases and feedback loops are documented in `docs/architecture.md` and `docs/DIRECTORY_MAP.md`. This document tracks what's shipped vs. still open beyond the six-phase vision.*

---

## Current State (v0.8.3+, unreleased)

LIMEN's six phases are individually complete, and as of commit `23e1eaa` the feedback paths that connect them at runtime are wired in too:

| Phase | Module(s) | Status |
|-------|-----------|--------|
| 1 — Lexicographic Compiler & IR | `limen.core` | ✅ |
| 2 — Stackelberg Co-Design | `limen.codesign` | ✅ (standalone; see "Shipped: Co-Design History Loop" below) |
| 3 — Multi-Backend Execution | `limen.backends`, `limen.analog` | ✅ |
| 4 — Distributed Compilation | `limen.distributed` | ✅ (auto-discovers peers; `server_addresses=` still overrides) |
| 5 — Gate-Model Pipeline + ECC | `limen.gates`, `limen.ecc`, `limen.pipeline` | ✅ |
| 6 — Crash-Resilient Job Lifecycle | `limen.router.job_state` | ✅ |

`run_route_request()` — QUBO + budget in, certified answer out — is the zero-manual-steps entry point tying all six together. It now handles every `RoutePlan` shape, including oversized problems that require circuit cutting.

| Phase | Module(s) | Status |
|-------|-----------|--------|
| 7 — QUBO Auto-Formulation (structured-input tier) | `limen.formulation` | ✅ (Steps 1-2, 4 of the 5-step plan below) |

---

## Shipped: QUBO Auto-Formulation, Structured-Input Tier (Phase 7, Step 1-2 + 4)

`limen.formulation.ConstraintCompiler` closes the "real gap" this roadmap previously only scoped: hand-deriving penalty terms from constraints. A caller declares typed constraints (`Equality`, `Inequality`, `OneHot`, `AtMostK`, `AtLeastK`, `AllDifferent`) over binary variable names, queues them alongside an objective QUBO, and gets back a validated `LogicalGraph` — the same IR `limen.frontends.pyqubo.from_qubo_dict` produces, so it drops straight into the existing compilation pipeline.

- **Step 1 (input contract):** structured typed dataclasses, not free-text — `limen/formulation/constraints.py`. Natural-language input is explicitly deferred (see "Research-Track" section note below on why NL sits *on top of*, not *inside*, the certified pipeline).
- **Step 2 (constraint compiler):** `limen/formulation/compiler.py`. Two primitives — squared-equality penalty expansion and binary-slack-encoded inequality reduction — implement every constraint type. `AtMostK(k=1)`/`AtLeastK(k=0)` special-case to zero-auxiliary-variable penalties (exact pairwise/empty forms) rather than paying for a slack encoding they don't need.
- **Step 3 (penalty-weight selection):** `limen/formulation/penalty.py`. This turned out to be a genuinely separate problem from the Stackelberg co-design loop, as suspected — co-design tunes chain-strength/embedding *after* a QUBO exists, against a fixed hardware graph; `default_penalty_weight` picks the penalty *before* the QUBO exists, from the objective's own coefficients, with no hardware dependency. The two don't compose into one search.
- **Step 4 (validation loop):** `tests/test_formulation.py`, following the same discover-don't-confirm philosophy as `tests/test_physics_validation.py` — every test brute-forces the compiled QUBO and checks the ground state actually satisfies the original constraint, across randomized objectives, rather than asserting one baked-in answer.
- **Step 5 (NL layer on top):** not started — deliberately deferred per the original scoping, since it changes LIMEN's trust surface (an LLM's interpretation would sit inside vs. outside the certified pipeline). A future NL layer should translate text into the Step 1 typed constraints, not bypass them.

---

## Shipped: Cut-Circuit → Certificate Bridge (Phase 4 × Phase 5)

When `RoutePlan.use_cutting = True`, the problem is too large for any single backend and is fragmented via `limen.cutting`. `run_pipeline_from_plan()` and `run_route_request()` dispatch this case to `run_cut_route_request()` (`limen/pipeline.py`), which:

1. Decomposes the QUBO into Ising terms and compiles the same QAOA circuit `run_pipeline` would use.
2. Reconstructs every qubit's `<Z_i>` marginal via `limen.cutting` (one cut + dispatch + reconstruct round trip per qubit).
3. Decodes a solution bitstring from those marginals by threshold rounding and computes its exact classical energy.
4. Certifies the result with the same surface-code ECC term `run_pipeline` uses (`limen.ecc.certificate.certify_logical_qubit`, reused unmodified).

The return type is `CuttingCertificate` (`limen.cutting.certificate`), not `EndToEndCertificate` — deliberately a different shape, since circuit cutting reconstructs Pauli-observable expectation values rather than a sampled solution. `is_optimal` is always `None` and `reconstructed_expected_energy` is an explicitly documented mean-field approximation.

---

## Shipped: Router Awareness of gRPC Peers (Phase 4 × Phase 6)

`run_route_request()` auto-discovers `limen.distributed` peers from the environment: when `server_addresses=None`, it falls back to `NodeConfig.from_env().known_peers` (the `LIMEN_KNOWN_PEERS` env var) if `LIMEN_NODE_ID` is set, so a deployed LIMEN cluster gets distributed compilation without every caller hand-wiring the peer list. `RoutePlan.server_addresses` records the decision so a plan is self-describing for async re-execution. Explicitly passing `server_addresses=` still overrides auto-discovery.

---

## Shipped: Automatic Substrate Selection (Phase 3)

`limen.router.problem_profile.ProblemProfile` computes structural signals (edge density, a heuristic `frustration_index`) once per QUBO. `BackendProfile.substrate_affinity: dict[str, float]` scores each backend's fit against those signals. `_select_backend()` in `budget_router.py` uses this strictly as a **tiebreaker**, applied only after all existing cost/capacity/validation filtering — regression tests confirm it never overrides those criteria, only decides between backends already tied on them.

---

## Shipped: Co-Design History Loop (Phase 2 × Phase 6)

`limen.codesign.solver.codesign_from_history(results_dir, encoding)` scans `results/` for the most recent cert on a target backend and seeds a fresh `run_codesign()` call from the best prior chain-strength. `CoDesignResult` gained `to_dict()`/`from_dict()` so state round-trips through `results/`.

This is **intentionally standalone** rather than wired into `run_route_request()`'s D-Wave dispatch: that path compiles against a complete hardware graph, where `chain_strength` is provably inert. Wiring it into `run_route_request()` would have been a no-op dressed up as a feature. Direct callers with a real sparse hardware graph (e.g. `examples/dwave_codesign_qpu.py`) get the benefit; the fully-connected router path does not need it.

---

## Blocked (Not an Engineering Task Right Now)

- **Atom Computing hardware access** (`limen/backends/azure_atom.py`) — dormant, blocker re-verified 2026-07-08. Atom Computing is absent from both Azure Quantum's live provider list and AWS Braket's device list (checked directly against current docs, not carried over from a stale earlier finding). The AWS Braket fallback previously queued up for this is **moot** — Atom Computing isn't a Braket provider either, so a `braket_atom.py` twin would be dead code against a nonexistent target. The only remaining path is Atom Computing's own direct enterprise access program (an application/approval step, not code). Revisit only when that access exists or Atom Computing lands on a public marketplace; don't re-litigate the Braket option again without checking whether the provider list changed.

## Research-Track (Not on This Roadmap)

- **General analog universality** (arbitrary non-diagonal, time-dependent Hamiltonians) — open research, not an engineering backlog item. Scoped and tracked separately in `limen/docs/universality_theorem.md`, which proves restricted-class results (Theorems 1–5, diagonal quadratic/Ising forms only) and explicitly flags the general case as open. If this is ever worked further, the actionable slice is a narrow Theorem 6 for one restricted extension (e.g. time-dependent-but-diagonal, or a bounded non-diagonal perturbation class), scoped and proved the way Theorems 1–5 were — a math-first task, kept off this engineering roadmap on purpose so it doesn't rot alongside phase-tracked work.

## Not Planned

The following are explicitly out of scope for LIMEN's compilation-stack role:

- **Full fault-tolerant lattice-surgery compilation** of QAOA circuits — the ECC module certifies the logical error rate of protecting *the solution qubits*; it does not fault-tolerantly compile *the QAOA ansatz itself* (a full FTQC compiler is a separate research programme).
- **Variational parameter optimisation beyond grid search** — COBYLA and gradient-based optimisers exist in `examples/cobyla_multi_backend.py` as demonstration scripts; they are not part of the deterministic compilation pipeline by design.
- **Learned routing models** — the router's cost model is currently a pure function of run history averages. A learned model (e.g. GP surrogate for queue time) is documented as a planned successor in `limen/router/budget_router.py` but is not scheduled.
