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

## Not Planned

The following are explicitly out of scope for LIMEN's compilation-stack role:

- **Full fault-tolerant lattice-surgery compilation** of QAOA circuits — the ECC module certifies the logical error rate of protecting *the solution qubits*; it does not fault-tolerantly compile *the QAOA ansatz itself* (a full FTQC compiler is a separate research programme).
- **Variational parameter optimisation beyond grid search** — COBYLA and gradient-based optimisers exist in `examples/cobyla_multi_backend.py` as demonstration scripts; they are not part of the deterministic compilation pipeline by design.
- **Learned routing models** — the router's cost model is currently a pure function of run history averages. A learned model (e.g. GP surrogate for queue time) is documented as a planned successor in `limen/router/budget_router.py` but is not scheduled.
