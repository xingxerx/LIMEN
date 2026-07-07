# LIMEN Roadmap

> *The following describes work planned beyond v0.8.3. All currently implemented phases are documented in `docs/architecture.md` and `docs/DIRECTORY_MAP.md`. This document covers the integration gaps and extensions needed to realise the closed-loop quantum systems deployment framework described in the six-phase vision.*

---

## Current State (v0.8.3)

LIMEN's six phases are individually complete:

| Phase | Module(s) | Status |
|-------|-----------|--------|
| 1 — Lexicographic Compiler & IR | `limen.core` | ✅ |
| 2 — Stackelberg Co-Design | `limen.codesign` | ✅ (standalone) |
| 3 — Multi-Backend Execution | `limen.backends`, `limen.analog` | ✅ |
| 4 — Distributed Compilation | `limen.distributed` | ✅ (requires `server_addresses=`) |
| 5 — Gate-Model Pipeline + ECC | `limen.gates`, `limen.ecc`, `limen.pipeline` | ✅ |
| 6 — Crash-Resilient Job Lifecycle | `limen.router.job_state` | ✅ |

What is not yet wired is the **feedback paths** that let each phase inform the next at runtime — the loops that turn six independent tools into a single autonomous framework.

---

## Planned: Co-Design Feedback Loop (Phase 2 × Phase 6)

### What it does

After `run_route_request()` completes and writes a cert to `results/`, the execution metadata from that cert (measured logical error rate, chain-break fraction) should automatically seed a `run_codesign()` call that updates the penalty coefficients and chain strength for the *next* run on the same backend.

### Gap

`run_codesign()` is today a standalone function. The router reads history into fleet profiles (`informed_fleet()`) but there is no path from a completed cert back into the Stackelberg solver. The co-design loop only runs if the user explicitly calls it via `examples/codesign_demo.py` or `examples/dwave_codesign_qpu.py`.

### Planned implementation

1. **`codesign_from_history(results_dir, encoding) → CoDesignResult | None`** — a new function in `limen.codesign` that scans `results/` for the most recent cert on the target backend, extracts the chain-break fraction (D-Wave path) or measured logical error (IBM path), and runs `run_codesign()` with that as the initial `chain_break_fraction_fn` prior.

2. **`run_route_request(..., enable_codesign: bool = False)`** — when `True`, calls `codesign_from_history()` after plan selection and feeds the result's `encoding` into the `run_pipeline()` call, replacing the default lexicographic embedding.

3. **Persistence** — `CoDesignResult` gets a `to_dict()` / `from_dict()` pair so the converged `encoding` and κ history can be written alongside the cert and loaded on the next call without re-running the full loop.

### Why it matters

Without this loop, penalty coefficients are static defaults regardless of how many QPU runs have accumulated in `results/`. With it, the compiler automatically routes around noisy qubits and drift after each completed job.

---

## Planned: Router Awareness of gRPC Peers (Phase 4 × Phase 6)

### What it does

`run_route_request()` should be able to discover available `limen.distributed` peer nodes from the environment and automatically include them in the compilation plan — so that a caller deploying a cluster of LIMEN nodes gets distributed compilation for free without hand-wiring `server_addresses=`.

### Gap

`run_route_request()` accepts `server_addresses` as of v0.8.3 and forwards them to `run_pipeline()`. However:
- There is no automatic peer discovery from `LIMEN_KNOWN_PEERS` at the `run_route_request()` level.
- The `RoutePlan` does not record the distributed split decision, so the plan is not self-describing for async re-execution.

### Planned implementation

1. **Auto-discovery from env** — when `server_addresses=None`, `run_route_request()` falls back to `NodeConfig.from_env().known_peers` if the distributed extra is installed. The caller gets distributed compilation just by setting `LIMEN_KNOWN_PEERS`.

2. **`RoutePlan.server_addresses`** — add an optional field to `RoutePlan` so `run_pipeline_from_plan()` can reconstruct the distributed execution without the caller re-supplying the peer list.

---

## Planned: Automatic Substrate Selection (Phase 3)

### What it does

The router currently selects backends by cost tier and qubit count (`budget_router.route()`). A future extension would score each backend against the problem's structure and prefer the native substrate:

- Dense, unfrustrated Ising → D-Wave annealer
- Planar, unit-disk interaction graph → Rydberg neutral-atom array (QuEra Aquila)
- Gate circuit within qubit budget → IBM / IonQ gate-model
- Problem exceeds any single device → circuit cutting or distributed annealing

### Gap

`BackendProfile` has no substrate-affinity field, and `route()` has no problem-structure scoring. The user must select a backend explicitly via `RouteRequest` or accept the cost-tier default.

### Planned implementation

1. **`ProblemProfile`** — a lightweight struct computed from the QUBO before routing: variable count, edge density, frustration index, max coupling magnitude.

2. **`BackendProfile.substrate_affinity: dict[str, float]`** — a scored mapping from problem-profile features to expected performance on this backend (e.g. `{"low_frustration": 0.9, "planar": 0.7}`).

3. **`route()` substrate scoring pass** — after cost-tier filtering, rank remaining candidates by affinity dot-product with the problem profile.

---

## Planned: Cut-Circuit → `EndToEndCertificate` Bridge (Phase 4 × Phase 5)

### What it does

When `RoutePlan.use_cutting = True`, the problem is too large for any single backend and must be fragmented via `limen.cutting`. Currently `run_pipeline_from_plan()` raises `NotImplementedError` in this case because `CutDispatchResult` (reconstructed expectation value) and `EndToEndCertificate` (sampled solution bitstring + ECC cert) have incompatible output shapes.

### Planned implementation

1. **`CuttingCertificate`** — a new datatype returned by cut-circuit runs, containing: reconstructed expectation value, per-partition job IDs, partition plan, and an optional ECC budget note.

2. **`run_cut_route_request()`** — a new orchestrator for the cutting path that takes a `RoutePlan` with `use_cutting=True`, calls `find_cuts_and_partition` + `run_cut_circuit`, reconstructs the expectation value, and wraps it in a `CuttingCertificate`.

3. **`run_route_request()` dispatch** — when `plan.use_cutting`, forward to `run_cut_route_request()` instead of raising.

---

## Not Planned

The following are explicitly out of scope for LIMEN's compilation-stack role:

- **Full fault-tolerant lattice-surgery compilation** of QAOA circuits — the ECC module certifies the logical error rate of protecting *the solution qubits*; it does not fault-tolerantly compile *the QAOA ansatz itself* (a full FTQC compiler is a separate research programme).
- **Variational parameter optimisation beyond grid search** — COBYLA and gradient-based optimisers exist in `examples/cobyla_multi_backend.py` as demonstration scripts; they are not part of the deterministic compilation pipeline by design.
- **Learned routing models** — the router's cost model is currently a pure function of run history averages. A learned model (e.g. GP surrogate for queue time) is documented as a planned successor in `limen/router/budget_router.py` but is not scheduled.
