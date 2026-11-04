# Changelog

All notable changes to `limen-compiler`. Versions follow semantic
versioning; before 1.0, minor versions may change behaviour.

## 0.9.0 (unreleased)

### Added
- Licensing module (`limen.licensing`): offline, ML-DSA-65 signed license
  keys with Free, Pro and Team tiers. See [docs/LICENSING.md](docs/LICENSING.md).
- Local GPU statevector tier 0 in the budget router (`aer_gpu`), opt-in via
  `LIMEN_LOCAL_GPU=1`, VRAM-sized at call time.
- Router memory ledger writes by default on `run_route_request`
  (`MEMORY_AUTO`).
- Surface-code logical-error certificate applied to every hardware tier and
  certificate shape.
- P1 Problem schema and TSP encoder (eil51 prefix).

### Changed
- `signed_certificates` (signed ledger entries) now needs a Pro or Team
  license. Unsigned certificates stay free.
- `peer_mesh` (multi-node peer registration) now needs a Team license.
- `limend` cert summary reports `success_probability`, `fidelity_target` and
  `target_met`; `fidelity_estimate` is a deprecated alias for one release.
- Raised `grpcio` and `protobuf` floors to match generated code.
- Rotated the LIMEN issuer public key.

### Fixed
- Seven tests never ran on any machine: three Rust-path test files guarded
  on `importorskip("limen_core")` instead of `limen.limen_core`, and two
  qiskit tests in `test_quantum_channel.py` were hardcoded `skipif(True)`.
  All now run and pass. Full suite with every extra: 570 passed, 0 skipped.
- D-Wave test skip scoped to its class instead of the whole file.
- Transpile cache LRU order stable on coarse clocks.
- Nested `pytest.approx` in the identity delta model test.

## 0.8.4 (2026-07-22)

- Guarded the optional `pqc` import.
- README and package name corrected to `limen-compiler`.
- Added the SQLite router memory ledger (`limen/router/memory.py`).

## 0.8.3 (2026-07-12)

- First PyPI release via Trusted Publishing, multi-platform wheels.
