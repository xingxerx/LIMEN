# LIMEN v0.4.0 — Phase 3 completion: apply notes

Drop-in files (repo-relative paths, copy over D:\LIMEN):

0. `limen/__init__.py` — version → "0.4.0"; lazy exports for
   `CompilationCertificate` / `certify_ising`.
1. `limen/analog/certificate.py` — NEW. Theorem 1 implementation.
2. `limen/analog/backends/neutral_atom.py` — REPLACED. Adds
   `delta_model: HardwareDeltaModel | None = None` param, certificate,
   native-realizability classification. `NeutralAtomResult` gains a
   `certificate` field (default None) before `metadata`.
3. `limen/analog/backends/photonic.py` — REPLACED. **Bug fix:** spectral
   scaling is now 1.1 × Gershgorin row-sum bound (old entry-wise max|J|
   scaling produced spectral radius > 1 on dense couplings, e.g. ring-3 —
   an invalid GBS device). Adds certificate; `metadata["scale_rule"]` =
   "gershgorin_row_sum_x1.1". `metadata["scale"]` values change for any
   instance with >1 coupling per site.
4. `limen/analog/backends/bec.py` — NEW. SubstrateType.BEC backend
   (Duan-Demler-Lukin superexchange mapping, Mott-regime validity flag).
5. `limen/analog/backends/__init__.py` — REPLACED. Exports BECResult/run_bec.
6. `limen/docs/universality_theorem.md` — NEW. Theorems 1-5.
7. `tests/test_certificate.py`, `tests/test_bec.py`,
   `tests/test_phase3_completion.py` — NEW (17 tests).

Manual edits in your repo (small, do by hand):

0. `pyproject.toml`: version = "0.4.0"
1. `README.md` roadmap, Phase 3 line → checked:
   "[x] Constructive universality theorem — restricted quadratic class with
   compilation certificates (limen/docs/universality_theorem.md); general
   analog universality remains open research."
   Status line: v0.4.0; test count: your current suite (54 at v0.3.0) + 17
   new = 71 expected (Rust-parity test skips until `maturin develop`).
2. `limen/docs/architecture.md`: replace the "pending research" sentence in
   the Phase 3 universality gap section with:
   "Resolved for the restricted quadratic class — see
   limen/docs/universality_theorem.md. General analog universality remains
   open."

Verify on Windows (PowerShell, repo root):

    python -m pytest tests/ -q

Expected: all pass; `test_parity_with_rust_extension` skips unless the Rust
core is built (`maturin develop`). Nothing in this change touches Rust.

Honest scoping (also stated inside the theorem doc): the theorem is an
engineering-grade constructive result for LIMEN's diagonal quadratic class —
exact certificates, per-substrate realizability conditions, and the
LHZ parity route (cited, not yet implemented as a compiler pass). It is not
peer-reviewed novel mathematics and does not claim universality for arbitrary
analog Hamiltonians. The LHZ compiler pass is the natural v0.5 item.
