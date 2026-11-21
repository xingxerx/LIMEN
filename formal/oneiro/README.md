Oneiro safety proofs and QFLTL guard

- Lean 4 models live under `formal/oneiro/Oneiro/` with a minimal `lake` build:
  - `Awareness.lean` — Awareness Timer sanitization and the start gate (never starts while a session is active)
  - `WBTB.lean` — QFLTL WBTB sanitization and a minimal lifecycle guard (wbtb_* only inside an active session)
- Python QFLTL checker lives at `limen/qfltl/guard.py` with tests in `tests/test_qfltl_guard.py`.

Mapping to Oneiro:
- Awareness Timer: mirrors `AwarenessTimerConfig.sanitized()` and the UI gate in `AwarenessTimerView` that prevents starting while `workoutManager.running` is true.
- WBTB: mirrors `QfltlWbtbConfig.sanitized()` and the lifecycle notifications `wbtb_wake` → `wbtb_start` → `wbtb_resume` logged between `session_start` and `session_end`.

Build locally:

```bash
cd formal/oneiro
elan toolchain install
lake build
```

Run the QFLTL checker tests:

```bash
python -m pytest tests/test_qfltl_guard.py -q
```
