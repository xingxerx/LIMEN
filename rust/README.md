# LIMEN Rust IR workspace

Separate Cargo workspace from the root `limen_core` PyO3 crate; the maturin build is untouched.
Python LIMEN stays the product. These crates are the engine it will call through a PyO3 shim.

| Crate | Unlock |
|---|---|
| `limen-ir` | One versioned, canonical circuit format shared by LIMEN, AQRK and Metality; hash = cache and certificate key |
| `limen-qasm` | Lossless OpenQASM 3 exchange, so Qiskit, Cirq and Braket can read and write LIMEN circuits without a custom syntax |
| `limen-pass` | Transforms that cannot silently break a circuit: level and invariant checks after every pass |
| `limen-backend` | `Backend` trait so hardware (AQRK drivers) and Qiskit (PyO3 adapter) plug in behind one interface |

## Rules
- IR is additive only; bump `SCHEMA_VER` only when an old reader would misread a file.
- No backend details in the IR. Device facts live in `Target`.
- Three levels, lowered one step at a time: Logical -> Physical -> Pulse.
- `#![forbid(unsafe_code)]` workspace-wide; drivers will get their own crate with documented exceptions.
- `serde_json` runs with `float_roundtrip` so canonical hashes are stable (default parser can drift one ULP).
- A crate without an unlock line in this table gets deleted.

## Run
```powershell
cargo test --manifest-path D:\LIMEN\rust\Cargo.toml
```

## Next
1. `limen-py`: maturin/PyO3 module exposing `to_qiskit` / `from_qiskit` via OpenQASM 3.
2. Equivalence checker (statevector compare, <=12 qubits) as a pass test harness.
3. Sign IR hashes with the existing ML-DSA-65 path.
4. `AqrkBackend` behind `Backend` once hardware or a simulator target exists.
