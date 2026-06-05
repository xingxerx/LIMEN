# Contributing to LIMEN

LIMEN is early-stage infrastructure. The most valuable contributions right now fall into three categories — engineering (new backends, validator improvements, problem formulations), research (the constructive universality theorem for analog substrates), and documentation.

## Getting Started

You will need Python 3.10+ and a Rust toolchain.

Install Rust via rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then set up the project:

```bash
pip install maturin
git clone https://github.com/xingxerx/LIMEN.git
cd LIMEN
maturin develop          # builds the Rust extension (limen_core)
pip install -e .         # installs Python package in editable mode
```

Verify the setup:

```bash
python -c "import limen; from limen_core import StackelbergSolver; print(limen.__version__)"
```

**Windows:** the Rust extension builds cleanly under WSL (Ubuntu). PowerShell 7 with the Rust toolchain installed natively also works — run `maturin develop` inside the repo root.

Optional SDK installs for hardware backend tests:

```bash
pip install limen[dwave]    # D-Wave Ocean SDK
pip install limen[ibm]      # Qiskit + Qiskit Aer
pip install limen[pyqubo]   # PyQUBO frontend
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected: 25 tests collected, with co-design tests skipping if `limen_core` is not built and SDK tests skipping if the relevant SDK is not installed.

Test files and what they cover:

- `tests/test_core.py` — core IR, compiler, validator pipeline
- `tests/test_backends_offline.py` — backend import guards, Ising energy math
- `tests/test_codesign.py` — Stackelberg loop, κ stability (requires `limen_core`)
- `tests/test_analog.py` — Hamiltonian IR, analog backend stubs
- `tests/test_backend_dwave.py` — D-Wave adapter (requires Ocean SDK)
- `tests/test_backend_qiskit.py` — Qiskit adapter (requires Qiskit)

## Adding a New Backend Adapter

A backend adapter converts a `PhysicalEncoding` into hardware-specific instructions and returns a result dataclass. The contract is:

0. Create `limen/backends/<name>.py`
1. Guard all SDK imports inside `try/except ImportError` with a clear install message directed at the user
2. Define a `<Name>Result` dataclass with at minimum: `samples`, `energies`, `best_assignment`, `best_energy`
3. Define `run_<name>(encoding: PhysicalEncoding, ...) -> <Name>Result`
4. Add lazy exports to `limen/backends/__init__.py`
5. Add tests to `tests/test_backend_<name>.py` with `pytest.importorskip` at module level

Reference implementations: `limen/backends/dwave.py` and `limen/backends/qiskit_backend.py`.

## Adding a New Frontend Adapter

A frontend adapter converts a domain-specific problem representation into a `LogicalGraph`. The contract is:

0. Create `limen/frontends/<name>.py`
1. Import any optional dependencies lazily with a clear `ImportError` message
2. Return a validated `LogicalGraph` from `limen.core.ir`
3. Add tests

Reference implementation: `limen/frontends/pyqubo.py`.

## The Phase 3 Research Problem

The analog substrate layer (`limen/analog/`) is waiting for one thing: a constructive universality theorem.

The current `from_physical_encoding()` function produces a correct Z-basis Ising Hamiltonian from any QUBO. What does not yet exist is a proof — and corresponding algorithm — that maps this Hamiltonian onto:

- **Neutral-atom arrays:** Rydberg blockade Hamiltonians with spatial layout constraints (blockade radius, connectivity geometry, available gate operations)
- **Photonic processors:** continuous-variable Hamiltonians (Gaussian boson sampling, Kerr interactions, homodyne measurement readout)
- **BEC simulators:** Bose-Einstein condensate interaction Hamiltonians with tunable scattering lengths

A contribution here is not code — it is a proof that such a mapping exists and is constructive, plus an implementation of the mapping algorithm as a backend adapter. The interface contract is already defined in `limen/analog/hamiltonian.py`. A valid research contribution would implement `run_neutral_atom()` or `run_photonic()` by replacing the `NotImplementedError` with a real compilation algorithm.

For the precise mathematical specification of what needs to be proved, see `limen/docs/architecture.md`.

## Code Style

Python code uses dataclasses, type hints, and docstrings on every public symbol. Every new file gets an Apache 2.0 header. Rust code uses standard `rustfmt` formatting with doc comments on every public item. No print statements in library code. Suggestion lists start from 0.

## Pull Request Conventions

One PR per logical change. PR titles follow the pattern `<scope>: <description>` — for example, `backends: add IonQ adapter`. All tests must pass (`python -m pytest tests/ -v`). New features need new tests. Run `maturin develop` and confirm it succeeds before pushing any Rust changes.

## License

All contributions are Apache 2.0. The patent grant clause is intentional — quantum computing is a patent-dense field.
