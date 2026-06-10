# limen/analog — Layout Reference

## Correct structure

    limen/analog/
      __init__.py              — package init, imports from backends/
      hamiltonian.py           — HamiltonianIR, HamiltonianTerm, SubstrateType
      delta_model.py           — HardwareDeltaModel, DeltaModelRegistry
      backends/
        __init__.py            — exports all backend symbols
        classical_sim.py       — exact Ising diagonalisation (≤20 sites)
        neutral_atom.py        — Rydberg heuristic compiler
        photonic.py            — GBS adjacency encoder

## Files that must NOT exist

The following files are WRONG and must never be created:
  limen/analog/base.py
  limen/analog/neutral_atom.py     ← only valid at limen/analog/backends/
  limen/analog/photonic.py         ← only valid at limen/analog/backends/

Creating these files at limen/analog/ (not limen/analog/backends/) violates
Invariant 4 (optional Rust) and breaks test_analog.py (collapses 18 tests
to 2). All analog backend code lives exclusively under limen/analog/backends/.

## test_analog.py imports

Correct:
  from limen.analog.backends.neutral_atom import ...
  from limen.analog.backends.photonic import ...
  from limen.analog.backends.classical_sim import ...

Wrong (will break tests):
  from limen.analog.neutral_atom import ...
  from limen.analog.photonic import ...
  from limen.analog.base import ...
