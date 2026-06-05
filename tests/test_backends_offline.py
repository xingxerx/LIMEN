"""Offline tests for backend adapters — run without any SDK installed."""

import sys
import unittest.mock

import pytest

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.backends.dwave import run_dwave
from limen.backends.qiskit_backend import _qubo_to_ising, run_qiskit

_TRIVIAL_QUBO = {("q0", "q0"): -1.0, ("q1", "q1"): -1.0, ("q0", "q1"): 2.0}


def _make_encoding(qubo: dict):
    return compile_lexicographic(from_qubo_dict(qubo), default_hardware_graph(8))


# ── D-Wave ImportError guard ──────────────────────────────────────────────

def test_dwave_import_error_with_helpful_message():
    """run_dwave raises ImportError with pip install hint when SDK is absent."""
    encoding = _make_encoding(_TRIVIAL_QUBO)
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith(("dwave", "dimod", "neal"))}
    try:
        with unittest.mock.patch.dict("sys.modules", {"dimod": None, "dwave": None, "neal": None}):
            with pytest.raises(ImportError, match="pip install"):
                run_dwave(encoding, num_reads=10)
    finally:
        sys.modules.update(saved)


# ── Qiskit ImportError guard ──────────────────────────────────────────────

def test_qiskit_import_error_with_helpful_message():
    """run_qiskit raises ImportError with pip install hint when SDK is absent."""
    encoding = _make_encoding(_TRIVIAL_QUBO)
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("qiskit")}
    try:
        with unittest.mock.patch.dict("sys.modules", {"qiskit": None}):
            with pytest.raises(ImportError, match="pip install"):
                run_qiskit(encoding, num_shots=10, algorithm="exact")
    finally:
        sys.modules.update(saved)


# ── _qubo_to_ising correctness (no Qiskit required) ─────────────────────

def test_qubo_to_ising_energy_equivalence():
    """QUBO and Ising energies must differ by the same constant at every assignment."""
    from itertools import product

    qubo = _TRIVIAL_QUBO
    h, J = _qubo_to_ising(qubo)
    variables = sorted({v for pair in qubo for v in pair})

    offsets = []
    for bits in product((0, 1), repeat=len(variables)):
        assignment = dict(zip(variables, bits))
        qubo_e = sum(w * assignment[i] * assignment[j] for (i, j), w in qubo.items())
        spin = {v: 2 * assignment[v] - 1 for v in variables}
        ising_e = (
            sum(h[v] * spin[v] for v in variables)
            + sum(w * spin[i] * spin[j] for (i, j), w in J.items())
        )
        offsets.append(qubo_e - ising_e)

    assert all(abs(o - offsets[0]) < 1e-10 for o in offsets), \
        f"Offset not constant across assignments: {offsets}"
