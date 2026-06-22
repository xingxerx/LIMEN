"""SDK-dependent tests for the D-Wave backend adapter.

Skipped cleanly when dwave Ocean SDK is not installed.
See tests/test_backends_offline.py for ImportError guard tests.
"""

import pytest

pytest.importorskip("dwave", reason="dwave Ocean SDK not installed")

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.backends.dwave import DWaveResult, run_dwave

_TRIVIAL_QUBO = {("q0", "q0"): -1.0, ("q1", "q1"): -1.0, ("q0", "q1"): 2.0}
_MAXCUT_QUBO = {
    ("A", "B"): 1.0, ("A", "C"): 1.0, ("B", "C"): 1.0,
    ("B", "D"): 1.0,
    ("A", "A"): -2.0, ("B", "B"): -3.0, ("C", "C"): -2.0, ("D", "D"): -1.0,
}


def _make_encoding(qubo: dict):
    return compile_lexicographic(from_qubo_dict(qubo), default_hardware_graph(8))


def test_returns_correct_types():
    """run_dwave returns a DWaveResult with correctly-typed fields."""
    encoding = _make_encoding(_TRIVIAL_QUBO)
    result = run_dwave(encoding, num_reads=50, seed=0)

    assert isinstance(result, DWaveResult)
    assert len(result.samples) == 50
    assert len(result.energies) == 50
    assert isinstance(result.best_assignment, dict)
    assert isinstance(result.best_energy, float)
    assert isinstance(result.timing, dict)
    assert all(v in (0, 1) for v in result.best_assignment.values())


def test_deterministic_same_seed():
    """Same seed and encoding must produce identical best_assignment."""
    encoding = _make_encoding(_TRIVIAL_QUBO)
    r1 = run_dwave(encoding, num_reads=100, seed=7)
    r2 = run_dwave(encoding, num_reads=100, seed=7)
    assert r1.best_assignment == r2.best_assignment
    assert r1.best_energy == r2.best_energy


def test_maxcut_best_energy_non_positive():
    """Sampler must find a cut (energy <= 0) on a 4-node Max-Cut."""
    encoding = _make_encoding(_MAXCUT_QUBO)
    result = run_dwave(encoding, num_reads=200, seed=42)

    assert result.best_energy <= 0.0
    assert all(v in (0, 1) for v in result.best_assignment.values())


def test_simulator_embedding_is_none():
    """Simulator path must leave embedding=None (no Pegasus embedding needed)."""
    encoding = _make_encoding(_TRIVIAL_QUBO)
    result = run_dwave(encoding, num_reads=10, seed=0)
    assert result.embedding is None


def test_pegasus_hardware_graph_structure():
    """pegasus_hardware_graph returns a non-empty adjacency dict with symmetric edges."""
    pytest.importorskip("dwave_networkx", reason="dwave-networkx not installed")
    from limen.backends.dwave import pegasus_hardware_graph

    g = pegasus_hardware_graph(m=2)  # small m for speed
    assert len(g) > 0
    for node, neighbours in g.items():
        assert isinstance(node, str)
        for nb in neighbours:
            assert node in g[nb], f"edge {node}-{nb} not symmetric"


def test_find_pegasus_embedding_raises_on_bad_sampler():
    """_find_pegasus_embedding must raise RuntimeError when no embedding exists."""
    pytest.importorskip("minorminer", reason="minorminer not installed")
    from unittest.mock import MagicMock

    from dimod import BinaryQuadraticModel  # type: ignore[import]
    from limen.backends.dwave import _find_pegasus_embedding

    # A BQM with no quadratic edges on an empty edge list forces minorminer
    # to return an empty embedding, which must surface as RuntimeError.
    bqm = BinaryQuadraticModel({"x": -1.0}, {}, 0.0, "BINARY")
    fake_sampler = MagicMock()
    fake_sampler.edgelist = []

    with pytest.raises(RuntimeError, match="minorminer could not find"):
        _find_pegasus_embedding(bqm, fake_sampler)
