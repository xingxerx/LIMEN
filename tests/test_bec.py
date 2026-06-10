# Copyright 2026 LIMEN Contributors. Apache 2.0.
"""Tests for the BEC superexchange backend."""

import pytest

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.analog.backends.bec import BECResult, run_bec
from limen.analog.backends.classical_sim import IsingSimulationResult
from limen.analog.hamiltonian import SubstrateType, from_physical_encoding

TRIVIAL_QUBO = {
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x0", "x1"):  2.0,
}


def _make_ir():
    graph = from_qubo_dict(TRIVIAL_QUBO)
    encoding = compile_lexicographic(graph, default_hardware_graph(4))
    return from_physical_encoding(encoding, substrate=SubstrateType.BEC)


def test_run_bec_returns_result():
    result = run_bec(_make_ir())
    assert isinstance(result, BECResult)
    assert result.available is True
    assert result.simulated is True


def test_run_bec_parameter_shapes():
    ir = _make_ir()
    result = run_bec(ir)
    assert len(result.potential_offsets) == ir.n_sites
    assert set(result.tunneling_amplitudes) == set(result.coupling_signs)
    assert result.on_site_interaction > 0.0


def test_run_bec_superexchange_inverse_relation():
    ir = _make_ir()
    result = run_bec(ir)
    # For every pair: 4 t^2 / U must reproduce |J_target|.
    target_J = {}
    for term in ir.terms:
        if len(term.operators) == 2:
            (si, _), (sj, _) = term.operators
            key = (min(si, sj), max(si, sj))
            target_J[key] = target_J.get(key, 0.0) + term.coefficient
    for key, t in result.tunneling_amplitudes.items():
        reconstructed = 4.0 * t * t / result.on_site_interaction
        assert reconstructed == pytest.approx(abs(target_J[key]))
        assert result.coupling_signs[key] == (1 if target_J[key] >= 0 else -1)


def test_run_bec_includes_simulation():
    result = run_bec(_make_ir())
    assert result.simulation is not None
    assert isinstance(result.simulation, IsingSimulationResult)


def test_run_bec_certificate_is_exact():
    result = run_bec(_make_ir())
    cert = result.certificate
    assert cert is not None
    assert cert.l1_bound == pytest.approx(0.0)
    assert cert.operator_norm == pytest.approx(0.0)
    assert cert.natively_realizable is True
