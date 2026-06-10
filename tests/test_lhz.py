# Copyright 2026 LIMEN Contributors. Apache 2.0.
"""Tests for the LHZ parity compiler pass (Theorem 3)."""

import pytest

from limen.analog.hamiltonian import HamiltonianIR, HamiltonianTerm, SubstrateType
from limen.analog.lhz import has_negative_couplings, lhz_parity_pass


def _zt(site: int, coeff: float) -> HamiltonianTerm:
    return HamiltonianTerm(coefficient=coeff, operators=[(site, "Z")])


def _zzt(i: int, j: int, coeff: float) -> HamiltonianTerm:
    return HamiltonianTerm(coefficient=coeff, operators=[(i, "Z"), (j, "Z")])


def _make_ir(*terms: HamiltonianTerm, n: int = 0) -> HamiltonianIR:
    return HamiltonianIR(terms=list(terms), n_sites=n, substrate=SubstrateType.NEUTRAL_ATOM)


# ---------------------------------------------------------------------------
# has_negative_couplings
# ---------------------------------------------------------------------------

def test_has_negative_couplings():
    assert has_negative_couplings(_make_ir(_zzt(0, 1, -1.0))) is True
    assert has_negative_couplings(_make_ir(_zzt(0, 1, 1.0))) is False
    assert has_negative_couplings(_make_ir(_zt(0, -2.0))) is False  # linear, not ZZ


# ---------------------------------------------------------------------------
# Qubit count and qubit map
# ---------------------------------------------------------------------------

def test_qubit_count():
    assert lhz_parity_pass(_make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, -1.0), _zzt(1, 2, 0.5), n=3)).n_physical == 3
    assert lhz_parity_pass(_make_ir(
        _zzt(0, 1, 1.0), _zzt(0, 2, -1.0), _zzt(0, 3, 0.5),
        _zzt(1, 2, -0.5), _zzt(1, 3, 0.3), _zzt(2, 3, -0.2), n=4,
    )).n_physical == 6


def test_qubit_map_spans_all_pairs():
    result = lhz_parity_pass(_make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, -1.0), _zzt(1, 2, 0.5), n=3))
    assert set(result.qubit_map.keys()) == {(0, 1), (0, 2), (1, 2)}
    assert sorted(result.qubit_map.values()) == [0, 1, 2]


# ---------------------------------------------------------------------------
# Local fields from logical couplings and linear terms
# ---------------------------------------------------------------------------

def test_couplings_become_local_fields():
    ir = _make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, -2.0), _zzt(1, 2, 0.5), n=3)
    result = lhz_parity_pass(ir)
    fields = {
        tuple(t.metadata["logical_pair"]): t.coefficient
        for t in result.encoded_ir.terms
        if t.metadata.get("source") == "lhz_coupling"
    }
    assert fields[(0, 1)] == pytest.approx(1.0)
    assert fields[(0, 2)] == pytest.approx(-2.0)
    assert fields[(1, 2)] == pytest.approx(0.5)


def test_linear_term_mapped_to_parity_qubit():
    ir = _make_ir(_zt(1, 0.8), _zzt(0, 1, -1.0), n=2)
    result = lhz_parity_pass(ir)
    lin = [t for t in result.encoded_ir.terms if t.metadata.get("source") == "lhz_linear"]
    assert len(lin) == 1
    assert lin[0].coefficient == pytest.approx(0.8)
    assert lin[0].operators == [(result.qubit_map[(0, 1)], "Z")]


def test_h0_dropped_as_constant():
    ir = _make_ir(_zt(0, 3.0), _zzt(0, 1, -1.0), n=2)
    result = lhz_parity_pass(ir)
    assert result.h0_constant == pytest.approx(3.0)
    assert all(t.metadata.get("source") != "lhz_linear" for t in result.encoded_ir.terms)


# ---------------------------------------------------------------------------
# Plaquette constraints
# ---------------------------------------------------------------------------

def test_plaquette_count():
    # n=2 → 0; n=3 → 1; n=4 → 3
    assert len(lhz_parity_pass(_make_ir(_zzt(0, 1, -1.0), n=2)).plaquettes) == 0
    assert len(lhz_parity_pass(_make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, -1.0), _zzt(1, 2, 0.5), n=3)).plaquettes) == 1
    ir4 = _make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, -1.0), _zzt(0, 3, 0.5),
                   _zzt(1, 2, -0.5), _zzt(1, 3, 0.3), _zzt(2, 3, -0.2), n=4)
    assert len(lhz_parity_pass(ir4).plaquettes) == 3


def test_plaquette_term_is_zzz_with_correct_penalty():
    ir = _make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, -1.0), _zzt(1, 2, 0.5), n=3)
    result = lhz_parity_pass(ir)
    plaq_terms = [t for t in result.encoded_ir.terms if t.metadata.get("source") == "lhz_plaquette"]
    assert len(plaq_terms) == 1
    assert len(plaq_terms[0].operators) == 3
    assert all(op == "Z" for _, op in plaq_terms[0].operators)
    assert plaq_terms[0].coefficient == pytest.approx(-result.penalty_strength / 2.0)


def test_plaquette_qubit_indices_match_map():
    ir = _make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, -1.0), _zzt(1, 2, 0.5), n=3)
    result = lhz_parity_pass(ir)
    p01, p02, p12 = result.qubit_map[(0, 1)], result.qubit_map[(0, 2)], result.qubit_map[(1, 2)]
    assert result.plaquettes[0] == (p01, p02, p12)


# ---------------------------------------------------------------------------
# Penalty strength
# ---------------------------------------------------------------------------

def test_penalty_above_lanthaler_bound():
    # Lanthaler & Lechner: P > 2 * max|J|
    ir = _make_ir(_zzt(0, 1, 2.0), _zzt(0, 2, -2.0), _zzt(1, 2, 1.0), n=3)
    result = lhz_parity_pass(ir)
    assert result.penalty_strength > 2.0 * 2.0


def test_penalty_factor_le2_raises():
    with pytest.raises(ValueError, match="penalty_factor"):
        lhz_parity_pass(_make_ir(_zzt(0, 1, -1.0), n=2), penalty_factor=2.0)


# ---------------------------------------------------------------------------
# Edge cases and metadata
# ---------------------------------------------------------------------------

def test_empty_ir():
    result = lhz_parity_pass(HamiltonianIR(terms=[], n_sites=0))
    assert result.n_physical == 0 and result.n_logical == 0

def test_negative_couplings_in_metadata():
    ir = _make_ir(_zzt(0, 1, -2.0), _zzt(0, 2, 1.0), n=3)
    result = lhz_parity_pass(ir)
    assert [0, 1] in result.metadata["negative_couplings"]
    assert [0, 2] not in result.metadata["negative_couplings"]

def test_theorem_reference_in_metadata():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    assert "theorem-3" in lhz_parity_pass(ir).encoded_ir.metadata.get("theorem", "")

def test_to_dict_roundtrip():
    ir = _make_ir(_zzt(0, 1, -1.0), _zzt(0, 2, 0.5), _zzt(1, 2, -0.3), n=3)
    d = lhz_parity_pass(ir).to_dict()
    assert d["n_logical"] == 3 and d["n_physical"] == 3 and len(d["plaquettes"]) == 1
