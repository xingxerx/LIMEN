# Copyright 2026 LIMEN Contributors. Apache 2.0.
"""Tests for the LHZ parity compiler pass (Theorem 3)."""

import math

import pytest

from limen.analog.hamiltonian import HamiltonianIR, HamiltonianTerm, SubstrateType
from limen.analog.lhz import LHZResult, has_negative_couplings, lhz_parity_pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ir(*terms: HamiltonianTerm, n_sites: int = 0) -> HamiltonianIR:
    return HamiltonianIR(terms=list(terms), n_sites=n_sites, substrate=SubstrateType.NEUTRAL_ATOM)


def _zt(site: int, coeff: float) -> HamiltonianTerm:
    return HamiltonianTerm(coefficient=coeff, operators=[(site, "Z")])


def _zzt(i: int, j: int, coeff: float) -> HamiltonianTerm:
    return HamiltonianTerm(coefficient=coeff, operators=[(i, "Z"), (j, "Z")])


# ---------------------------------------------------------------------------
# has_negative_couplings
# ---------------------------------------------------------------------------

def test_no_negative():
    ir = _make_ir(_zzt(0, 1, 1.0), _zzt(1, 2, 0.5))
    assert has_negative_couplings(ir) is False


def test_has_negative():
    ir = _make_ir(_zzt(0, 1, -1.0), _zzt(1, 2, 0.5))
    assert has_negative_couplings(ir) is True


def test_linear_only_not_negative():
    ir = _make_ir(_zt(0, -2.0), _zt(1, 1.0))
    assert has_negative_couplings(ir) is False


# ---------------------------------------------------------------------------
# Basic structure: 3-spin fully-connected problem
# ---------------------------------------------------------------------------

def test_three_spin_qubit_count():
    # n=3 logical spins → K=3*(3-1)/2=3 parity qubits
    ir = _make_ir(_zzt(0, 1, -1.0), _zzt(0, 2, 0.5), _zzt(1, 2, -0.3), n_sites=3)
    result = lhz_parity_pass(ir)
    assert result.n_logical == 3
    assert result.n_physical == 3
    assert result.encoded_ir.n_sites == 3


def test_four_spin_qubit_count():
    # n=4 → K=6
    ir = _make_ir(
        _zzt(0, 1, 1.0), _zzt(0, 2, -1.0), _zzt(0, 3, 0.5),
        _zzt(1, 2, -0.5), _zzt(1, 3, 0.3), _zzt(2, 3, -0.2),
        n_sites=4,
    )
    result = lhz_parity_pass(ir)
    assert result.n_logical == 4
    assert result.n_physical == 6
    assert len(result.qubit_map) == 6


def test_qubit_map_complete():
    ir = _make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, -1.0), _zzt(1, 2, 0.5), n_sites=3)
    result = lhz_parity_pass(ir)
    expected_pairs = {(0, 1), (0, 2), (1, 2)}
    assert set(result.qubit_map.keys()) == expected_pairs
    # Qubit indices are unique and span 0..K-1
    indices = sorted(result.qubit_map.values())
    assert indices == list(range(result.n_physical))


# ---------------------------------------------------------------------------
# Local-field terms: J_ij → local field on parity qubit
# ---------------------------------------------------------------------------

def test_coupling_becomes_local_field():
    ir = _make_ir(_zzt(0, 1, -2.5), n_sites=2)
    result = lhz_parity_pass(ir)
    # Only one parity qubit (0,1)
    assert result.n_physical == 1
    lf_terms = [
        t for t in result.encoded_ir.terms
        if t.metadata.get("source") == "lhz_coupling"
    ]
    assert len(lf_terms) == 1
    assert lf_terms[0].coefficient == pytest.approx(-2.5)
    assert lf_terms[0].operators == [(0, "Z")]


def test_all_couplings_become_local_fields():
    ir = _make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, -2.0), _zzt(1, 2, 0.5), n_sites=3)
    result = lhz_parity_pass(ir)
    coupling_terms = {
        tuple(t.metadata["logical_pair"]): t.coefficient
        for t in result.encoded_ir.terms
        if t.metadata.get("source") == "lhz_coupling"
    }
    assert coupling_terms[(0, 1)] == pytest.approx(1.0)
    assert coupling_terms[(0, 2)] == pytest.approx(-2.0)
    assert coupling_terms[(1, 2)] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Linear terms: h_i → parity qubit (0,i), i>0; h_0 dropped
# ---------------------------------------------------------------------------

def test_linear_term_encoded_on_parity_qubit():
    ir = _make_ir(_zt(1, 0.8), _zzt(0, 1, -1.0), n_sites=2)
    result = lhz_parity_pass(ir)
    lin_terms = [t for t in result.encoded_ir.terms if t.metadata.get("source") == "lhz_linear"]
    assert len(lin_terms) == 1
    assert lin_terms[0].coefficient == pytest.approx(0.8)
    assert lin_terms[0].metadata["logical_site"] == 1
    # Parity qubit for (0,1)
    pq = result.qubit_map[(0, 1)]
    assert lin_terms[0].operators == [(pq, "Z")]


def test_h0_dropped_as_constant():
    ir = _make_ir(_zt(0, 3.0), _zzt(0, 1, -1.0), n_sites=2)
    result = lhz_parity_pass(ir)
    lin_terms = [t for t in result.encoded_ir.terms if t.metadata.get("source") == "lhz_linear"]
    # h_0 must not appear in IR
    assert len(lin_terms) == 0
    assert result.h0_constant == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Plaquette constraints
# ---------------------------------------------------------------------------

def test_plaquette_count_three_spins():
    # n=3: independent plaquettes = (n-1)(n-2)/2 = 1
    ir = _make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, -1.0), _zzt(1, 2, 0.5), n_sites=3)
    result = lhz_parity_pass(ir)
    assert len(result.plaquettes) == 1


def test_plaquette_count_four_spins():
    # n=4: independent plaquettes = (3)(2)/2 = 3
    ir = _make_ir(
        _zzt(0, 1, 1.0), _zzt(0, 2, -1.0), _zzt(0, 3, 0.5),
        _zzt(1, 2, -0.5), _zzt(1, 3, 0.3), _zzt(2, 3, -0.2),
        n_sites=4,
    )
    result = lhz_parity_pass(ir)
    assert len(result.plaquettes) == 3  # (3)(2)/2


def test_plaquette_penalty_terms_exist():
    ir = _make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, -1.0), _zzt(1, 2, 0.5), n_sites=3)
    result = lhz_parity_pass(ir)
    plaq_terms = [
        t for t in result.encoded_ir.terms
        if t.metadata.get("source") == "lhz_plaquette"
    ]
    assert len(plaq_terms) == 1
    # Coefficient must be -P/2
    assert plaq_terms[0].coefficient == pytest.approx(-result.penalty_strength / 2.0)
    # Must be a 3-body ZZZ term
    assert len(plaq_terms[0].operators) == 3
    assert all(op == "Z" for _, op in plaq_terms[0].operators)


def test_plaquette_indices_match_qubit_map():
    ir = _make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, -1.0), _zzt(1, 2, 0.5), n_sites=3)
    result = lhz_parity_pass(ir)
    # The one plaquette should be (p_{0,1}, p_{0,2}, p_{1,2})
    p01 = result.qubit_map[(0, 1)]
    p02 = result.qubit_map[(0, 2)]
    p12 = result.qubit_map[(1, 2)]
    assert result.plaquettes[0] == (p01, p02, p12)


# ---------------------------------------------------------------------------
# Penalty strength
# ---------------------------------------------------------------------------

def test_penalty_strength_above_lanthaler_bound():
    # Lanthaler & Lechner: P > 2 * max|J|
    J_max = 2.0
    ir = _make_ir(_zzt(0, 1, J_max), _zzt(0, 2, -J_max), _zzt(1, 2, 1.0), n_sites=3)
    result = lhz_parity_pass(ir)
    assert result.penalty_strength > 2.0 * J_max


def test_penalty_factor_default_is_three():
    ir = _make_ir(_zzt(0, 1, 1.0), n_sites=2)
    result = lhz_parity_pass(ir)
    assert result.penalty_strength == pytest.approx(3.0 * 1.0)


def test_penalty_factor_invalid_raises():
    ir = _make_ir(_zzt(0, 1, -1.0), n_sites=2)
    with pytest.raises(ValueError, match="penalty_factor"):
        lhz_parity_pass(ir, penalty_factor=2.0)


# ---------------------------------------------------------------------------
# Substrate and metadata pass-through
# ---------------------------------------------------------------------------

def test_substrate_preserved():
    ir = _make_ir(_zzt(0, 1, -1.0), n_sites=2)
    ir.substrate = SubstrateType.NEUTRAL_ATOM
    result = lhz_parity_pass(ir)
    assert result.encoded_ir.substrate == SubstrateType.NEUTRAL_ATOM


def test_metadata_contains_theorem_reference():
    ir = _make_ir(_zzt(0, 1, -1.0), n_sites=2)
    result = lhz_parity_pass(ir)
    assert "theorem-3" in result.encoded_ir.metadata.get("theorem", "")


def test_negative_couplings_flagged_in_metadata():
    ir = _make_ir(_zzt(0, 1, -2.0), _zzt(0, 2, 1.0), n_sites=3)
    result = lhz_parity_pass(ir)
    neg = result.metadata["negative_couplings"]
    assert [0, 1] in neg
    assert [0, 2] not in neg


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_ir_returns_zero_physical():
    ir = HamiltonianIR(terms=[], n_sites=0)
    result = lhz_parity_pass(ir)
    assert result.n_physical == 0
    assert result.n_logical == 0
    assert result.encoded_ir.n_sites == 0


def test_single_pair_no_plaquettes():
    # n=2: K=1, (n-1)(n-2)/2 = 0 plaquettes
    ir = _make_ir(_zzt(0, 1, -1.0), n_sites=2)
    result = lhz_parity_pass(ir)
    assert result.n_physical == 1
    assert len(result.plaquettes) == 0


def test_all_positive_couplings_still_encodes():
    # LHZ pass should work even when all couplings are positive
    ir = _make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, 0.5), _zzt(1, 2, 2.0), n_sites=3)
    result = lhz_parity_pass(ir)
    assert result.n_physical == 3


def test_to_dict_roundtrip_keys():
    ir = _make_ir(_zzt(0, 1, -1.0), _zzt(0, 2, 0.5), _zzt(1, 2, -0.3), n_sites=3)
    result = lhz_parity_pass(ir)
    d = result.to_dict()
    assert d["n_logical"] == 3
    assert d["n_physical"] == 3
    assert len(d["plaquettes"]) == 1
    assert d["penalty_strength"] == pytest.approx(result.penalty_strength)


def test_passthrough_higher_body_terms():
    xterm = HamiltonianTerm(coefficient=0.5, operators=[(0, "X")])
    ir = _make_ir(_zzt(0, 1, -1.0), xterm, n_sites=2)
    result = lhz_parity_pass(ir)
    passthrough = [t for t in result.encoded_ir.terms if t.metadata.get("lhz_passthrough")]
    assert len(passthrough) == 1
    assert passthrough[0].coefficient == pytest.approx(0.5)
