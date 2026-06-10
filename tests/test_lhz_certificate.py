# Copyright 2026 LIMEN Contributors. Apache 2.0.
"""Tests for the LHZ penalty-strength certificate (certify_lhz)."""

import pytest

from limen.analog.certificate import certify_ising
from limen.analog.hamiltonian import HamiltonianIR, HamiltonianTerm, SubstrateType
from limen.analog.lhz import LHZCertificate, certify_lhz, lhz_parity_pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zzt(i: int, j: int, coeff: float) -> HamiltonianTerm:
    return HamiltonianTerm(coefficient=coeff, operators=[(i, "Z"), (j, "Z")])


def _zt(i: int, coeff: float) -> HamiltonianTerm:
    return HamiltonianTerm(coefficient=coeff, operators=[(i, "Z")])


def _make_ir(*terms: HamiltonianTerm, n: int = 0) -> HamiltonianIR:
    return HamiltonianIR(terms=list(terms), n_sites=n, substrate=SubstrateType.NEUTRAL_ATOM)


# ---------------------------------------------------------------------------
# penalty_gap and error_tolerance
# ---------------------------------------------------------------------------

def test_penalty_gap_positive_by_construction():
    ir = _make_ir(_zzt(0, 1, -2.0), _zzt(0, 2, 1.5), _zzt(1, 2, -0.5), n=3)
    result = lhz_parity_pass(ir, penalty_factor=3.0)
    cert = certify_lhz(result)
    assert cert.penalty_gap > 0.0


def test_penalty_gap_equals_P_minus_twice_max_J():
    # max|J| = 2.0, P = 3 * 2.0 = 6.0, gap = 6 - 2*2 = 2.0
    ir = _make_ir(_zzt(0, 1, -2.0), n=2)
    result = lhz_parity_pass(ir, penalty_factor=3.0)
    cert = certify_lhz(result)
    assert cert.max_logical_coupling == pytest.approx(2.0)
    assert cert.penalty_strength == pytest.approx(6.0)
    assert cert.penalty_gap == pytest.approx(6.0 - 2.0 * 2.0)


def test_error_tolerance_is_half_penalty_gap():
    ir = _make_ir(_zzt(0, 1, -1.0), _zzt(0, 2, 0.5), _zzt(1, 2, -0.3), n=3)
    result = lhz_parity_pass(ir)
    cert = certify_lhz(result)
    assert cert.error_tolerance == pytest.approx(cert.penalty_gap / 2.0)


def test_larger_penalty_factor_gives_larger_gap():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    cert3 = certify_lhz(lhz_parity_pass(ir, penalty_factor=3.0))
    cert5 = certify_lhz(lhz_parity_pass(ir, penalty_factor=5.0))
    assert cert5.penalty_gap > cert3.penalty_gap


def test_h_i_included_in_max_coupling():
    # h_2 = 3.0 > |J_{01}| = 1.0; max_coupling should be 3.0
    ir = _make_ir(_zzt(0, 1, -1.0), _zt(2, 3.0), n=3)
    result = lhz_parity_pass(ir, penalty_factor=3.0)
    cert = certify_lhz(result)
    assert cert.max_logical_coupling == pytest.approx(3.0)


def test_h0_constant_included_in_max_coupling():
    # h_0 = 5.0 is dropped from IR but must still bound max_coupling
    ir = _make_ir(_zt(0, 5.0), _zzt(0, 1, -1.0), n=2)
    result = lhz_parity_pass(ir, penalty_factor=3.0)
    cert = certify_lhz(result)
    assert cert.max_logical_coupling == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# sufficient_for_correctness: no compilation certificate
# ---------------------------------------------------------------------------

def test_no_certificate_sufficient_true():
    ir = _make_ir(_zzt(0, 1, -1.0), _zzt(0, 2, 0.5), _zzt(1, 2, -0.3), n=3)
    result = lhz_parity_pass(ir)
    cert = certify_lhz(result)
    assert cert.compilation_certificate is None
    assert cert.sufficient_for_correctness is True


def test_no_certificate_note_mentions_perfect_compilation():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    cert = certify_lhz(lhz_parity_pass(ir))
    assert any("perfect" in n.lower() for n in cert.notes)


# ---------------------------------------------------------------------------
# sufficient_for_correctness: with compilation certificate
# ---------------------------------------------------------------------------

def test_small_error_cert_sufficient():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    result = lhz_parity_pass(ir, penalty_factor=5.0)
    # P=5, max|J|=1, gap=3, tolerance=1.5
    # The encoded IR has 1 parity qubit (site 0) with local field -1.0.
    # Small detuning error of 0.05 → l1_bound=0.05 << 1.5.
    small_cert = certify_ising(
        target_h={0: -1.0}, target_J={},
        compiled_h={0: -1.05}, compiled_J={},
        n_sites=result.n_physical,
    )
    cert = certify_lhz(result, compilation_certificate=small_cert)
    assert cert.sufficient_for_correctness is True
    assert cert.compilation_certificate is not None


def test_large_error_cert_not_sufficient():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    result = lhz_parity_pass(ir, penalty_factor=3.0)
    # P=3, max|J|=1, gap=1, tolerance=0.5
    # Large detuning error of 1.0 → l1_bound=1.0 > 0.5.
    large_cert = certify_ising(
        target_h={0: -1.0}, target_J={},
        compiled_h={0: -2.0}, compiled_J={},
        n_sites=result.n_physical,
    )
    cert = certify_lhz(result, compilation_certificate=large_cert)
    assert cert.sufficient_for_correctness is False


def test_notes_report_certified_when_sufficient():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    result = lhz_parity_pass(ir, penalty_factor=10.0)
    # n_physical=1 parity qubit; certify on that 1-qubit encoded system
    tiny_cert = certify_ising({0: 1.0}, {}, {0: 1.001}, {}, n_sites=1)
    cert = certify_lhz(result, compilation_certificate=tiny_cert)
    assert cert.sufficient_for_correctness is True
    assert any("certified correct" in n for n in cert.notes)


def test_notes_report_cannot_certify_when_not_sufficient():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    result = lhz_parity_pass(ir, penalty_factor=3.0)
    # P=3, max|J|=1, gap=1, tolerance=0.5; give l1_bound=1.0 > 0.5
    big_cert = certify_ising({0: 1.0}, {}, {0: 5.0}, {}, n_sites=1)
    cert = certify_lhz(result, compilation_certificate=big_cert)
    assert cert.sufficient_for_correctness is False
    assert any("cannot be certified" in n for n in cert.notes)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_to_dict_without_cert():
    ir = _make_ir(_zzt(0, 1, -1.0), _zzt(0, 2, 0.5), _zzt(1, 2, -0.3), n=3)
    cert = certify_lhz(lhz_parity_pass(ir))
    d = cert.to_dict()
    assert d["compilation_certificate"] is None
    assert d["sufficient_for_correctness"] is True
    assert d["penalty_gap"] == pytest.approx(cert.penalty_gap)


def test_to_dict_with_cert():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    result = lhz_parity_pass(ir)
    # 1 parity qubit in encoded_ir; certify a small local-field error
    comp_cert = certify_ising({0: 1.0}, {}, {0: 1.02}, {}, n_sites=1)
    lhz_cert = certify_lhz(result, compilation_certificate=comp_cert)
    d = lhz_cert.to_dict()
    assert d["compilation_certificate"] is not None
    assert "l1_bound" in d["compilation_certificate"]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_metadata_has_theorem_reference():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    cert = certify_lhz(lhz_parity_pass(ir))
    assert "theorem-3" in cert.metadata.get("theorem", "")


def test_metadata_has_site_counts():
    ir = _make_ir(_zzt(0, 1, -1.0), _zzt(0, 2, 0.5), _zzt(1, 2, 0.3), n=3)
    result = lhz_parity_pass(ir)
    cert = certify_lhz(result)
    assert cert.metadata["n_logical"] == 3
    assert cert.metadata["n_physical"] == 3


def test_notes_include_reference():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    cert = certify_lhz(lhz_parity_pass(ir))
    combined = " ".join(cert.notes)
    assert "Lanthaler" in combined
