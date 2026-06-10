# Copyright 2026 LIMEN Contributors. Apache 2.0.
"""Tests for the LHZ penalty-strength certificate (certify_lhz)."""

import pytest

from limen.analog.certificate import certify_ising
from limen.analog.hamiltonian import HamiltonianIR, HamiltonianTerm, SubstrateType
from limen.analog.lhz import certify_lhz, lhz_parity_pass


def _zzt(i: int, j: int, coeff: float) -> HamiltonianTerm:
    return HamiltonianTerm(coefficient=coeff, operators=[(i, "Z"), (j, "Z")])


def _zt(i: int, coeff: float) -> HamiltonianTerm:
    return HamiltonianTerm(coefficient=coeff, operators=[(i, "Z")])


def _make_ir(*terms: HamiltonianTerm, n: int = 0) -> HamiltonianIR:
    return HamiltonianIR(terms=list(terms), n_sites=n, substrate=SubstrateType.NEUTRAL_ATOM)


# ---------------------------------------------------------------------------
# penalty_gap and error_tolerance
# ---------------------------------------------------------------------------

def test_penalty_gap_equals_P_minus_twice_max_J():
    # max|J|=2, P=3*2=6, gap=6-4=2
    ir = _make_ir(_zzt(0, 1, -2.0), n=2)
    cert = certify_lhz(lhz_parity_pass(ir, penalty_factor=3.0))
    assert cert.max_logical_coupling == pytest.approx(2.0)
    assert cert.penalty_strength == pytest.approx(6.0)
    assert cert.penalty_gap == pytest.approx(2.0)
    assert cert.error_tolerance == pytest.approx(1.0)


def test_larger_penalty_factor_gives_larger_gap():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    gap3 = certify_lhz(lhz_parity_pass(ir, penalty_factor=3.0)).penalty_gap
    gap5 = certify_lhz(lhz_parity_pass(ir, penalty_factor=5.0)).penalty_gap
    assert gap5 > gap3


def test_h_i_and_h0_included_in_max_coupling():
    # h_0=5 is gauge-dropped from IR but must bound max_coupling; h_1=3 is kept
    ir = _make_ir(_zt(0, 5.0), _zt(1, 3.0), _zzt(0, 1, -1.0), n=2)
    cert = certify_lhz(lhz_parity_pass(ir, penalty_factor=3.0))
    assert cert.max_logical_coupling == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# sufficient_for_correctness
# ---------------------------------------------------------------------------

def test_no_cert_sufficient():
    ir = _make_ir(_zzt(0, 1, -1.0), _zzt(0, 2, 0.5), _zzt(1, 2, -0.3), n=3)
    cert = certify_lhz(lhz_parity_pass(ir))
    assert cert.compilation_certificate is None
    assert cert.sufficient_for_correctness is True


def test_small_error_cert_sufficient():
    # P=5, max|J|=1, gap=3, tolerance=1.5; error=0.05 < 1.5
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    result = lhz_parity_pass(ir, penalty_factor=5.0)
    small_cert = certify_ising({0: -1.0}, {}, {0: -1.05}, {}, n_sites=result.n_physical)
    cert = certify_lhz(result, compilation_certificate=small_cert)
    assert cert.sufficient_for_correctness is True


def test_large_error_cert_not_sufficient():
    # P=3, max|J|=1, gap=1, tolerance=0.5; error=1.0 > 0.5
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    result = lhz_parity_pass(ir, penalty_factor=3.0)
    large_cert = certify_ising({0: -1.0}, {}, {0: -2.0}, {}, n_sites=result.n_physical)
    cert = certify_lhz(result, compilation_certificate=large_cert)
    assert cert.sufficient_for_correctness is False


# ---------------------------------------------------------------------------
# Serialization and metadata
# ---------------------------------------------------------------------------

def test_to_dict_with_cert():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    result = lhz_parity_pass(ir)
    comp_cert = certify_ising({0: -1.0}, {}, {0: -1.02}, {}, n_sites=1)
    d = certify_lhz(result, compilation_certificate=comp_cert).to_dict()
    assert d["compilation_certificate"] is not None
    assert "l1_bound" in d["compilation_certificate"]
    assert d["sufficient_for_correctness"] is True


def test_metadata_has_theorem_and_site_counts():
    ir = _make_ir(_zzt(0, 1, -1.0), _zzt(0, 2, 0.5), _zzt(1, 2, 0.3), n=3)
    result = lhz_parity_pass(ir)
    cert = certify_lhz(result)
    assert "theorem-3" in cert.metadata.get("theorem", "")
    assert cert.metadata["n_logical"] == 3
    assert cert.metadata["n_physical"] == 3
