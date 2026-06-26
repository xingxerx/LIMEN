# Copyright 2026 LIMEN Contributors. Apache 2.0.
"""Phase 3 completion tests: delta-model threading, realizability
classification, and the photonic spectral-radius regression fix."""

import pytest

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.analog.backends.neutral_atom import run_neutral_atom
from limen.analog.backends.photonic import run_photonic
from limen.analog.delta_model import DeviceDrift, HardwareDeltaModel
from limen.analog.hamiltonian import (
    HamiltonianIR,
    HamiltonianTerm,
    SubstrateType,
    from_physical_encoding,
)

TRIVIAL_QUBO = {
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x0", "x1"):  2.0,
}


def _make_ir(substrate=SubstrateType.NEUTRAL_ATOM):
    graph = from_qubo_dict(TRIVIAL_QUBO)
    encoding = compile_lexicographic(graph, default_hardware_graph(4))
    return from_physical_encoding(encoding, substrate=substrate)


def _ring_ir(n: int, J: float = 1.0) -> HamiltonianIR:
    terms = [
        HamiltonianTerm(coefficient=J, operators=[(i, "Z"), ((i + 1) % n, "Z")])
        for i in range(n)
    ]
    return HamiltonianIR(terms=terms, n_sites=n, substrate=SubstrateType.PHOTONIC)


# -- Delta-model threading ----------------------------------------------

def test_identity_delta_model_matches_no_model():
    ir = _make_ir()
    base = run_neutral_atom(ir)
    ident = run_neutral_atom(
        ir, delta_model=HardwareDeltaModel.identity("dev-id", SubstrateType.NEUTRAL_ATOM, ir.n_sites)
    )
    assert ident.atom_positions == pytest.approx(base.atom_positions)
    assert ident.detunings == pytest.approx(base.detunings)
    for key in base.realized_couplings:
        assert ident.realized_couplings[key] == pytest.approx(base.realized_couplings[key])
    assert ident.metadata["delta_model_device"] == "dev-id"
    assert base.metadata["delta_model_device"] is None


def test_drifted_delta_model_changes_submitted_parameters():
    ir = _make_ir()
    base = run_neutral_atom(ir)
    drift = DeviceDrift(
        site_detuning_offsets={0: 0.5},
        coupling_scale_errors={(0, 1): 0.2},
    )
    model = HardwareDeltaModel(
        device_id="dev-drift", substrate=SubstrateType.NEUTRAL_ATOM,
        drift=drift, n_sites=ir.n_sites,
    )
    drifted = run_neutral_atom(ir, delta_model=model)
    # Detuning at site 0 must be pre-distorted by -0.5 relative to its own
    # uncorrected value; positions also change, so just assert difference.
    assert drifted.detunings != pytest.approx(base.detunings)
    assert drifted.metadata["delta_model_device"] == "dev-drift"
    assert drifted.certificate is not None


def test_negative_coupling_flags_not_natively_realizable():
    terms = [
        HamiltonianTerm(coefficient=-0.5, operators=[(0, "Z"), (1, "Z")]),
    ]
    ir = HamiltonianIR(terms=terms, n_sites=2, substrate=SubstrateType.NEUTRAL_ATOM)
    result = run_neutral_atom(ir)
    cert = result.certificate
    assert cert is not None
    assert cert.natively_realizable is False
    # vdW realises a positive coupling against a negative target: real error.
    assert cert.operator_norm is not None
    assert cert.operator_norm > 0.0
    assert any("parity" in note.lower() for note in cert.notes)


def test_lhz_fallback_populated_on_negative_coupling():
    """When natively_realizable=False, result must carry an LHZ encoding."""
    from limen.analog.lhz import LHZCertificate, LHZResult

    terms = [
        HamiltonianTerm(coefficient=-1.0, operators=[(0, "Z"), (1, "Z")]),
        HamiltonianTerm(coefficient=0.5, operators=[(1, "Z"), (2, "Z")]),
    ]
    ir = HamiltonianIR(terms=terms, n_sites=3, substrate=SubstrateType.NEUTRAL_ATOM)
    result = run_neutral_atom(ir)

    assert result.certificate is not None
    assert result.certificate.natively_realizable is False

    assert isinstance(result.lhz_result, LHZResult)
    assert isinstance(result.lhz_certificate, LHZCertificate)

    enc = result.lhz_result
    # n_physical = number of unique (i,j) pairs = 2
    assert enc.n_physical == 2
    assert enc.n_logical == 3
    # penalty gap must be positive (encoding is self-consistent)
    assert result.lhz_certificate.penalty_gap > 0.0

    assert result.metadata["lhz_fallback_applied"] is True
    assert result.metadata["lhz_n_physical"] == 2


def test_lhz_fallback_absent_when_natively_realizable():
    """Positive-only problems must not trigger LHZ encoding."""
    terms = [
        HamiltonianTerm(coefficient=1.0, operators=[(0, "Z"), (1, "Z")]),
    ]
    ir = HamiltonianIR(terms=terms, n_sites=2, substrate=SubstrateType.NEUTRAL_ATOM)
    result = run_neutral_atom(ir)

    assert result.lhz_result is None
    assert result.lhz_certificate is None
    assert result.metadata["lhz_fallback_applied"] is False


def test_neutral_atom_certificate_norm_bounded_by_l1():
    result = run_neutral_atom(_make_ir())
    cert = result.certificate
    assert cert is not None
    assert cert.operator_norm is not None
    assert cert.operator_norm <= cert.l1_bound + 1e-12
    assert cert.n_sites == 2


# -- Photonic spectral-radius regression (Gershgorin fix) ----------------

def test_photonic_dense_ring_spectral_radius_below_one():
    # Regression: entry-wise max|J| scaling gave rho(A) ~ 1.8 on rings.
    for n in (3, 4, 6):
        result = run_photonic(_ring_ir(n))
        assert result.spectral_radius < 1.0, f"ring-{n} violated rho < 1"
        assert result.metadata["scale_rule"] == "gershgorin_row_sum_x1.1"


def test_photonic_certificate_encoding_exact():
    result = run_photonic(_make_ir(SubstrateType.PHOTONIC))
    cert = result.certificate
    assert cert is not None
    assert cert.l1_bound == pytest.approx(0.0)
    assert cert.operator_norm == pytest.approx(0.0)
    assert any("heuristic" in note.lower() for note in cert.notes)


def test_top_level_certificate_exports():
    import limen

    assert limen.__version__ == "0.4.0"
    cert_cls = limen.CompilationCertificate
    fn = limen.certify_ising
    cert = fn({0: 1.0}, {}, {0: 1.0}, {}, n_sites=1)
    assert isinstance(cert, cert_cls)
