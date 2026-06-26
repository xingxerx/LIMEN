# Copyright 2026 LIMEN Contributors. Apache 2.0.
"""Tests for the automatic LHZ parity fallback in run_neutral_atom (Theorem 3 route)."""

from itertools import combinations

import pytest

from limen.analog.backends.neutral_atom import (
    _C6_MHZ_UM6,
    check_plaquette_geometry,
    run_neutral_atom,
)
from limen.analog.hamiltonian import HamiltonianIR, HamiltonianTerm, SubstrateType

pytest.importorskip("numpy", reason="numpy required for geometry tests")


def _zzt(i: int, j: int, coeff: float) -> HamiltonianTerm:
    return HamiltonianTerm(coefficient=coeff, operators=[(i, "Z"), (j, "Z")])


def _make_ir(*terms: HamiltonianTerm, n: int) -> HamiltonianIR:
    return HamiltonianIR(terms=list(terms), n_sites=n, substrate=SubstrateType.NEUTRAL_ATOM)


def _tetrahedron_J(d_um: float) -> dict[tuple[int, int], float]:
    """All-equal-distance coupling dict for 4 atoms (regular tetrahedron, needs R³)."""
    J_val = _C6_MHZ_UM6 / (4.0 * d_um ** 6)
    return {(i, j): J_val for i, j in combinations(range(4), 2)}


def test_geometrically_frustrated_input_routes_through_lhz():
    # Same frustrated input as test_tetrahedron_flags_not_natively_realizable:
    # the heuristic certificate is still flagged not-natively-realizable...
    J = _tetrahedron_J(5.0)
    ir = _make_ir(*[_zzt(i, j, v) for (i, j), v in J.items()], n=4)
    res = run_neutral_atom(ir)
    assert res.geometry.embeddable is False
    assert res.certificate.natively_realizable is False

    # ...but the automatic LHZ fallback now provides a real, certified
    # recursive compilation instead of leaving the caller with just a
    # known-bad heuristic certificate.
    assert res.lhz_result is not None
    assert res.lhz_certificate is not None

    encoded_cert = res.lhz_certificate.compilation_certificate
    assert encoded_cert is not None
    # All logical couplings became local fields on parity qubits, which the
    # detuning formula realizes exactly (Theorem 2 part 1) — near-zero error.
    assert encoded_cert.natively_realizable is True
    assert encoded_cert.l1_bound < 1e-6
    if encoded_cert.operator_norm is not None:
        assert encoded_cert.operator_norm < 1e-6

    assert res.lhz_certificate.sufficient_for_correctness is True


def test_negative_coupling_also_routes_through_lhz():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    res = run_neutral_atom(ir)
    assert res.certificate.natively_realizable is False
    assert res.lhz_result is not None
    assert res.lhz_certificate is not None
    assert res.lhz_certificate.compilation_certificate.natively_realizable is True


def test_natively_realizable_input_skips_lhz():
    ir = _make_ir(_zzt(0, 1, 1.0), n=2)
    res = run_neutral_atom(ir)
    assert res.certificate.natively_realizable is True
    assert res.lhz_result is None
    assert res.lhz_certificate is None
    assert res.plaquette_geometry is None


def test_lhz_fallback_includes_plaquette_geometry():
    J = _tetrahedron_J(5.0)
    ir = _make_ir(*[_zzt(i, j, v) for (i, j), v in J.items()], n=4)
    res = run_neutral_atom(ir)
    assert res.lhz_result is not None

    geo = res.plaquette_geometry
    assert geo is not None
    assert geo.n_plaquettes == len(res.lhz_result.plaquettes)
    assert geo.max_spread is not None
    assert len(geo.per_plaquette_spread) == geo.n_plaquettes
    # Scope limitation must always be surfaced, regardless of outcome.
    assert any("does not certify" in note for note in geo.notes)


def test_check_plaquette_geometry_no_plaquettes():
    geo = check_plaquette_geometry([], [])
    assert geo.plausible is True
    assert geo.n_plaquettes == 0
    assert geo.max_spread is None


def test_check_plaquette_geometry_equilateral_is_plausible():
    # An equilateral triangle of parity qubits is the ideal symmetric case.
    positions = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.8660254)]
    geo = check_plaquette_geometry([(0, 1, 2)], positions)
    assert geo.plausible is True
    assert geo.per_plaquette_spread[0] < 0.01


def test_check_plaquette_geometry_degenerate_triangle_is_implausible():
    # Three collinear, unevenly-spaced points: highly asymmetric triangle.
    positions = [(0.0, 0.0), (1.0, 0.0), (10.0, 0.0)]
    geo = check_plaquette_geometry([(0, 1, 2)], positions, max_relative_spread=0.5)
    assert geo.plausible is False
    assert any("not roughly" in note for note in geo.notes)
