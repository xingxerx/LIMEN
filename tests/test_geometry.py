# Copyright 2026 LIMEN Contributors. Apache 2.0.
"""Tests for the Schoenberg/MDS geometric embeddability check (Theorem 2 geometry condition)."""

import math

import pytest

from limen.analog.backends.neutral_atom import (
    GeometricEmbeddabilityResult,
    _C6_MHZ_UM6,
    _target_radius,
    check_geometric_embeddability,
    run_neutral_atom,
)
from limen.analog.hamiltonian import HamiltonianIR, HamiltonianTerm, SubstrateType

numpy = pytest.importorskip("numpy", reason="numpy required for geometry tests")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zzt(i: int, j: int, coeff: float) -> HamiltonianTerm:
    return HamiltonianTerm(coefficient=coeff, operators=[(i, "Z"), (j, "Z")])


def _make_ir(*terms: HamiltonianTerm, n: int) -> HamiltonianIR:
    return HamiltonianIR(terms=list(terms), n_sites=n, substrate=SubstrateType.NEUTRAL_ATOM)


def _equilateral_J(n: int, r_um: float) -> dict[tuple[int, int], float]:
    """Return J dict for n atoms uniformly spaced at distance r_um on a circle."""
    from itertools import combinations
    import math
    J = {}
    angles = [2 * math.pi * k / n for k in range(n)]
    positions = [(r_um * math.cos(a), r_um * math.sin(a)) for a in angles]
    for i, j in combinations(range(n), 2):
        dx = positions[i][0] - positions[j][0]
        dy = positions[i][1] - positions[j][1]
        d = math.sqrt(dx * dx + dy * dy)
        J[(i, j)] = _C6_MHZ_UM6 / (4.0 * d ** 6)
    return J


# ---------------------------------------------------------------------------
# check_geometric_embeddability: trivial / small cases
# ---------------------------------------------------------------------------

def test_empty_target_J_trivially_embeddable():
    result = check_geometric_embeddability({})
    assert result.embeddable is True
    assert result.checked is False
    assert "trivially" in " ".join(result.notes).lower()


def test_single_pair_trivially_embeddable():
    J = {(0, 1): 1.0}
    result = check_geometric_embeddability(J)
    assert result.embeddable is True
    assert result.checked is False  # n < 4


def test_three_sites_trivially_embeddable():
    J = {(0, 1): 1.0, (0, 2): 0.5, (1, 2): 0.8}
    result = check_geometric_embeddability(J)
    assert result.embeddable is True
    assert result.checked is False
    assert result.n_constrained_sites == 3


def test_negative_couplings_ignored():
    # Negative J pairs are excluded; only positive pairs checked.
    J = {(0, 1): -1.0, (0, 2): -0.5}
    result = check_geometric_embeddability(J)
    assert result.embeddable is True
    assert result.checked is False  # no positive pairs → trivial


# ---------------------------------------------------------------------------
# check_geometric_embeddability: 2-D realizable configurations
# ---------------------------------------------------------------------------

def test_four_atoms_on_circle_embeddable():
    # 4 atoms at corners of a square — definitely 2-D embeddable.
    r = 5.0
    J = _equilateral_J(4, r)
    result = check_geometric_embeddability(J)
    assert result.checked is True
    assert result.embeddable is True
    assert result.psd_satisfied is True
    assert result.gram_rank is not None
    assert result.gram_rank <= 2


def test_five_atoms_on_circle_embeddable():
    # 5 atoms on a ring — all lie in the plane.
    J = _equilateral_J(5, 6.0)
    result = check_geometric_embeddability(J)
    assert result.checked is True
    assert result.embeddable is True


def test_gram_min_eigenvalue_nonnegative_for_embeddable():
    J = _equilateral_J(4, 5.0)
    result = check_geometric_embeddability(J)
    assert result.gram_min_eigenvalue is not None
    assert result.gram_min_eigenvalue >= -1e-6


# ---------------------------------------------------------------------------
# check_geometric_embeddability: geometrically frustrated configuration
# ---------------------------------------------------------------------------

def test_geometrically_frustrated_not_embeddable():
    # 4 atoms requiring pairwise distances that can't live in R².
    # Construction: assign distances consistent with an R³ tetrahedron
    # (all equal edges), which is NOT 2-D embeddable.
    # For a regular tetrahedron with edge d, the Cayley–Menger determinant
    # is non-zero for 4 points iff they span R³.
    # We pick a coupling strength corresponding to edge d, then give the
    # 4th point a distance to all others that forces a 3-D embedding.
    d = 5.0  # target distance in µm for all pairs
    J_regular = _C6_MHZ_UM6 / (4.0 * d ** 6)
    # All 6 pairs of 4 atoms at equal required distance (regular tetrahedron)
    from itertools import combinations
    J = {(i, j): J_regular for i, j in combinations(range(4), 2)}
    result = check_geometric_embeddability(J)
    assert result.checked is True
    # A regular tetrahedron requires rank-3 embedding; gram_rank should be 3
    assert result.gram_rank is not None
    assert result.gram_rank > 2
    assert result.embeddable is False


def test_negative_gram_eigenvalue_flags_not_psd():
    # Construct a clearly impossible distance set: violates the triangle
    # inequality so the squared-distance matrix can't be PSD.
    # Triangle with sides 1, 1, 100 (violates d(0,2) ≤ d(0,1)+d(1,2)).
    d01, d12, d02 = 1.0, 1.0, 100.0
    J01 = _C6_MHZ_UM6 / (4.0 * d01 ** 6)
    J12 = _C6_MHZ_UM6 / (4.0 * d12 ** 6)
    J02 = _C6_MHZ_UM6 / (4.0 * d02 ** 6)
    # Need a 4th site to trigger the full check (n ≥ 4).
    # Add site 3 equidistant at d=1 from 0 and 1.
    J03 = _C6_MHZ_UM6 / (4.0 * 1.0 ** 6)
    J13 = _C6_MHZ_UM6 / (4.0 * 1.0 ** 6)
    J = {(0, 1): J01, (1, 2): J12, (0, 2): J02, (0, 3): J03, (1, 3): J13}
    result = check_geometric_embeddability(J)
    assert result.checked is True
    assert result.gram_min_eigenvalue is not None
    # The triangle-inequality violation should produce a negative eigenvalue.
    assert not result.psd_satisfied or not result.embeddable


# ---------------------------------------------------------------------------
# Integration with run_neutral_atom
# ---------------------------------------------------------------------------

def test_run_neutral_atom_geometry_field_present():
    ir = _make_ir(_zzt(0, 1, 1.0), _zzt(0, 2, 0.5), n=3)
    result = run_neutral_atom(ir)
    assert result.geometry is not None
    assert isinstance(result.geometry, GeometricEmbeddabilityResult)


def test_run_neutral_atom_geometry_not_checked_below_4():
    # 2 or 3 atoms: geometry check should be skipped (trivially embeddable).
    ir = _make_ir(_zzt(0, 1, 1.0), n=2)
    result = run_neutral_atom(ir)
    assert result.geometry is not None
    assert result.geometry.checked is False
    assert result.geometry.embeddable is True


def test_run_neutral_atom_natively_realizable_positive_embeddable():
    # 4 atoms on a ring with all-positive couplings: should be natively realizable.
    J = _equilateral_J(4, 5.0)
    terms = [_zzt(i, j, v) for (i, j), v in J.items()]
    ir = _make_ir(*terms, n=4)
    result = run_neutral_atom(ir)
    assert result.certificate is not None
    # With positive, 2-D embeddable couplings: natively_realizable should be True.
    assert result.certificate.natively_realizable is True
    assert result.geometry is not None
    assert result.geometry.embeddable is True


def test_run_neutral_atom_geometry_flags_not_realizable():
    # Regular tetrahedron requires R³; should flag natively_realizable=False.
    d = 5.0
    J_val = _C6_MHZ_UM6 / (4.0 * d ** 6)
    from itertools import combinations
    terms = [_zzt(i, j, J_val) for i, j in combinations(range(4), 2)]
    ir = _make_ir(*terms, n=4)
    result = run_neutral_atom(ir)
    assert result.geometry is not None
    assert result.geometry.checked is True
    assert result.geometry.embeddable is False
    assert result.certificate is not None
    assert result.certificate.natively_realizable is False


def test_run_neutral_atom_geometry_metadata_keys():
    J = _equilateral_J(4, 5.0)
    terms = [_zzt(i, j, v) for (i, j), v in J.items()]
    ir = _make_ir(*terms, n=4)
    result = run_neutral_atom(ir)
    assert "geometry_embeddable" in result.metadata
    assert "geometry_checked" in result.metadata
    assert "gram_min_eigenvalue" in result.metadata
    assert "gram_rank" in result.metadata


def test_negative_coupling_still_not_natively_realizable():
    # Negative coupling: sign failure should still flag correctly
    # (regardless of geometry).
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    result = run_neutral_atom(ir)
    assert result.certificate is not None
    assert result.certificate.natively_realizable is False
