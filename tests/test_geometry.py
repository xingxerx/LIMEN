# Copyright 2026 LIMEN Contributors. Apache 2.0.
"""Tests for the Schoenberg/MDS geometric embeddability check (Theorem 2 geometry condition)."""

import math
from itertools import combinations

import pytest

from limen.analog.backends.neutral_atom import (
    _C6_MHZ_UM6,
    check_geometric_embeddability,
    run_neutral_atom,
)
from limen.analog.hamiltonian import HamiltonianIR, HamiltonianTerm, SubstrateType

pytest.importorskip("numpy", reason="numpy required for geometry tests")


def _zzt(i: int, j: int, coeff: float) -> HamiltonianTerm:
    return HamiltonianTerm(coefficient=coeff, operators=[(i, "Z"), (j, "Z")])


def _make_ir(*terms: HamiltonianTerm, n: int) -> HamiltonianIR:
    return HamiltonianIR(terms=list(terms), n_sites=n, substrate=SubstrateType.NEUTRAL_ATOM)


def _ring_J(n: int, r_um: float) -> dict[tuple[int, int], float]:
    """Coupling dict for n atoms uniformly placed on a circle of radius r_um."""
    angles = [2 * math.pi * k / n for k in range(n)]
    pos = [(r_um * math.cos(a), r_um * math.sin(a)) for a in angles]
    J = {}
    for i, j in combinations(range(n), 2):
        d = math.hypot(pos[i][0] - pos[j][0], pos[i][1] - pos[j][1])
        J[(i, j)] = _C6_MHZ_UM6 / (4.0 * d ** 6)
    return J


def _tetrahedron_J(d_um: float) -> dict[tuple[int, int], float]:
    """All-equal-distance coupling dict for 4 atoms (regular tetrahedron, needs R³)."""
    J_val = _C6_MHZ_UM6 / (4.0 * d_um ** 6)
    return {(i, j): J_val for i, j in combinations(range(4), 2)}


# ---------------------------------------------------------------------------
# Trivial / small cases (not checked by MDS)
# ---------------------------------------------------------------------------

def test_empty_and_small_are_trivially_embeddable():
    assert check_geometric_embeddability({}).embeddable is True
    assert check_geometric_embeddability({(0, 1): 1.0}).checked is False
    assert check_geometric_embeddability(
        {(0, 1): 1.0, (0, 2): 0.5, (1, 2): 0.8}
    ).checked is False


def test_negative_couplings_excluded_from_check():
    # No positive pairs → trivially embeddable without running MDS.
    result = check_geometric_embeddability({(0, 1): -1.0, (0, 2): -0.5})
    assert result.embeddable is True
    assert result.checked is False


# ---------------------------------------------------------------------------
# 2-D realizable configurations
# ---------------------------------------------------------------------------

def test_ring_of_four_is_embeddable():
    result = check_geometric_embeddability(_ring_J(4, 5.0))
    assert result.checked is True
    assert result.embeddable is True
    assert result.gram_rank <= 2


def test_ring_of_five_is_embeddable():
    result = check_geometric_embeddability(_ring_J(5, 6.0))
    assert result.checked is True
    assert result.embeddable is True


# ---------------------------------------------------------------------------
# Geometrically frustrated configurations
# ---------------------------------------------------------------------------

def test_regular_tetrahedron_not_2d_embeddable():
    # All 6 pairs at equal distance → needs R³ (rank-3 Gram matrix).
    result = check_geometric_embeddability(_tetrahedron_J(5.0))
    assert result.checked is True
    assert result.gram_rank > 2
    assert result.embeddable is False


def test_triangle_inequality_violation_not_embeddable():
    # Triangle with sides 1, 1, 100 — impossible in any Euclidean space.
    # Add a 4th site to force the full MDS check.
    d_short, d_long = 1.0, 100.0
    J = {
        (0, 1): _C6_MHZ_UM6 / (4 * d_short ** 6),
        (1, 2): _C6_MHZ_UM6 / (4 * d_short ** 6),
        (0, 2): _C6_MHZ_UM6 / (4 * d_long  ** 6),
        (0, 3): _C6_MHZ_UM6 / (4 * d_short ** 6),
        (1, 3): _C6_MHZ_UM6 / (4 * d_short ** 6),
    }
    result = check_geometric_embeddability(J)
    assert result.checked is True
    assert not result.embeddable


# ---------------------------------------------------------------------------
# Integration with run_neutral_atom
# ---------------------------------------------------------------------------

def test_2d_embeddable_ring_is_natively_realizable():
    J = _ring_J(4, 5.0)
    ir = _make_ir(*[_zzt(i, j, v) for (i, j), v in J.items()], n=4)
    res = run_neutral_atom(ir)
    assert res.geometry.embeddable is True
    assert res.certificate.natively_realizable is True


def test_tetrahedron_flags_not_natively_realizable():
    J = _tetrahedron_J(5.0)
    ir = _make_ir(*[_zzt(i, j, v) for (i, j), v in J.items()], n=4)
    res = run_neutral_atom(ir)
    assert res.geometry.embeddable is False
    assert res.certificate.natively_realizable is False


def test_geometry_not_checked_for_small_n():
    ir = _make_ir(_zzt(0, 1, 1.0), n=2)
    assert run_neutral_atom(ir).geometry.checked is False


def test_negative_coupling_still_not_natively_realizable():
    ir = _make_ir(_zzt(0, 1, -1.0), n=2)
    assert run_neutral_atom(ir).certificate.natively_realizable is False
