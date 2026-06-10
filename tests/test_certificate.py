# Copyright 2026 LIMEN Contributors. Apache 2.0.
"""Tests for limen.analog.certificate (Theorem 1 implementation)."""

from itertools import product

import pytest

from limen.analog.certificate import CompilationCertificate, certify_ising


def _brute_force_norm(dh, dJ, n):
    """Independent brute-force operator norm for verification."""
    best = 0.0
    for spins in product((1, -1), repeat=n):
        e = sum(v * spins[i] for i, v in dh.items())
        e += sum(v * spins[i] * spins[j] for (i, j), v in dJ.items())
        best = max(best, abs(e))
    return best


def test_exact_norm_matches_independent_brute_force():
    target_h = {0: 1.0, 1: -0.5}
    target_J = {(0, 1): 2.0, (1, 2): -1.0}
    compiled_h = {0: 1.2, 1: -0.5}
    compiled_J = {(0, 1): 1.8, (1, 2): -1.1}
    cert = certify_ising(target_h, target_J, compiled_h, compiled_J, n_sites=3)

    dh = {0: 0.2}
    dJ = {(0, 1): -0.2, (1, 2): -0.1}
    assert cert.operator_norm == pytest.approx(_brute_force_norm(dh, dJ, 3))


def test_operator_norm_bounded_by_l1():
    target_h = {0: 0.3, 1: 0.7, 2: -0.2}
    target_J = {(0, 1): 1.0, (0, 2): -0.4, (1, 2): 0.6}
    compiled_h = {0: 0.5, 1: 0.6, 2: 0.0}
    compiled_J = {(0, 1): 0.9, (0, 2): -0.6, (1, 2): 0.65}
    cert = certify_ising(target_h, target_J, compiled_h, compiled_J, n_sites=3)
    assert cert.operator_norm is not None
    assert cert.operator_norm <= cert.l1_bound + 1e-12


def test_zero_error_gives_zero_certificate():
    h = {0: 1.0, 1: 2.0}
    J = {(0, 1): -3.0}
    cert = certify_ising(h, J, dict(h), dict(J), n_sites=2)
    assert cert.l1_bound == pytest.approx(0.0)
    assert cert.operator_norm == pytest.approx(0.0)
    assert cert.max_linear_error == pytest.approx(0.0)
    assert cert.max_quadratic_error == pytest.approx(0.0)


def test_large_instance_skips_exact_norm():
    h = {i: 1.0 for i in range(25)}
    compiled_h = {i: 1.1 for i in range(25)}
    cert = certify_ising(h, {}, compiled_h, {}, n_sites=25)
    assert cert.operator_norm is None
    assert cert.l1_bound == pytest.approx(25 * 0.1)
    assert any("cutoff" in note for note in cert.notes)


def test_roundtrip_serialization():
    cert = certify_ising(
        {0: 1.0}, {(0, 1): 2.0}, {0: 1.5}, {(0, 1): 1.5},
        n_sites=2, natively_realizable=False,
    )
    restored = CompilationCertificate.from_dict(cert.to_dict())
    assert restored.l1_bound == pytest.approx(cert.l1_bound)
    assert restored.operator_norm == pytest.approx(cert.operator_norm)
    assert restored.n_sites == cert.n_sites
    assert restored.natively_realizable == cert.natively_realizable
    assert restored.notes == cert.notes


def test_not_realizable_adds_parity_note():
    cert = certify_ising({}, {(0, 1): -1.0}, {}, {(0, 1): 1.0},
                         n_sites=2, natively_realizable=False)
    assert cert.natively_realizable is False
    assert any("parity" in note.lower() for note in cert.notes)
    # |dJ| = 2 on a single pair: exact norm equals L1 bound here.
    assert cert.operator_norm == pytest.approx(2.0)
    assert cert.l1_bound == pytest.approx(2.0)
