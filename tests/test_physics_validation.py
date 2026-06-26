"""Physics-validation tests: LIMEN must *discover* results, not confirm known ones.

Unlike tests/test_pipeline.py (fixed integration smoke tests with baked-in
answers), every test here checks a property of the underlying physics across
randomized or swept inputs: QAOA finding optima it was never told, surface-code
logical-error suppression scaling with the physical error rate, the distance-3
correction boundary, and BB84 eavesdropping detection.
"""

import random

import pytest

from limen.ecc.certificate import certify_logical_qubit
from limen.ecc.decoder import LookupDecoder
from limen.ecc.encoder import run_logical_roundtrip, verify_corrects_all_weight_one
from limen.ecc.surface_code import build_surface_code
from limen.pipeline import run_pipeline


def _random_qubo(rng: random.Random, n: int) -> dict[tuple[str, str], float]:
    """Random QUBO over n variables with all linear terms and dense couplings."""
    names = [f"x{i}" for i in range(n)]
    qubo: dict[tuple[str, str], float] = {
        (names[i], names[i]): rng.uniform(-2.0, 2.0) for i in range(n)
    }
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.8:
                qubo[(names[i], names[j])] = rng.uniform(-2.0, 2.0)
    return qubo


# ---------------------------------------------------------------------------
# 1. Random QUBO vs. brute force - QAOA discovers optima, never beats them
# ---------------------------------------------------------------------------

def test_qaoa_never_beats_brute_force():
    """The recovered energy can never be below the exact global minimum."""
    rng = random.Random(7)
    for _ in range(40):
        qubo = _random_qubo(rng, rng.randint(2, 4))
        cert = run_pipeline(qubo, qaoa_layers=2, grid_size=16, encode_logical=False)
        assert cert.classical_energy is not None
        assert cert.energy >= cert.classical_energy - 1e-9


def test_qaoa_solution_energy_is_consistent():
    """cert.energy must equal the energy of cert.solution recomputed from the QUBO."""
    rng = random.Random(11)
    for _ in range(20):
        qubo = _random_qubo(rng, 3)
        cert = run_pipeline(qubo, qaoa_layers=2, grid_size=16, encode_logical=False)
        recomputed = sum(
            w * cert.solution[i] * cert.solution[j] for (i, j), w in qubo.items()
        )
        assert recomputed == pytest.approx(cert.energy)


def test_qaoa_discovers_two_variable_optima():
    """On 2-variable problems, p=2 QAOA grid search must find every optimum."""
    rng = random.Random(23)
    for _ in range(30):
        qubo = _random_qubo(rng, 2)
        cert = run_pipeline(qubo, qaoa_layers=2, grid_size=24, encode_logical=False)
        assert cert.is_optimal, f"failed to find optimum for {qubo}"
        assert 0.0 < cert.success_probability <= 1.0 + 1e-9


def test_qaoa_finds_most_optima_three_to_four_vars():
    """Across larger random problems QAOA should still find most exact optima."""
    rng = random.Random(31)
    optimal = 0
    trials = 30
    for _ in range(trials):
        qubo = _random_qubo(rng, rng.randint(3, 4))
        cert = run_pipeline(qubo, qaoa_layers=2, grid_size=18, encode_logical=False)
        optimal += int(bool(cert.is_optimal))
    assert optimal / trials >= 0.6


# ---------------------------------------------------------------------------
# 2. Logical-error-rate sweep - the surface-code math is real
# ---------------------------------------------------------------------------

def test_logical_error_rate_monotonic_in_physical_rate():
    """Higher physical error rate must give a higher logical error rate."""
    rates = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
    logical = [
        run_pipeline({("a", "a"): -1.0}, physical_error_rate=p).logical_error_rate
        for p in rates
    ]
    assert all(b > a for a, b in zip(logical, logical[1:]))


def test_distance_three_suppresses_errors_super_linearly():
    """A real distance-3 code suppresses errors: logical << physical, and the
    suppression ratio improves as the physical rate shrinks (quadratic-leading)."""
    patch = build_surface_code(3)
    decoder = LookupDecoder(patch)
    small, big = 0.001, 0.01
    low = certify_logical_qubit(patch, decoder, small).logical_error_rate
    high = certify_logical_qubit(patch, decoder, big).logical_error_rate
    assert low < small and high < big
    # Leading-order p^2 scaling: a 10x drop in p gives a >10x drop in logical rate.
    assert low / high < 0.1


# ---------------------------------------------------------------------------
# 3. Code-distance boundary - corrects all weight-1, fails some weight-2
# ---------------------------------------------------------------------------

def test_corrects_every_weight_one_error():
    patch = build_surface_code(3)
    decoder = LookupDecoder(patch)
    assert verify_corrects_all_weight_one(patch, decoder)


def test_some_weight_two_errors_are_uncorrectable():
    """Distance 3 cannot correct all weight-2 errors; at least one must survive
    as a logical error, proving the d=3 boundary is genuinely enforced."""
    patch = build_surface_code(3)
    decoder = LookupDecoder(patch)
    n = len(patch.data_qubits)
    logical_failures = [
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if run_logical_roundtrip(patch, decoder, [i, j]).logical_error
    ]
    assert logical_failures, "expected at least one uncorrectable weight-2 error"


# ---------------------------------------------------------------------------
# 4. BB84 eavesdropping detection (requires qiskit)
# ---------------------------------------------------------------------------

def test_eavesdropping_raises_qber_and_aborts():
    pytest.importorskip("qiskit", reason="qiskit not installed")
    pytest.importorskip("qiskit_aer", reason="qiskit-aer not installed")
    from limen.communication.channel import QuantumChannel

    clean = QuantumChannel(backend_name="statevector", seed=99).qkd_bb84(
        key_length=200, eavesdrop_rate=0.0
    )
    tapped = QuantumChannel(backend_name="statevector", seed=99).qkd_bb84(
        key_length=200, eavesdrop_rate=1.0
    )
    assert clean.qber == 0.0 and clean.secure and clean.shared_key is not None
    assert tapped.qber > clean.qber
    assert tapped.qber > 0.11 and not tapped.secure and tapped.shared_key is None
