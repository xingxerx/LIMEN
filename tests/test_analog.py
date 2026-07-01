# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Tests for the Phase 3 analog interface layer.

All tests run offline — no external hardware or SDK dependencies required.
"""

import pytest

import math

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.analog.backends.classical_sim import IsingSimulationResult, run_ising_simulation
from limen.analog.backends.neutral_atom import (
    NeutralAtomResult,
    _C6_MHZ_UM6,
    _jacobi_eigenvalues,
    check_geometric_embeddability,
    run_neutral_atom,
)
from limen.analog.backends.photonic import PhotonicResult, run_photonic
from limen.analog.delta_model import DeviceDrift, HardwareDeltaModel
from limen.analog.hamiltonian import (
    HamiltonianIR,
    HamiltonianTerm,
    SubstrateType,
    from_physical_encoding,
)

TRIVIAL_QUBO: dict[tuple[str, str], float] = {
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x0", "x1"):  2.0,
}


def _make_encoding():
    graph = from_qubo_dict(TRIVIAL_QUBO)
    return compile_lexicographic(graph, default_hardware_graph(4))


def test_from_physical_encoding_returns_hamiltonian_ir():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    assert isinstance(ir, HamiltonianIR)
    assert ir.n_sites == 2
    assert len(ir.terms) > 0
    assert ir.substrate == SubstrateType.UNSPECIFIED


def test_substrate_hint_preserved():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    assert ir.substrate == SubstrateType.NEUTRAL_ATOM
    assert ir.metadata["substrate"] == "neutral_atom"


def test_hamiltonian_term_roundtrip():
    term = HamiltonianTerm(coefficient=0.5, operators=[(0, "Z"), (1, "Z")])
    term2 = HamiltonianTerm.from_dict(term.to_dict())
    assert term2.coefficient == term.coefficient
    assert term2.operators == term.operators


def test_hamiltonian_ir_roundtrip():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    ir2 = HamiltonianIR.from_dict(ir.to_dict())
    assert ir2.n_sites == ir.n_sites
    assert len(ir2.terms) == len(ir.terms)
    assert ir2.substrate == ir.substrate


def test_all_terms_have_valid_operators():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    for term in ir.terms:
        assert term.coefficient != 0.0
        for site_idx, op_str in term.operators:
            assert isinstance(site_idx, int)
            assert op_str in ("Z", "X", "Y", "I")


def test_run_ising_simulation_returns_result():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    result = run_ising_simulation(ir)
    assert isinstance(result, IsingSimulationResult)
    assert result.n_sites == 2
    assert result.available is True
    assert result.simulated is True


def test_run_ising_simulation_ground_state_trivial_qubo():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    result = run_ising_simulation(ir)
    assert result.ground_state_energy < 0.0
    assert len(result.ground_state_assignment) == 2


def test_run_ising_simulation_too_many_sites():
    ir = HamiltonianIR(
        terms=[HamiltonianTerm(coefficient=1.0, operators=[(i, "Z")]) for i in range(5)],
        n_sites=5,
    )
    with pytest.raises(ValueError, match="exceeds max_sites"):
        run_ising_simulation(ir, max_sites=4)


def test_run_ising_simulation_excited_states():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    result = run_ising_simulation(ir)
    assert len(result.excited_states) > 0
    gs_e = result.ground_state_energy
    for e, _ in result.excited_states:
        assert e > gs_e


def test_run_neutral_atom_returns_result():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    result = run_neutral_atom(ir)
    assert isinstance(result, NeutralAtomResult)
    assert result.available is True
    assert result.simulated is True


def test_run_neutral_atom_positions():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    result = run_neutral_atom(ir)
    assert len(result.atom_positions) == ir.n_sites
    for x, y in result.atom_positions:
        assert isinstance(x, float)
        assert isinstance(y, float)


def test_run_neutral_atom_detunings():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    result = run_neutral_atom(ir)
    assert len(result.detunings) == ir.n_sites


def test_run_neutral_atom_includes_simulation():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    result = run_neutral_atom(ir)
    assert result.simulation is not None
    assert isinstance(result.simulation, IsingSimulationResult)


def test_run_neutral_atom_applies_rabi_correction():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    drift = DeviceDrift(global_rabi_error=0.25)
    model = HardwareDeltaModel(
        device_id="dev-rabi-neutral-atom",
        substrate=SubstrateType.NEUTRAL_ATOM,
        drift=drift,
        n_sites=ir.n_sites,
    )
    baseline = run_neutral_atom(ir)
    corrected = run_neutral_atom(ir, delta_model=model)
    assert baseline.rabi_frequency == pytest.approx(1.0)
    assert corrected.rabi_frequency == pytest.approx(1.0 / 1.25)
    assert corrected.metadata["rabi_frequency_mhz"] == pytest.approx(corrected.rabi_frequency)


def test_run_neutral_atom_no_delta_model_uses_default_rabi():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    result = run_neutral_atom(ir)
    assert result.rabi_frequency == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Geometric embeddability check (Theorem 2 geometry condition) and its
# pure-Python Jacobi eigenvalue fallback for environments without numpy.
# ---------------------------------------------------------------------------

def test_jacobi_eigenvalues_matches_known_2x2():
    # [[2, 1], [1, 2]] has eigenvalues 1 and 3.
    eigvals = sorted(_jacobi_eigenvalues([[2.0, 1.0], [1.0, 2.0]]))
    assert eigvals == pytest.approx([1.0, 3.0])


def test_jacobi_eigenvalues_matches_diagonal_matrix():
    eigvals = sorted(_jacobi_eigenvalues([[5.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 3.0]]))
    assert eigvals == pytest.approx([-2.0, 3.0, 5.0])


def test_check_geometric_embeddability_no_positive_couplings():
    result = check_geometric_embeddability({})
    assert result.checked is False
    assert result.embeddable is True


def test_check_geometric_embeddability_trivial_triangle():
    # 3 constrained sites always lie in a plane.
    target_J = {(0, 1): 1.0, (1, 2): 1.0, (0, 2): 1.0}
    result = check_geometric_embeddability(target_J)
    assert result.checked is False
    assert result.embeddable is True
    assert result.n_constrained_sites == 3


def _square_positions() -> list[tuple[float, float]]:
    return [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _target_j_from_positions(
    positions: list[tuple[float, float]], pairs: list[tuple[int, int]], c6: float = _C6_MHZ_UM6
) -> dict[tuple[int, int], float]:
    """Build target_J so the required distance for each pair matches the actual
    Euclidean distance between the given 2-D positions exactly."""
    target_j: dict[tuple[int, int], float] = {}
    for i, j in pairs:
        dx = positions[i][0] - positions[j][0]
        dy = positions[i][1] - positions[j][1]
        d = math.hypot(dx, dy)
        target_j[(i, j)] = c6 / (4.0 * d**6)
    return target_j


def test_check_geometric_embeddability_flat_square_is_embeddable():
    positions = _square_positions()
    all_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    target_j = _target_j_from_positions(positions, all_pairs)
    result = check_geometric_embeddability(target_j)
    assert result.checked is True
    assert result.psd_satisfied is True
    assert result.embeddable is True
    assert result.gram_rank is not None and result.gram_rank <= 2


def test_check_geometric_embeddability_regular_tetrahedron_is_not_embeddable():
    # 4 sites with all six pairwise distances equal require a 3-D embedding
    # (regular tetrahedron) -- not achievable in a 2-D atom array.
    target_j = {
        (0, 1): 1.0, (0, 2): 1.0, (0, 3): 1.0,
        (1, 2): 1.0, (1, 3): 1.0, (2, 3): 1.0,
    }
    result = check_geometric_embeddability(target_j)
    assert result.checked is True
    assert result.embeddable is False
    assert result.gram_rank == 3


def test_check_geometric_embeddability_pure_python_matches_numpy():
    positions = _square_positions()
    all_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    target_j = _target_j_from_positions(positions, all_pairs)

    numpy_result = check_geometric_embeddability(target_j)

    import builtins
    real_import = builtins.__import__

    def _blocked_numpy_import(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("numpy blocked for this test")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = _blocked_numpy_import
    try:
        pure_python_result = check_geometric_embeddability(target_j)
    finally:
        builtins.__import__ = real_import

    assert pure_python_result.checked is True
    assert pure_python_result.embeddable == numpy_result.embeddable
    assert pure_python_result.psd_satisfied == numpy_result.psd_satisfied
    assert pure_python_result.gram_rank == numpy_result.gram_rank
    assert pure_python_result.gram_min_eigenvalue == pytest.approx(
        numpy_result.gram_min_eigenvalue, abs=1e-6
    )


def test_run_photonic_returns_result():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.PHOTONIC)
    result = run_photonic(ir)
    assert isinstance(result, PhotonicResult)
    assert result.available is True
    assert result.simulated is True


def test_run_photonic_adjacency_matrix_shape():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.PHOTONIC)
    result = run_photonic(ir)
    n = ir.n_sites
    assert len(result.adjacency_matrix) == n
    for row in result.adjacency_matrix:
        assert len(row) == n


def test_run_photonic_spectral_radius_valid():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.PHOTONIC)
    result = run_photonic(ir)
    assert result.spectral_radius < 1.0


def test_run_photonic_squeezing_params():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.PHOTONIC)
    result = run_photonic(ir)
    assert len(result.squeezing_params) == ir.n_sites
    for r in result.squeezing_params:
        assert r >= 0.0


def test_run_photonic_includes_simulation():
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.PHOTONIC)
    result = run_photonic(ir)
    assert result.simulation is not None
    assert isinstance(result.simulation, IsingSimulationResult)
