# Copyright 2026 LIMEN Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the Phase 3 analog interface layer.

All tests run offline — no external hardware or SDK dependencies required.
"""

import pytest

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.analog.backends.classical_sim import IsingSimulationResult, run_ising_simulation
from limen.analog.backends.neutral_atom import NeutralAtomResult, run_neutral_atom
from limen.analog.backends.photonic import PhotonicResult, run_photonic
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


# ── Hamiltonian IR ────────────────────────────────────────────────────

def test_from_physical_encoding_returns_hamiltonian_ir():
    """from_physical_encoding returns a valid HamiltonianIR."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    assert isinstance(ir, HamiltonianIR)
    assert ir.n_sites == 2
    assert len(ir.terms) > 0
    assert ir.substrate == SubstrateType.UNSPECIFIED


def test_substrate_hint_preserved():
    """Substrate hint is stored on HamiltonianIR and its metadata."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    assert ir.substrate == SubstrateType.NEUTRAL_ATOM
    assert ir.metadata["substrate"] == "neutral_atom"


def test_hamiltonian_term_roundtrip():
    """HamiltonianTerm serialises and deserialises correctly."""
    term = HamiltonianTerm(coefficient=0.5, operators=[(0, "Z"), (1, "Z")])
    term2 = HamiltonianTerm.from_dict(term.to_dict())
    assert term2.coefficient == term.coefficient
    assert term2.operators == term.operators


def test_hamiltonian_ir_roundtrip():
    """HamiltonianIR serialises and deserialises correctly (no source_encoding)."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    ir2 = HamiltonianIR.from_dict(ir.to_dict())
    assert ir2.n_sites == ir.n_sites
    assert len(ir2.terms) == len(ir.terms)
    assert ir2.substrate == ir.substrate


def test_all_terms_have_valid_operators():
    """Every term produced by from_physical_encoding has valid operator strings."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    for term in ir.terms:
        assert term.coefficient != 0.0
        for site_idx, op_str in term.operators:
            assert isinstance(site_idx, int)
            assert op_str in ("Z", "X", "Y", "I")


# ── Classical simulation ──────────────────────────────────────────────

def test_run_ising_simulation_returns_result():
    """run_ising_simulation returns an IsingSimulationResult."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    result = run_ising_simulation(ir)
    assert isinstance(result, IsingSimulationResult)
    assert result.n_sites == 2
    assert result.available is True
    assert result.simulated is True


def test_run_ising_simulation_ground_state_trivial_qubo():
    """Classical simulation finds the correct ground state for the trivial QUBO.

    QUBO: x0^2 - x0 + x1^2 - x1 + 2*x0*x1
    Minimum energy is at (x0=1, x1=0) or (x0=0, x1=1) with energy -1.
    """
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    result = run_ising_simulation(ir)
    assert result.ground_state_energy < 0.0
    assert len(result.ground_state_assignment) == 2


def test_run_ising_simulation_too_many_sites():
    """run_ising_simulation raises ValueError when n_sites exceeds max_sites."""
    ir = HamiltonianIR(
        terms=[HamiltonianTerm(coefficient=1.0, operators=[(i, "Z")]) for i in range(5)],
        n_sites=5,
    )
    with pytest.raises(ValueError, match="exceeds max_sites"):
        run_ising_simulation(ir, max_sites=4)


def test_run_ising_simulation_excited_states():
    """run_ising_simulation returns at least one excited state."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    result = run_ising_simulation(ir)
    assert len(result.excited_states) > 0
    gs_e = result.ground_state_energy
    for e, _ in result.excited_states:
        assert e > gs_e


# ── Neutral-atom backend ──────────────────────────────────────────────

def test_run_neutral_atom_returns_result():
    """run_neutral_atom returns a NeutralAtomResult."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    result = run_neutral_atom(ir)
    assert isinstance(result, NeutralAtomResult)
    assert result.available is True
    assert result.simulated is True


def test_run_neutral_atom_positions():
    """run_neutral_atom returns one position per site."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    result = run_neutral_atom(ir)
    assert len(result.atom_positions) == ir.n_sites
    for x, y in result.atom_positions:
        assert isinstance(x, float)
        assert isinstance(y, float)


def test_run_neutral_atom_detunings():
    """run_neutral_atom returns one detuning per site."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    result = run_neutral_atom(ir)
    assert len(result.detunings) == ir.n_sites


def test_run_neutral_atom_includes_simulation():
    """run_neutral_atom includes a classical simulation for small instances."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    result = run_neutral_atom(ir)
    assert result.simulation is not None
    assert isinstance(result.simulation, IsingSimulationResult)


# ── Photonic backend ──────────────────────────────────────────────────

def test_run_photonic_returns_result():
    """run_photonic returns a PhotonicResult."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.PHOTONIC)
    result = run_photonic(ir)
    assert isinstance(result, PhotonicResult)
    assert result.available is True
    assert result.simulated is True


def test_run_photonic_adjacency_matrix_shape():
    """run_photonic returns an n×n adjacency matrix."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.PHOTONIC)
    result = run_photonic(ir)
    n = ir.n_sites
    assert len(result.adjacency_matrix) == n
    for row in result.adjacency_matrix:
        assert len(row) == n


def test_run_photonic_spectral_radius_valid():
    """GBS adjacency matrix spectral radius is < 1 (required for valid GBS)."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.PHOTONIC)
    result = run_photonic(ir)
    assert result.spectral_radius < 1.0


def test_run_photonic_squeezing_params():
    """run_photonic returns one squeezing parameter per mode."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.PHOTONIC)
    result = run_photonic(ir)
    assert len(result.squeezing_params) == ir.n_sites
    for r in result.squeezing_params:
        assert r >= 0.0


def test_run_photonic_includes_simulation():
    """run_photonic includes a classical simulation for small instances."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.PHOTONIC)
    result = run_photonic(ir)
    assert result.simulation is not None
    assert isinstance(result.simulation, IsingSimulationResult)
