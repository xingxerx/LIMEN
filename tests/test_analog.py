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
from limen.analog.backends.neutral_atom import run_neutral_atom
from limen.analog.backends.photonic import run_photonic
from limen.analog.hamiltonian import (
    HamiltonianIR,
    HamiltonianTerm,
    SubstrateType,
    from_physical_encoding,
)

TRIVIAL_QUBO: dict[tuple[str, str], float] = {
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x0", "x1"): 2.0,
}


def _make_encoding():
    graph = from_qubo_dict(TRIVIAL_QUBO)
    return compile_lexicographic(graph, default_hardware_graph(4))


# ── Test 0 ───────────────────────────────────────────────────────────

def test_from_physical_encoding_returns_hamiltonian_ir():
    """from_physical_encoding returns a valid HamiltonianIR."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    assert isinstance(ir, HamiltonianIR)
    assert ir.n_sites == 2
    assert len(ir.terms) > 0
    assert ir.substrate == SubstrateType.UNSPECIFIED


# ── Test 1 ───────────────────────────────────────────────────────────

def test_substrate_hint_preserved():
    """Substrate hint is stored on HamiltonianIR and its metadata."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    assert ir.substrate == SubstrateType.NEUTRAL_ATOM
    assert ir.metadata["substrate"] == "neutral_atom"


# ── Test 2 ───────────────────────────────────────────────────────────

def test_hamiltonian_term_roundtrip():
    """HamiltonianTerm serialises and deserialises correctly."""
    term = HamiltonianTerm(coefficient=0.5, operators=[(0, "Z"), (1, "Z")])
    term2 = HamiltonianTerm.from_dict(term.to_dict())
    assert term2.coefficient == term.coefficient
    assert term2.operators == term.operators


# ── Test 3 ───────────────────────────────────────────────────────────

def test_hamiltonian_ir_roundtrip():
    """HamiltonianIR serialises and deserialises correctly (no source_encoding)."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    ir2 = HamiltonianIR.from_dict(ir.to_dict())
    assert ir2.n_sites == ir.n_sites
    assert len(ir2.terms) == len(ir.terms)
    assert ir2.substrate == ir.substrate


# ── Test 4 ───────────────────────────────────────────────────────────

def test_run_neutral_atom_raises():
    """run_neutral_atom raises NotImplementedError mentioning universality."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    with pytest.raises(NotImplementedError, match="constructive universality"):
        run_neutral_atom(ir)


# ── Test 5 ───────────────────────────────────────────────────────────

def test_run_photonic_raises():
    """run_photonic raises NotImplementedError mentioning universality."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    with pytest.raises(NotImplementedError, match="constructive universality"):
        run_photonic(ir)


# ── Test 6 ───────────────────────────────────────────────────────────

def test_all_terms_have_valid_operators():
    """Every term produced by from_physical_encoding has valid operator strings."""
    encoding = _make_encoding()
    ir = from_physical_encoding(encoding)
    for term in ir.terms:
        assert term.coefficient != 0.0
        for site_idx, op_str in term.operators:
            assert isinstance(site_idx, int)
            assert op_str in ("Z", "X", "Y", "I")
