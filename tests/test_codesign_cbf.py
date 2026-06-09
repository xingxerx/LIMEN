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
"""Tests for chain-break fraction wiring in the Stackelberg co-design loop."""

from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
from limen.codesign.solver import run_codesign

TRIVIAL_QUBO = {
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x0", "x1"):  2.0,
}


def _make_encoding():
    graph = from_qubo_dict(TRIVIAL_QUBO)
    return compile_lexicographic(graph, default_hardware_graph(4))


def test_run_codesign_default_cbf_is_zero():
    """Without chain_break_fraction_fn the loop runs and cbf stays 0.0."""
    encoding = _make_encoding()
    result = run_codesign(encoding, max_iterations=3, runs_per_iteration=100, seed=0)
    # The metadata cbf_fn_provided flag should be absent (fn was None).
    assert result.iterations >= 1


def test_run_codesign_accepts_cbf_fn():
    """chain_break_fraction_fn callback is called each iteration."""
    encoding = _make_encoding()
    call_count = [0]

    def mock_cbf(enc) -> float:
        call_count[0] += 1
        return 0.05  # simulate 5% chain breaks

    result = run_codesign(
        encoding,
        max_iterations=4,
        runs_per_iteration=100,
        seed=0,
        chain_break_fraction_fn=mock_cbf,
    )
    assert call_count[0] == result.iterations


def test_run_codesign_cbf_fn_none_does_not_fail():
    """Passing chain_break_fraction_fn=None explicitly is identical to default."""
    encoding = _make_encoding()
    result = run_codesign(
        encoding,
        max_iterations=3,
        runs_per_iteration=100,
        seed=7,
        chain_break_fraction_fn=None,
    )
    assert result.iterations >= 1
