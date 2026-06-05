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
"""Analog interface demo for LIMEN.

Shows the path from a compiled PhysicalEncoding to a HamiltonianIR and
illustrates what neutral-atom and photonic backend submission will look like
when the constructive universality theorem is implemented. All three substrate
paths are demonstrated: the HamiltonianIR is always produced; the hardware
submission stubs raise NotImplementedError and are caught gracefully.

Two problems are demonstrated:
- A trivial 2-variable QUBO (smallest meaningful instance)
- A 4-node path-graph Max-Cut (shows n_sites > 2)
"""

from limen import (
    compile_lexicographic,
    default_hardware_graph,
    from_qubo_dict,
)
from limen.analog.backends.neutral_atom import run_neutral_atom
from limen.analog.backends.photonic import run_photonic
from limen.analog.hamiltonian import SubstrateType, from_physical_encoding

# ── Problem definitions ───────────────────────────────────────────────

TRIVIAL_QUBO: dict[tuple[str, str], float] = {
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x0", "x1"):  2.0,
}

PATH_EDGES = [("A", "B"), ("B", "C"), ("C", "D")]


def build_max_cut_qubo(
    edges: list[tuple[str, str]],
) -> dict[tuple[str, str], float]:
    """Convert a Max-Cut instance to a QUBO dict."""
    qubo: dict[tuple[str, str], float] = {}
    for u, v in edges:
        qubo[(u, v)] = qubo.get((u, v), 0.0) + 1.0
        qubo[(u, u)] = qubo.get((u, u), 0.0) - 1.0
        qubo[(v, v)] = qubo.get((v, v), 0.0) - 1.0
    return qubo


# ── Report helpers ────────────────────────────────────────────────────

def _stub_status(fn, ir) -> str:
    """Call a backend stub and return a one-line status string."""
    try:
        fn(ir)
        return "Unexpectedly succeeded"
    except NotImplementedError as exc:
        first_sentence = str(exc).split(".")[0]
        return f"Not implemented — {first_sentence}"


def _report_problem(name: str, qubo: dict[tuple[str, str], float]) -> None:
    """Compile, map to HamiltonianIR, and print a full report for one problem."""
    W = 65

    graph = from_qubo_dict(qubo)
    encoding = compile_lexicographic(graph, default_hardware_graph(8))

    ir = from_physical_encoding(encoding)
    ir_na = from_physical_encoding(encoding, substrate=SubstrateType.NEUTRAL_ATOM)
    ir_ph = from_physical_encoding(encoding, substrate=SubstrateType.PHOTONIC)

    n_vars = len(graph.variables)
    linear_terms = sum(1 for t in ir.terms if len(t.operators) == 1)
    quadratic_terms = sum(1 for t in ir.terms if len(t.operators) == 2)

    print(f"── LIMEN Analog Demo — {name} {'─' * max(0, W - 24 - len(name))}")
    print(f"  Variables   : {n_vars}")
    print(f"  QUBO terms  : {len(encoding.qubo)}")
    print(f"  Chain str.  : {encoding.chain_strength:.4f}")
    print()
    print("  Hamiltonian IR (unspecified substrate)")
    print(f"    Sites     : {ir.n_sites}")
    print(f"    Terms     : {len(ir.terms)}")
    print(f"    Substrate : {ir.substrate.value}")
    print(f"    Status    : {ir.metadata['status']}")
    print()
    print("  Term breakdown:")
    print(f"    Linear (Z)       : {linear_terms}")
    print(f"    Quadratic (ZZ)   : {quadratic_terms}")
    print()
    print("  Neutral-atom submission:")
    print(f"    Status: {_stub_status(run_neutral_atom, ir_na)}")
    print()
    print("  Photonic submission:")
    print(f"    Status: {_stub_status(run_photonic, ir_ph)}")
    print()
    print(f"── End {'─' * (W - 7)}")
    print()


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    """Run the analog interface demo for both example problems."""
    _report_problem("trivial QUBO (2 vars)", TRIVIAL_QUBO)
    _report_problem("4-node path Max-Cut", build_max_cut_qubo(PATH_EDGES))


if __name__ == "__main__":
    main()
