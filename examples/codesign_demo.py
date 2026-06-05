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
"""Co-design demo for LIMEN — closed-loop comparison with the open-loop pipeline.

Uses the same 5-node Max-Cut problem as examples/max_cut.py so the two scripts
are directly comparable: max_cut.py shows a single open-loop validate pass while
this script runs the Stackelberg co-design loop that iteratively adjusts chain
strength to maximise the calibration margin κ.

If the limen_core Rust extension has not been built (maturin develop), the
co-design section is skipped and the script exits cleanly with code 0.
"""

from limen import (
    compile_lexicographic,
    default_hardware_graph,
    from_qubo_dict,
    validate,
)

# ── Problem definition ────────────────────────────────────────────────

NODES = ["A", "B", "C", "D", "E"]
EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E")]


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


# ── Pipeline ──────────────────────────────────────────────────────────

def main() -> None:
    W = 65

    # 1. Build QUBO and compile base encoding.
    qubo = build_max_cut_qubo(EDGES)
    graph = from_qubo_dict(qubo)
    encoding = compile_lexicographic(graph, default_hardware_graph(8))

    # 2. Baseline validation (open-loop).
    baseline = validate(encoding, runs=1000, seed=42)

    # 3. Co-design loop.
    codesign_available = True
    cd_result = None
    try:
        from limen.codesign.solver import run_codesign

        cd_result = run_codesign(
            encoding,
            target_kappa=0.85,
            max_iterations=20,
            runs_per_iteration=500,
            seed=42,
        )
    except ImportError:
        codesign_available = False

    # 4. Report.
    print(f"── LIMEN Co-Design Demo {'─' * (W - 24)}")
    print(f"  Problem : {len(NODES)} nodes, {len(EDGES)} edges (Max-Cut)")
    print()
    print("  Baseline (open-loop)")
    print(f"    Chain strength : {encoding.chain_strength:.4f}")
    print(f"    Confidence     : {baseline.confidence * 100:.1f}%")
    print(f"    Best energy    : {baseline.best_energy:.4f}")
    print()

    if codesign_available and cd_result is not None:
        initial_cs = cd_result.chain_strength_history[0]
        final_cs = cd_result.encoding.chain_strength
        final_confidence = (
            cd_result.confidence_history[-1]
            if cd_result.confidence_history
            else baseline.confidence
        )

        print("  Co-Design Loop")
        print(f"    Iterations     : {cd_result.iterations}")
        print(f"    Converged      : {cd_result.converged}")
        print(f"    Final κ        : {cd_result.kappa:.4f}")
        print(f"    κ std dev      : {cd_result.kappa_std:.4f}")
        print(f"    Chain strength : {initial_cs:.4f} → {final_cs:.4f}")
        print(
            f"    Confidence     : {baseline.confidence * 100:.1f}%"
            f" → {final_confidence * 100:.1f}%"
        )
        print()
        print("  Chain strength history:")
        for cs in cd_result.chain_strength_history:
            print(f"    {cs:.4f}")
    else:
        print("  Co-Design Loop  [limen_core not built — skipped]")

    print()
    print(f"── End {'─' * (W - 7)}")


if __name__ == "__main__":
    main()
