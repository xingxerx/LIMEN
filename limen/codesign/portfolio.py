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
"""Portfolio compilation layer for LIMEN.

Runs the Stackelberg co-design loop across multiple named backend slots,
ranks the results by calibration margin κ, and emits switching conditions
so the caller can select the best-performing backend at runtime.
"""

from dataclasses import dataclass, field
from typing import Any

from limen.core.compiler import PhysicalEncoding
from limen.codesign.solver import CoDesignResult, run_codesign


@dataclass
class SwitchingCondition:
    """A ranked backend candidate from a portfolio compilation run.

    Attributes:
        backend: Name of the backend slot (e.g. ``"dwave"``, ``"qiskit_exact"``).
        encoding: The best PhysicalEncoding produced by co-design for this slot.
        kappa: Calibration margin κ achieved after co-design.
        condition: Human-readable switching condition string.
        priority: Rank within the portfolio (0 = highest κ, preferred).
    """

    backend: str
    encoding: PhysicalEncoding
    kappa: float
    condition: str
    priority: int


@dataclass
class PortfolioResult:
    """The result of a portfolio compilation run across multiple backends.

    Attributes:
        candidates: All backend candidates sorted by descending κ (best first).
        best_backend: Name of the highest-κ backend.
        best_kappa: Calibration margin of the best backend.
        metadata: Arbitrary annotations from the portfolio run.
    """

    candidates: list[SwitchingCondition]
    best_backend: str
    best_kappa: float
    metadata: dict[str, Any] = field(default_factory=dict)


def compile_portfolio(
    encoding: PhysicalEncoding,
    backends: list[str],
    target_kappa: float = 0.85,
    max_iterations: int = 50,
    runs_per_iteration: int = 500,
    seed: int = 42,
) -> PortfolioResult:
    """Run co-design across multiple backends and rank them by calibration margin.

    Each backend in ``backends`` is treated as a named slot. The same
    PhysicalEncoding is optimised independently for each slot using
    ``run_codesign``, and the results are sorted by κ descending so the
    caller can select the best candidate or implement runtime switching.

    Args:
        encoding: Base PhysicalEncoding to optimise for each backend.
        backends: List of backend slot names (e.g. ``["dwave", "qiskit_exact"]``).
        target_kappa: Co-design convergence target passed to each solver run.
        max_iterations: Iteration cap for each co-design run.
        runs_per_iteration: Validation runs per iteration.
        seed: Base RNG seed; each backend uses ``seed + idx`` for independence.

    Returns:
        A PortfolioResult with candidates sorted by descending κ.
    """
    codesign_results: list[tuple[str, CoDesignResult]] = []
    for idx, backend in enumerate(backends):
        result = run_codesign(
            encoding,
            target_kappa=target_kappa,
            max_iterations=max_iterations,
            runs_per_iteration=runs_per_iteration,
            seed=seed + idx * 1000,
        )
        codesign_results.append((backend, result))

    codesign_results.sort(key=lambda x: x[1].kappa, reverse=True)

    candidates = [
        SwitchingCondition(
            backend=backend,
            encoding=result.encoding,
            kappa=result.kappa,
            condition=f"kappa >= {result.kappa:.3f}",
            priority=priority,
        )
        for priority, (backend, result) in enumerate(codesign_results)
    ]

    return PortfolioResult(
        candidates=candidates,
        best_backend=candidates[0].backend,
        best_kappa=candidates[0].kappa,
        metadata={
            "backends": backends,
            "target_kappa": target_kappa,
        },
    )
