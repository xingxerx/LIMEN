# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Structured Problem schema for the Discovery Loop (P1).

A Problem is the typed container above a raw QUBO: identity, class,
variables, constraints, objective, and an optional classical
``known_optimum`` used to score approximation ratios. Encoders (see
``limen.frontends.tsp``) turn a Problem into a QUBO dict that
``RouteRequest`` / ``run_route_request`` already accept — no second
QUBO stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from limen.formulation.constraints import Constraint


@dataclass
class Problem:
    """Optimization problem for Discovery Loop encode -> route -> score.

    Attributes:
        id: Stable instance identifier (e.g. ``"tsp/eil51/n=2"``).
        problem_class: Family tag (``"tsp"``, ``"qap"``, ...).
        variables: Binary variable names after encoding. Empty is allowed
            on structured inputs whose encoder invents the one-hot layout.
        constraints: Typed formulation constraints. May be empty when the
            encoder expands constraints into QUBO penalties itself (TSP).
        objective: Objective QUBO terms ``{(u, v): weight}``. Empty before
            encoding when the objective lives in ``metadata`` (coords).
        known_optimum: Optional classical optimum for approx_ratio
            (tour length for TSP).
        metadata: Encoder-specific payload (coordinates, distance matrix,
            source instance name, ...).
    """

    id: str
    problem_class: str
    variables: list[str] = field(default_factory=list)
    constraints: list[Constraint] = field(default_factory=list)
    objective: dict[tuple[str, str], float] = field(default_factory=dict)
    known_optimum: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Problem.id must be non-empty")
        if not self.problem_class:
            raise ValueError("Problem.problem_class must be non-empty")
