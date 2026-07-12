# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""QUBO auto-formulation for LIMEN: typed constraints in, LogicalGraph out.

This is the structured-input tier of constraint-based QUBO formulation
(see docs/ROADMAP.md, Phase 7): a caller declares named constraints over
binary variables (Equality, Inequality, OneHot, AtMostK, AtLeastK,
AllDifferent) instead of hand-deriving penalty terms, and
ConstraintCompiler turns them into QUBO penalty terms merged with an
objective. Natural-language input is deliberately not in scope here — an
NL layer would sit on top of this module, translating text into the same
typed constraints, so its ambiguity never enters the certified pipeline
below this layer.
"""

from limen.formulation.compiler import ConstraintCompiler, expand_equality_penalty
from limen.formulation.constraints import (
    AllDifferent,
    AtLeastK,
    AtMostK,
    Constraint,
    Equality,
    Inequality,
    OneHot,
)
from limen.formulation.penalty import default_penalty_weight

__all__ = [
    "ConstraintCompiler",
    "expand_equality_penalty",
    "default_penalty_weight",
    "Constraint",
    "Equality",
    "Inequality",
    "OneHot",
    "AtMostK",
    "AtLeastK",
    "AllDifferent",
]
