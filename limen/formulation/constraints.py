# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Typed constraint contract for limen.formulation.

This is the "structured input" tier of QUBO auto-formulation: constraints
are named, typed dataclasses over binary variables, not raw penalty terms
and not free-text. A ConstraintCompiler (see compiler.py) turns these into
QUBO penalty terms and merges them with an objective into a LogicalGraph.

Every constraint type here reduces to one of two primitives internally:
    - a linear equality (Sum c_i x_i == rhs), penalized as its square, or
    - a linear inequality (Sum c_i x_i <= rhs / >= rhs), penalized via a
      binary-encoded slack variable that turns it into an equality.
Both primitives are implemented once in compiler.py; the dataclasses below
just describe *what* is being constrained.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Equality:
    """Sum of coeffs[v] * v over the given variables must equal rhs.

    Attributes:
        coeffs: Mapping of variable name to its coefficient in the sum.
        rhs: The required value of the sum.
        name: Optional label, used in error messages and metadata.
    """

    coeffs: dict[str, float]
    rhs: float
    name: str | None = None


@dataclass
class Inequality:
    """Sum of coeffs[v] * v over the given variables must satisfy <= or >= rhs.

    Realized via a binary-encoded slack variable, so it costs
    ceil(log2(bound + 1)) auxiliary variables where bound is the maximum
    possible slack (see compiler.py for the derivation). Coefficients and
    rhs should be integers for the slack encoding to be exact; non-integer
    inputs raise at compile time.

    Attributes:
        coeffs: Mapping of variable name to its coefficient in the sum.
        rhs: The bound on the sum.
        sense: Either "<=" or ">=".
        name: Optional label, used in error messages and metadata.
    """

    coeffs: dict[str, float]
    rhs: float
    sense: str = "<="
    name: str | None = None

    def __post_init__(self) -> None:
        if self.sense not in ("<=", ">="):
            raise ValueError(f"Inequality.sense must be '<=' or '>=', got {self.sense!r}")


@dataclass
class OneHot:
    """Exactly one variable in the group must be 1.

    A named special case of Equality (coeffs all 1, rhs 1) for the common
    "pick exactly one option" pattern (e.g. one city per tour position).

    Attributes:
        variables: The group of mutually-exclusive variables.
        name: Optional label, used in error messages and metadata.
    """

    variables: list[str]
    name: str | None = None


@dataclass
class AtMostK:
    """At most k of the given variables may be 1.

    Attributes:
        variables: The group of variables.
        k: The maximum number that may be 1.
        name: Optional label, used in error messages and metadata.
    """

    variables: list[str]
    k: int
    name: str | None = None


@dataclass
class AtLeastK:
    """At least k of the given variables must be 1.

    Attributes:
        variables: The group of variables.
        k: The minimum number that must be 1.
        name: Optional label, used in error messages and metadata.
    """

    variables: list[str]
    k: int
    name: str | None = None


@dataclass
class AllDifferent:
    """Each item in a set must be assigned to a distinct slot.

    Expects one binary variable per (item, slot) pair, named by
    ``var_name(item, slot)`` (default: "item__slot"). Compiles to two
    families of constraints: each item is assigned exactly one slot
    (OneHot per item/row), and each slot is used by at most one item
    (AtMostK(k=1) per slot/column) — the standard one-hot-assignment QUBO
    pattern used for permutation-shaped problems (e.g. VRP-style routing).

    Attributes:
        items: The set of items to assign.
        slots: The set of slots items can be assigned to.
        var_name: Function mapping (item, slot) to the binary variable name
            representing "item is assigned to slot". Defaults to
            f"{item}__{slot}".
        name: Optional label, used in error messages and metadata.
    """

    items: list[str]
    slots: list[str]
    var_name: "callable[[str, str], str]" = field(
        default=lambda item, slot: f"{item}__{slot}"
    )
    name: str | None = None


Constraint = Equality | Inequality | OneHot | AtMostK | AtLeastK | AllDifferent
