"""Logical graph intermediate representation for LIMEN."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Variable:
    """A variable in the logical graph.

    Attributes:
        name: Unique identifier for the variable.
        domain: The variable's domain. Currently only 'binary' is supported.
    """

    name: str
    domain: str = "binary"

    def to_dict(self) -> dict[str, Any]:
        """Serialize this variable to a plain dict."""
        return {"name": self.name, "domain": self.domain}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Variable":
        """Deserialize a Variable from a plain dict."""
        return cls(name=d["name"], domain=d["domain"])


@dataclass
class Interaction:
    """A weighted interaction between two variables.

    An interaction where i == j represents a linear (single-variable) term.

    Attributes:
        i: Name of the first variable.
        j: Name of the second variable (may equal i for linear terms).
        weight: Coefficient of this interaction term.
    """

    i: str
    j: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize this interaction to a plain dict."""
        return {"i": self.i, "j": self.j, "weight": self.weight}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Interaction":
        """Deserialize an Interaction from a plain dict."""
        return cls(i=d["i"], j=d["j"], weight=float(d["weight"]))


@dataclass
class LogicalGraph:
    """A logical graph composed of binary variables and their interactions.

    The graph represents an objective function as a sum of weighted
    interaction terms between variables.

    Attributes:
        variables: Ordered list of variables in the graph.
        interactions: List of pairwise (or self) interaction terms.
        metadata: Arbitrary key/value annotations for the graph.
    """

    variables: list[Variable] = field(default_factory=list)
    interactions: list[Interaction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this graph to a plain Python dict.

        Returns:
            A dict with keys 'variables', 'interactions', and 'metadata'.
        """
        return {
            "variables": [v.to_dict() for v in self.variables],
            "interactions": [ix.to_dict() for ix in self.interactions],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LogicalGraph":
        """Deserialize a LogicalGraph from a plain Python dict.

        Args:
            d: Dict previously produced by to_dict().

        Returns:
            A new LogicalGraph instance.
        """
        return cls(
            variables=[Variable.from_dict(v) for v in d.get("variables", [])],
            interactions=[Interaction.from_dict(ix) for ix in d.get("interactions", [])],
            metadata=dict(d.get("metadata", {})),
        )

    def validate(self) -> list[str]:
        """Check the graph for structural errors.

        Checks performed:
            - Every variable name referenced in an interaction exists in
              the variables list.
            - No two interactions share the same (i, j) pair.

        Returns:
            A list of error strings. An empty list means the graph is valid.
        """
        errors: list[str] = []
        known = {v.name for v in self.variables}

        seen: set[tuple[str, str]] = set()
        for idx, ix in enumerate(self.interactions):
            if ix.i not in known:
                errors.append(
                    f"Interaction[{idx}]: unknown variable '{ix.i}'"
                )
            if ix.j not in known:
                errors.append(
                    f"Interaction[{idx}]: unknown variable '{ix.j}'"
                )
            key = (min(ix.i, ix.j), max(ix.i, ix.j))
            if key in seen:
                errors.append(
                    f"Interaction[{idx}]: duplicate interaction ({ix.i!r}, {ix.j!r})"
                )
            else:
                seen.add(key)

        return errors
