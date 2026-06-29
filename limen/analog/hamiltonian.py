# Copyright (C) 2026 Jemone McCubbin / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Hamiltonian IR — hardware-agnostic intermediate representation for analog substrates.

Defines the boundary between the LIMEN compiler and analog hardware backends.
The constructive universality theorem required to guarantee correct compilation
onto neutral-atom, photonic, or BEC substrates is pending research. This module
defines the interface contract that theorem will satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from limen.core.compiler import PhysicalEncoding


class SubstrateType(Enum):
    """Supported analog substrate types.

    Attributes:
        NEUTRAL_ATOM: Neutral-atom arrays (e.g. QuEra Aquila, Pasqal).
        PHOTONIC: Continuous-variable photonic processors.
        BEC: Bose-Einstein condensate analog simulators.
        UNSPECIFIED: Substrate not yet determined.
    """

    NEUTRAL_ATOM = "neutral_atom"
    PHOTONIC = "photonic"
    BEC = "bec"
    UNSPECIFIED = "unspecified"


@dataclass
class HamiltonianTerm:
    """A single term in a Hamiltonian expressed as a tensor product of operators.

    Attributes:
        coefficient: Real-valued coupling strength.
        operators: List of (site_index, operator_string) pairs.
            operator_string is substrate-dependent (e.g. "Z", "X",
            "sigma_plus", "n" for number operator).
        metadata: Arbitrary term annotations.
    """

    coefficient: float
    operators: list[tuple[int, str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain Python dict."""
        return {
            "coefficient": self.coefficient,
            "operators": list(self.operators),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HamiltonianTerm:
        """Deserialize from a plain Python dict."""
        return cls(
            coefficient=float(d["coefficient"]),
            operators=[tuple(op) for op in d["operators"]],
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class HamiltonianIR:
    """Hardware-agnostic Hamiltonian intermediate representation.

    Stores an optimization problem as a sum of HamiltonianTerms.
    This IR is the boundary between the LIMEN compiler and analog
    substrate backends. It is intentionally substrate-agnostic —
    the backend is responsible for mapping operators to physical
    interactions.

    Attributes:
        terms: List of Hamiltonian terms summing to the full objective.
        n_sites: Number of physical sites (qubits, atoms, modes).
        substrate: Target substrate type (may be UNSPECIFIED).
        source_encoding: The PhysicalEncoding this IR was derived from,
            or None if constructed directly.
        metadata: Arbitrary annotations.

    Status:
        PLACEHOLDER. The constructive universality theorem required to
        guarantee correct compilation onto analog substrates is pending
        research. This IR defines the interface contract; implementations
        of the compilation theorem will populate it.
    """

    terms: list[HamiltonianTerm] = field(default_factory=list)
    n_sites: int = 0
    substrate: SubstrateType = SubstrateType.UNSPECIFIED
    source_encoding: PhysicalEncoding | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain Python dict."""
        return {
            "terms": [t.to_dict() for t in self.terms],
            "n_sites": self.n_sites,
            "substrate": self.substrate.value,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HamiltonianIR:
        """Deserialize from a plain Python dict."""
        return cls(
            terms=[HamiltonianTerm.from_dict(t) for t in d.get("terms", [])],
            n_sites=int(d.get("n_sites", 0)),
            substrate=SubstrateType(d.get("substrate", "unspecified")),
            metadata=dict(d.get("metadata", {})),
        )


def from_physical_encoding(
    encoding: PhysicalEncoding,
    substrate: SubstrateType = SubstrateType.UNSPECIFIED,
) -> HamiltonianIR:
    """Convert a PhysicalEncoding to a HamiltonianIR via direct Ising mapping.

    Applies the standard QUBO → Ising substitution x_i = (1 + σ_i^z) / 2
    to produce Z and ZZ Hamiltonian terms. This mapping is exact for
    gate-model and annealing hardware. Analog substrate compilation
    (neutral-atom, photonic) requires additional universality work not
    yet implemented.

    Args:
        encoding: A compiled PhysicalEncoding from the LIMEN compiler.
        substrate: Target substrate hint (does not affect the mapping).

    Returns:
        A HamiltonianIR with Z and ZZ terms derived from the QUBO,
        annotated with substrate type and source encoding.

    Note:
        PLACEHOLDER for analog backends. The returned IR is correct as
        a Z-basis Ising Hamiltonian but does not account for substrate-
        specific constraints (blockade radius, mode coupling, etc.).
    """
    variables = sorted({v for pair in encoding.qubo for v in pair})
    var_idx = {v: idx for idx, v in enumerate(variables)}
    terms: list[HamiltonianTerm] = []

    for (i, j), weight in encoding.qubo.items():
        if i == j:
            # Linear term: Q_ii * x_i → (Q_ii / 2) * Z_i + const
            h = weight / 2.0
            if h != 0.0:
                terms.append(HamiltonianTerm(
                    coefficient=h,
                    operators=[(var_idx[i], "Z")],
                    metadata={"source": "linear"},
                ))
        else:
            # Quadratic term: Q_ij * x_i * x_j → (Q_ij / 4) * Z_i Z_j + ...
            J = weight / 4.0
            if J != 0.0:
                terms.append(HamiltonianTerm(
                    coefficient=J,
                    operators=[(var_idx[i], "Z"), (var_idx[j], "Z")],
                    metadata={"source": "quadratic"},
                ))
            # Single-site contributions from quadratic terms.
            for v in (i, j):
                h_contrib = weight / 4.0
                terms.append(HamiltonianTerm(
                    coefficient=h_contrib,
                    operators=[(var_idx[v], "Z")],
                    metadata={"source": "quadratic_linear_contribution"},
                ))

    return HamiltonianIR(
        terms=terms,
        n_sites=len(variables),
        substrate=substrate,
        source_encoding=encoding,
        metadata={
            "compiler": "ising_mapping",
            "n_variables": len(variables),
            "substrate": substrate.value,
            "status": "placeholder",
        },
    )
