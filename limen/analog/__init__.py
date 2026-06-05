"""Analog substrate interface layer for LIMEN.

This package defines the Hamiltonian IR interface and stub backends for
neutral-atom and photonic hardware. The constructive universality theorem
required to compile arbitrary QUBOs onto these substrates is pending
research. These interfaces exist to receive that theorem when it arrives.
"""

from limen.analog.hamiltonian import (
    HamiltonianIR,
    HamiltonianTerm,
    SubstrateType,
    from_physical_encoding,
)

__all__ = [
    "HamiltonianIR",
    "HamiltonianTerm",
    "SubstrateType",
    "from_physical_encoding",
]
