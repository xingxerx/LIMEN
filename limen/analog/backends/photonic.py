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
"""Photonic backend stub for LIMEN.

Placeholder adapter for continuous-variable photonic processors.
Real compilation requires mapping the HamiltonianIR onto optical mode
interactions (squeezed states, beamsplitters, homodyne measurement).
That mapping is pending the constructive universality theorem.

This stub validates the HamiltonianIR structure and raises NotImplementedError
with a clear explanation of what research is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from limen.analog.hamiltonian import HamiltonianIR


@dataclass
class PhotonicResult:
    """Result placeholder for a photonic hardware run.

    Attributes:
        hamiltonian: The HamiltonianIR submitted.
        available: False — hardware compilation not yet implemented.
        simulated: False — no simulator path exists yet.
        message: Human-readable status message.
        metadata: Arbitrary annotations.
    """

    hamiltonian: HamiltonianIR
    available: bool
    simulated: bool
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


def run_photonic(
    hamiltonian: HamiltonianIR,
) -> PhotonicResult:
    """Stub: submit a HamiltonianIR to a photonic processor.

    Status: PLACEHOLDER. Raises NotImplementedError with a clear message
    explaining what is missing.

    Args:
        hamiltonian: A HamiltonianIR from limen.analog.hamiltonian.

    Raises:
        NotImplementedError: Always. Describes what research is required.
    """
    raise NotImplementedError(
        "Photonic backend compilation is not yet implemented. "
        "Required: constructive universality theorem mapping Ising Z/ZZ "
        "terms onto continuous-variable optical Hamiltonians "
        "(Gaussian boson sampling / Kerr interactions). "
        "See Phase 3 roadmap."
    )
