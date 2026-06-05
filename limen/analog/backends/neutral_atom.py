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
"""Neutral-atom backend stub for LIMEN.

Placeholder adapter for neutral-atom array processors (QuEra Aquila,
Pasqal, etc.). Real compilation requires mapping the HamiltonianIR onto
Rydberg blockade interactions with hardware-specific spatial constraints.
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
class NeutralAtomResult:
    """Result placeholder for a neutral-atom hardware run.

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


def run_neutral_atom(
    hamiltonian: HamiltonianIR,
) -> NeutralAtomResult:
    """Stub: submit a HamiltonianIR to a neutral-atom processor.

    Status: PLACEHOLDER. Raises NotImplementedError with a clear message
    explaining what is missing.

    Args:
        hamiltonian: A HamiltonianIR from limen.analog.hamiltonian.

    Raises:
        NotImplementedError: Always. Describes what research is required.
    """
    raise NotImplementedError(
        "Neutral-atom backend compilation is not yet implemented. "
        "Required: constructive universality theorem mapping Ising Z/ZZ "
        "terms onto Rydberg blockade Hamiltonians with spatial layout "
        "constraints. See Phase 3 roadmap."
    )
