# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Classical Ising simulation backend for LIMEN.

Implements exact diagonalization of a HamiltonianIR for instances up to
MAX_SITES sites by enumerating all 2^n spin configurations. This is the
reference implementation for all analog substrate types: it finds the
true ground state of the Z-basis Ising Hamiltonian without any hardware.

The simulation is substrate-agnostic — it does not matter whether the
HamiltonianIR came from a neutral-atom, photonic, or BEC path. The
physics is identical: minimise Σ_i h_i σ_i^z + Σ_{i<j} J_ij σ_i^z σ_j^z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from limen.analog.hamiltonian import HamiltonianIR

MAX_SITES: int = 20


@dataclass
class IsingSimulationResult:
    """Result of a classical exact-diagonalisation run.

    Attributes:
        hamiltonian: The HamiltonianIR that was simulated.
        ground_state_energy: Minimum energy found over all spin configurations.
        ground_state_assignment: Binary assignment {site_index: 0|1} for the
            ground state, using x_i = (1 + σ_i^z) / 2.
        excited_states: Up to five distinct excited energy levels, each as
            (energy, assignment) tuples, sorted ascending.
        n_sites: Number of sites in the simulation.
        available: Always True — the simulator has no hardware dependency.
        simulated: Always True.
        message: Human-readable summary.
        metadata: Diagnostic annotations.
    """

    hamiltonian: HamiltonianIR
    ground_state_energy: float
    ground_state_assignment: dict[int, int]
    excited_states: list[tuple[float, dict[int, int]]]
    n_sites: int
    available: bool = True
    simulated: bool = True
    message: str = "Classical exact diagonalisation"
    metadata: dict[str, Any] = field(default_factory=dict)


def _eval_energy(terms: list, spins: tuple[int, ...]) -> float:
    """Evaluate total Hamiltonian energy for one spin configuration."""
    energy = 0.0
    for term in terms:
        contrib = term.coefficient
        for site_idx, op_str in term.operators:
            if op_str == "Z":
                contrib *= spins[site_idx]
            # "I" contributes factor 1; "X"/"Y" are off-diagonal — skip in Z-basis
        energy += contrib
    return energy


def run_ising_simulation(
    hamiltonian: HamiltonianIR,
    max_sites: int = MAX_SITES,
) -> IsingSimulationResult:
    """Run classical exact diagonalisation of the Z-basis Ising Hamiltonian.

    Enumerates all 2^n_sites spin-±1 configurations and finds the global
    energy minimum. Valid for any substrate type — the simulation is
    substrate-agnostic and requires no hardware or SDK.

    Args:
        hamiltonian: A HamiltonianIR from limen.analog.hamiltonian.
        max_sites: Maximum number of sites. Defaults to 20 (1M evaluations).
            Raise this carefully — runtime is O(2^n_sites).

    Returns:
        IsingSimulationResult with the ground state, energy, and the five
        lowest distinct excited levels.

    Raises:
        ValueError: If hamiltonian.n_sites > max_sites.
    """
    n = hamiltonian.n_sites
    if n > max_sites:
        raise ValueError(
            f"n_sites={n} exceeds max_sites={max_sites}. "
            "Exact diagonalisation is exponential — increase max_sites with care."
        )
    if n == 0:
        return IsingSimulationResult(
            hamiltonian=hamiltonian,
            ground_state_energy=0.0,
            ground_state_assignment={},
            excited_states=[],
            n_sites=0,
            metadata={"n_configurations": 0},
        )

    configs: list[tuple[float, tuple[int, ...]]] = []
    for spins in product((1, -1), repeat=n):
        configs.append((_eval_energy(hamiltonian.terms, spins), spins))

    configs.sort(key=lambda x: x[0])

    # x_i = (1 + s_i) / 2  →  s=+1 maps to x=1, s=-1 maps to x=0
    ground_assignment = {i: (1 + configs[0][1][i]) // 2 for i in range(n)}

    excited: list[tuple[float, dict[int, int]]] = []
    seen: set[float] = {configs[0][0]}
    for e, spins in configs[1:]:
        if e not in seen and len(excited) < 5:
            seen.add(e)
            excited.append((e, {i: (1 + spins[i]) // 2 for i in range(n)}))

    energy_gap = excited[0][0] - configs[0][0] if excited else 0.0

    return IsingSimulationResult(
        hamiltonian=hamiltonian,
        ground_state_energy=configs[0][0],
        ground_state_assignment=ground_assignment,
        excited_states=excited,
        n_sites=n,
        metadata={
            "n_configurations": 2**n,
            "n_terms": len(hamiltonian.terms),
            "energy_gap": energy_gap,
            "substrate": hamiltonian.substrate.value,
        },
    )
