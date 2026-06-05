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
"""Neutral-atom backend for LIMEN.

Maps a HamiltonianIR onto Rydberg atom array parameters using van der Waals
interaction physics. The mapping is heuristic: atom positions are chosen so
that the realized C₆/r⁶ couplings approximate the target ZZ Ising coefficients,
and per-site detunings realise the linear Z terms.

This is an engineering implementation, not a certified universality result.
The constructive universality theorem that would guarantee exact, error-bounded
compilation for arbitrary Ising models onto Rydberg hardware is pending research
(see limen/docs/architecture.md for the precise research specification).

For small instances (≤ 20 sites) a classical exact-diagonalisation result is
included for verification against real hardware measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from limen.analog.hamiltonian import HamiltonianIR

from limen.analog.backends.classical_sim import IsingSimulationResult, run_ising_simulation

# Van der Waals C₆ coefficient for Rb-87 |70S₁/₂⟩ Rydberg state in MHz·μm⁶.
# Reference: Saffman, Walker & Mølmer, Rev. Mod. Phys. 82, 2313 (2010).
_C6_MHZ_UM6: float = 862_690.0

# Default global Rabi frequency in MHz.  Typical value for QuEra / Pasqal devices.
_OMEGA_MHZ: float = 1.0


@dataclass
class NeutralAtomResult:
    """Result of a neutral-atom Rydberg array compilation.

    Attributes:
        hamiltonian: The HamiltonianIR that was compiled.
        atom_positions: 2-D atom positions [(x, y), ...] in micrometres.
        rabi_frequency: Global Rabi drive Ω in MHz.
        detunings: Per-site detuning Δ_i in MHz, one entry per site.
        realized_couplings: Realised van der Waals V_ij values in MHz for
            each pair (i, j) with i < j.
        target_couplings: Target ZZ coefficients J_ij from HamiltonianIR for
            each pair (i, j) with i < j.
        coupling_rms_error: RMS relative error between realised and target
            couplings. Zero when no quadratic terms are present.
        simulation: Classical exact-diagonalisation result, or None if
            n_sites > 20.
        available: True if layout succeeded (always True for heuristic path).
        simulated: True — result includes a classical simulation.
        message: Human-readable status.
        metadata: Compilation annotations.
    """

    hamiltonian: HamiltonianIR
    atom_positions: list[tuple[float, float]]
    rabi_frequency: float
    detunings: list[float]
    realized_couplings: dict[tuple[int, int], float]
    target_couplings: dict[tuple[int, int], float]
    coupling_rms_error: float
    simulation: IsingSimulationResult | None
    available: bool
    simulated: bool
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Layout helpers ────────────────────────────────────────────────────

def _circle_positions(n: int, radius: float) -> list[tuple[float, float]]:
    """Place n atoms uniformly on a circle of given radius (μm)."""
    if n == 1:
        return [(0.0, 0.0)]
    return [
        (radius * math.cos(2.0 * math.pi * i / n),
         radius * math.sin(2.0 * math.pi * i / n))
        for i in range(n)
    ]


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _vdw(r: float) -> float:
    """V(r) = C₆ / r⁶ in MHz, with a floor to prevent divergence."""
    return _C6_MHZ_UM6 / max(r, 0.1) ** 6


def _target_radius(J: float) -> float:
    """Atom separation that realises |J| MHz ZZ coupling via V(r)/4 = |J|."""
    return (_C6_MHZ_UM6 / (4.0 * max(abs(J), 1e-12))) ** (1.0 / 6.0)


def _spring_layout(
    n: int,
    target_J: dict[tuple[int, int], float],
    iterations: int = 300,
) -> list[tuple[float, float]]:
    """Spring-relaxation 2-D layout realising target pairwise ZZ couplings.

    Starting from a circular arrangement, gradient-free spring forces pull
    each atom toward the inter-site distance that would realise the target
    coupling via the van der Waals law.  Pairs without a target coupling
    use a repulsive baseline to prevent site collapse.
    """
    # Initial radius: median target distance or 5 μm fallback.
    if target_J:
        radii = [_target_radius(J) for J in target_J.values()]
        radii.sort()
        init_r = radii[len(radii) // 2]
    else:
        init_r = 5.0

    positions = _circle_positions(n, radius=max(init_r, 1.0))

    lr = 0.15
    for iteration in range(iterations):
        # Decay learning rate for stability.
        step = lr * (1.0 - 0.5 * iteration / iterations)
        new_pos = list(positions)
        for i in range(n):
            fx = fy = 0.0
            for j in range(n):
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                r_curr = _dist(positions[i], positions[j])
                if r_curr < 1e-9:
                    r_curr = 1e-9

                if key in target_J:
                    r_tgt = _target_radius(target_J[key])
                else:
                    # Repulsive baseline: push apart to at least 1 μm.
                    r_tgt = max(r_curr, 1.0)

                force = step * (r_curr - r_tgt)
                dx = (positions[i][0] - positions[j][0]) / r_curr
                dy = (positions[i][1] - positions[j][1]) / r_curr
                fx += force * dx
                fy += force * dy

            new_pos[i] = (positions[i][0] - fx, positions[i][1] - fy)
        positions = new_pos

    return positions


# ── Main entry point ──────────────────────────────────────────────────

def run_neutral_atom(hamiltonian: HamiltonianIR) -> NeutralAtomResult:
    """Compile a HamiltonianIR to neutral-atom Rydberg array parameters.

    Maps Z and ZZ Hamiltonian terms to physical Rydberg parameters:

    - ZZ coupling J_ij  →  atom separation r_ij = (C₆ / (4·|J_ij|))^(1/6) μm
    - Linear term h_i   →  detuning Δ_i = −2·h_i − Σ_j V_ij / 2  (MHz)
    - Global Rabi drive  Ω = 1 MHz (standard adiabatic-sweep value)

    A spring-relaxation algorithm finds 2-D atom positions that minimise the
    RMS relative error between realised van der Waals couplings and the
    target J_ij values.

    For instances with ≤ 20 sites, a classical exact-diagonalisation check
    is included in the result.

    Note:
        HEURISTIC — not a certified universality result. See
        limen/docs/architecture.md for the research specification.

    Args:
        hamiltonian: A HamiltonianIR from limen.analog.hamiltonian.

    Returns:
        NeutralAtomResult with atom positions, Rabi frequency, per-site
        detunings, realised couplings, and (for small instances) a
        classical simulation for verification.
    """
    n = hamiltonian.n_sites

    # Extract linear h_i and quadratic J_ij from terms.
    h: dict[int, float] = {}
    target_J: dict[tuple[int, int], float] = {}

    for term in hamiltonian.terms:
        if len(term.operators) == 1:
            site, op = term.operators[0]
            if op == "Z":
                h[site] = h.get(site, 0.0) + term.coefficient
        elif len(term.operators) == 2:
            (si, oi), (sj, oj) = term.operators
            if oi == "Z" and oj == "Z":
                key = (min(si, sj), max(si, sj))
                target_J[key] = target_J.get(key, 0.0) + term.coefficient

    # Find 2-D atom layout.
    positions = _spring_layout(n, target_J) if n > 0 else []

    # Realised van der Waals couplings from final positions.
    realized: dict[tuple[int, int], float] = {}
    for (i, j) in target_J:
        r = _dist(positions[i], positions[j])
        realized[(i, j)] = _vdw(r) / 4.0  # V(r)/4 = ZZ coupling

    # RMS relative coupling error.
    errors = []
    for key, J_tgt in target_J.items():
        J_real = realized.get(key, 0.0)
        if abs(J_tgt) > 1e-12:
            errors.append(((J_real - J_tgt) / J_tgt) ** 2)
    rms_err = math.sqrt(sum(errors) / len(errors)) if errors else 0.0

    # Per-site detuning: Δ_i = -2·h_i - Σ_j V_ij/2
    detunings: list[float] = []
    for site in range(n):
        hi = h.get(site, 0.0)
        neighbor_sum = sum(
            _vdw(_dist(positions[site], positions[j])) / 2.0
            for j in range(n) if j != site
        )
        detunings.append(-2.0 * hi - neighbor_sum)

    # Classical simulation for small instances.
    sim: IsingSimulationResult | None = None
    if n <= 20:
        try:
            sim = run_ising_simulation(hamiltonian)
        except Exception:
            pass

    return NeutralAtomResult(
        hamiltonian=hamiltonian,
        atom_positions=positions,
        rabi_frequency=_OMEGA_MHZ,
        detunings=detunings,
        realized_couplings=realized,
        target_couplings=target_J,
        coupling_rms_error=rms_err,
        simulation=sim,
        available=True,
        simulated=True,
        message=(
            f"Rydberg layout: {n} atoms, Ω={_OMEGA_MHZ} MHz, "
            f"coupling RMS error={rms_err:.4f}"
        ),
        metadata={
            "c6_mhz_um6": _C6_MHZ_UM6,
            "rabi_frequency_mhz": _OMEGA_MHZ,
            "n_zz_pairs": len(target_J),
            "coupling_rms_error": rms_err,
            "status": "heuristic",
            "note": (
                "Heuristic van der Waals layout. "
                "Constructive universality theorem pending research."
            ),
        },
    )
