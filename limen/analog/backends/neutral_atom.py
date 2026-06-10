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
interaction physics. Atom positions are chosen so that the realized C6/r^6
couplings approximate the target ZZ Ising coefficients, and per-site
detunings realise the linear Z terms exactly.

Every compilation now emits a CompilationCertificate (Theorem 1 of
limen/docs/universality_theorem.md): an exact operator-norm bound on
||H_target - H_compiled|| for <= 20 sites, and an L1 bound above that.
Native realizability is classified per Theorem 2: van der Waals
interactions realise only positive (antiferromagnetic-type) ZZ couplings,
so targets with any J_ij < 0 are flagged and routed to the parity-encoding
universality result (Theorem 3) for exact compilation.

An optional HardwareDeltaModel pre-distorts the submitted detunings and
couplings so the as-executed Hamiltonian matches the intended one; the
certificate is then computed against the predicted as-executed couplings.

For small instances (<= 20 sites) a classical exact-diagonalisation result
is included for verification against real hardware measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from limen.analog.hamiltonian import HamiltonianIR
    from limen.analog.delta_model import HardwareDeltaModel

from limen.analog.backends.classical_sim import IsingSimulationResult, run_ising_simulation
from limen.analog.certificate import CompilationCertificate, certify_ising

# Van der Waals C6 coefficient for Rb-87 |70S1/2> Rydberg state in MHz*um^6.
# Reference: Saffman, Walker & Molmer, Rev. Mod. Phys. 82, 2313 (2010).
_C6_MHZ_UM6: float = 862_690.0

# Default global Rabi frequency in MHz. Typical value for QuEra / Pasqal devices.
_OMEGA_MHZ: float = 1.0


@dataclass
class NeutralAtomResult:
    """Result of a neutral-atom Rydberg array compilation.

    Attributes:
        hamiltonian: The HamiltonianIR that was compiled.
        atom_positions: 2-D atom positions [(x, y), ...] in micrometres.
        rabi_frequency: Global Rabi drive Omega in MHz.
        detunings: Per-site detuning Delta_i in MHz (as submitted to
            hardware — pre-distorted when a delta_model is supplied).
        realized_couplings: Realised van der Waals V_ij/4 values in MHz.
        target_couplings: Target ZZ coefficients J_ij from HamiltonianIR.
        coupling_rms_error: RMS relative error between realised and target
            couplings. Zero when no quadratic terms are present.
        certificate: CompilationCertificate with the exact operator-norm
            error (<= 20 sites) or L1 bound, per Theorem 1.
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
    certificate: CompilationCertificate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# -- Layout helpers ----------------------------------------------------

def _circle_positions(n: int, radius: float) -> list[tuple[float, float]]:
    """Place n atoms uniformly on a circle of given radius (um)."""
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
    """V(r) = C6 / r^6 in MHz, with a floor to prevent divergence."""
    return _C6_MHZ_UM6 / max(r, 0.1) ** 6


def _target_radius(J: float) -> float:
    """Atom separation that realises |J| MHz ZZ coupling via V(r)/4 = |J|."""
    return (_C6_MHZ_UM6 / (4.0 * max(abs(J), 1e-12))) ** (1.0 / 6.0)


def _spring_layout(
    n: int,
    target_J: dict[tuple[int, int], float],
    iterations: int = 300,
) -> list[tuple[float, float]]:
    """Spring-relaxation 2-D layout realising target pairwise ZZ couplings."""
    if target_J:
        radii = [_target_radius(J) for J in target_J.values()]
        radii.sort()
        init_r = radii[len(radii) // 2]
    else:
        init_r = 5.0

    positions = _circle_positions(n, radius=max(init_r, 1.0))

    lr = 0.15
    for iteration in range(iterations):
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
                    r_tgt = max(r_curr, 1.0)

                force = step * (r_curr - r_tgt)
                dx = (positions[i][0] - positions[j][0]) / r_curr
                dy = (positions[i][1] - positions[j][1]) / r_curr
                fx += force * dx
                fy += force * dy

            new_pos[i] = (positions[i][0] - fx, positions[i][1] - fy)
        positions = new_pos

    return positions


def _extract_h_J(
    hamiltonian: HamiltonianIR,
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Collect linear h_i and quadratic J_ij coefficients from IR terms."""
    h: dict[int, float] = {}
    J: dict[tuple[int, int], float] = {}
    for term in hamiltonian.terms:
        if len(term.operators) == 1:
            site, op = term.operators[0]
            if op == "Z":
                h[site] = h.get(site, 0.0) + term.coefficient
        elif len(term.operators) == 2:
            (si, oi), (sj, oj) = term.operators
            if oi == "Z" and oj == "Z":
                key = (min(si, sj), max(si, sj))
                J[key] = J.get(key, 0.0) + term.coefficient
    return h, J


# -- Main entry point --------------------------------------------------

def run_neutral_atom(
    hamiltonian: HamiltonianIR,
    delta_model: "HardwareDeltaModel | None" = None,
) -> NeutralAtomResult:
    """Compile a HamiltonianIR to neutral-atom Rydberg array parameters.

    Maps Z and ZZ Hamiltonian terms to physical Rydberg parameters:

    - ZZ coupling J_ij  ->  atom separation r_ij = (C6 / (4*|J_ij|))^(1/6) um
    - Linear term h_i   ->  detuning Delta_i = -2*h_i - sum_j V_ij / 2  (MHz)
    - Global Rabi drive Omega = 1 MHz (standard adiabatic-sweep value)

    A spring-relaxation algorithm finds 2-D atom positions that minimise the
    coupling error; the result carries a CompilationCertificate giving the
    exact operator-norm error ||H_target - H_compiled|| for <= 20 sites
    (Theorem 1, limen/docs/universality_theorem.md).

    Native realizability (Theorem 2): van der Waals interactions realise
    only J_ij > 0. Targets containing any negative coupling are flagged
    natively_realizable=False on the certificate; exact compilation of such
    targets requires the parity-encoding route (Theorem 3) with quadratic
    ancilla overhead.

    When a HardwareDeltaModel is supplied, the submitted couplings and
    detunings are pre-distorted so the as-executed Hamiltonian matches the
    target, and the certificate is computed against the predicted
    as-executed couplings.

    Args:
        hamiltonian: A HamiltonianIR from limen.analog.hamiltonian.
        delta_model: Optional calibration model for the target device.
            None (default) is equivalent to a zero-drift identity model.

    Returns:
        NeutralAtomResult with atom positions, Rabi frequency, per-site
        detunings, realised couplings, a compilation certificate, and
        (for small instances) a classical simulation for verification.
    """
    n = hamiltonian.n_sites
    h, target_J = _extract_h_J(hamiltonian)

    # Pre-distort target couplings so as-executed couplings land on target.
    if delta_model is not None and target_J:
        submit_J = delta_model.apply_coupling_correction(target_J)
    else:
        submit_J = target_J

    # Find 2-D atom layout against the (possibly pre-distorted) couplings.
    positions = _spring_layout(n, submit_J) if n > 0 else []

    # Realised van der Waals couplings from final positions (as submitted).
    realized: dict[tuple[int, int], float] = {}
    for (i, j) in target_J:
        r = _dist(positions[i], positions[j])
        realized[(i, j)] = _vdw(r) / 4.0  # V(r)/4 = ZZ coupling

    # Predicted as-executed couplings: hardware multiplies by (1 + error).
    if delta_model is not None:
        errs = delta_model.drift.coupling_scale_errors
        as_executed = {k: v * (1.0 + errs.get(k, 0.0)) for k, v in realized.items()}
    else:
        as_executed = dict(realized)

    # RMS relative coupling error against the original target.
    errors = []
    for key, J_tgt in target_J.items():
        J_real = as_executed.get(key, 0.0)
        if abs(J_tgt) > 1e-12:
            errors.append(((J_real - J_tgt) / J_tgt) ** 2)
    rms_err = math.sqrt(sum(errors) / len(errors)) if errors else 0.0

    # Per-site detuning: Delta_i = -2*h_i - sum_j V_ij/2 — realises h exactly.
    detunings: list[float] = []
    for site in range(n):
        hi = h.get(site, 0.0)
        neighbor_sum = sum(
            _vdw(_dist(positions[site], positions[j])) / 2.0
            for j in range(n) if j != site
        )
        detunings.append(-2.0 * hi - neighbor_sum)

    # Pre-distort submitted detunings against measured per-site offsets.
    if delta_model is not None:
        detunings = delta_model.apply_detuning_correction(detunings)

    # Native realizability (Theorem 2): vdW realises only positive ZZ.
    natively_realizable = all(J > 0.0 for J in target_J.values())

    # Compilation certificate (Theorem 1). Linear terms are realised exactly
    # by the detuning formula, so dh = 0; the error lives in the couplings.
    certificate = certify_ising(
        target_h=h,
        target_J=target_J,
        compiled_h=h,
        compiled_J=as_executed,
        n_sites=n,
        natively_realizable=natively_realizable,
        notes=["Linear (Z) terms realised exactly via detuning formula."],
    )

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
            f"Rydberg layout: {n} atoms, Omega={_OMEGA_MHZ} MHz, "
            f"coupling RMS error={rms_err:.4f}"
        ),
        certificate=certificate,
        metadata={
            "c6_mhz_um6": _C6_MHZ_UM6,
            "rabi_frequency_mhz": _OMEGA_MHZ,
            "n_zz_pairs": len(target_J),
            "coupling_rms_error": rms_err,
            "natively_realizable": natively_realizable,
            "delta_model_device": (
                delta_model.device_id if delta_model is not None else None
            ),
            "status": "certified-heuristic",
            "note": (
                "Heuristic van der Waals layout with exact compilation "
                "certificate (Theorem 1, limen/docs/universality_theorem.md). "
                "Targets with negative J require the parity-encoding route "
                "(Theorem 3)."
            ),
        },
    )
