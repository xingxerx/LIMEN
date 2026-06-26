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

A geometric embeddability check (Theorem 2, geometry condition) is run
before the spring layout: the required inter-atom distances are derived from
the target couplings, and the Schoenberg/MDS test checks whether those
distances can be realized in the 2-D plane. A failed geometry check flags
natively_realizable=False even when all couplings are positive. The check
uses numpy when available; without it the check is skipped.

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
from limen.analog.lhz import LHZCertificate, LHZResult, certify_lhz, lhz_parity_pass

# Van der Waals C6 coefficient for Rb-87 |70S1/2> Rydberg state in MHz*um^6.
# Reference: Saffman, Walker & Molmer, Rev. Mod. Phys. 82, 2313 (2010).
_C6_MHZ_UM6: float = 862_690.0

# Default global Rabi frequency in MHz. Typical value for QuEra / Pasqal devices.
_OMEGA_MHZ: float = 1.0


@dataclass
class GeometricEmbeddabilityResult:
    """Result of the Schoenberg/MDS 2-D embeddability check (Theorem 2 geometry condition).

    Attributes:
        embeddable: True when the required inter-atom distances are realizable
            in the 2-D Euclidean plane (Gram matrix PSD with rank ≤ 2).
        psd_satisfied: True when the Gram matrix is PSD (necessary for ANY
            Euclidean embedding, not just 2-D).
        gram_min_eigenvalue: Smallest eigenvalue of the doubly-centered
            squared-distance Gram matrix. Negative values indicate geometric
            frustration. None when numpy is not available.
        gram_rank: Number of eigenvalues above the PSD tolerance (estimate of
            the embedding dimension required). None when numpy is not available.
        checked: True when the test was actually run (numpy available and at
            least 4 constrained sites were present).
        n_constrained_sites: Number of sites involved in constrained pairs.
        notes: Human-readable observations.
    """

    embeddable: bool
    psd_satisfied: bool
    gram_min_eigenvalue: float | None
    gram_rank: int | None
    checked: bool
    n_constrained_sites: int
    notes: list[str] = field(default_factory=list)


def check_geometric_embeddability(
    target_J: dict[tuple[int, int], float],
    c6: float = _C6_MHZ_UM6,
    tol: float = 1e-8,
) -> GeometricEmbeddabilityResult:
    """Check whether the required coupling distances can be realized in the 2-D plane.

    Implements the Schoenberg test (Theorem 2 geometry condition from
    limen/docs/universality_theorem.md): given target ZZ coupling strengths
    J_ij > 0, the required inter-atom distances are d_ij = (C6 / (4 J_ij))^{1/6}.
    The distance set is 2-D Euclidean-embeddable iff the doubly-centered
    squared-distance matrix G = -½ J D² J (J = I - (1/n)·11^T) is positive
    semidefinite with rank ≤ 2.

    Only positive couplings are checked; negative couplings are an independent
    sign obstruction already handled by the native-realizability sign check.

    Requires numpy for the eigenvalue computation. When numpy is unavailable
    or fewer than 4 constrained sites are present (trivially 2-D embeddable),
    the result is returned with checked=False.

    Args:
        target_J: Dict mapping (i, j) pairs (i < j) to ZZ coupling strengths.
            Only positive entries are considered.
        c6: Van der Waals C6 coefficient in MHz·μm^6. Default: Rb-87 value.
        tol: Tolerance for treating eigenvalues as zero. Default 1e-8.

    Returns:
        GeometricEmbeddabilityResult.
    """
    positive_J = {k: v for k, v in target_J.items() if v > 0.0}

    if not positive_J:
        return GeometricEmbeddabilityResult(
            embeddable=True, psd_satisfied=True,
            gram_min_eigenvalue=None, gram_rank=None,
            checked=False, n_constrained_sites=0,
            notes=["No positive-coupling pairs; geometry check trivially satisfied."],
        )

    # Collect unique sites involved in positive-coupling pairs.
    sites_set: set[int] = set()
    for i, j in positive_J:
        sites_set.update((i, j))
    sites = sorted(sites_set)
    n = len(sites)
    site_idx = {s: k for k, s in enumerate(sites)}

    if n < 4:
        # n ≤ 3 constrained sites are always 2-D embeddable (a triangle always
        # lies in a plane).
        return GeometricEmbeddabilityResult(
            embeddable=True, psd_satisfied=True,
            gram_min_eigenvalue=None, gram_rank=None,
            checked=False, n_constrained_sites=n,
            notes=[
                f"Only {n} constrained site(s); 2-D embeddability holds trivially "
                f"(any ≤ 3 points lie in a plane)."
            ],
        )

    try:
        import numpy as np
    except ImportError:
        return GeometricEmbeddabilityResult(
            embeddable=True, psd_satisfied=True,
            gram_min_eigenvalue=None, gram_rank=None,
            checked=False, n_constrained_sites=n,
            notes=["numpy not available; geometry check skipped."],
        )

    # Build n×n squared-distance matrix D²; unconstrained pairs get 0 (free).
    D2 = np.zeros((n, n), dtype=float)
    for (a, b), J_val in positive_J.items():
        ia, ib = site_idx[a], site_idx[b]
        d_req = (c6 / (4.0 * J_val)) ** (1.0 / 6.0)
        D2[ia, ib] = D2[ib, ia] = d_req * d_req

    # Doubly-center: G = -½ J D² J   where J = I - (1/n)·11^T.
    # Equivalent row/column mean subtraction:
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    grand_mean = D2.mean()
    G = -0.5 * (D2 - row_mean - col_mean + grand_mean)

    eigvals = np.linalg.eigvalsh(G)
    min_eigval = float(eigvals.min())
    gram_rank = int(np.sum(eigvals > tol))
    psd_ok = min_eigval >= -tol
    embeddable_2d = psd_ok and gram_rank <= 2

    notes: list[str] = []
    if not psd_ok:
        notes.append(
            f"Gram matrix has negative eigenvalue {min_eigval:.3e} (< -{tol}); "
            f"required distances are not realizable in any Euclidean space. "
            f"Native neutral-atom compilation requires the LHZ parity-encoding route."
        )
    elif gram_rank > 2:
        notes.append(
            f"Gram matrix rank {gram_rank} > 2; required distances need a "
            f"{gram_rank}-D embedding, not achievable in a 2-D atom array. "
            f"Native neutral-atom compilation requires the LHZ parity-encoding route."
        )
    else:
        notes.append(
            f"Gram matrix is PSD with rank {gram_rank} ≤ 2; "
            f"required distances are 2-D Euclidean-embeddable."
        )

    return GeometricEmbeddabilityResult(
        embeddable=embeddable_2d,
        psd_satisfied=psd_ok,
        gram_min_eigenvalue=min_eigval,
        gram_rank=gram_rank,
        checked=True,
        n_constrained_sites=n,
        notes=notes,
    )


@dataclass
class PlaquetteGeometryResult:
    """Geometric proximity check for LHZ plaquette ancilla siting.

    This does NOT certify physical realizability of the 3-body plaquette
    constraint — the ancilla-gadget construction itself (constraint-qubit
    coupling strengths, its own penalty bound) is not modeled anywhere in
    this codebase; see LHZCertificate.metadata['plaquette_realizability'].
    It only checks a necessary (not sufficient) geometric precondition for
    the LHZ gadget construction (Lechner, Hauke & Zoller 2015): that the
    three parity qubits of each plaquette sit at roughly equal pairwise
    distances, as required to host a single ancilla coupled symmetrically
    to all three.

    Attributes:
        plausible: True when every plaquette triple's relative side-length
            spread is within max_relative_spread.
        n_plaquettes: Number of plaquettes checked.
        max_spread: Largest relative spread (over all plaquettes) between the
            longest and shortest side of the constraint triangle. None when
            there are no plaquettes.
        per_plaquette_spread: Relative spread for each plaquette, in the same
            order as LHZResult.plaquettes.
        notes: Human-readable observations, including the scope limitation.
    """

    plausible: bool
    n_plaquettes: int
    max_spread: float | None
    per_plaquette_spread: list[float]
    notes: list[str] = field(default_factory=list)


_PLAQUETTE_SCOPE_NOTE = (
    "Geometric proximity only — does not certify the LHZ ancilla-gadget "
    "construction (constraint-qubit coupling strengths, its own penalty "
    "bound). See LHZCertificate.metadata['plaquette_realizability']."
)


def check_plaquette_geometry(
    plaquettes: list[tuple[int, int, int]],
    atom_positions: list[tuple[float, float]],
    max_relative_spread: float = 0.5,
) -> PlaquetteGeometryResult:
    """Check whether plaquette parity qubits sit close enough for a shared ancilla.

    Args:
        plaquettes: List of (pa, pb, pc) parity-qubit index triples, as
            produced by lhz_parity_pass.
        atom_positions: 2-D positions of the parity qubits (e.g. from
            run_neutral_atom applied to the encoded IR).
        max_relative_spread: Maximum allowed (max_side - min_side) / mean_side
            for a plaquette's constraint triangle to be considered plausible.

    Returns:
        PlaquetteGeometryResult.
    """
    if not plaquettes:
        return PlaquetteGeometryResult(
            plausible=True, n_plaquettes=0, max_spread=None,
            per_plaquette_spread=[],
            notes=["No plaquettes to check.", _PLAQUETTE_SCOPE_NOTE],
        )

    spreads: list[float] = []
    for (pa, pb, pc) in plaquettes:
        a, b, c = atom_positions[pa], atom_positions[pb], atom_positions[pc]
        sides = [_dist(a, b), _dist(a, c), _dist(b, c)]
        mean_side = sum(sides) / 3.0
        spread = (max(sides) - min(sides)) / mean_side if mean_side > 1e-9 else 0.0
        spreads.append(spread)

    max_spread = max(spreads)
    plausible = max_spread <= max_relative_spread

    notes = [
        f"Max relative side-length spread across {len(plaquettes)} "
        f"plaquette(s): {max_spread:.3f} (threshold {max_relative_spread}).",
        _PLAQUETTE_SCOPE_NOTE,
    ]
    if not plausible:
        notes.append(
            "At least one plaquette's three parity qubits are not roughly "
            "equidistant — a shared ancilla site would need asymmetric "
            "coupling strengths, which the LHZ fixed-lattice construction "
            "does not provide for."
        )

    return PlaquetteGeometryResult(
        plausible=plausible,
        n_plaquettes=len(plaquettes),
        max_spread=max_spread,
        per_plaquette_spread=spreads,
        notes=notes,
    )


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
        lhz_result: Set when the target was not natively realizable
            (certificate.natively_realizable is False) and compilation was
            automatically routed through the LHZ parity encoding (Theorem 3).
            None when the heuristic van der Waals layout sufficed.
        lhz_certificate: Penalty-gap certificate for the LHZ route, wrapping
            the certified compilation of the encoded (local-field) problem.
            None unless lhz_result is set.
        plaquette_geometry: Geometric proximity check (necessary, not
            sufficient) for siting an LHZ ancilla gadget per plaquette.
            None unless lhz_result is set; see PlaquetteGeometryResult's
            scope limitation.
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
    geometry: GeometricEmbeddabilityResult | None = None
    lhz_result: LHZResult | None = None
    lhz_certificate: LHZCertificate | None = None


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

    # Geometric embeddability check (Theorem 2 geometry condition): run before
    # the spring layout so we can flag non-realizable geometry early.
    geo = check_geometric_embeddability(
        {k: v for k, v in target_J.items() if v > 0.0}
    )

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

    # Native realizability (Theorem 2): vdW realises only positive ZZ (sign
    # condition), AND the required distances must be 2-D embeddable (geometry
    # condition checked above).
    sign_ok = all(J > 0.0 for J in target_J.values())
    natively_realizable = sign_ok and (not geo.checked or geo.embeddable)

    # Compilation certificate (Theorem 1). Linear terms are realised exactly
    # by the detuning formula, so dh = 0; the error lives in the couplings.
    cert_notes = ["Linear (Z) terms realised exactly via detuning formula."]
    if geo.checked and not geo.embeddable:
        cert_notes += geo.notes
    certificate = certify_ising(
        target_h=h,
        target_J=target_J,
        compiled_h=h,
        compiled_J=as_executed,
        n_sites=n,
        natively_realizable=natively_realizable,
        notes=cert_notes,
    )

    # Classical simulation for small instances.
    sim: IsingSimulationResult | None = None
    if n <= 20:
        try:
            sim = run_ising_simulation(hamiltonian)
        except Exception:
            pass

    # Automatic LHZ fallback (Theorem 3): when the heuristic van der Waals
    # layout cannot natively realize the target (negative coupling or a
    # non-2D-embeddable geometry), encode the problem into local fields via
    # the parity transform and recursively compile *that* — local fields are
    # always natively realizable (Theorem 2 part 1), so this terminates in
    # one recursion and yields a real, certified compilation rather than
    # leaving the caller with only a known-bad heuristic certificate.
    lhz_result: LHZResult | None = None
    lhz_certificate: LHZCertificate | None = None
    plaquette_geometry: PlaquetteGeometryResult | None = None
    if not natively_realizable:
        lhz_result = lhz_parity_pass(hamiltonian)
        encoded_compiled = run_neutral_atom(lhz_result.encoded_ir, delta_model=delta_model)
        lhz_certificate = certify_lhz(
            lhz_result,
            compilation_certificate=encoded_compiled.certificate,
        )
        plaquette_geometry = check_plaquette_geometry(
            lhz_result.plaquettes, encoded_compiled.atom_positions
        )

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
        geometry=geo,
        metadata={
            "c6_mhz_um6": _C6_MHZ_UM6,
            "rabi_frequency_mhz": _OMEGA_MHZ,
            "n_zz_pairs": len(target_J),
            "coupling_rms_error": rms_err,
            "natively_realizable": natively_realizable,
            "sign_ok": sign_ok,
            "geometry_embeddable": geo.embeddable,
            "geometry_checked": geo.checked,
            "gram_min_eigenvalue": geo.gram_min_eigenvalue,
            "gram_rank": geo.gram_rank,
            "delta_model_device": (
                delta_model.device_id if delta_model is not None else None
            ),
            "status": "certified-heuristic",
            "lhz_fallback_applied": lhz_result is not None,
            "lhz_n_physical": lhz_result.n_physical if lhz_result is not None else None,
            "note": (
                "Heuristic van der Waals layout with exact compilation "
                "certificate (Theorem 1, limen/docs/universality_theorem.md). "
                "Targets with negative J or non-2D-embeddable distances are "
                "automatically LHZ-encoded (Theorem 3); see result.lhz_result "
                "and result.lhz_certificate for the parity-encoded Hamiltonian "
                "and its correctness certificate."
            ),
        },
        lhz_result=lhz_result,
        lhz_certificate=lhz_certificate,
        plaquette_geometry=plaquette_geometry,
    )
