# Copyright (C) 2026 Jemone McCubbin / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""LHZ parity encoding for analog Ising Hamiltonians (Theorem 3).

Implements the Lanthaler-Lechner-Zoller parity encoding that maps a
logical Ising problem with arbitrary coupling signs onto a physical
Hamiltonian with only local fields and three-body plaquette constraints.
See limen/docs/universality_theorem.md#theorem-3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from limen.analog.certificate import CompilationCertificate
from limen.analog.hamiltonian import HamiltonianIR, HamiltonianTerm, SubstrateType


def has_negative_couplings(ir: HamiltonianIR) -> bool:
    """Return True if any ZZ term has a negative coefficient."""
    for term in ir.terms:
        ops = term.operators
        if (
            len(ops) == 2
            and ops[0][1] == "Z"
            and ops[1][1] == "Z"
            and term.coefficient < 0
        ):
            return True
    return False


@dataclass
class LHZResult:
    """Output of lhz_parity_pass: the encoded Hamiltonian and encoding metadata."""

    encoded_ir: HamiltonianIR
    n_logical: int
    n_physical: int
    qubit_map: dict[tuple[int, int], int]
    plaquettes: list[tuple]
    penalty_strength: float
    h0_constant: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_logical": self.n_logical,
            "n_physical": self.n_physical,
            "qubit_map": {str(list(k)): v for k, v in self.qubit_map.items()},
            "plaquettes": [list(p) for p in self.plaquettes],
            "penalty_strength": self.penalty_strength,
            "h0_constant": self.h0_constant,
            "metadata": self.metadata,
        }


def lhz_parity_pass(ir: HamiltonianIR, penalty_factor: float = 3.0) -> LHZResult:
    """Apply the LHZ parity encoding to a Z-basis Ising Hamiltonian.

    Maps logical couplings J_ij onto local fields on parity qubits σ_ij,
    and emits plaquette constraint terms to enforce σ_ij = s_i * s_j.

    Args:
        ir: Input HamiltonianIR containing Z and ZZ terms.
        penalty_factor: Multiplier for the plaquette penalty strength.
            Must be > 2.0 (Lanthaler-Lechner bound).

    Returns:
        LHZResult with the encoded Hamiltonian and encoding metadata.
    """
    if penalty_factor <= 2.0:
        raise ValueError("penalty_factor must be > 2.0 (Lanthaler-Lechner bound)")

    # Step 0: Classify terms into h (linear Z), J (quadratic ZZ), pass-throughs.
    h: dict[int, float] = {}
    J: dict[tuple[int, int], float] = {}
    passthrough: list[HamiltonianTerm] = []

    for term in ir.terms:
        ops = term.operators
        if len(ops) == 1 and ops[0][1] == "Z":
            site = ops[0][0]
            h[site] = h.get(site, 0.0) + term.coefficient
        elif len(ops) == 2 and ops[0][1] == "Z" and ops[1][1] == "Z":
            i, j = ops[0][0], ops[1][0]
            if i > j:
                i, j = j, i
            J[(i, j)] = J.get((i, j), 0.0) + term.coefficient
        else:
            pt_meta = dict(term.metadata)
            pt_meta["lhz_passthrough"] = True
            passthrough.append(
                HamiltonianTerm(
                    coefficient=term.coefficient,
                    operators=list(term.operators),
                    metadata=pt_meta,
                )
            )

    # Step 1: Build qubit_map from sorted pairs.
    pairs_sorted = sorted(J.keys())
    qubit_map: dict[tuple[int, int], int] = {pair: idx for idx, pair in enumerate(pairs_sorted)}
    K = len(qubit_map)

    logical_sites = sorted(set(site for pair in J for site in pair))
    n_logical = len(logical_sites)

    # Step 2: Encoded terms — logical couplings become local fields.
    all_terms: list[HamiltonianTerm] = []

    for (i, j), coeff in sorted(J.items()):
        all_terms.append(
            HamiltonianTerm(
                coefficient=coeff,
                operators=[(qubit_map[(i, j)], "Z")],
                metadata={"source": "lhz_coupling", "logical_pair": [i, j]},
            )
        )

    # Linear terms h[i] (i > 0) map to parity qubit (0, i) via gauge s_0 = 1.
    h0_constant = h.get(0, 0.0)
    for i, coeff in sorted(h.items()):
        if i == 0:
            continue
        pair = (0, i)
        if pair in qubit_map:
            all_terms.append(
                HamiltonianTerm(
                    coefficient=coeff,
                    operators=[(qubit_map[pair], "Z")],
                    metadata={"source": "lhz_linear", "logical_site": i},
                )
            )

    # Step 3: Penalty strength — floor at 1.0.
    if J:
        penalty_strength = max(penalty_factor * max(abs(v) for v in J.values()), 1.0)
    else:
        penalty_strength = 1.0

    # Step 4: Plaquettes anchored at the lowest logical site (spanning basis).
    # Generates (n-1)(n-2)/2 independent plaquette constraints for a complete graph.
    plaquettes: list[tuple] = []
    if logical_sites:
        a = logical_sites[0]
        rest = logical_sites[1:]
        for b_idx in range(len(rest)):
            b = rest[b_idx]
            for c_idx in range(b_idx + 1, len(rest)):
                c = rest[c_idx]
                if (a, b) in qubit_map and (a, c) in qubit_map and (b, c) in qubit_map:
                    pa = qubit_map[(a, b)]
                    pb = qubit_map[(a, c)]
                    pc = qubit_map[(b, c)]
                    all_terms.append(
                        HamiltonianTerm(
                            coefficient=-penalty_strength / 2.0,
                            operators=[(pa, "Z"), (pb, "Z"), (pc, "Z")],
                            metadata={"source": "lhz_plaquette"},
                        )
                    )
                    plaquettes.append((pa, pb, pc))

    # Step 5: Pass-through terms.
    all_terms.extend(passthrough)

    # Step 6: Build encoded IR.
    encoded_ir = HamiltonianIR(
        terms=all_terms,
        n_sites=K,
        substrate=ir.substrate,
        metadata={
            "theorem": "universality_theorem.md#theorem-3",
            "n_logical": n_logical,
            "n_physical": K,
        },
    )

    # Step 7: Result metadata.
    metadata: dict[str, Any] = {
        "negative_couplings": [[i, j] for (i, j), v in J.items() if v < 0],
        "penalty_strength": penalty_strength,
        "n_logical": n_logical,
        "n_physical": K,
    }

    return LHZResult(
        encoded_ir=encoded_ir,
        n_logical=n_logical,
        n_physical=K,
        qubit_map=qubit_map,
        plaquettes=plaquettes,
        penalty_strength=penalty_strength,
        h0_constant=h0_constant,
        metadata=metadata,
    )


@dataclass
class LHZCertificate:
    """Certificate quantifying the LHZ penalty gap and correctness guarantee."""

    max_logical_coupling: float
    penalty_strength: float
    penalty_gap: float
    error_tolerance: float
    compilation_certificate: CompilationCertificate | None
    sufficient_for_correctness: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_logical_coupling": self.max_logical_coupling,
            "penalty_strength": self.penalty_strength,
            "penalty_gap": self.penalty_gap,
            "error_tolerance": self.error_tolerance,
            "compilation_certificate": (
                self.compilation_certificate.to_dict()
                if self.compilation_certificate is not None
                else None
            ),
            "sufficient_for_correctness": self.sufficient_for_correctness,
            "metadata": self.metadata,
        }


def certify_lhz(
    lhz_result: LHZResult,
    compilation_certificate: CompilationCertificate | None = None,
) -> LHZCertificate:
    """Compute the LHZ penalty-gap certificate for Theorem 3 correctness.

    The penalty gap P - 2*max|J_ij| guarantees that the parity encoding
    correctly recovers the logical ground state when the physical compilation
    error is smaller than the gap divided by 2.

    Args:
        lhz_result: Output of lhz_parity_pass.
        compilation_certificate: Optional compilation error certificate from
            certify_ising; used to check sufficient_for_correctness.

    Returns:
        LHZCertificate with penalty gap, error tolerance, and correctness flag.
    """
    coupling_values: list[float] = []
    for term in lhz_result.encoded_ir.terms:
        src = term.metadata.get("source")
        if src in ("lhz_coupling", "lhz_linear"):
            coupling_values.append(abs(term.coefficient))

    coupling_values.append(abs(lhz_result.h0_constant))
    max_logical_coupling = max(coupling_values) if coupling_values else 0.0

    penalty_strength = lhz_result.penalty_strength
    penalty_gap = penalty_strength - 2.0 * max_logical_coupling
    error_tolerance = penalty_gap / 2.0

    if compilation_certificate is None:
        sufficient_for_correctness = True
    else:
        sufficient_for_correctness = compilation_certificate.l1_bound < error_tolerance

    metadata: dict[str, Any] = {
        "theorem": "universality_theorem.md#theorem-3",
        "n_logical": lhz_result.n_logical,
        "n_physical": lhz_result.n_physical,
        "plaquette_realizability": (
            "unverified — this certificate covers only coefficient-exactness "
            "of the encoded local fields (h/J); it does not certify physical "
            "realizability of the 3-body plaquette penalty terms themselves"
        ),
    }

    return LHZCertificate(
        max_logical_coupling=max_logical_coupling,
        penalty_strength=penalty_strength,
        penalty_gap=penalty_gap,
        error_tolerance=error_tolerance,
        compilation_certificate=compilation_certificate,
        sufficient_for_correctness=sufficient_for_correctness,
        metadata=metadata,
    )
