# Copyright (C) 2026 Jemone McCubbin / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Compilation certificates for analog substrate compilation.

Implements Theorem 1 of limen/docs/universality_theorem.md: for diagonal
(Z-basis) Ising Hamiltonians, the operator-norm compilation error

    ||H_target - H_compiled||_op

is exactly computable by enumeration for n <= 20 sites, and is always
bounded above by the L1 norm of the coefficient errors:

    ||dH||_op = max_{s in {-1,+1}^n} |sum_i dh_i s_i + sum_{i<j} dJ_ij s_i s_j|
             <= sum_i |dh_i| + sum_{i<j} |dJ_ij|

Both quantities follow directly from the fact that all Z and ZZ terms
commute and are simultaneously diagonal in the computational basis, so the
operator norm of the difference equals the maximum absolute eigenvalue,
which is realised on a spin configuration. This is the computable error
bound ("compilation certificate") required by the constructive universality
specification in limen/docs/architecture.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from limen.quantum_channel.qkd import QKDResult
    from limen.quantum_channel.teleport import TeleportResult

MAX_EXACT_SITES: int = 20


@dataclass
class CompilationCertificate:
    """A computable bound on the compilation error of an analog encoding.

    Attributes:
        l1_bound: Upper bound on the operator-norm error: the L1 norm of
            all linear and quadratic coefficient errors. Always finite and
            cheap to compute (Theorem 1, inequality part).
        operator_norm: The exact operator-norm error, computed by
            enumerating all 2^n spin configurations. None when
            n_sites > MAX_EXACT_SITES (Theorem 1, equality part).
        n_sites: Number of sites the certificate covers.
        max_linear_error: Largest single |dh_i| coefficient error.
        max_quadratic_error: Largest single |dJ_ij| coefficient error.
        natively_realizable: True when the target Hamiltonian lies inside
            the substrate's natively realizable class (e.g. all J_ij > 0
            for van der Waals neutral-atom arrays — Theorem 2). When False,
            exact compilation requires the parity-encoding route (Theorem 3)
            and the certificate quantifies the approximation gap instead.
        notes: Human-readable observations.
        metadata: Arbitrary annotations.
    """

    l1_bound: float
    operator_norm: float | None
    n_sites: int
    max_linear_error: float
    max_quadratic_error: float
    natively_realizable: bool
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe plain Python dict."""
        return {
            "l1_bound": self.l1_bound,
            "operator_norm": self.operator_norm,
            "n_sites": self.n_sites,
            "max_linear_error": self.max_linear_error,
            "max_quadratic_error": self.max_quadratic_error,
            "natively_realizable": self.natively_realizable,
            "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CompilationCertificate:
        """Deserialize from a plain Python dict produced by to_dict()."""
        return cls(
            l1_bound=float(d["l1_bound"]),
            operator_norm=(
                float(d["operator_norm"]) if d.get("operator_norm") is not None else None
            ),
            n_sites=int(d["n_sites"]),
            max_linear_error=float(d["max_linear_error"]),
            max_quadratic_error=float(d["max_quadratic_error"]),
            natively_realizable=bool(d["natively_realizable"]),
            notes=list(d.get("notes", [])),
            metadata=dict(d.get("metadata", {})),
        )


def _delta_energy(
    dh: dict[int, float],
    dJ: dict[tuple[int, int], float],
    spins: tuple[int, ...],
) -> float:
    """Evaluate the error Hamiltonian dH at one spin configuration."""
    energy = sum(v * spins[i] for i, v in dh.items())
    energy += sum(v * spins[i] * spins[j] for (i, j), v in dJ.items())
    return energy


def certify_ising(
    target_h: dict[int, float],
    target_J: dict[tuple[int, int], float],
    compiled_h: dict[int, float],
    compiled_J: dict[tuple[int, int], float],
    n_sites: int,
    natively_realizable: bool = True,
    notes: list[str] | None = None,
    max_exact_sites: int = MAX_EXACT_SITES,
) -> CompilationCertificate:
    """Produce a compilation certificate for a compiled Ising Hamiltonian.

    Computes the coefficient errors dh = compiled_h - target_h and
    dJ = compiled_J - target_J, then:

    - Always computes the L1 bound sum|dh| + sum|dJ| (valid upper bound on
      the operator-norm error for any n — Theorem 1).
    - For n_sites <= max_exact_sites, also computes the exact operator norm
      by enumerating all 2^n spin configurations.

    Args:
        target_h: Target linear Ising coefficients {site: h_i}.
        target_J: Target quadratic Ising coefficients {(i, j): J_ij}, i < j.
        compiled_h: As-compiled (or predicted as-executed) linear coefficients.
        compiled_J: As-compiled quadratic coefficients.
        n_sites: Number of sites in the Hamiltonian.
        natively_realizable: Whether the target lies in the substrate's
            natively realizable class (substrate-specific — see Theorem 2).
        notes: Optional extra notes to attach.
        max_exact_sites: Enumeration cutoff for the exact norm (default 20).

    Returns:
        A CompilationCertificate. operator_norm is None above the cutoff.
    """
    dh = {
        i: compiled_h.get(i, 0.0) - target_h.get(i, 0.0)
        for i in set(target_h) | set(compiled_h)
    }
    dh = {i: v for i, v in dh.items() if v != 0.0}
    dJ = {
        k: compiled_J.get(k, 0.0) - target_J.get(k, 0.0)
        for k in set(target_J) | set(compiled_J)
    }
    dJ = {k: v for k, v in dJ.items() if v != 0.0}

    l1 = sum(abs(v) for v in dh.values()) + sum(abs(v) for v in dJ.values())
    max_lin = max((abs(v) for v in dh.values()), default=0.0)
    max_quad = max((abs(v) for v in dJ.values()), default=0.0)

    op_norm: float | None = None
    if n_sites <= max_exact_sites:
        if n_sites == 0 or (not dh and not dJ):
            op_norm = 0.0
        else:
            try:
                from limen_core import exact_ising_norm as _rust_norm
                op_norm = _rust_norm(
                    list(dh.items()),
                    list(dJ.items()),
                    n_sites,
                )
            except ImportError:
                op_norm = max(
                    abs(_delta_energy(dh, dJ, spins))
                    for spins in product((1, -1), repeat=n_sites)
                )

    cert_notes = list(notes) if notes else []
    if op_norm is None:
        cert_notes.append(
            f"n_sites={n_sites} exceeds exact-enumeration cutoff "
            f"({max_exact_sites}); operator_norm reported as L1 bound only."
        )
    if not natively_realizable:
        cert_notes.append(
            "Target lies outside the substrate's natively realizable class. "
            "Exact compilation requires the parity-encoding route "
            "(Theorem 3, limen/docs/universality_theorem.md); the certificate "
            "above bounds the native-approximation error."
        )

    return CompilationCertificate(
        l1_bound=l1,
        operator_norm=op_norm,
        n_sites=n_sites,
        max_linear_error=max_lin,
        max_quadratic_error=max_quad,
        natively_realizable=natively_realizable,
        notes=cert_notes,
        metadata={"theorem": "universality_theorem.md#theorem-1"},
    )


def attach_quantum_channel_results(
    cert: CompilationCertificate,
    qkd_result: QKDResult | None = None,
    teleport_result: TeleportResult | None = None,
) -> None:
    """Record quantum-channel primitive results on a certificate's metadata.

    Lets a CompilationCertificate double as the single source of truth for
    an entire quantum network compilation run: the analog/Ising compilation
    error bound alongside the QKD and teleportation primitives that moved
    data over the same channel.
    """
    if qkd_result is not None:
        cert.metadata["qber"] = qkd_result.qber
        cert.metadata["qkd_secure"] = qkd_result.secure
        cert.metadata["qkd_backend"] = qkd_result.backend
        cert.metadata["qkd_job_id"] = qkd_result.job_id
    if teleport_result is not None:
        cert.metadata["teleport_fidelity"] = teleport_result.fidelity_estimate
        cert.metadata["teleport_success"] = teleport_result.success
        cert.metadata["teleport_backend"] = teleport_result.backend
        cert.metadata["teleport_job_id"] = teleport_result.job_id
