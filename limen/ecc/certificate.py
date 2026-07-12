"""Logical error rate certification for a surface-code-protected logical qubit.

A parallel concept to limen.analog.certificate.CompilationCertificate
(Hamiltonian coefficient error / Theorem 1) - this module quantifies
quantum logical error rate instead, and the two are not composed.

Exact brute-force enumeration over all 2^(d^2) X-error patterns, not
the asymptotic surface-code threshold formula - consistent with how
CompilationCertificate does exact enumeration for n<=20.

Scope limit: independent per-qubit bit-flip (X-error) noise model only.
Y-error/depolarizing noise and Z-stabilizer-side (Z-error) decoding are
out of scope for this milestone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

from limen.ecc.decoder import LookupDecoder, compute_syndrome
from limen.ecc.surface_code import SurfaceCodePatch


@dataclass
class LogicalErrorCertificate:
    """Logical error rate estimate for one surface-code-protected logical qubit.

    Attributes:
        logical_error_rate: Probability that decoding fails to recover
            the encoded logical state, under the independent bit-flip
            noise model.
        physical_error_rate: Per-qubit independent bit-flip probability
            used for the estimate.
        distance: Code distance.
        n_physical_qubits: Number of data qubits (ancillas not counted;
            they are not subject to data-qubit noise in this model).
        n_logical_qubits: Number of logical qubits encoded (always 1
            for a single SurfaceCodePatch).
        decoder: Name of the decoder used.
        notes: Human-readable scope/assumption notes.
        metadata: Arbitrary annotations.
    """

    logical_error_rate: float
    physical_error_rate: float
    distance: int
    n_physical_qubits: int
    n_logical_qubits: int = 1
    decoder: str = "LookupDecoder"
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def certify_logical_qubit(
    patch: SurfaceCodePatch, decoder: LookupDecoder, physical_error_rate: float
) -> LogicalErrorCertificate:
    """Compute the exact logical error rate for a surface-code patch.

    Brute-forces all 2^(d^2) independent-bit-flip X-error patterns,
    weights each by its exact binomial probability, decodes via
    `decoder`, and sums the probability mass of patterns where
    `error XOR correction` overlaps `patch.logical_z` on an odd number
    of qubits (an undetected residual logical bit-flip).

    Args:
        patch: The SurfaceCodePatch being certified.
        decoder: A LookupDecoder built for the same patch.
        physical_error_rate: Independent per-qubit bit-flip probability.

    Returns:
        A LogicalErrorCertificate with the computed logical_error_rate.
    """
    n = len(patch.data_qubits)
    p = physical_error_rate
    logical_z_set = set(patch.logical_z)

    try:
        from limen.limen_core import logical_failure_probability as _rust_failure
    except ImportError:
        _rust_failure = None

    if _rust_failure is not None:
        # The Rust path rebuilds the same minimum-weight lookup table
        # internally (identical tie-breaking), so the whole 2^n
        # enumerate-decode-score loop runs natively.
        failure_probability = _rust_failure(
            n, patch.z_stabilizers, patch.logical_z, p
        )
    else:
        failure_probability = 0.0
        for bits in product((0, 1), repeat=n):
            weight = sum(bits)
            prob = (p**weight) * ((1 - p) ** (n - weight))
            syndrome = compute_syndrome(bits, patch.z_stabilizers)
            correction = decoder.decode(syndrome)
            residual = set(i for i, b in enumerate(bits) if b)
            residual.symmetric_difference_update(correction)
            if len(residual & logical_z_set) % 2 == 1:
                failure_probability += prob

    return LogicalErrorCertificate(
        logical_error_rate=failure_probability,
        physical_error_rate=physical_error_rate,
        distance=patch.distance,
        n_physical_qubits=n,
        decoder=type(decoder).__name__,
        notes=[
            "Independent per-qubit bit-flip (X-error) noise model only; "
            "Y-error/depolarizing noise and Z-error decoding are out of scope.",
            "Exact brute-force enumeration, not the asymptotic threshold formula.",
        ],
    )
