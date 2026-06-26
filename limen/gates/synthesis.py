"""Unitary synthesis pass for the gate-model IR.

Only the well-defined closed-form case is implemented: an arbitrary
single-qubit unitary decomposes exactly into one `u(theta, phi, lambda)`
gate via the standard ZYZ Euler-angle construction. Multi-qubit unitary
synthesis (KAK decomposition, gate-count-optimal compilation) is real
numerical-research scope on its own and is explicitly out of scope here
- see limen/docs/architecture.md for how the project documents this kind
of gap rather than half-implementing it.

A global phase is discarded during decomposition. This is standard for
single-qubit gate synthesis: global phase has no effect on measurement
statistics for a qubit that is never used as a control.
"""

from __future__ import annotations

import cmath
import math

from limen.exceptions import GateSynthesisError
from limen.gates.ir import GateInstruction

_TOLERANCE = 1e-6


def decompose_unitary_1q(matrix: list[list[complex]]) -> GateInstruction:
    """Decompose a 2x2 unitary matrix into a single `u` GateInstruction.

    Args:
        matrix: A 2x2 unitary matrix, as a list of two rows of two
            complex (or real) entries each.

    Returns:
        A GateInstruction("u", [0], [theta, phi, lambda]) such that the
        standard U(theta, phi, lambda) gate matrix equals `matrix` up to
        a global phase.

    Raises:
        GateSynthesisError: If `matrix` is not 2x2, or is not unitary.
            For unitaries on 2 or more qubits, use qiskit.transpile or
            qiskit.circuit.library.UnitaryGate instead.
    """
    if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise GateSynthesisError(
            "decompose_unitary_1q only supports 2x2 (single-qubit) unitaries; "
            "multi-qubit unitary synthesis is out of scope for LIMEN's own "
            "synthesis pass - use qiskit.transpile or "
            "qiskit.circuit.library.UnitaryGate for those cases."
        )

    a, b = complex(matrix[0][0]), complex(matrix[0][1])
    c, d = complex(matrix[1][0]), complex(matrix[1][1])

    if not _is_unitary(a, b, c, d):
        raise GateSynthesisError("decompose_unitary_1q requires a unitary matrix")

    norm = cmath.exp(-1j * cmath.phase(a)) if abs(a) > _TOLERANCE else 1.0 + 0j
    a2, b2, c2, d2 = a * norm, b * norm, c * norm, d * norm

    theta = 2 * math.acos(max(-1.0, min(1.0, abs(a2))))
    sin_half = math.sin(theta / 2)

    if sin_half > _TOLERANCE:
        lam = cmath.phase(-b2)
        phi = cmath.phase(c2)
    else:
        lam = cmath.phase(d2) if abs(d2) > _TOLERANCE else 0.0
        phi = 0.0

    return GateInstruction(name="u", qubits=[0], params=[theta, phi, lam])


def u_matrix(theta: float, phi: float, lam: float) -> list[list[complex]]:
    """Return the standard U(theta, phi, lambda) gate matrix.

    Inverse of decompose_unitary_1q up to the discarded global phase:
    u_matrix(*decompose_unitary_1q(m).params) reconstructs m up to a
    global phase factor.
    """
    cos_half = math.cos(theta / 2)
    sin_half = math.sin(theta / 2)
    return [
        [complex(cos_half), -cmath.exp(1j * lam) * sin_half],
        [cmath.exp(1j * phi) * sin_half, cmath.exp(1j * (phi + lam)) * cos_half],
    ]


def _is_unitary(a: complex, b: complex, c: complex, d: complex) -> bool:
    """Check M^dagger @ M == I for a 2x2 matrix [[a, b], [c, d]]."""
    m00 = a.conjugate() * a + c.conjugate() * c
    m11 = b.conjugate() * b + d.conjugate() * d
    m01 = a.conjugate() * b + c.conjugate() * d
    return (
        abs(m00 - 1.0) < 1e-6
        and abs(m11 - 1.0) < 1e-6
        and abs(m01) < 1e-6
    )
