"""Distance-3 rotated surface code construction.

Builds the stabilizer structure algorithmically from the standard
bulk/boundary lattice rule, rather than transcribing a specific named
qubit layout from memory. Correctness (that the result is actually a
valid distance-3 CSS code) is proven by brute-force enumeration in
tests/test_surface_code.py, not assumed from this construction alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SurfaceCodePatch:
    """One logical qubit encoded in a rotated surface code patch.

    Attributes:
        distance: Code distance (only distance=3 is tested/supported).
        data_qubits: Indices of the d^2 data qubits (0..d^2-1).
        x_stabilizers: Each a list of data-qubit indices forming an
            X-type stabilizer's support.
        z_stabilizers: Each a list of data-qubit indices forming a
            Z-type stabilizer's support.
        logical_x: Data-qubit indices forming a representative logical
            X operator (one full row).
        logical_z: Data-qubit indices forming a representative logical
            Z operator (one full column).
    """

    distance: int
    data_qubits: list[int] = field(default_factory=list)
    x_stabilizers: list[list[int]] = field(default_factory=list)
    z_stabilizers: list[list[int]] = field(default_factory=list)
    logical_x: list[int] = field(default_factory=list)
    logical_z: list[int] = field(default_factory=list)


def build_surface_code(distance: int = 3) -> SurfaceCodePatch:
    """Build a rotated surface code patch encoding one logical qubit.

    Data qubits sit on a distance x distance grid, indexed row-major
    (qubit at row r, col c has index r*distance + c). Stabilizers are
    placed at lattice sites between data qubits: weight-4 in the bulk,
    weight-2 at the boundary. Boundary stabilizers alternate by edge
    orientation so that exactly (distance^2 - 1) independent stabilizers
    result, matching the [[distance^2, 1, distance]] code parameters.

    Args:
        distance: Code distance. Only distance=3 is tested/supported;
            other odd distances use the same construction but are
            unverified by this module's test suite.

    Returns:
        A SurfaceCodePatch with data qubits, stabilizers, and
        representative logical operators.
    """
    d = distance

    def qubit_index(r: int, c: int) -> int:
        return r * d + c

    def neighbors(r: int, c: int) -> list[tuple[int, int]]:
        candidates = [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
        return [(rr, cc) for rr, cc in candidates if 0 <= rr < d and 0 <= cc < d]

    x_stabilizers: list[list[int]] = []
    z_stabilizers: list[list[int]] = []

    for r in range(-1, d):
        for c in range(-1, d):
            support = neighbors(r, c)
            if len(support) == 4:
                # Bulk site: checkerboard parity decides stabilizer type.
                qubits = [qubit_index(rr, cc) for rr, cc in support]
                if (r + c) % 2 == 0:
                    z_stabilizers.append(qubits)
                else:
                    x_stabilizers.append(qubits)
            elif len(support) == 2:
                # Boundary site: keep exactly one weight-2 stabilizer per
                # edge. Top/bottom edges (varying column) are Z-type;
                # left/right edges (varying row) are X-type. Of the two
                # candidate sites along each edge, keep the one whose
                # bulk-style parity matches the edge's assigned type, so
                # boundary stabilizers extend the bulk checkerboard
                # pattern rather than duplicating or skipping it.
                is_horizontal_edge = r in (-1, d - 1)
                qubits = [qubit_index(rr, cc) for rr, cc in support]
                edge_parity = (r + c) % 2
                if is_horizontal_edge:
                    if edge_parity == 0:
                        z_stabilizers.append(qubits)
                else:
                    if edge_parity == 1:
                        x_stabilizers.append(qubits)
            # len(support) in (0, 1): corner sites, not a stabilizer.

    data_qubits = [qubit_index(r, c) for r in range(d) for c in range(d)]
    logical_x = [qubit_index(0, c) for c in range(d)]
    logical_z = [qubit_index(r, 0) for r in range(d)]

    return SurfaceCodePatch(
        distance=d,
        data_qubits=data_qubits,
        x_stabilizers=x_stabilizers,
        z_stabilizers=z_stabilizers,
        logical_x=logical_x,
        logical_z=logical_z,
    )
