"""Exact minimum-weight lookup decoder for a distance-3 surface code.

Built by exhaustively enumerating all possible X-error patterns on the
data qubits and grouping by the Z-stabilizer syndrome they produce -
exact for small instances, the same brute-force-at-small-scale
philosophy as limen.validator.validator's <=20-variable brute force.
"""

from __future__ import annotations

from itertools import product

from limen.ecc.surface_code import SurfaceCodePatch


def compute_syndrome(error: tuple[int, ...], z_stabilizers: list[list[int]]) -> tuple[int, ...]:
    """Compute the Z-stabilizer syndrome produced by an X-error pattern.

    error[i] is 1 if data qubit i is bit-flipped. A stabilizer's bit is
    the parity (XOR) of the error bits on its support.
    """
    syndrome = []
    for stabilizer in z_stabilizers:
        parity = 0
        for q in stabilizer:
            parity ^= error[q]
        syndrome.append(parity)
    return tuple(syndrome)


def _weight(error: tuple[int, ...]) -> int:
    return sum(error)


class LookupDecoder:
    """Minimum-weight syndrome-to-correction lookup table for a SurfaceCodePatch."""

    def __init__(self, patch: SurfaceCodePatch) -> None:
        """Build the lookup table by exhaustive enumeration over all X-error patterns.

        Args:
            patch: The SurfaceCodePatch to decode for.
        """
        self.patch = patch
        n = len(patch.data_qubits)
        table: dict[tuple[int, ...], tuple[int, ...]] = {}
        for bits in product((0, 1), repeat=n):
            syndrome = compute_syndrome(bits, patch.z_stabilizers)
            current = table.get(syndrome)
            if current is None or _weight(bits) < _weight(current):
                table[syndrome] = bits
        self._table = table
        self._n = n

    def syndrome_for(self, error: list[int]) -> tuple[int, ...]:
        """Return the Z-stabilizer syndrome for a list of flipped data-qubit indices."""
        bits = tuple(1 if i in error else 0 for i in range(self._n))
        return compute_syndrome(bits, self.patch.z_stabilizers)

    def decode(self, syndrome: tuple[int, ...]) -> list[int]:
        """Return the minimum-weight correction (flipped qubit indices) for a syndrome.

        Args:
            syndrome: A tuple of 0/1 values, one per Z-stabilizer, in the
                same order as patch.z_stabilizers.

        Returns:
            List of data-qubit indices to flip to correct the error.
        """
        bits = self._table.get(syndrome, tuple([0] * self._n))
        return [i for i, b in enumerate(bits) if b]
