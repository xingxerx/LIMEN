# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Pure-Python fallback for limen_core.qubo_energy_spectrum.

Used when limen_core is not built; see limen.qubo_spectrum for the
shared wrapper that dispatches between this and the Rust extension.
"""

from itertools import product


def qubo_energy_spectrum(
    qubo: list[tuple[tuple[int, int], float]],
    n_vars: int,
) -> tuple[list[int], float, list[float]]:
    """Exhaustively enumerate a QUBO's energy spectrum (index-based).

    Args:
        qubo: QUBO terms as [((var_i, var_j), weight)] using 0-based
            variable indices. i == j encodes a linear term.
        n_vars: Number of variables. Must be <= 20.

    Returns:
        A (best_assignment_bits, best_energy, sorted_distinct_energies)
        tuple, matching limen_core.qubo_energy_spectrum's contract exactly.

    Raises:
        ValueError: If n_vars > 20.
    """
    if n_vars > 20:
        raise ValueError("SizeViolation: qubo_energy_spectrum requires n_vars <= 20")

    if n_vars == 0:
        return [], 0.0, [0.0]

    best_energy = float("inf")
    best_bits: list[int] = [0] * n_vars
    distinct_energies: set[float] = set()

    for bits in product((0, 1), repeat=n_vars):
        energy = sum(w * bits[i] * bits[j] for (i, j), w in qubo)
        if energy < best_energy:
            best_energy = energy
            best_bits = list(bits)
        distinct_energies.add(energy)

    return best_bits, best_energy, sorted(distinct_energies)
