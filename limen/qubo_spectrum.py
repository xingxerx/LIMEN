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
"""Shared QUBO exhaustive-enumeration helper.

Consolidates the O(2^n) brute-force QUBO enumeration that previously existed
independently in three places:

- ``limen.validator.validator.brute_force_solve``
- ``limen.codesign.solver._second_best_energy``
- ``limen.backends.qiskit_backend._enumerate_assignments`` (+ its
  ``_qubo_energy`` usage inside ``ibm_noise_fn``)

All three now call :func:`qubo_energy_spectrum` once per QUBO, which uses the
``limen_core`` Rust extension when built and falls back to an equivalent
pure-Python implementation (:mod:`limen._qubo_spectrum_pyfallback`)
otherwise.
"""

from dataclasses import dataclass

try:
    from limen_core import qubo_energy_spectrum as _qubo_energy_spectrum_rust

    _RUST_AVAILABLE = True
except ImportError:
    from limen._qubo_spectrum_pyfallback import (
        qubo_energy_spectrum as _qubo_energy_spectrum_rust,
    )

    _RUST_AVAILABLE = False


@dataclass
class QuboSpectrum:
    """The result of exhaustively enumerating a QUBO's energy spectrum.

    Attributes:
        variables: Sorted variable names, in the order assignment bits map.
        best_assignment: Variable name -> 0/1 dict for the minimum-energy
            assignment.
        best_energy: The minimum energy found.
        distinct_energies: Ascending list of unique energies across all
            2^n assignments.
    """

    variables: list[str]
    best_assignment: dict[str, int]
    best_energy: float
    distinct_energies: list[float]


def qubo_energy_spectrum(
    qubo: dict[tuple[str, str], float],
) -> QuboSpectrum | None:
    """Exhaustively enumerate a QUBO's energy spectrum.

    Args:
        qubo: QUBO dict mapping (variable, variable) pairs to float weights.
            Diagonal entries (i, i) encode linear terms.

    Returns:
        A QuboSpectrum, or None if the problem has more than 20 variables
        (too large to enumerate classically).
    """
    variables: list[str] = sorted({name for pair in qubo for name in pair})
    if len(variables) > 20:
        return None

    if not variables:
        return QuboSpectrum(
            variables=[], best_assignment={}, best_energy=0.0, distinct_energies=[0.0]
        )

    index_of = {name: idx for idx, name in enumerate(variables)}
    indexed_terms = [
        ((index_of[i], index_of[j]), w) for (i, j), w in qubo.items()
    ]

    best_bits, best_energy, distinct_energies = _qubo_energy_spectrum_rust(
        indexed_terms, len(variables)
    )

    best_assignment = dict(zip(variables, (int(b) for b in best_bits)))

    return QuboSpectrum(
        variables=variables,
        best_assignment=best_assignment,
        best_energy=best_energy,
        distinct_energies=list(distinct_energies),
    )
