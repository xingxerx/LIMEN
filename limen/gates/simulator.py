"""Pure-Python statevector simulator for CircuitIR.

The offline classical backend for the gate-model track, mirroring what
limen.analog.backends.classical_sim is for the diagonal Ising track: it
lets CircuitIR be executed, tested, and composed without Qiskit installed.
limen.gates.qiskit_exec remains the path to real/accelerated hardware.

Convention: a basis-state index s assigns qubit q the bit (s >> q) & 1,
and bitstrings are emitted qubit-0-first (bitstring[q] is qubit q). This
differs from Qiskit's little-endian get_counts ordering; callers that mix
the two backends must reverse accordingly.
"""

from __future__ import annotations

import cmath
import math
import random

from limen.gates.ir import CircuitIR
from limen.gates.synthesis import u_matrix

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _single_qubit_matrix(name: str, params: list[float]) -> list[list[complex]]:
    """Return the 2x2 matrix for a single-qubit gate in KNOWN_GATES."""
    if name == "h":
        return [[_INV_SQRT2, _INV_SQRT2], [_INV_SQRT2, -_INV_SQRT2]]
    if name == "x":
        return [[0, 1], [1, 0]]
    if name == "y":
        return [[0, -1j], [1j, 0]]
    if name == "z":
        return [[1, 0], [0, -1]]
    if name == "s":
        return [[1, 0], [0, 1j]]
    if name == "t":
        return [[1, 0], [0, cmath.exp(1j * math.pi / 4)]]
    if name == "rx":
        c, s = math.cos(params[0] / 2), math.sin(params[0] / 2)
        return [[c, -1j * s], [-1j * s, c]]
    if name == "ry":
        c, s = math.cos(params[0] / 2), math.sin(params[0] / 2)
        return [[c, -s], [s, c]]
    if name == "rz":
        return [[cmath.exp(-1j * params[0] / 2), 0], [0, cmath.exp(1j * params[0] / 2)]]
    if name == "u":
        return u_matrix(params[0], params[1], params[2])
    raise ValueError(f"'{name}' is not a single-qubit gate")


def _apply_1q(state: list[complex], n: int, q: int, m: list[list[complex]]) -> None:
    """Apply a 2x2 gate to qubit q of an n-qubit state, in place."""
    a, b = m[0]
    c, d = m[1]
    step = 1 << q
    for base in range(0, 1 << n, step << 1):
        for s0 in range(base, base + step):
            s1 = s0 | step
            v0, v1 = state[s0], state[s1]
            state[s0] = a * v0 + b * v1
            state[s1] = c * v0 + d * v1


def _apply_2q(state: list[complex], n: int, name: str, qubits: list[int]) -> None:
    """Apply a two-qubit gate (cx, cz, swap) to the state, in place."""
    a, b = qubits
    abit, bbit = 1 << a, 1 << b
    for s in range(1 << n):
        if name == "cx":
            if (s & abit) and not (s & bbit):
                t = s | bbit
                state[s], state[t] = state[t], state[s]
        elif name == "cz":
            if (s & abit) and (s & bbit):
                state[s] = -state[s]
        elif name == "swap":
            if (s & abit) and not (s & bbit):
                t = (s & ~abit) | bbit
                state[s], state[t] = state[t], state[s]


def statevector(circuit: CircuitIR) -> list[complex]:
    """Return the final statevector of `circuit` starting from |0...0>.

    Raises:
        ValueError: If circuit.validate() reports any errors.
    """
    errors = circuit.validate()
    if errors:
        raise ValueError(f"invalid CircuitIR: {errors}")

    n = circuit.n_qubits
    state: list[complex] = [0j] * (1 << n)
    state[0] = 1.0 + 0j

    for ins in circuit.instructions:
        if len(ins.qubits) == 1:
            _apply_1q(state, n, ins.qubits[0], _single_qubit_matrix(ins.name, ins.params))
        else:
            _apply_2q(state, n, ins.name, ins.qubits)
    return state


def _index_to_bitstring(s: int, n: int) -> str:
    """Render basis index s as a qubit-0-first bitstring of length n."""
    return "".join(str((s >> q) & 1) for q in range(n))


def probabilities(circuit: CircuitIR, threshold: float = 1e-12) -> dict[str, float]:
    """Return the exact measurement distribution of `circuit`.

    Keys are qubit-0-first bitstrings; values are probabilities. Outcomes
    with probability <= `threshold` are omitted.
    """
    state = statevector(circuit)
    n = circuit.n_qubits
    dist: dict[str, float] = {}
    for s, amp in enumerate(state):
        p = (amp.conjugate() * amp).real
        if p > threshold:
            dist[_index_to_bitstring(s, n)] = p
    return dist


def sample_counts(circuit: CircuitIR, shots: int = 1000, seed: int = 42) -> dict[str, int]:
    """Sample measurement counts from `circuit` using a seeded RNG.

    Keys are qubit-0-first bitstrings. Deterministic for a fixed seed.
    """
    dist = probabilities(circuit)
    outcomes = sorted(dist)
    weights = [dist[o] for o in outcomes]
    rng = random.Random(seed)
    counts: dict[str, int] = {}
    for outcome in rng.choices(outcomes, weights=weights, k=shots):
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts
