# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""QuEra Aquila backend adapter for LIMEN, via the AWS Braket SDK.

Aquila is a programmable *analog* neutral-atom device, not a gate-model or
quantum-annealing QUBO sampler like the qpu/dwave backends. Two physical
degrees of freedom are programmable per shot:

  - a single global Rabi drive (amplitude/phase), ramped adiabatically, and
  - the 2D positions of the atoms, which fix the Rydberg (van der Waals)
    interaction strength between every pair of atoms (~ C6 / r^6) — a
    *position-derived* coupling, not an arbitrary tunable per-edge weight.

This module approximates a QUBO's quadratic couplings with the same
unit-disk-graph trick used in QuEra's own Maximum-Independent-Set examples:
any two variables with a non-zero coupling are placed within the Rydberg
blockade radius (full blockade, antiferromagnetic), and every other pair is
placed beyond it (negligible interaction). Linear (diagonal) QUBO terms are
NOT encoded — Aquila also supports per-atom local detuning ("shifting
fields") for that, but wiring it up is left as future work; see
limen/docs/architecture.md for how LIMEN documents such gaps. The atom
placement here additionally only checks each variable against its immediate
predecessor in sorted order, so it correctly realizes chain/sparse-coupling
graphs (Max-Cut on a cycle, TSP edge terms, etc.) but not dense graphs with
non-adjacent couplings — a real unit-disk embedding solver would be needed
for the general case.

All Braket SDK imports are guarded so this module loads cleanly even when
the SDK is not installed. Import errors surface at call time with a clear
install hint.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from limen.core.compiler import PhysicalEncoding

_INSTALL_MSG = (
    "The Amazon Braket SDK is required to use the QuEra Aquila backend. "
    "Install it with: pip install limen[braket]  "
    "(or: pip install amazon-braket-sdk)"
)

DEFAULT_AQUILA_ARN = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"

_BLOCKADE_RADIUS_UM = 8.0  # coupled pairs are placed this close (within blockade)
_FAR_SEPARATION_UM = 20.0  # uncoupled pairs are placed this far apart (no interaction)

_DRIVE_TIME_S = 4.0e-6
_RAMP_TIME_S = 0.1e-6
_RABI_FREQUENCY_RAD_S = 15.8e6
_DETUNING_START_RAD_S = -50.0e6
_DETUNING_END_RAD_S = 50.0e6


@dataclass
class BraketResult:
    """The result of an analog Hamiltonian simulation run on Aquila.

    Attributes:
        samples: One dict per valid shot, mapping variable name to binary
            value (0 or 1). Shots where an atom was lost ('e' in the
            decoded measurement) are post-selected out.
        energies: QUBO energy of each sample, same order as samples.
        best_assignment: The lowest-energy sample.
        best_energy: The energy of best_assignment.
        device_arn: The Braket device ARN that was run against.
        shots: Number of shots requested.
        task_arn: The Braket quantum task ARN/ID, when available.
        metadata: Diagnostic info: num_atoms and how many shots were
            discarded to atom loss.
    """

    samples: list[dict[str, int]]
    energies: list[float]
    best_assignment: dict[str, int]
    best_energy: float
    device_arn: str
    shots: int
    task_arn: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _import_ahs_types():
    try:
        from braket.ahs.analog_hamiltonian_simulation import (  # type: ignore[import]
            AnalogHamiltonianSimulation,
        )
        from braket.ahs.atom_arrangement import AtomArrangement  # type: ignore[import]
        from braket.ahs.driving_field import DrivingField  # type: ignore[import]
        from braket.timings.time_series import TimeSeries  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc
    return AtomArrangement, DrivingField, AnalogHamiltonianSimulation, TimeSeries


def _import_device(device_arn: str, use_qpu: bool) -> Any:
    if use_qpu:
        try:
            from braket.aws import AwsDevice  # type: ignore[import]
        except ModuleNotFoundError as exc:
            raise ImportError(_INSTALL_MSG) from exc
        return AwsDevice(device_arn)
    try:
        from braket.devices import LocalSimulator  # type: ignore[import]
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc
    return LocalSimulator("braket_ahs")


def _atom_positions(
    order: list[str], coupling: dict[tuple[str, str], float]
) -> dict[str, tuple[float, float]]:
    """Lay out atoms on a 1D chain, pulling coupled neighbours within blockade radius.

    See the module docstring for the unit-disk-graph approximation this
    implements and its limitation to chain/sparse coupling graphs.
    """
    coupled = {
        frozenset((i, j)) for (i, j), w in coupling.items() if i != j and abs(w) > 1e-12
    }
    positions: dict[str, tuple[float, float]] = {}
    prev: str | None = None
    for v in order:
        if prev is None:
            positions[v] = (0.0, 0.0)
        else:
            px, _ = positions[prev]
            step = _BLOCKADE_RADIUS_UM * 0.6 if frozenset((prev, v)) in coupled else _FAR_SEPARATION_UM
            positions[v] = (px + step, 0.0)
        prev = v
    return positions


def _decode_counts(result: Any) -> Counter:
    """Decode Aquila's per-shot pre/post measurement sequences into state strings.

    Each character is 'e' (atom lost/empty site), 'r' (Rydberg state), or
    'g' (ground state) — the same decode used in QuEra's own Braket example
    notebooks.
    """
    states = ["e", "r", "g"]
    counts: Counter = Counter()
    for shot in result.measurements:
        idx = []
        for pre_i, post_i in zip(shot.pre_sequence, shot.post_sequence):
            if pre_i == 0:
                idx.append(0)
            elif post_i == 0:
                idx.append(1)
            else:
                idx.append(2)
        counts["".join(states[i] for i in idx)] += 1
    return counts


def run_braket(
    encoding: PhysicalEncoding,
    device_arn: str = DEFAULT_AQUILA_ARN,
    shots: int = 100,
    use_qpu: bool = False,
) -> BraketResult:
    """Submit a PhysicalEncoding to QuEra Aquila (or its local simulator).

    Args:
        encoding: A compiled PhysicalEncoding from the LIMEN compiler. Only
            its `.qubo` (physical-label keyed) is used; couplings are
            approximated via atom placement (see module docstring).
        device_arn: Braket device ARN. Defaults to Aquila; pass a simulator
            ARN or leave use_qpu=False to run locally instead.
        shots: Number of shots to run.
        use_qpu: If True, submit to the real AwsDevice at device_arn
            (requires AWS credentials with Braket access). If False
            (default), runs on Braket's local AHS simulator.

    Returns:
        A BraketResult with all valid (non-atom-loss) samples sorted by
        the order they were measured, plus the best (lowest-energy) one.

    Raises:
        ImportError: If the Amazon Braket SDK is not installed.
        RuntimeError: If every shot lost an atom and no valid sample exists.
    """
    AtomArrangement, DrivingField, AnalogHamiltonianSimulation, TimeSeries = _import_ahs_types()

    order = sorted({name for pair in encoding.qubo for name in pair})
    positions = _atom_positions(order, encoding.qubo)

    register = AtomArrangement()
    for v in order:
        x, y = positions[v]
        register.add([x * 1e-6, y * 1e-6])

    amplitude = TimeSeries()
    amplitude.put(0.0, 0.0)
    amplitude.put(_RAMP_TIME_S, _RABI_FREQUENCY_RAD_S)
    amplitude.put(_DRIVE_TIME_S - _RAMP_TIME_S, _RABI_FREQUENCY_RAD_S)
    amplitude.put(_DRIVE_TIME_S, 0.0)

    detuning = TimeSeries()
    detuning.put(0.0, _DETUNING_START_RAD_S)
    detuning.put(_RAMP_TIME_S, _DETUNING_START_RAD_S)
    detuning.put(_DRIVE_TIME_S - _RAMP_TIME_S, _DETUNING_END_RAD_S)
    detuning.put(_DRIVE_TIME_S, _DETUNING_END_RAD_S)

    phase = TimeSeries()
    phase.put(0.0, 0.0)
    phase.put(_DRIVE_TIME_S, 0.0)

    drive = DrivingField(amplitude=amplitude, phase=phase, detuning=detuning)
    program = AnalogHamiltonianSimulation(register=register, hamiltonian=drive)

    device = _import_device(device_arn, use_qpu)
    task = device.run(program, shots=shots)
    result = task.result()

    counts = _decode_counts(result)
    samples: list[dict[str, int]] = []
    energies: list[float] = []
    discarded = 0
    for state, n in counts.items():
        if "e" in state:
            discarded += n
            continue
        assignment = {v: (1 if ch == "r" else 0) for v, ch in zip(order, state)}
        e = sum(w * assignment[i] * assignment[j] for (i, j), w in encoding.qubo.items())
        for _ in range(n):
            samples.append(assignment)
            energies.append(e)

    if not samples:
        raise RuntimeError(
            "All shots lost an atom (measured state 'e'); no valid samples to report. "
            "Try increasing shots or check the device's calibration."
        )

    best_idx = min(range(len(energies)), key=energies.__getitem__)
    return BraketResult(
        samples=samples,
        energies=energies,
        best_assignment=samples[best_idx],
        best_energy=energies[best_idx],
        device_arn=device_arn,
        shots=shots,
        task_arn=getattr(task, "id", None),
        metadata={"num_atoms": len(order), "discarded_atom_loss_shots": discarded},
    )
