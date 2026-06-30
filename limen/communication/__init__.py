# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Communication package for LIMEN."""

from limen.communication.channel import (
    ChannelDeltaModel,
    FeedforwardTransport,
    QKDResult,
    QuantumChannel,
    SiftedKeyResult,
    TeleportationResult,
    TeleportResult,
    bb84_circuit,
    estimate_fidelity,
    run_teleport_qpu,
    sift_and_evaluate,
    simulate_feedforward_teleport,
    teleport_circuit,
)

__all__ = [
    # High-level (simulator) API
    "QuantumChannel",
    "TeleportationResult",
    "QKDResult",
    # Hardware-level channel model
    "ChannelDeltaModel",
    # Pure-Python feedforward teleportation (no Qiskit)
    "FeedforwardTransport",
    "simulate_feedforward_teleport",
    # Low-level QPU-execution API
    "TeleportResult",
    "SiftedKeyResult",
    "teleport_circuit",
    "bb84_circuit",
    "run_teleport_qpu",
    "estimate_fidelity",
    "sift_and_evaluate",
]
