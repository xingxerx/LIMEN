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
"""Communication package for LIMEN."""

from limen.communication.channel import (
    ChannelDeltaModel,
    QKDResult,
    QuantumChannel,
    SiftedKeyResult,
    TeleportationResult,
    TeleportResult,
    bb84_circuit,
    estimate_fidelity,
    run_teleport_qpu,
    sift_and_evaluate,
    teleport_circuit,
)

__all__ = [
    # High-level (simulator) API
    "QuantumChannel",
    "TeleportationResult",
    "QKDResult",
    # Hardware-level channel model
    "ChannelDeltaModel",
    # Low-level QPU-execution API
    "TeleportResult",
    "SiftedKeyResult",
    "teleport_circuit",
    "bb84_circuit",
    "run_teleport_qpu",
    "estimate_fidelity",
    "sift_and_evaluate",
]
