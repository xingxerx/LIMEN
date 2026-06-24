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
"""Tests for the QuantumChannel communication module."""

import math
import pytest

# Skip tests if qiskit or qiskit_aer are not installed
pytest.importorskip("qiskit", reason="qiskit not installed")
pytest.importorskip("qiskit_aer", reason="qiskit-aer not installed")

from limen.communication.channel import QuantumChannel, TeleportationResult, QKDResult


def test_quantum_channel_init():
    """Verify that QuantumChannel initializes correctly with parameters."""
    channel = QuantumChannel(backend_name="statevector", seed=101)
    assert channel.backend_name == "statevector"
    assert not channel.use_runtime
    assert channel.seed == 101


def test_teleportation_state_0():
    """Verify quantum teleportation of the |0> state."""
    channel = QuantumChannel(backend_name="statevector", seed=42)
    # State |0>: theta = 0, phi = 0
    res = channel.teleport(0.0, 0.0, shots=100, inverse_prep=True)

    assert isinstance(res, TeleportationResult)
    assert res.input_state == {"theta": 0.0, "phi": 0.0}
    assert res.fidelity == 1.0
    assert res.verification_success_rate == 1.0
    assert res.circuit_depth is not None
    assert res.circuit_depth > 0


def test_teleportation_state_1():
    """Verify quantum teleportation of the |1> state."""
    channel = QuantumChannel(backend_name="statevector", seed=42)
    # State |1>: theta = pi, phi = 0
    res = channel.teleport(math.pi, 0.0, shots=100, inverse_prep=True)

    assert isinstance(res, TeleportationResult)
    assert res.fidelity == 1.0
    assert res.verification_success_rate == 1.0


def test_teleportation_superposition_state():
    """Verify quantum teleportation of a superposition state (|+>)."""
    channel = QuantumChannel(backend_name="statevector", seed=42)
    # State |+>: theta = pi/2, phi = 0
    res = channel.teleport(math.pi / 2.0, 0.0, shots=100, inverse_prep=True)

    assert isinstance(res, TeleportationResult)
    assert res.fidelity == 1.0


def test_teleportation_no_inverse_prep():
    """Verify that teleportation returns no fidelity score if inverse_prep is False."""
    channel = QuantumChannel(backend_name="statevector", seed=42)
    res = channel.teleport(math.pi / 2.0, 0.0, shots=100, inverse_prep=False)

    assert isinstance(res, TeleportationResult)
    assert res.fidelity is None
    assert res.verification_success_rate is None
    assert res.measured_counts is not None


def test_qkd_bb84_noiseless():
    """Verify BB84 key distribution on a noiseless simulator has 0% QBER and is secure."""
    channel = QuantumChannel(backend_name="statevector", seed=123)
    res = channel.qkd_bb84(key_length=40, eavesdrop_rate=0.0, shots=1)

    assert isinstance(res, QKDResult)
    assert res.protocol == "bb84"
    assert res.raw_key_length == 40
    assert res.sifted_key_length > 0
    assert res.qber == 0.0
    assert res.secure
    assert res.shared_key is not None
    assert len(res.shared_key) > 0
    assert all(c in ("0", "1") for c in res.shared_key)


def test_qkd_bb84_eavesdropping():
    """Verify BB84 key distribution under active eavesdropping detects errors and aborts."""
    channel = QuantumChannel(backend_name="statevector", seed=456)
    # With 100% eavesdropping, QBER will be high
    res = channel.qkd_bb84(key_length=150, eavesdrop_rate=1.0, shots=1)

    assert isinstance(res, QKDResult)
    assert res.qber > 0.11  # elevated QBER expected
    assert not res.secure
    assert res.shared_key is None
