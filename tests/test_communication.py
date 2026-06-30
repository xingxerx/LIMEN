# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Tests for the QuantumChannel communication module."""

import math
import pytest

from limen.communication.channel import (
    ChannelDeltaModel,
    FeedforwardTransport,
    QKDResult,
    TeleportationResult,
    simulate_feedforward_teleport,
)

# ---------------------------------------------------------------------------
# Pure-Python feedforward tests (no Qiskit required)
# ---------------------------------------------------------------------------

def test_feedforward_teleport_state_zero():
    """|0⟩ teleports with fidelity 1 regardless of correction outcome."""
    result, transport = simulate_feedforward_teleport(0.0, 0.0, seed=42)
    assert isinstance(result, TeleportationResult)
    assert isinstance(transport, FeedforwardTransport)
    assert result.fidelity is not None
    assert abs(result.fidelity - 1.0) < 1e-9


def test_feedforward_teleport_state_one():
    """|1⟩ (theta=pi) teleports with fidelity 1."""
    result, transport = simulate_feedforward_teleport(math.pi, 0.0, seed=42)
    assert abs(result.fidelity - 1.0) < 1e-9


def test_feedforward_teleport_plus_state():
    """|+⟩ (theta=pi/2) teleports with fidelity 1."""
    result, transport = simulate_feedforward_teleport(math.pi / 2, 0.0, seed=7)
    assert abs(result.fidelity - 1.0) < 1e-9


def test_feedforward_teleport_arbitrary_state():
    """Arbitrary state teleports with fidelity 1 (no noise in simulator)."""
    result, transport = simulate_feedforward_teleport(1.1, 0.7, seed=99)
    assert abs(result.fidelity - 1.0) < 1e-9


def test_feedforward_transport_correction_is_valid():
    """The correction field is one of the four expected Paulis."""
    _, transport = simulate_feedforward_teleport(0.5, 0.3, seed=5)
    assert transport.correction in {"I", "X", "Z", "XZ"}


def test_feedforward_transport_alice_bits_binary():
    """Alice's measurement bits are always 0 or 1."""
    _, transport = simulate_feedforward_teleport(1.0, 0.5, seed=17)
    m0, m1 = transport.alice_bits
    assert m0 in (0, 1)
    assert m1 in (0, 1)


def test_feedforward_transport_no_channel_delta():
    """Without a ChannelDeltaModel, coherence fields are None."""
    _, transport = simulate_feedforward_teleport(0.0, 0.0, seed=0)
    assert transport.within_coherence is None
    assert transport.fidelity_penalty is None
    assert transport.transport_latency_ms is None


def test_feedforward_transport_with_channel_delta_within_coherence():
    """Fast channel (1 µs latency, T2=100 µs) is within coherence."""
    delta = ChannelDeltaModel(latency_ms=0.001, t2_us=100.0)
    _, transport = simulate_feedforward_teleport(0.0, 0.0, channel_delta=delta, seed=0)
    assert transport.within_coherence is True
    assert transport.fidelity_penalty is not None
    assert transport.fidelity_penalty > 0.99  # negligible decay
    assert transport.transport_latency_ms == pytest.approx(0.001)


def test_feedforward_transport_with_channel_delta_outside_coherence():
    """Slow channel (2 ms latency, T2=1 µs) exceeds coherence time."""
    delta = ChannelDeltaModel(latency_ms=2.0, t2_us=1.0)
    _, transport = simulate_feedforward_teleport(0.0, 0.0, channel_delta=delta, seed=0)
    assert transport.within_coherence is False
    assert transport.fidelity_penalty < 0.01  # severe decay


def test_feedforward_transport_to_dict():
    """FeedforwardTransport serialises to dict correctly."""
    delta = ChannelDeltaModel(latency_ms=0.5, t2_us=50.0)
    _, transport = simulate_feedforward_teleport(0.0, 0.0, channel_delta=delta, seed=0)
    d = transport.to_dict()
    assert "alice_bits" in d
    assert "correction" in d
    assert "within_coherence" in d
    assert "fidelity_penalty" in d
    assert "transport_latency_ms" in d


def test_feedforward_result_metadata_contains_correction():
    """TeleportationResult.metadata captures alice_bits and correction."""
    result, transport = simulate_feedforward_teleport(0.3, 1.2, seed=13)
    assert result.metadata["backend"] == "statevector_simulator"
    assert "alice_bits" in result.metadata
    assert result.metadata["correction"] == transport.correction


# ---------------------------------------------------------------------------
# Qiskit-gated tests (skip if Qiskit / Aer not installed)
# ---------------------------------------------------------------------------
pytest.importorskip("qiskit", reason="qiskit not installed")
pytest.importorskip("qiskit_aer", reason="qiskit-aer not installed")

from limen.communication.channel import QuantumChannel  # noqa: E402


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
