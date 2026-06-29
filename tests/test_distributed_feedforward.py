# Copyright (C) 2026 Jemone McCubbin / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""End-to-end test of cross-node classical feedforward transport (Milestone 3).

Spins up a real local gRPC Coordination server (Bob) and points a
CoordinationClient (Alice's process) at it, mirroring the pattern in
tests/test_distributed_server.py.
"""

import math
import unittest

import pytest

grpc = pytest.importorskip("grpc")

from limen.communication.channel import (
    ChannelDeltaModel,
    FeedforwardTransport,
    TeleportationResult,
    correction_for_bits,
    run_distributed_feedforward_teleport,
    simulate_feedforward_teleport,
)
from limen.distributed.client import CoordinationClient
from limen.distributed.config import NodeConfig
from limen.distributed.registry import NodeRegistry
from limen.distributed.server import serve


class TestTransportFeedforwardRPC(unittest.TestCase):
    """Direct RPC round trip: client.transport_feedforward against a real server."""

    def setUp(self):
        self.registry = NodeRegistry()
        config = NodeConfig(node_id="bob", host="127.0.0.1", port=0)
        self.server, self.port = serve(config, self.registry, port=0)
        self.addCleanup(lambda: self.server.stop(grace=0))
        self.client = CoordinationClient(f"127.0.0.1:{self.port}")
        self.addCleanup(self.client.close)

    def test_correction_matches_local_helper_for_all_bit_combinations(self):
        for m0 in (0, 1):
            for m1 in (0, 1):
                correction, _ = self.client.transport_feedforward(m0, m1)
                self.assertEqual(correction, correction_for_bits(m0, m1))

    def test_measured_latency_feeds_channel_delta_model(self):
        correction, channel_delta = self.client.transport_feedforward(
            0, 1, t2_us=1_000_000.0
        )
        self.assertEqual(correction, "X")
        self.assertIsInstance(channel_delta, ChannelDeltaModel)
        self.assertGreaterEqual(channel_delta.latency_ms, 0.0)
        # A huge T2 relative to a fast localhost RPC should stay within coherence.
        self.assertTrue(channel_delta.within_coherence())
        self.assertGreater(channel_delta.fidelity_penalty(), 0.99)

    def test_tiny_t2_breaks_coherence(self):
        # A vanishingly small T2 should make even a fast localhost RPC exceed it.
        _, channel_delta = self.client.transport_feedforward(1, 1, t2_us=1e-9)
        self.assertFalse(channel_delta.within_coherence())


class TestRunDistributedFeedforwardTeleport(unittest.TestCase):
    """Full Alice-side-simulate + cross-node-transport + Bob-side-correction flow."""

    def setUp(self):
        self.registry = NodeRegistry()
        config = NodeConfig(node_id="bob", host="127.0.0.1", port=0)
        self.server, self.port = serve(config, self.registry, port=0)
        self.addCleanup(lambda: self.server.stop(grace=0))
        self.peer_address = f"127.0.0.1:{self.port}"

    def test_distributed_teleport_matches_local_simulation(self):
        theta, phi, seed = 1.1, 0.7, 99

        local_result, local_transport = simulate_feedforward_teleport(
            theta, phi, seed=seed
        )
        dist_result, dist_transport = run_distributed_feedforward_teleport(
            theta, phi, self.peer_address, seed=seed
        )

        self.assertIsInstance(dist_result, TeleportationResult)
        self.assertIsInstance(dist_transport, FeedforwardTransport)

        # Same RNG seed -> same Alice bits -> same correction -> same fidelity.
        self.assertEqual(dist_transport.alice_bits, local_transport.alice_bits)
        self.assertEqual(dist_transport.correction, local_transport.correction)
        self.assertAlmostEqual(dist_result.fidelity, local_result.fidelity, places=9)

    def test_distributed_teleport_state_zero_fidelity_one(self):
        result, transport = run_distributed_feedforward_teleport(
            0.0, 0.0, self.peer_address, seed=42
        )
        self.assertAlmostEqual(result.fidelity, 1.0, places=9)
        self.assertIn(transport.correction, ("I", "X", "Z", "XZ"))

    def test_distributed_transport_latency_is_measured_not_modelled(self):
        _, transport = run_distributed_feedforward_teleport(
            math.pi / 2, 0.3, self.peer_address, seed=7
        )
        self.assertIsNotNone(transport.transport_latency_ms)
        self.assertGreaterEqual(transport.transport_latency_ms, 0.0)
        self.assertIsNotNone(transport.within_coherence)
        self.assertIsNotNone(transport.fidelity_penalty)


if __name__ == "__main__":
    unittest.main()
