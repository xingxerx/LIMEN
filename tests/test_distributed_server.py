"""End-to-end test of the Coordination gRPC service against a real server."""

import unittest

import pytest

grpc = pytest.importorskip("grpc")

from limen.analog.delta_model import HardwareDeltaModel
from limen.analog.hamiltonian import SubstrateType
from limen.distributed.client import CoordinationClient
from limen.distributed.config import NodeConfig
from limen.distributed.node import NodeInfo
from limen.distributed.registry import NodeRegistry
from limen.distributed.server import serve


class TestCoordinationServer(unittest.TestCase):

    def setUp(self):
        self.registry = NodeRegistry()
        config = NodeConfig(node_id="node-a", host="127.0.0.1", port=0)
        self.server, self.port = serve(config, self.registry, port=0)
        self.addCleanup(lambda: self.server.stop(grace=0))
        self.client = CoordinationClient(f"127.0.0.1:{self.port}")
        self.addCleanup(self.client.close)

    def test_register_heartbeat_and_list_peers(self):
        peer = NodeInfo(node_id="node-b", host="127.0.0.1", port=6000, device_ids=["qpu-1"])
        self.assertTrue(self.client.register(peer))
        self.assertTrue(self.client.heartbeat("node-b"))
        self.assertFalse(self.client.heartbeat("unknown-node"))

        peers = self.client.list_peers()
        self.assertEqual([p.node_id for p in peers], ["node-b"])

    def test_sync_calibration_round_trips_model(self):
        model = HardwareDeltaModel.identity("qpu-1", SubstrateType.NEUTRAL_ATOM, 4)
        self.registry.local.register(model)

        fetched = self.client.sync_calibration("qpu-1")
        self.assertEqual(fetched.to_dict(), model.to_dict())

    def test_sync_calibration_unknown_device_raises(self):
        with self.assertRaises(grpc.RpcError):
            self.client.sync_calibration("does-not-exist")


if __name__ == "__main__":
    unittest.main()
