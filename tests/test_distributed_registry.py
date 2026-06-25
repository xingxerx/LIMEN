"""Unit tests for NodeRegistry peer tracking and calibration caching."""

import unittest

from limen.analog.delta_model import DeltaModelRegistry, HardwareDeltaModel
from limen.analog.hamiltonian import SubstrateType
from limen.distributed.node import NodeInfo
from limen.distributed.registry import NodeRegistry


def _node(node_id: str, device_ids=None) -> NodeInfo:
    return NodeInfo(node_id=node_id, host="127.0.0.1", port=50051, device_ids=device_ids or [])


class TestNodeRegistry(unittest.TestCase):

    def test_add_and_list_peers(self):
        reg = NodeRegistry()
        reg.add_peer(_node("node-b"))
        reg.add_peer(_node("node-a"))
        peers = reg.list_peers()
        self.assertEqual([p.node_id for p in peers], ["node-a", "node-b"])

    def test_heartbeat_unknown_peer_returns_false(self):
        reg = NodeRegistry()
        self.assertFalse(reg.heartbeat("ghost"))

    def test_heartbeat_known_peer_returns_true(self):
        reg = NodeRegistry()
        reg.add_peer(_node("node-a"))
        self.assertTrue(reg.heartbeat("node-a"))

    def test_evict_stale_removes_old_peers(self):
        reg = NodeRegistry(peer_ttl=10.0)
        reg.add_peer(_node("node-a"))
        evicted = reg.evict_stale(now=reg._peers["node-a"].last_seen + 100.0)
        self.assertEqual(evicted, ["node-a"])
        self.assertEqual(reg.list_peers(), [])

    def test_evict_stale_keeps_fresh_peers(self):
        reg = NodeRegistry(peer_ttl=10.0)
        reg.add_peer(_node("node-a"))
        evicted = reg.evict_stale(now=reg._peers["node-a"].last_seen + 1.0)
        self.assertEqual(evicted, [])
        self.assertEqual(len(reg.list_peers()), 1)

    def test_peer_for_device_finds_serving_peer(self):
        reg = NodeRegistry()
        reg.add_peer(_node("node-a", device_ids=["qpu-1"]))
        peer = reg.peer_for_device("qpu-1")
        self.assertIsNotNone(peer)
        self.assertEqual(peer.node_id, "node-a")

    def test_peer_for_device_returns_none_when_unserved(self):
        reg = NodeRegistry()
        reg.add_peer(_node("node-a", device_ids=["qpu-1"]))
        self.assertIsNone(reg.peer_for_device("qpu-2"))

    def test_resolve_prefers_local_over_cache(self):
        local = DeltaModelRegistry()
        model = HardwareDeltaModel.identity("qpu-1", SubstrateType.NEUTRAL_ATOM, 4)
        local.register(model)
        reg = NodeRegistry(local_delta_registry=local)
        self.assertIs(reg.resolve("qpu-1"), model)

    def test_resolve_falls_back_to_cache(self):
        reg = NodeRegistry()
        model = HardwareDeltaModel.identity("qpu-2", SubstrateType.NEUTRAL_ATOM, 4)
        reg.cache_calibration(model)
        self.assertEqual(reg.resolve("qpu-2"), model)

    def test_resolve_returns_none_for_unknown_device(self):
        reg = NodeRegistry()
        self.assertIsNone(reg.resolve("unknown"))

    def test_cached_calibration_expires_after_ttl(self):
        reg = NodeRegistry(cache_ttl=5.0)
        model = HardwareDeltaModel.identity("qpu-3", SubstrateType.NEUTRAL_ATOM, 4)
        reg.cache_calibration(model)
        reg._device_cache["qpu-3"].cached_at -= 100.0
        self.assertIsNone(reg.cached_calibration("qpu-3"))


if __name__ == "__main__":
    unittest.main()
