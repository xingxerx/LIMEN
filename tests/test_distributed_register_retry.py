# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Confirms register_with_peers() retries transient failures with backoff.

Covers the multi-node bring-up race described in the LIMEN distributed
hardening task: a peer listed in known_peers may not be listening yet when
this node starts. _register_with_retry should retry with exponential
backoff and recover once the peer comes up, rather than permanently
missing registration after a single failed attempt.
"""

from __future__ import annotations

import unittest
from unittest import mock

import pytest

grpc = pytest.importorskip("grpc")

from limen.distributed.config import NodeConfig
from limen.distributed.node import NodeInfo
from limen.distributed.registry import NodeRegistry
from limen.distributed.server import (
    _REGISTER_BASE_DELAY_S,
    _register_with_retry,
    register_with_peers,
    serve,
)

pytestmark = pytest.mark.network


class _FlakyThenOkClient:
    """Fails register() on the first N *instances* with grpc.RpcError, then succeeds.

    register_with_peers/_register_with_retry constructs a fresh
    CoordinationClient for each attempt (mirroring real gRPC channel
    reconnect semantics), so "fails N times then succeeds" is modeled as
    a shared counter across instances rather than per-instance call count.
    """

    instances: list["_FlakyThenOkClient"] = []
    shared_fail_budget = 0

    def __init__(self, address: str, fail_count: int = 2):
        self.address = address
        self.calls = 0
        self.closed = False
        _FlakyThenOkClient.instances.append(self)

    def register(self, info: NodeInfo) -> bool:
        self.calls += 1
        if _FlakyThenOkClient.shared_fail_budget > 0:
            _FlakyThenOkClient.shared_fail_budget -= 1
            raise grpc.RpcError("simulated transient failure")
        return True

    def close(self) -> None:
        self.closed = True


class TestRegisterWithRetryMocked(unittest.TestCase):
    """Fast, deterministic test using a mock client that fails N times."""

    def setUp(self):
        _FlakyThenOkClient.instances = []
        _FlakyThenOkClient.shared_fail_budget = 0

    def test_recovers_after_transient_failures(self):
        _FlakyThenOkClient.shared_fail_budget = 2

        def factory(address):
            return _FlakyThenOkClient(address)

        with mock.patch(
            "limen.distributed.client.CoordinationClient", side_effect=factory
        ):
            self_info = NodeInfo(node_id="a", host="127.0.0.1", port=1234)
            # Use a tiny base delay so the test doesn't actually wait long.
            ok = _register_with_retry(
                "127.0.0.1:9999",
                self_info,
                max_attempts=5,
                base_delay_s=0.01,
                max_delay_s=0.05,
            )

        self.assertTrue(ok)
        self.assertEqual(len(_FlakyThenOkClient.instances), 3)
        self.assertTrue(_FlakyThenOkClient.instances[-1].calls == 1)
        self.assertTrue(all(c.closed for c in _FlakyThenOkClient.instances))

    def test_gives_up_after_exhausting_attempts_without_raising(self):
        _FlakyThenOkClient.shared_fail_budget = 999

        def factory(address):
            return _FlakyThenOkClient(address)

        with mock.patch(
            "limen.distributed.client.CoordinationClient", side_effect=factory
        ):
            self_info = NodeInfo(node_id="a", host="127.0.0.1", port=1234)
            ok = _register_with_retry(
                "127.0.0.1:9999",
                self_info,
                max_attempts=3,
                base_delay_s=0.01,
                max_delay_s=0.02,
            )

        self.assertFalse(ok)
        self.assertEqual(len(_FlakyThenOkClient.instances), 3)


class TestRegisterWithPeersRealServerStartedLate(unittest.TestCase):
    """End-to-end: peer's server starts slightly after registration begins."""

    def test_register_with_peers_recovers_once_peer_comes_up(self):
        import threading
        import time

        registry_b = NodeRegistry()
        config_a = NodeConfig(
            node_id="node-a", host="127.0.0.1", port=0, known_peers=[]
        )

        # Reserve a port up front but don't start the server yet, so the
        # first registration attempt(s) hit "connection refused".
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        free_port = sock.getsockname()[1]
        sock.close()

        config_a.known_peers = [f"127.0.0.1:{free_port}"]

        server_holder = {}

        def start_peer_late():
            time.sleep(_REGISTER_BASE_DELAY_S * 2)
            config_b = NodeConfig(node_id="node-b", host="127.0.0.1", port=free_port)
            server, _ = serve(config_b, registry_b, port=free_port)
            server_holder["server"] = server

        t = threading.Thread(target=start_peer_late)
        t.start()
        try:
            register_with_peers(config_a, NodeRegistry())
        finally:
            t.join()
            if "server" in server_holder:
                server_holder["server"].stop(grace=0)

        self.assertIn("node-a", [p.node_id for p in registry_b.list_peers()])


if __name__ == "__main__":
    unittest.main()
