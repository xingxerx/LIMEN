# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Confirms the standard grpc.health.v1 Health service is wired into serve().

Connects a grpc_health.v1.health_pb2_grpc.HealthStub to a real local
Coordination server (started the same way as tests/test_distributed_server.py)
and checks that it reports SERVING, the standard mechanism k8s probes and
gRPC load balancers rely on.
"""

from __future__ import annotations

import unittest

import pytest

grpc = pytest.importorskip("grpc")
health_pb2 = pytest.importorskip("grpc_health.v1.health_pb2")
health_pb2_grpc = pytest.importorskip("grpc_health.v1.health_pb2_grpc")

from limen.distributed.config import NodeConfig
from limen.distributed.registry import NodeRegistry
from limen.distributed.server import serve

pytestmark = pytest.mark.network


class TestHealthService(unittest.TestCase):
    def setUp(self):
        self.registry = NodeRegistry()
        config = NodeConfig(node_id="node-a", host="127.0.0.1", port=0)
        self.server, self.port = serve(config, self.registry, port=0)
        self.addCleanup(lambda: self.server.stop(grace=0))
        self.channel = grpc.insecure_channel(f"127.0.0.1:{self.port}")
        self.addCleanup(self.channel.close)
        self.health_stub = health_pb2_grpc.HealthStub(self.channel)

    def test_health_check_reports_serving(self):
        response = self.health_stub.Check(health_pb2.HealthCheckRequest(service=""))
        self.assertEqual(response.status, health_pb2.HealthCheckResponse.SERVING)


if __name__ == "__main__":
    unittest.main()
