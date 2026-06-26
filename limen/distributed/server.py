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
"""gRPC server for the LIMEN node Coordination service."""

from __future__ import annotations

import json
import logging
from concurrent import futures

import grpc

from limen.core.compiler import compile_lexicographic
from limen.core.ir import LogicalGraph
from limen.distributed import marshal
from limen.distributed.config import NodeConfig
from limen.distributed.node import NodeInfo
from limen.distributed.partition import namespaced_hardware_graph
from limen.distributed.proto import coordination_pb2 as pb
from limen.distributed.proto import coordination_pb2_grpc as pb_grpc
from limen.distributed.registry import NodeRegistry

logger = logging.getLogger("limen.distributed")


class CoordinationServicer(pb_grpc.CoordinationServicer):
    """Implements Register/Heartbeat/SyncCalibration/ListPeers against a NodeRegistry."""

    def __init__(self, registry: NodeRegistry) -> None:
        self.registry = registry

    def Register(self, request: pb.NodeInfo, context) -> pb.RegisterAck:
        self.registry.add_peer(marshal.node_info_from_proto(request))
        return pb.RegisterAck(accepted=True)

    def Heartbeat(self, request: pb.HeartbeatRequest, context) -> pb.HeartbeatAck:
        alive = self.registry.heartbeat(request.node_id)
        return pb.HeartbeatAck(alive=alive)

    def SyncCalibration(
        self, request: pb.SyncCalibrationRequest, context
    ) -> pb.HardwareDeltaModelProto:
        model = self.registry.local.get(request.device_id)
        if model is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown device_id: {request.device_id}")
            raise AssertionError("unreachable: context.abort raises")
        return marshal.delta_model_to_proto(model)

    def ListPeers(self, request: pb.Empty, context) -> pb.NodeList:
        peers = [marshal.node_info_to_proto(p) for p in self.registry.list_peers()]
        return pb.NodeList(nodes=peers)

    def CompilePartition(
        self, request: pb.CompilePartitionRequest, context
    ) -> pb.CompilePartitionResult:
        graph = LogicalGraph.from_dict(json.loads(request.graph_json))
        hardware_graph = namespaced_hardware_graph(len(graph.variables), request.hardware_prefix)
        encoding = compile_lexicographic(graph, hardware_graph)
        return pb.CompilePartitionResult(encoding_json=json.dumps(encoding.to_dict()))


def serve(
    config: NodeConfig, registry: NodeRegistry, port: int | None = None
) -> tuple[grpc.Server, int]:
    """Start a Coordination gRPC server bound to config.host:<port or config.port>.

    Pass port=0 to bind an OS-assigned ephemeral port (used by tests).

    Returns the started grpc.Server and the port it actually bound to;
    callers are responsible for calling server.stop() / wait_for_termination().
    """
    bind_port = config.port if port is None else port
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_CoordinationServicer_to_server(CoordinationServicer(registry), server)
    bound_port = server.add_insecure_port(f"{config.host}:{bind_port}")
    server.start()
    logger.info("coordination server listening on %s:%d", config.host, bound_port)
    return server, bound_port


def register_with_peers(config: NodeConfig, registry: NodeRegistry) -> None:
    """Self-register with each peer in config.known_peers.

    Each side's startup calls this, so for two nodes with each other in
    their KNOWN_PEERS list, registration ends up symmetric: A's Register
    call populates B's registry with A, and B's own register_with_peers
    call populates A's registry with B.
    """
    from limen.distributed.client import CoordinationClient

    self_info = NodeInfo(
        node_id=config.node_id, host=config.host, port=config.port, device_ids=config.device_ids
    )
    for address in config.known_peers:
        client = CoordinationClient(address)
        try:
            client.register(self_info)
        finally:
            client.close()


def main() -> None:
    """Entry point for `python -m limen.distributed.server`."""
    logging.basicConfig(level=logging.INFO)
    config = NodeConfig.from_env()
    registry = NodeRegistry()
    server, _ = serve(config, registry)
    register_with_peers(config, registry)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=2)


if __name__ == "__main__":
    main()
