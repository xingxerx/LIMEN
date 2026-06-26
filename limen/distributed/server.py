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
import time
from concurrent import futures

import grpc

from limen.communication.channel import correction_for_bits
from limen.core.compiler import compile_lexicographic
from limen.core.ir import LogicalGraph
from limen.distributed import marshal
from limen.distributed.config import NodeConfig
from limen.distributed.node import NodeInfo
from limen.distributed.partition import namespaced_hardware_graph
from limen.distributed.proto import coordination_pb2 as pb
from limen.distributed.proto import coordination_pb2_grpc as pb_grpc
from limen.distributed.registry import NodeRegistry

try:
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc

    _HEALTH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the optional dep
    _HEALTH_AVAILABLE = False

logger = logging.getLogger("limen.distributed")

# Retry/backoff defaults for register_with_peers().
_REGISTER_MAX_ATTEMPTS = 5
_REGISTER_BASE_DELAY_S = 0.1
_REGISTER_MAX_DELAY_S = 2.0


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

    def TransportFeedforward(
        self, request: pb.FeedforwardRequest, context
    ) -> pb.FeedforwardResponse:
        """Apply Bob-side Pauli correction for Alice's transported (m0, m1).

        Reuses limen.communication.channel.correction_for_bits so this
        node computes the exact same correction simulate_feedforward_teleport
        would for the same measurement outcomes.
        """
        correction = correction_for_bits(request.m0, request.m1)
        return pb.FeedforwardResponse(correction=correction)


def serve(
    config: NodeConfig, registry: NodeRegistry, port: int | None = None
) -> tuple[grpc.Server, int]:
    """Start a Coordination gRPC server bound to config.host:<port or config.port>.

    Pass port=0 to bind an OS-assigned ephemeral port (used by tests).

    When config.tls_cert_path and config.tls_key_path are both set, the
    server binds with TLS (grpc.ssl_server_credentials) instead of
    cleartext. If config.tls_ca_path is additionally set, client
    certificates are required and verified against that CA (mutual TLS).
    When TLS paths are unset (the default), behavior is unchanged from
    before TLS support existed: the server binds with add_insecure_port.

    If the grpc_health.v1 health-checking package is installed, the
    standard Health service is also registered and marked SERVING once
    the server starts, so orchestrators/load balancers can probe node
    liveness via the interoperable grpc.health.v1.Health API.

    Returns the started grpc.Server and the port it actually bound to;
    callers are responsible for calling server.stop() / wait_for_termination().
    """
    bind_port = config.port if port is None else port
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_CoordinationServicer_to_server(CoordinationServicer(registry), server)

    health_servicer = None
    if _HEALTH_AVAILABLE:
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    address = f"{config.host}:{bind_port}"
    if config.tls_cert_path and config.tls_key_path:
        with open(config.tls_cert_path, "rb") as f:
            cert_chain = f.read()
        with open(config.tls_key_path, "rb") as f:
            private_key = f.read()
        root_certs = None
        require_client_auth = False
        if config.tls_ca_path:
            with open(config.tls_ca_path, "rb") as f:
                root_certs = f.read()
            require_client_auth = True
        credentials = grpc.ssl_server_credentials(
            [(private_key, cert_chain)],
            root_certificates=root_certs,
            require_client_auth=require_client_auth,
        )
        bound_port = server.add_secure_port(address, credentials)
    else:
        bound_port = server.add_insecure_port(address)

    server.start()

    if health_servicer is not None:
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)

    logger.info("coordination server listening on %s:%d", config.host, bound_port)
    return server, bound_port


def _register_with_retry(
    address: str,
    self_info: NodeInfo,
    max_attempts: int = _REGISTER_MAX_ATTEMPTS,
    base_delay_s: float = _REGISTER_BASE_DELAY_S,
    max_delay_s: float = _REGISTER_MAX_DELAY_S,
) -> bool:
    """Register self_info with one peer, retrying with exponential backoff.

    Common at multi-node bring-up: a peer listed in LIMEN_KNOWN_PEERS may
    not be listening yet when this node starts. Each failed attempt is
    logged as a warning; after max_attempts failures this peer is given up
    on (logged as an error) without raising, so one unreachable peer never
    blocks startup or registration with the remaining peers.

    Returns True if registration succeeded, False if all attempts failed.
    """
    from limen.distributed.client import CoordinationClient

    delay = base_delay_s
    for attempt in range(1, max_attempts + 1):
        client = CoordinationClient(address)
        try:
            client.register(self_info)
            return True
        except grpc.RpcError as exc:
            if attempt == max_attempts:
                logger.error(
                    "giving up registering with peer %s after %d attempts: %s",
                    address,
                    max_attempts,
                    exc,
                )
                return False
            logger.warning(
                "attempt %d/%d to register with peer %s failed: %s; retrying in %.2fs",
                attempt,
                max_attempts,
                address,
                exc,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, max_delay_s)
        finally:
            client.close()
    return False  # pragma: no cover - loop always returns/continues above


def register_with_peers(config: NodeConfig, registry: NodeRegistry) -> None:
    """Self-register with each peer in config.known_peers.

    Each side's startup calls this, so for two nodes with each other in
    their KNOWN_PEERS list, registration ends up symmetric: A's Register
    call populates B's registry with A, and B's own register_with_peers
    call populates A's registry with B.

    Each peer is registered with exponential-backoff retry (see
    _register_with_retry) so a peer that isn't listening yet at startup
    doesn't permanently miss registration, and a single unreachable peer
    doesn't prevent registering with the rest.
    """
    self_info = NodeInfo(
        node_id=config.node_id, host=config.host, port=config.port, device_ids=config.device_ids
    )
    for address in config.known_peers:
        _register_with_retry(address, self_info)


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
