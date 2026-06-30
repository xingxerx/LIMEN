# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Thin client for talking to a peer LIMEN node's Coordination service."""

from __future__ import annotations

import json
import time

import grpc

from limen.analog.delta_model import HardwareDeltaModel
from limen.communication.channel import ChannelDeltaModel
from limen.core.compiler import PhysicalEncoding
from limen.core.ir import LogicalGraph
from limen.distributed import marshal
from limen.distributed.node import NodeInfo
from limen.distributed.proto import coordination_pb2 as pb
from limen.distributed.proto import coordination_pb2_grpc as pb_grpc


class CoordinationClient:
    """Wraps a gRPC channel to a single peer's Coordination service."""

    def __init__(
        self,
        address: str,
        ca_cert_path: str | None = None,
        client_cert_path: str | None = None,
        client_key_path: str | None = None,
    ) -> None:
        """Connect to a peer's Coordination service at *address*.

        Args:
            address: "host:port" of the peer.
            ca_cert_path: Optional path to a PEM CA bundle used to verify
                the server's certificate. When set, the channel is created
                with ``grpc.secure_channel`` over TLS instead of the
                default ``grpc.insecure_channel``.
            client_cert_path: Optional path to a PEM client certificate,
                for mutual TLS. Requires ca_cert_path and client_key_path
                to also be set.
            client_key_path: Optional path to a PEM client private key,
                paired with client_cert_path for mutual TLS.

        When ca_cert_path is unset (the default), behavior is unchanged
        from before TLS support existed: an insecure channel is used.
        """
        self.address = address
        if ca_cert_path:
            with open(ca_cert_path, "rb") as f:
                root_certs = f.read()
            private_key = None
            cert_chain = None
            if client_cert_path and client_key_path:
                with open(client_key_path, "rb") as f:
                    private_key = f.read()
                with open(client_cert_path, "rb") as f:
                    cert_chain = f.read()
            credentials = grpc.ssl_channel_credentials(
                root_certificates=root_certs,
                private_key=private_key,
                certificate_chain=cert_chain,
            )
            self._channel = grpc.secure_channel(address, credentials)
        else:
            self._channel = grpc.insecure_channel(address)
        self._stub = pb_grpc.CoordinationStub(self._channel)

    def register(self, info: NodeInfo) -> bool:
        ack = self._stub.Register(marshal.node_info_to_proto(info))
        return ack.accepted

    def heartbeat(self, node_id: str) -> bool:
        ack = self._stub.Heartbeat(pb.HeartbeatRequest(node_id=node_id))
        return ack.alive

    def sync_calibration(self, device_id: str) -> HardwareDeltaModel:
        msg = self._stub.SyncCalibration(pb.SyncCalibrationRequest(device_id=device_id))
        return marshal.delta_model_from_proto(msg)

    def list_peers(self) -> list[NodeInfo]:
        response = self._stub.ListPeers(pb.Empty())
        return [marshal.node_info_from_proto(n) for n in response.nodes]

    def compile_partition(
        self, partition_id: str, graph: LogicalGraph, hardware_prefix: str
    ) -> PhysicalEncoding:
        request = pb.CompilePartitionRequest(
            partition_id=partition_id,
            graph_json=json.dumps(graph.to_dict()),
            hardware_prefix=hardware_prefix,
        )
        response = self._stub.CompilePartition(request)
        return PhysicalEncoding.from_dict(json.loads(response.encoding_json))

    def transport_feedforward(
        self,
        m0: int,
        m1: int,
        theta: float = 0.0,
        phi: float = 0.0,
        t2_us: float = 100.0,
    ) -> tuple[str, ChannelDeltaModel]:
        """Send Alice's Bell-measurement bits to this peer and get Bob's correction.

        Measures actual round-trip wall-clock latency of the
        ``TransportFeedforward`` RPC with ``time.perf_counter()`` and uses it
        to build a :class:`ChannelDeltaModel` (paired with *t2_us*), so
        callers can evaluate ``within_coherence()`` / ``fidelity_penalty()``
        against the real network latency rather than a modelled constant.

        Args:
            m0: Alice's measurement outcome for qubit 0.
            m1: Alice's measurement outcome for qubit 1.
            theta: Input-state polar angle, echoed to the peer for context.
            phi: Input-state azimuthal angle, echoed to the peer for context.
            t2_us: T2 coherence time (microseconds) for the resulting
                :class:`ChannelDeltaModel`.

        Returns:
            ``(correction, channel_delta)`` where *correction* is the Pauli
            string ("I", "X", "Z", or "XZ") the peer applied.
        """
        request = pb.FeedforwardRequest(m0=m0, m1=m1, theta=theta, phi=phi)
        start = time.perf_counter()
        response = self._stub.TransportFeedforward(request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        channel_delta = ChannelDeltaModel(latency_ms=latency_ms, t2_us=t2_us)
        return response.correction, channel_delta

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "CoordinationClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
