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
"""Thin client for talking to a peer LIMEN node's Coordination service."""

from __future__ import annotations

import json

import grpc

from limen.analog.delta_model import HardwareDeltaModel
from limen.core.compiler import PhysicalEncoding
from limen.core.ir import LogicalGraph
from limen.distributed import marshal
from limen.distributed.node import NodeInfo
from limen.distributed.proto import coordination_pb2 as pb
from limen.distributed.proto import coordination_pb2_grpc as pb_grpc


class CoordinationClient:
    """Wraps a gRPC channel to a single peer's Coordination service."""

    def __init__(self, address: str) -> None:
        self.address = address
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

    def close(self) -> None:
        self._channel.close()

    def __enter__(self) -> "CoordinationClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
