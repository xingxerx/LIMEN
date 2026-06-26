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
"""Node identity for LIMEN multi-node deployments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NodeInfo:
    """Identity and reachability of a LIMEN node.

    Attributes:
        node_id: Unique identifier for this node.
        host: Hostname or IP address other nodes use to reach it.
        port: gRPC port the Coordination service listens on.
        device_ids: HardwareDeltaModel device IDs this node can serve.
    """

    node_id: str
    host: str
    port: int
    device_ids: list[str] = field(default_factory=list)

    def address(self) -> str:
        """Return the "host:port" string used to open a gRPC channel."""
        return f"{self.host}:{self.port}"

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "device_ids": list(self.device_ids),
        }

    @classmethod
    def from_dict(cls, d: dict) -> NodeInfo:
        return cls(
            node_id=str(d["node_id"]),
            host=str(d["host"]),
            port=int(d["port"]),
            device_ids=list(d.get("device_ids", [])),
        )
