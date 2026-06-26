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
"""Environment-driven configuration for a LIMEN distributed node.

Follows the same env-var convention as IBM_QUANTUM_TOKEN/IBM_QUANTUM_CRN
used elsewhere in the project (see limen/backends/qiskit_backend.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class NodeConfig:
    """Static configuration for this process's node identity and peers.

    Attributes:
        node_id: Unique identifier for this node.
        host: Hostname/IP this node's gRPC server binds to and advertises.
        port: Port this node's gRPC server listens on.
        device_ids: HardwareDeltaModel device IDs this node serves.
        known_peers: "host:port" addresses to register with on startup.
    """

    node_id: str
    host: str
    port: int
    device_ids: list[str] = field(default_factory=list)
    known_peers: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> NodeConfig:
        """Build a NodeConfig from LIMEN_NODE_* environment variables.

        Reads:
            LIMEN_NODE_ID: required, unique node identifier.
            LIMEN_NODE_HOST: defaults to "0.0.0.0".
            LIMEN_NODE_PORT: defaults to 50051.
            LIMEN_NODE_DEVICE_IDS: comma-separated device IDs (optional).
            LIMEN_KNOWN_PEERS: comma-separated "host:port" peer addresses.

        Raises:
            ValueError: if LIMEN_NODE_ID is unset.
        """
        node_id = os.environ.get("LIMEN_NODE_ID")
        if not node_id:
            raise ValueError("LIMEN_NODE_ID environment variable is required")

        host = os.environ.get("LIMEN_NODE_HOST", "0.0.0.0")
        port = int(os.environ.get("LIMEN_NODE_PORT", "50051"))
        device_ids = _split_csv(os.environ.get("LIMEN_NODE_DEVICE_IDS", ""))
        known_peers = _split_csv(os.environ.get("LIMEN_KNOWN_PEERS", ""))

        return cls(
            node_id=node_id,
            host=host,
            port=port,
            device_ids=device_ids,
            known_peers=known_peers,
        )


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
