# Copyright (C) 2026 Jemone McCubbin / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
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
        tls_cert_path: Path to this node's PEM-encoded TLS server certificate.
            When unset (default), the server binds with ``add_insecure_port``
            exactly as before TLS support was added.
        tls_key_path: Path to this node's PEM-encoded TLS private key.
            Required alongside tls_cert_path to enable TLS.
        tls_ca_path: Optional path to a PEM-encoded CA bundle used to verify
            client certificates (mutual TLS). When set, the server requires
            and verifies client certificates signed by this CA.
    """

    node_id: str
    host: str
    port: int
    device_ids: list[str] = field(default_factory=list)
    known_peers: list[str] = field(default_factory=list)
    tls_cert_path: str | None = None
    tls_key_path: str | None = None
    tls_ca_path: str | None = None

    @classmethod
    def from_env(cls) -> NodeConfig:
        """Build a NodeConfig from LIMEN_NODE_* environment variables.

        Reads:
            LIMEN_NODE_ID: required, unique node identifier.
            LIMEN_NODE_HOST: defaults to "0.0.0.0".
            LIMEN_NODE_PORT: defaults to 50051.
            LIMEN_NODE_DEVICE_IDS: comma-separated device IDs (optional).
            LIMEN_KNOWN_PEERS: comma-separated "host:port" peer addresses.
            LIMEN_TLS_CERT: optional path to a PEM server certificate. When
                set together with LIMEN_TLS_KEY, the server binds with TLS
                instead of cleartext.
            LIMEN_TLS_KEY: optional path to a PEM private key for LIMEN_TLS_CERT.
            LIMEN_TLS_CA: optional path to a PEM CA bundle for verifying
                client certificates (mutual TLS).

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
        tls_cert_path = os.environ.get("LIMEN_TLS_CERT") or None
        tls_key_path = os.environ.get("LIMEN_TLS_KEY") or None
        tls_ca_path = os.environ.get("LIMEN_TLS_CA") or None

        return cls(
            node_id=node_id,
            host=host,
            port=port,
            device_ids=device_ids,
            known_peers=known_peers,
            tls_cert_path=tls_cert_path,
            tls_key_path=tls_key_path,
            tls_ca_path=tls_ca_path,
        )


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
