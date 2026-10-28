# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Environment-driven configuration for a LIMEN distributed node.

Follows the same env-var convention as IBM_QUANTUM_TOKEN/IBM_QUANTUM_CRN
used elsewhere in the project (see limen/backends/qiskit_backend.py).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger("limen.distributed")

# Explicit opt-in for serving cleartext gRPC on a non-loopback address.
ALLOW_INSECURE_NETWORK_ENV = "LIMEN_ALLOW_INSECURE_NETWORK"


# Tailscale hands out addresses from the carrier-grade NAT range. A tailnet address is
# network-reachable like any other: WireGuard encrypts the link, but it does not authenticate
# the caller to this service, so it gets no exemption from the TLS requirement.
TAILNET = ipaddress.ip_network("100.64.0.0/10")


class InsecureBindError(ValueError):
    """Refusing to serve cleartext gRPC on a network-reachable address."""


def is_tailnet_host(host: str) -> bool:
    """True for an IPv4 address in Tailscale's range (100.64.0.0/10)."""
    try:
        ip = ipaddress.ip_address(host.strip().strip("[]"))
    except ValueError:
        return False
    return ip.version == 4 and ip in TAILNET


def _tailscale_ipv4() -> str | None:
    """This machine's Tailscale IPv4 address, or None if it has none.

    Asks the ``tailscale`` CLI first, then falls back to the addresses the
    machine's own hostname resolves to."""
    exe = shutil.which("tailscale")
    if exe:
        try:
            out = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=3)
            for line in out.stdout.split():
                if is_tailnet_host(line):
                    return line
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            if is_tailnet_host(info[4][0]):
                return info[4][0]
    except OSError:
        pass
    return None


def default_node_host() -> str:
    """The address a node binds when LIMEN_NODE_HOST is unset: its own Tailscale
    address if it has one (so peers on the tailnet can reach it), else loopback.
    A Tailscale address still needs TLS, see :meth:`NodeConfig.check_bind_policy`."""
    return _tailscale_ipv4() or "127.0.0.1"


def is_loopback_host(host: str) -> bool:
    """True only for an address that cannot be reached from another machine.
    Wildcards (``0.0.0.0``, ``::``, empty) and every other hostname or IP
    count as network-reachable."""
    host = host.strip().strip("[]")
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass
class NodeConfig:
    """Static configuration for this process's node identity and peers.

    Attributes:
        node_id: Unique identifier for this node.
        host: Hostname/IP this node's gRPC server binds to and advertises.
            from_env() defaults to this machine's Tailscale address, or
            loopback if it has none. Any non-loopback address, Tailscale
            included, needs TLS (see tls_cert_path).
        port: Port this node's gRPC server listens on.
        device_ids: HardwareDeltaModel device IDs this node serves.
        known_peers: "host:port" addresses to register with on startup.
        tls_cert_path: Path to this node's PEM-encoded TLS server certificate.
            When unset (default), the server binds with ``add_insecure_port``,
            which is only allowed on a loopback host (see :meth:`check_bind_policy`).
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

    @property
    def tls_enabled(self) -> bool:
        return bool(self.tls_cert_path and self.tls_key_path)

    def check_bind_policy(self) -> None:
        """Refuse cleartext gRPC on a network-reachable address.

        Loopback may run without TLS. Anything else, a Tailscale address
        included, needs TLS (a certificate and key) unless
        ``LIMEN_ALLOW_INSECURE_NETWORK=1`` is set, which is logged as a warning
        every time. WireGuard encrypts a tailnet link but does not authenticate
        callers to this service, so a tailnet address is not exempt. TLS without
        a client CA is allowed but warned about: any peer that can reach the
        port can call it.
        """
        if is_loopback_host(self.host):
            return
        if not self.tls_enabled:
            if os.environ.get(ALLOW_INSECURE_NETWORK_ENV) == "1":
                logger.warning(
                    "serving cleartext gRPC on %s because %s=1; anyone who can reach this "
                    "port can call it. Configure LIMEN_TLS_CERT/LIMEN_TLS_KEY (and LIMEN_TLS_CA).",
                    self.host, ALLOW_INSECURE_NETWORK_ENV,
                )
                return
            raise InsecureBindError(
                f"refusing to serve cleartext gRPC on {self.host!r}: set LIMEN_TLS_CERT and "
                f"LIMEN_TLS_KEY, bind a loopback address, or set {ALLOW_INSECURE_NETWORK_ENV}=1 "
                "to accept the risk"
            )
        if not self.tls_ca_path:
            logger.warning(
                "serving TLS on %s without LIMEN_TLS_CA: server identity is protected but "
                "clients are not authenticated (no mutual TLS).",
                self.host,
            )

    @classmethod
    def from_env(cls) -> NodeConfig:
        """Build a NodeConfig from LIMEN_NODE_* environment variables.

        Reads:
            LIMEN_NODE_ID: required, unique node identifier.
            LIMEN_NODE_HOST: defaults to this machine's Tailscale address, or
                "127.0.0.1" if it has none. A non-loopback address (Tailscale
                included) needs LIMEN_TLS_CERT/LIMEN_TLS_KEY, otherwise the
                server refuses to start; set LIMEN_NODE_HOST=127.0.0.1 for a
                local-only node. See :meth:`check_bind_policy`.
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

        host = os.environ.get("LIMEN_NODE_HOST") or default_node_host()
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
