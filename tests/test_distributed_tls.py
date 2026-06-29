# Copyright (C) 2026 Jemone McCubbin / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""TLS round-trip test for the Coordination gRPC service.

Generates a short-lived self-signed certificate at test time (via the
`cryptography` library) and confirms a real grpc.Server bound with
add_secure_port can be reached by a CoordinationClient configured with
the matching CA cert, mirroring the real-server pattern in
tests/test_distributed_server.py.
"""

from __future__ import annotations

import datetime
import unittest

import pytest

grpc = pytest.importorskip("grpc")
cryptography = pytest.importorskip("cryptography")

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from limen.analog.delta_model import HardwareDeltaModel
from limen.analog.hamiltonian import SubstrateType
from limen.distributed.client import CoordinationClient
from limen.distributed.config import NodeConfig
from limen.distributed.registry import NodeRegistry
from limen.distributed.server import serve


def _generate_self_signed_cert(common_name: str = "127.0.0.1") -> tuple[bytes, bytes]:
    """Generate a short-lived self-signed cert/key pair (PEM), for tests only."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(minutes=30))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(common_name), x509.IPAddress(__import__("ipaddress").ip_address(common_name))]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return cert_pem, key_pem


class TestCoordinationServerTLS(unittest.TestCase):
    def setUp(self):
        cert_pem, key_pem = _generate_self_signed_cert("127.0.0.1")

        import tempfile

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        cert_path = f"{self.tmpdir.name}/server.crt"
        key_path = f"{self.tmpdir.name}/server.key"
        with open(cert_path, "wb") as f:
            f.write(cert_pem)
        with open(key_path, "wb") as f:
            f.write(key_pem)

        self.registry = NodeRegistry()
        config = NodeConfig(
            node_id="node-a",
            host="127.0.0.1",
            port=0,
            tls_cert_path=cert_path,
            tls_key_path=key_path,
        )
        self.server, self.port = serve(config, self.registry, port=0)
        self.addCleanup(lambda: self.server.stop(grace=0))

        self.client = CoordinationClient(
            f"127.0.0.1:{self.port}", ca_cert_path=cert_path
        )
        self.addCleanup(self.client.close)

    def test_register_and_sync_calibration_over_tls(self):
        from limen.distributed.node import NodeInfo

        peer = NodeInfo(node_id="node-b", host="127.0.0.1", port=6000, device_ids=["qpu-1"])
        self.assertTrue(self.client.register(peer))

        model = HardwareDeltaModel.identity("qpu-1", SubstrateType.NEUTRAL_ATOM, 4)
        self.registry.local.register(model)
        fetched = self.client.sync_calibration("qpu-1")
        self.assertEqual(fetched.to_dict(), model.to_dict())

    def test_insecure_client_cannot_talk_to_tls_server(self):
        # An insecure client against a TLS-only server should fail, proving
        # the server is actually enforcing TLS rather than silently
        # accepting cleartext.
        plain_client = CoordinationClient(f"127.0.0.1:{self.port}")
        self.addCleanup(plain_client.close)
        with self.assertRaises(grpc.RpcError):
            plain_client.heartbeat("node-b")


if __name__ == "__main__":
    unittest.main()
