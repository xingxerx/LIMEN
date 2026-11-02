import base64
import time

import pytest

pqc = pytest.importorskip("limen.security.pqc", exc_type=ImportError)

from limen.licensing.license import PRO, TEAM, decode_key, encode_key  # noqa: E402


@pytest.fixture(scope="module")
def issuer():
    key = pqc.generate_signing_key()
    pub = base64.b64encode(pqc.public_key_to_bytes(key.public_key())).decode("ascii")
    return key, pub


def _payload(**kw):
    p = {"lid": "t1", "sub": "a@b.c", "tier": PRO, "features": [], "iat": 0, "exp": 0}
    p.update(kw)
    return p


def test_valid_pro_key(issuer):
    key, pub = issuer
    lic = decode_key(encode_key(key, _payload()), pub)
    assert lic.valid and lic.tier == PRO
    assert lic.has("signed_certificates")
    assert not lic.has("peer_mesh")


def test_team_has_peer_mesh(issuer):
    key, pub = issuer
    assert decode_key(encode_key(key, _payload(tier=TEAM)), pub).has("peer_mesh")


def test_tampered_payload_rejected(issuer):
    key, pub = issuer
    k = encode_key(key, _payload())
    head, body, sig = k.split(".")
    forged = encode_key(key, _payload(tier=TEAM)).split(".")[1]
    lic = decode_key(f"{head}.{forged}.{sig}", pub)
    assert not lic.valid and lic.reason == "bad signature"


def test_wrong_issuer_rejected(issuer):
    key, _ = issuer
    other = pqc.generate_signing_key()
    other_pub = base64.b64encode(pqc.public_key_to_bytes(other.public_key())).decode("ascii")
    assert not decode_key(encode_key(key, _payload()), other_pub).valid


def test_expired(issuer):
    key, pub = issuer
    lic = decode_key(encode_key(key, _payload(exp=100)), pub, now=time.time())
    assert not lic.valid and lic.reason == "expired"


def test_no_issuer_configured():
    assert decode_key("LIMEN1.a.b", "").reason == "no issuer public key configured"


def test_malformed():
    assert decode_key("garbage", "AAAA").reason == "malformed key"

from limen.licensing.license import FREE, LicenseError, License


def test_free_tier_blocks_signed_certificates(tmp_path, monkeypatch):
    pytest.importorskip("limen.security.pqc", exc_type=ImportError)
    from unittest import mock
    from limen.router.memory import RouterMemory

    mem = RouterMemory(tmp_path / "m.sqlite3")
    try:
        with mock.patch("limen.licensing.license.current_license", return_value=License(tier=FREE)):
            mem.append_certificate({"e": 1})  # unsigned stays free
            with pytest.raises(LicenseError):
                mem.append_certificate({"e": 2}, private_key=pqc.generate_signing_key())
    finally:
        mem.close()


def test_free_tier_blocks_peer_mesh_but_not_single_node():
    pytest.importorskip("grpc")
    from unittest import mock
    from limen.distributed.config import NodeConfig
    from limen.distributed.registry import NodeRegistry
    from limen.distributed.server import register_with_peers

    with mock.patch("limen.licensing.license.current_license", return_value=License(tier=FREE)):
        register_with_peers(NodeConfig(node_id="a", host="127.0.0.1", port=0, known_peers=[]), NodeRegistry())
        with pytest.raises(LicenseError):
            register_with_peers(
                NodeConfig(node_id="a", host="127.0.0.1", port=0, known_peers=["127.0.0.1:1"]), NodeRegistry()
            )
