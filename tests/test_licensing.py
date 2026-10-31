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
