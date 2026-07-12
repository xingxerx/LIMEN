"""Tests for limen.security.pqc: post-quantum (ML-DSA / FIPS 204) signing
of LIMEN's persisted/exchanged JSON artifacts. This is a purely additive
capability -- these tests exercise it in isolation, not through any
existing save/load path, which stays unsigned by default (see module
docstring in limen/security/pqc.py)."""

import pathlib
import tempfile
import unittest

from limen.security.pqc import (
    canonical_json_bytes,
    generate_signing_key,
    load_private_key,
    load_public_key,
    private_key_from_bytes,
    private_key_to_bytes,
    public_key_from_bytes,
    public_key_to_bytes,
    save_private_key,
    save_public_key,
    sign_bytes,
    sign_json,
    verify_bytes,
    verify_json,
)


class TestSignVerifyBytes(unittest.TestCase):

    def test_valid_signature_verifies(self):
        key = generate_signing_key()
        sig = sign_bytes(key, b"hello world")
        self.assertTrue(verify_bytes(key.public_key(), b"hello world", sig))

    def test_tampered_data_fails_verification(self):
        key = generate_signing_key()
        sig = sign_bytes(key, b"hello world")
        self.assertFalse(verify_bytes(key.public_key(), b"tampered", sig))

    def test_tampered_signature_fails_verification(self):
        key = generate_signing_key()
        sig = bytearray(sign_bytes(key, b"hello world"))
        sig[0] ^= 0xFF
        self.assertFalse(verify_bytes(key.public_key(), b"hello world", bytes(sig)))

    def test_wrong_public_key_fails_verification(self):
        key_a = generate_signing_key()
        key_b = generate_signing_key()
        sig = sign_bytes(key_a, b"hello world")
        self.assertFalse(verify_bytes(key_b.public_key(), b"hello world", sig))


class TestKeySerialization(unittest.TestCase):

    def test_private_key_round_trips_through_bytes(self):
        key = generate_signing_key()
        restored = private_key_from_bytes(private_key_to_bytes(key))
        sig = sign_bytes(restored, b"data")
        self.assertTrue(verify_bytes(key.public_key(), b"data", sig))

    def test_public_key_round_trips_through_bytes(self):
        key = generate_signing_key()
        restored_pub = public_key_from_bytes(public_key_to_bytes(key.public_key()))
        sig = sign_bytes(key, b"data")
        self.assertTrue(verify_bytes(restored_pub, b"data", sig))

    def test_private_key_round_trips_through_file(self):
        key = generate_signing_key()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "priv.key"
            save_private_key(path, key)
            restored = load_private_key(path)
        sig = sign_bytes(restored, b"data")
        self.assertTrue(verify_bytes(key.public_key(), b"data", sig))

    def test_public_key_round_trips_through_file(self):
        key = generate_signing_key()
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "pub.key"
            save_public_key(path, key.public_key())
            restored_pub = load_public_key(path)
        sig = sign_bytes(key, b"data")
        self.assertTrue(verify_bytes(restored_pub, b"data", sig))


class TestCanonicalJson(unittest.TestCase):

    def test_key_order_does_not_affect_encoding(self):
        a = {"b": 1, "a": 2}
        b = {"a": 2, "b": 1}
        self.assertEqual(canonical_json_bytes(a), canonical_json_bytes(b))

    def test_different_values_produce_different_encoding(self):
        self.assertNotEqual(
            canonical_json_bytes({"a": 1}), canonical_json_bytes({"a": 2})
        )


class TestSignVerifyJson(unittest.TestCase):

    def test_valid_signature_verifies_regardless_of_key_order(self):
        key = generate_signing_key()
        obj = {"job_id": "abc123", "status": "DONE", "shots": 1000}
        sig = sign_json(key, obj)
        reordered = {"shots": 1000, "status": "DONE", "job_id": "abc123"}
        self.assertTrue(verify_json(key.public_key(), reordered, sig))

    def test_tampered_field_fails_verification(self):
        key = generate_signing_key()
        obj = {"job_id": "abc123", "status": "DONE"}
        sig = sign_json(key, obj)
        tampered = {"job_id": "abc123", "status": "CANCELLED"}
        self.assertFalse(verify_json(key.public_key(), tampered, sig))

    def test_malformed_base64_signature_returns_false_not_raises(self):
        key = generate_signing_key()
        obj = {"a": 1}
        self.assertFalse(verify_json(key.public_key(), obj, "not-valid-base64!!!"))


if __name__ == "__main__":
    unittest.main()
