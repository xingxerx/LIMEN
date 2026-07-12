# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Post-quantum digital signatures for LIMEN's persisted/exchanged artifacts.

LIMEN already writes a family of trust-sensitive JSON artifacts to disk and
across the distributed gRPC layer: calibration snapshots
(limen.router.calibration), QPU job lifecycle state (limen.router.job_state),
route plans, and certificates. Classical TLS (limen.distributed.server/client)
protects those artifacts *in transit*, but says nothing about a snapshot
sitting in results/ for months, or a peer's CompilePartition RPC response --
and a sufficiently capable quantum computer breaks the classical RSA/ECDSA
signatures typically used to protect data at rest.

This module signs those artifacts with ML-DSA (FIPS 204, formerly
CRYSTALS-Dilithium), a NIST-standardized post-quantum signature scheme, via
the ``cryptography`` package's native ``hazmat.primitives.asymmetric.mldsa``
implementation (no third-party PQC library required). ML-DSA-65 (NIST
security category 3) is used throughout as a balanced default -- comparable
to a 192-bit classical security level, matching the mid-tier security
category most general-purpose signing guidance recommends over the smaller
(-44) or the more conservative, larger (-87) parameter sets.

Deliberately additive, not a replacement: nothing in the rest of LIMEN
requires a keypair to exist or a signature to be present. Every existing
JSON write path (save_state, fetch_backend_calibration's caller, RoutePlan/
EndToEndCertificate.to_dict) is unchanged; callers who want tamper-evidence
opt in by calling sign_json()/verify_json() themselves alongside their
existing read/write code.

Scope limit: signing-at-rest/in-transit tamper-evidence only. The
``cryptography`` version this module requires also exposes ML-KEM
(FIPS 203, post-quantum key encapsulation) via
``hazmat.primitives.asymmetric.mlkem``, but nothing here uses it --
there is no PQC encryption or key-exchange capability in this module,
and none of the classical TLS transport in limen.distributed is
touched (see that module's docstring for why: gRPC's TLS handshake is
handled by its C-core, not reachable from these primitives).
"""

from __future__ import annotations

import base64
import json
import pathlib
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import mldsa

PrivateKey = mldsa.MLDSA65PrivateKey
PublicKey = mldsa.MLDSA65PublicKey


def generate_signing_key() -> PrivateKey:
    """Generate a new ML-DSA-65 keypair. Returns the private key."""
    return mldsa.MLDSA65PrivateKey.generate()


def private_key_to_bytes(key: PrivateKey) -> bytes:
    """Serialize a private key to its 32-byte seed (compact, deterministic)."""
    return key.private_bytes_raw()


def private_key_from_bytes(data: bytes) -> PrivateKey:
    """Reconstruct a private key from a 32-byte seed."""
    return mldsa.MLDSA65PrivateKey.from_seed_bytes(data)


def public_key_to_bytes(key: PublicKey) -> bytes:
    """Serialize a public key to its raw bytes."""
    return key.public_bytes_raw()


def public_key_from_bytes(data: bytes) -> PublicKey:
    """Reconstruct a public key from raw bytes."""
    return mldsa.MLDSA65PublicKey.from_public_bytes(data)


def save_private_key(path: pathlib.Path | str, key: PrivateKey) -> None:
    """Write a private key seed to *path* as base64 text.

    The file contains secret key material -- callers are responsible for
    file permissions/access control; this function only handles encoding.
    """
    pathlib.Path(path).write_text(base64.b64encode(private_key_to_bytes(key)).decode("ascii"))


def load_private_key(path: pathlib.Path | str) -> PrivateKey:
    """Load a private key seed previously written by :func:`save_private_key`."""
    raw = base64.b64decode(pathlib.Path(path).read_text().strip())
    return private_key_from_bytes(raw)


def save_public_key(path: pathlib.Path | str, key: PublicKey) -> None:
    """Write a public key to *path* as base64 text."""
    pathlib.Path(path).write_text(base64.b64encode(public_key_to_bytes(key)).decode("ascii"))


def load_public_key(path: pathlib.Path | str) -> PublicKey:
    """Load a public key previously written by :func:`save_public_key`."""
    raw = base64.b64decode(pathlib.Path(path).read_text().strip())
    return public_key_from_bytes(raw)


def sign_bytes(private_key: PrivateKey, data: bytes) -> bytes:
    """Sign *data*, returning the raw ML-DSA-65 signature (3309 bytes)."""
    return private_key.sign(data)


def verify_bytes(public_key: PublicKey, data: bytes, signature: bytes) -> bool:
    """Verify *signature* over *data*. Returns False on any mismatch/error
    rather than raising -- callers treat "not verified" as one outcome,
    not a special error path."""
    try:
        public_key.verify(signature, data)
        return True
    except InvalidSignature:
        return False


def canonical_json_bytes(obj: dict[str, Any]) -> bytes:
    """Deterministic JSON encoding: sorted keys, no incidental whitespace.

    Two dicts with the same keys/values but different insertion order or
    formatting must produce byte-identical output, so signing/verification
    doesn't depend on how the caller happened to build or re-read the dict.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_json(private_key: PrivateKey, obj: dict[str, Any]) -> str:
    """Sign a JSON-serializable dict; returns a base64-encoded signature."""
    return base64.b64encode(sign_bytes(private_key, canonical_json_bytes(obj))).decode("ascii")


def verify_json(public_key: PublicKey, obj: dict[str, Any], signature_b64: str) -> bool:
    """Verify a base64-encoded signature (from :func:`sign_json`) over *obj*."""
    try:
        signature = base64.b64decode(signature_b64)
    except (ValueError, TypeError):
        return False
    return verify_bytes(public_key, canonical_json_bytes(obj), signature)
