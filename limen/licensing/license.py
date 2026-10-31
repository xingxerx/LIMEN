# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""License key format, verification and feature gating.

Key format:  LIMEN1.<base64url(payload_json)>.<base64url(ml-dsa-65 signature)>

Payload fields: lid (license id), sub (customer email or org), tier,
features (list), iat / exp (unix seconds; exp 0 = perpetual).

Only the PUBLIC issuer key ships in the package (limen/licensing/_issuer_key.py).
The private issuer seed lives off-repo (see scripts/issue_license.py) and must
never be committed.
"""

from __future__ import annotations

import base64
import functools
import json
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from limen.licensing._issuer_key import ISSUER_PUBLIC_KEY_B64

FREE = "free"
PRO = "pro"
TEAM = "team"

_PREFIX = "LIMEN1"

# Features unlocked per tier. Paid features are placeholders until the
# product decision on what is paid is made; nothing in LIMEN is gated yet.
TIER_FEATURES: dict[str, frozenset[str]] = {
    FREE: frozenset(),
    PRO: frozenset({"signed_certificates", "priority_routing"}),
    TEAM: frozenset({"signed_certificates", "priority_routing", "peer_mesh", "router_memory"}),
}


class LicenseError(Exception):
    """Raised when a paid feature is used without a valid license."""


@dataclass(frozen=True)
class License:
    tier: str = FREE
    sub: str = ""
    lid: str = ""
    iat: int = 0
    exp: int = 0
    features: frozenset[str] = field(default_factory=frozenset)
    valid: bool = True
    reason: str = ""

    def has(self, feature: str) -> bool:
        if not self.valid:
            return False
        return feature in self.features or feature in TIER_FEATURES.get(self.tier, frozenset())


_FREE_LICENSE = License()


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def encode_key(private_key: Any, payload: dict[str, Any]) -> str:
    """Issuer side: sign *payload* and return the license key string."""
    from limen.security import pqc

    body = _canonical(payload)
    sig = pqc.sign_bytes(private_key, body)
    return f"{_PREFIX}.{_b64e(body)}.{_b64e(sig)}"


def _invalid(reason: str) -> License:
    return License(valid=False, reason=reason)


def decode_key(key: str, public_key_b64: str | None = None, now: float | None = None) -> License:
    """Verify a license key offline. Never raises; returns License(valid=False)."""
    pub_b64 = ISSUER_PUBLIC_KEY_B64 if public_key_b64 is None else public_key_b64
    if not pub_b64:
        return _invalid("no issuer public key configured")
    parts = key.strip().split(".")
    if len(parts) != 3 or parts[0] != _PREFIX:
        return _invalid("malformed key")
    try:
        from limen.security import pqc
    except ImportError:
        return _invalid("pip install limen-compiler[pqc] to verify license keys")
    try:
        body, sig = _b64d(parts[1]), _b64d(parts[2])
        pub = pqc.public_key_from_bytes(base64.b64decode(pub_b64))
    except (ValueError, TypeError):
        return _invalid("malformed key")
    if not pqc.verify_bytes(pub, body, sig):
        return _invalid("bad signature")
    try:
        p = json.loads(body)
    except ValueError:
        return _invalid("malformed payload")
    exp = int(p.get("exp", 0))
    if exp and (now if now is not None else time.time()) > exp:
        return _invalid("expired")
    tier = p.get("tier", FREE)
    if tier not in TIER_FEATURES:
        return _invalid(f"unknown tier {tier!r}")
    return License(
        tier=tier,
        sub=str(p.get("sub", "")),
        lid=str(p.get("lid", "")),
        iat=int(p.get("iat", 0)),
        exp=exp,
        features=frozenset(p.get("features", [])),
    )


def _find_key() -> str | None:
    env = os.environ.get("LIMEN_LICENSE")
    if env:
        return env
    path = os.environ.get("LIMEN_LICENSE_FILE") or str(pathlib.Path.home() / ".limen" / "license.key")
    try:
        return pathlib.Path(path).read_text().strip() or None
    except OSError:
        return None


@functools.lru_cache(maxsize=1)
def current_license() -> License:
    """The license active on this machine (free tier if none is installed)."""
    key = _find_key()
    return _FREE_LICENSE if key is None else decode_key(key)


def require_feature(feature: str) -> Callable:
    """Decorator: raise LicenseError unless the active license has *feature*."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            lic = current_license()
            if not lic.has(feature):
                why = f" ({lic.reason})" if lic.reason else ""
                raise LicenseError(
                    f"'{feature}' needs a paid LIMEN license{why}. Set LIMEN_LICENSE or ~/.limen/license.key."
                )
            return fn(*args, **kwargs)

        return wrapper

    return deco
