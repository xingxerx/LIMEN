# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Offline, post-quantum-signed license keys for paid LIMEN tiers.

The free tier needs nothing. A paid tier is a small JSON payload signed with
ML-DSA-65 (reusing limen.security.pqc). Verification is fully offline: no
phone-home, no customer data leaves the machine.

Key lookup order: LIMEN_LICENSE env var (the key string itself), then
LIMEN_LICENSE_FILE, then ~/.limen/license.key.
"""

from limen.licensing.license import (
    FREE,
    PRO,
    TEAM,
    License,
    LicenseError,
    current_license,
    decode_key,
    encode_key,
    require_feature,
)

__all__ = [
    "FREE",
    "PRO",
    "TEAM",
    "License",
    "LicenseError",
    "current_license",
    "decode_key",
    "encode_key",
    "require_feature",
]
