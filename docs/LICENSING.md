# LIMEN Licensing and Tiers

LIMEN is source-available under the Elastic License 2.0 (ELv2). The core is
free to use, including in production. Paid tiers unlock a small set of
features by checking an offline license key. Nothing phones home.

## Tiers

| Tier | Price | What it adds |
|---|---|---|
| Free | $0 | Everything not listed below: IR, compilers, validator, hardware adapters, gate-model pipeline and unsigned certificates, budget router, router memory, circuit cutting, Stackelberg co-design, portfolio compilation |
| Pro | Contact | `signed_certificates`: ML-DSA-65 (FIPS 204) signed ledger entries via `RouterMemory.append_certificate` |
| Team | Contact | Everything in Pro, plus `peer_mesh`: multi-node registration via `limen.distributed.server.register_with_peers`, and written permission to offer LIMEN as a hosted or managed service (ELv2 otherwise prohibits this) |

`priority_routing` exists in the code as a reserved Pro feature. It is not
enforced or implemented yet and is not part of what a Pro license sells
today.

## Installing a key

A key looks like `LIMEN1.<payload>.<signature>`. Verification needs the
`pqc` extra:

```
pip install "limen-compiler[pqc]"
```

Then either set an environment variable:

```
$env:LIMEN_LICENSE = "LIMEN1.your-key-here"
```

or save the key to `~/.limen/license.key` (or point `LIMEN_LICENSE_FILE` at
another path).

Check what is active:

```
python -c "from limen.licensing.license import current_license as c; l=c(); print(l.tier, l.valid, l.reason)"
```

## How keys work

- Payload fields: `lid` (license id), `sub` (customer), `tier`, `features`,
  `iat` and `exp` (unix seconds, `exp` 0 means perpetual).
- The payload is signed with the issuer's ML-DSA-65 private key, which never
  ships in the package. Only the public key is in
  `limen/licensing/_issuer_key.py`.
- An invalid, expired or malformed key falls back to Free behaviour and
  reports why. Using a paid feature without a valid key raises
  `LicenseError` with that reason.

## Getting a license

Open an issue at https://github.com/xingxerx/LIMEN/issues titled
"License request" with your organization and intended use. Use is subject to
[ACCEPTABLE_USE.md](../ACCEPTABLE_USE.md).
