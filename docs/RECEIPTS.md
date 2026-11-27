# Signed run receipts

New in 0.9.3.

A receipt is one JSON file that shows a certified LIMEN run happened and has
not been edited since it was signed. You can hand it to a customer, an auditor
or a reviewer, and they can check it offline on their own machine without a
LIMEN license and without talking to you.

Signing receipts needs a Pro or Team license. Checking them is free.

## Install

Signing and checking both use ML-DSA-65 (FIPS 204), which comes with the `pqc`
extra:

```
pip install "limen-compiler[pqc]"
```

Without the extra, `limen keygen` and `limen verify` stop with a message that
names this install line.

## Sign a run (Pro or Team)

### 1. Create a signing key, once

```
limen keygen results
```

This writes two files into `results`:

| File | What to do with it |
|---|---|
| `limen_signing.key` | Private. Keep it on the machine that runs jobs. Never commit it or send it. |
| `limen_signing.pub` | Public. Give it to anyone who will check your receipts. |

It also prints a key id (the first 16 hex characters of the SHA-256 of the
public key). `keygen` refuses to overwrite a key that is already there.

### 2. Set your license

```
$env:LIMEN_LICENSE = "LIMEN1.your-key-here"
```

See [LICENSING.md](LICENSING.md) for the other ways to install a key.

### 3. Run with `run_signed`

```python
from limen import RouteRequest
from limen.receipt import run_signed

qubo = {
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x0", "x1"):  2.0,
}

request = RouteRequest(qubo=qubo, fidelity_target=0.9, credit_budget=0)
cert = run_signed(request, results_dir="results")

print(cert.metadata["receipt_path"])
# results\receipts\receipt_000001_3f9a1c2b7d4e.json
```

`run_signed` is `run_route_request` plus a receipt. Every other keyword
argument is passed straight through. The receipt lands in
`<results_dir>/receipts/`.

The signing key is looked up in this order:

1. the `signing_key=` argument (a key object or a path to a key file)
2. the `LIMEN_SIGNING_KEY` environment variable (a path)
3. `<results_dir>/limen_signing.key`

### If there is no key or no license

The run still completes and you still get your certificate. No receipt is
written, and `cert.notes` says why:

- `No receipt: no signing key found. Run limen keygen ...`
- `No receipt: ...` followed by the license reason

The run is still recorded in the hash-chained ledger, unsigned, exactly as on
the free tier.

## Check a receipt (free)

```
limen verify receipt_000001_3f9a1c2b7d4e.json --pubkey limen_signing.pub
```

```text
VALID  receipt seq 1 backend statevector
  payload hash : matches
  signature    : ML-DSA-65 verified
  signer key   : 25507c4e3af9b4eb (pinned to your key)
```

| Option | What it does |
|---|---|
| `--pubkey FILE` | Checks the signature against a public key you already trust. Use this whenever it matters who signed. |
| `--ledger FILE` | Also checks the receipt against the signer's `router_memory.sqlite3`: the hash chain must verify end to end and the entry must still be there with the same chain head. |

Exit codes: `0` valid, `1` invalid, `2` could not read the receipt or key.
That makes `limen verify` usable as a step in a script or a CI job.

### Why `--pubkey` matters

A receipt carries the public key it was signed with. If you check it without
`--pubkey`, the result only tells you the file has not changed since someone
signed it. Anyone can make a key and sign their own file. To know the receipt
came from a specific signer, get their `limen_signing.pub` through a channel
you trust and pass it with `--pubkey`. Compare the key id it prints with the
one the signer gave you.

## What a receipt proves, and what it does not

It proves:

- the certificate payload is byte for byte what was signed (any edit breaks
  the payload hash)
- the holder of the signing key signed it
- with `--ledger`, the run sits in the signer's append-only ledger at the
  position the receipt claims, and that ledger's history has not been altered

It does not prove:

- that the answer is optimal. That is what the certificate inside the receipt
  reports, with its own limits (see the README).
- who the signer is, unless you pin their public key.
- the time of the run beyond the signer's own clock (`recorded_at`).
- anything about a signer who controls the key and chooses to sign a false
  record. A receipt protects a record from being changed after the fact. It
  does not replace trust in the party that produced it.

## What is in the file

Format `limen-receipt/1`:

| Field | Meaning |
|---|---|
| `format` | `limen-receipt/1` |
| `alg` | `ML-DSA-65` |
| `seq` | Position of the run in the signer's ledger |
| `recorded_at` | When the ledger recorded it (unix seconds, signer's clock) |
| `backend` | Backend the run executed on |
| `payload` | The certificate as recorded |
| `payload_sha256` | SHA-256 of the canonical JSON of `payload` |
| `chain_sha256` | Ledger chain head at this entry, which binds it to all earlier entries |
| `signature` | ML-DSA-65 signature over `{payload_sha256, chain_sha256}` |
| `public_key` | Signer's public key, base64 |
| `key_fingerprint` | Key id, first 16 hex characters of the SHA-256 of the public key |

## Python API

Everything lives in `limen.receipt`:

```python
import json
from limen.receipt import verify_receipt, check_against_ledger
from limen.security.pqc import load_public_key

receipt = json.load(open("receipt_000001_3f9a1c2b7d4e.json"))
check = verify_receipt(receipt, load_public_key("limen_signing.pub"))

print(check.ok, check.reasons)
print(check_against_ledger(receipt, "results/router_memory.sqlite3"))  # [] means it agrees
```

`verify_receipt` never raises. Problems are listed in `check.reasons`.

For a long-running process, or to control where receipts go, use the ledger
class directly and pass it to `run_route_request`:

```python
from limen import run_route_request
from limen.receipt import SigningRouterMemory, resolve_signing_key

memory = SigningRouterMemory(
    "results/router_memory.sqlite3",
    signing_key=resolve_signing_key(None, "results"),
    receipts_dir="receipts_out",
)
cert = run_route_request(request, results_dir="results", memory=memory)
print(memory.last_receipt_path, memory.signing_skipped)
memory.close()
```

## Limits today

- The `limend` daemon does not sign receipts yet. Use `run_signed` or
  `SigningRouterMemory` from your own code.
- One signing key per results folder. There is no key rotation or revocation
  list. If a key leaks, make a new one in a new folder and tell the people who
  pinned the old key.
- Entries moved out of the live ledger by `archive_and_compact()` are reported
  by `--ledger` as not in the ledger. The receipt itself still verifies.
