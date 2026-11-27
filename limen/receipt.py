# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Portable signed run receipts.

A receipt is one JSON file that proves a certified LIMEN run happened and was
not edited afterwards. It carries the certificate payload, its SHA-256, the
ledger chain head it was witnessed under, an ML-DSA-65 signature over
``{payload_sha256, chain_sha256}`` (the same content RouterMemory signs) and
the signer's public key. Anyone can check it offline with ``limen verify``,
no LIMEN license needed: signing is the paid part, checking is free.

This module is additive. It reuses limen.security.pqc for the crypto,
limen.router.memory for the hash chain and limen.licensing for the
``signed_certificates`` gate, and it does not touch limen.pipeline:
:class:`SigningRouterMemory` is a RouterMemory that run_route_request already
accepts through its ``memory=`` argument, and :func:`run_signed` wraps the
call for the common case.

Trust note: a receipt embeds its public key, so verifying with no pinned key
only proves "unmodified since signed by whoever holds this key". To prove it
came from a key you trust, pass the signer's public key (``--pubkey``).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any

from limen.router.memory import (
    DEFAULT_FILENAME,
    LedgerEntry,
    RouterMemory,
    _canonical_json,
    _sha256_hex,
)

RECEIPT_FORMAT = "limen-receipt/1"
ALGORITHM = "ML-DSA-65"
KEY_FILE = "limen_signing.key"
PUB_FILE = "limen_signing.pub"
RECEIPTS_SUBDIR = "receipts"

_PQC_HINT = "post-quantum signing needs: pip install limen-compiler[pqc]"


def key_fingerprint(public_key_bytes: bytes) -> str:
    """Short, stable id for a public key (first 16 hex chars of its SHA-256)."""
    return hashlib.sha256(public_key_bytes).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Build, write, verify
# ---------------------------------------------------------------------------


def build_receipt(entry: LedgerEntry, public_key: Any) -> dict[str, Any]:
    """Turn a signed ledger entry into a self-contained receipt dict."""
    if entry.signature is None:
        raise ValueError("ledger entry is unsigned; there is nothing to put in a receipt")
    from limen.security.pqc import public_key_to_bytes

    raw = public_key_to_bytes(public_key)
    return {
        "format": RECEIPT_FORMAT,
        "alg": ALGORITHM,
        "seq": entry.seq,
        "recorded_at": entry.recorded_at,
        "backend": entry.backend,
        "payload": entry.payload,
        "payload_sha256": entry.payload_sha256,
        "chain_sha256": entry.chain_sha256,
        "signature": entry.signature,
        "public_key": base64.b64encode(raw).decode("ascii"),
        "key_fingerprint": key_fingerprint(raw),
    }


def write_receipt(directory: pathlib.Path | str, receipt: dict[str, Any]) -> pathlib.Path:
    """Write *receipt* under *directory* and return its path."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"receipt_{int(receipt['seq']):06d}_{receipt['payload_sha256'][:12]}.json"
    path = directory / name
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return path


@dataclass
class ReceiptCheck:
    """Outcome of :func:`verify_receipt`. ``ok`` is the single bottom line."""

    ok: bool
    signature_valid: bool = False
    payload_hash_valid: bool = False
    key_fingerprint: str | None = None
    key_pinned: bool = False
    reasons: list[str] = field(default_factory=list)


def verify_receipt(receipt: Any, public_key: Any = None) -> ReceiptCheck:
    """Check a receipt offline. Never raises; problems land in ``reasons``.

    With *public_key* (a limen.security.pqc public key) the signature must
    verify under that key. Without it, the key embedded in the receipt is
    used, which proves integrity but not who signed.
    """
    res = ReceiptCheck(ok=False)
    if not isinstance(receipt, dict) or receipt.get("format") != RECEIPT_FORMAT:
        res.reasons.append(f"not a {RECEIPT_FORMAT} document")
        return res
    try:
        payload = receipt["payload"]
        payload_sha = receipt["payload_sha256"]
        chain = receipt["chain_sha256"]
        signature = receipt["signature"]
        embedded_b64 = receipt["public_key"]
    except KeyError as exc:
        res.reasons.append(f"missing field {exc.args[0]!r}")
        return res
    if not isinstance(payload, dict) or not all(
        isinstance(v, str) for v in (payload_sha, chain, signature, embedded_b64)
    ):
        res.reasons.append("malformed receipt fields")
        return res

    recomputed = _sha256_hex(_canonical_json(payload).encode("utf-8"))
    res.payload_hash_valid = recomputed == payload_sha
    if not res.payload_hash_valid:
        res.reasons.append("payload does not match its recorded hash (edited after signing)")

    try:
        from limen.security import pqc
    except ImportError:
        res.reasons.append(_PQC_HINT)
        return res

    try:
        embedded_raw = base64.b64decode(embedded_b64, validate=True)
        key = pqc.public_key_from_bytes(embedded_raw)
    except Exception:
        res.reasons.append("embedded public key is unreadable")
        return res

    key_raw = embedded_raw
    if public_key is not None:
        res.key_pinned = True
        key = public_key
        key_raw = pqc.public_key_to_bytes(public_key)
        if key_raw != embedded_raw:
            res.reasons.append("receipt was signed by a different key than the one you pinned")
    res.key_fingerprint = key_fingerprint(key_raw)

    res.signature_valid = pqc.verify_json(
        key, {"payload_sha256": payload_sha, "chain_sha256": chain}, signature
    )
    if not res.signature_valid and not any("different key" in r for r in res.reasons):
        res.reasons.append("signature does not verify")
    res.ok = res.payload_hash_valid and res.signature_valid
    return res


def check_against_ledger(receipt: dict[str, Any], ledger_path: pathlib.Path | str) -> list[str]:
    """Cross-check a receipt against its RouterMemory ledger file.

    Returns a list of problems (empty means the ledger agrees): the hash
    chain must verify end to end, and the receipt's entry must still be in
    it with the same chain head.
    """
    path = pathlib.Path(ledger_path)
    if not path.is_file():
        return [f"ledger not found: {path}"]
    problems: list[str] = []
    mem = RouterMemory(path)
    try:
        if not mem.verify_ledger():
            problems.append("ledger hash chain does not verify (history was altered)")
        match = [e for e in mem.certificates() if e.payload_sha256 == receipt.get("payload_sha256")]
        if not match:
            problems.append("this receipt's entry is not in the ledger (or was archived)")
        elif match[0].chain_sha256 != receipt.get("chain_sha256"):
            problems.append("chain head differs from the ledger's copy of this entry")
    finally:
        mem.close()
    return problems


# ---------------------------------------------------------------------------
# Signing on the way into the ledger
# ---------------------------------------------------------------------------


class SigningRouterMemory(RouterMemory):
    """A RouterMemory that signs every certificate it appends and writes a
    receipt file for it.

    Pass it to run_route_request as ``memory=``. Signing needs a Pro or Team
    license (the ``signed_certificates`` feature). Without one, entries are
    still chained into the ledger unsigned (tamper-evident, same as today),
    no receipt is written, and ``signing_skipped`` says why.
    """

    def __init__(
        self,
        path: pathlib.Path | str,
        *,
        signing_key: Any = None,
        receipts_dir: pathlib.Path | str | None = None,
    ) -> None:
        super().__init__(path)
        self._signing_key = signing_key
        self.receipts_dir = (
            pathlib.Path(receipts_dir)
            if receipts_dir is not None
            else self.path.parent / RECEIPTS_SUBDIR
        )
        self.last_receipt_path: pathlib.Path | None = None
        self.signing_skipped: str | None = None

    def append_certificate(
        self,
        payload: dict[str, Any],
        *,
        backend: str | None = None,
        private_key: Any = None,
        recorded_at: float | None = None,
    ) -> LedgerEntry:
        explicit = private_key is not None
        key = private_key if explicit else self._signing_key
        self.last_receipt_path = None
        self.signing_skipped = None
        if key is not None and not explicit:
            from limen.licensing.license import LicenseError, check_feature

            try:
                check_feature("signed_certificates")
            except LicenseError as exc:
                self.signing_skipped = str(exc)
                key = None
        entry = super().append_certificate(
            payload, backend=backend, private_key=key, recorded_at=recorded_at
        )
        if key is not None:
            # The signed entry is already in the ledger. A receipt file that
            # cannot be written must not fail the run or hide that fact.
            try:
                receipt = build_receipt(entry, key.public_key())
                self.last_receipt_path = write_receipt(self.receipts_dir, receipt)
            except OSError as exc:
                self.signing_skipped = f"receipt file not written: {exc}"
        return entry


def resolve_signing_key(signing_key: Any, results_dir: pathlib.Path | str | None) -> Any:
    """A private key from: the argument (key object or key file path), else
    the LIMEN_SIGNING_KEY environment variable (a file path), else
    ``<results_dir>/limen_signing.key``. None when none of those exist.
    Raises ValueError when a key was named but cannot be read."""
    source: Any = signing_key
    if source is None:
        env = os.environ.get("LIMEN_SIGNING_KEY")
        if env:
            source = env
        elif results_dir is not None:
            candidate = pathlib.Path(results_dir) / KEY_FILE
            if candidate.is_file():
                source = candidate
    if source is None:
        return None
    if isinstance(source, (str, os.PathLike)):
        try:
            from limen.security.pqc import load_private_key

            return load_private_key(source)
        except ImportError as exc:
            raise ValueError(_PQC_HINT) from exc
        except Exception as exc:
            raise ValueError(f"cannot read signing key {source}: {exc}") from exc
    return source


def run_signed(
    request: Any,
    *,
    results_dir: pathlib.Path | str,
    signing_key: Any = None,
    **kwargs: Any,
) -> Any:
    """run_route_request plus a signed receipt.

    Everything in *kwargs* goes straight to
    :func:`limen.pipeline.run_route_request`. On success the certificate's
    ``metadata["receipt_path"]`` points at the receipt file. If no key is
    found, or the license does not cover signing, the run still completes
    and a note on the certificate says what to do.
    """
    from limen.pipeline import run_route_request

    if "memory" in kwargs:
        raise TypeError(
            "run_signed manages the ledger itself; for a custom setup call "
            "run_route_request(memory=SigningRouterMemory(...))"
        )
    results = pathlib.Path(results_dir)
    key = resolve_signing_key(signing_key, results)
    mem = SigningRouterMemory(results / DEFAULT_FILENAME, signing_key=key)
    try:
        cert = run_route_request(request, results_dir=results, memory=mem, **kwargs)
        if mem.last_receipt_path is not None:
            cert.metadata["receipt_path"] = str(mem.last_receipt_path)
        elif key is None:
            cert.notes.append(
                f"No receipt: no signing key found. Run `limen keygen {results}` to create one."
            )
        elif mem.signing_skipped:
            cert.notes.append(f"No receipt: {mem.signing_skipped}")
        else:
            cert.notes.append("No receipt: the ledger write did not complete; check the results folder.")
        return cert
    finally:
        mem.close()


# ---------------------------------------------------------------------------
# CLI: limen verify / limen keygen
# ---------------------------------------------------------------------------


def _cmd_verify(args: argparse.Namespace) -> int:
    try:
        receipt = json.loads(pathlib.Path(args.receipt).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"cannot read receipt: {exc}", file=sys.stderr)
        return 2
    pinned = None
    if args.pubkey:
        try:
            from limen.security.pqc import load_public_key

            pinned = load_public_key(args.pubkey)
        except ImportError:
            print(_PQC_HINT, file=sys.stderr)
            return 2
        except Exception as exc:
            print(f"cannot read public key: {exc}", file=sys.stderr)
            return 2

    check = verify_receipt(receipt, pinned)
    problems = list(check.reasons)
    if args.ledger and isinstance(receipt, dict):
        problems += check_against_ledger(receipt, args.ledger)
    ok = check.ok and not problems

    seq = receipt.get("seq") if isinstance(receipt, dict) else None
    backend = receipt.get("backend") if isinstance(receipt, dict) else None
    print(f"{'VALID' if ok else 'INVALID'}  receipt seq {seq} backend {backend}")
    print(f"  payload hash : {'matches' if check.payload_hash_valid else 'MISMATCH'}")
    print(f"  signature    : {ALGORITHM} {'verified' if check.signature_valid else 'NOT verified'}")
    if check.key_fingerprint:
        trust = "pinned to your key" if check.key_pinned else "taken from the receipt, pass --pubkey to pin it"
        print(f"  signer key   : {check.key_fingerprint} ({trust})")
    if args.ledger:
        print(f"  ledger       : {'agrees' if not problems else 'see problems'}")
    for problem in problems:
        print(f"  problem      : {problem}")
    return 0 if ok else 1


def _cmd_keygen(args: argparse.Namespace) -> int:
    try:
        from limen.security import pqc
    except ImportError:
        print(_PQC_HINT, file=sys.stderr)
        return 2
    directory = pathlib.Path(args.directory)
    directory.mkdir(parents=True, exist_ok=True)
    private_path, public_path = directory / KEY_FILE, directory / PUB_FILE
    if private_path.exists() or public_path.exists():
        print(f"refusing to overwrite an existing key in {directory}", file=sys.stderr)
        return 2
    key = pqc.generate_signing_key()
    pqc.save_private_key(private_path, key)
    pqc.save_public_key(public_path, key.public_key())
    try:
        os.chmod(private_path, 0o600)
    except OSError:
        pass
    fingerprint = key_fingerprint(pqc.public_key_to_bytes(key.public_key()))
    print(f"wrote {private_path} (keep private) and {public_path} (share this one)")
    print(f"key id {fingerprint}")
    print("signing receipts needs a Pro or Team license (LIMEN_LICENSE); verifying is free")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="limen", description="LIMEN signed run receipts.")
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify", help="check a receipt file offline")
    verify.add_argument("receipt", help="path to a receipt_*.json file")
    verify.add_argument("--pubkey", help="signer public key file to pin (limen_signing.pub)")
    verify.add_argument("--ledger", help="router_memory.sqlite3 to cross-check the receipt against")
    verify.set_defaults(func=_cmd_verify)

    keygen = sub.add_parser("keygen", help="create a signing keypair in a folder")
    keygen.add_argument("directory", help="folder to write limen_signing.key and limen_signing.pub into")
    keygen.set_defaults(func=_cmd_keygen)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
