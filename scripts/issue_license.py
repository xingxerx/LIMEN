"""Issue LIMEN license keys (issuer side only; never ship the seed).

The ML-DSA-65 issuer seed is never written in plain text. It is wrapped
with Windows DPAPI (Data Protection API, CurrentUser scope) plus your
issuer passphrase as extra entropy, so other code running as you cannot
unwrap it without the passphrase. Default location, outside the ATRIUM
tree and every xing-pc connector root:

    D:\\keys\\limen\\limen-issuer.seed.dpapi   (override: LIMEN_ISSUER_SEED)

  python scripts/issue_license.py --init
      generate a new seed straight into the DPAPI blob and write the new
      public key into limen/licensing/_issuer_key.py.

  python scripts/issue_license.py --backup E:\\limen-issuer.bak
      write a passphrase-encrypted offline backup (scrypt + AES-256-GCM).

  python scripts/issue_license.py --restore E:\\limen-issuer.bak
      rebuild the DPAPI blob from a backup (new Windows install).

  python scripts/issue_license.py --selftest
      check the seed matches the shipped public key and a fresh key verifies.

  python scripts/issue_license.py --sub buyer@example.com --tier team --days 365
      print a signed license key.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import getpass
import os
import pathlib
import sys
import time
import uuid

REPO = pathlib.Path(__file__).resolve().parent.parent
# Always use this repo's limen, never an installed limen-compiler wheel.
sys.path.insert(0, str(REPO))

from limen.licensing.license import TIER_FEATURES, decode_key, encode_key  # noqa: E402
from limen.security import pqc  # noqa: E402

DEFAULT_SEED = pathlib.Path(
    os.environ.get("LIMEN_ISSUER_SEED", r"D:\keys\limen\limen-issuer.seed.dpapi")
)
ISSUER_KEY_FILE = REPO / "limen" / "licensing" / "_issuer_key.py"

_ENTROPY_LABEL = b"LIMEN-issuer-seed-v1\x00"
_BACKUP_MAGIC = b"LIMENBK1"
_MIN_PASSPHRASE = 12


# ---------------------------------------------------------------- DPAPI


def _dpapi(data: bytes, entropy: bytes, protect: bool) -> bytes:
    """CryptProtectData / CryptUnprotectData via ctypes (no extra deps)."""
    if sys.platform != "win32":
        raise SystemExit("DPAPI is Windows-only: run this on xing-pc.")
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    def blob(buf: bytes) -> tuple[Blob, object]:
        arr = (ctypes.c_ubyte * len(buf)).from_buffer_copy(buf)
        return Blob(len(buf), ctypes.cast(arr, ctypes.POINTER(ctypes.c_ubyte))), arr

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    fn.argtypes = [
        ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.POINTER(Blob),
        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob),
    ]
    fn.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]

    blob_in, _keep_in = blob(data)
    blob_ent, _keep_ent = blob(entropy)
    blob_out = Blob()
    CRYPTPROTECT_UI_FORBIDDEN = 0x1
    ok = fn(ctypes.byref(blob_in), None, ctypes.byref(blob_ent), None, None,
            CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(blob_out))
    if not ok:
        err = ctypes.get_last_error()
        if not protect:
            raise SystemExit(f"could not unwrap the seed (wrong passphrase or Windows user?) [error {err}]")
        raise SystemExit(f"DPAPI protect failed [error {err}]")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(blob_out.pbData, ctypes.c_void_p))


# ---------------------------------------------------------------- helpers


def _passphrase(prompt: str, confirm: bool) -> bytes:
    pw = getpass.getpass(prompt)
    if confirm:
        if len(pw) < _MIN_PASSPHRASE:
            raise SystemExit(f"passphrase must be at least {_MIN_PASSPHRASE} characters")
        if getpass.getpass("Repeat: ") != pw:
            raise SystemExit("passphrases did not match")
    return pw.encode("utf-8")


def _entropy(pw: bytes) -> bytes:
    return _ENTROPY_LABEL + pw


def _write_new(path: pathlib.Path, data: bytes) -> None:
    """Create *path* exclusively; never overwrite key material."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError:
        raise SystemExit(f"{path} already exists; refusing to overwrite") from None
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def _check_outside_tree(path: pathlib.Path) -> None:
    tree = REPO.parent.resolve()
    if path.resolve().is_relative_to(tree):
        raise SystemExit(f"{path} is inside {tree}; keep the issuer seed outside the ATRIUM tree")


def _pub_b64(key: pqc.PrivateKey) -> str:
    return base64.b64encode(pqc.public_key_to_bytes(key.public_key())).decode("ascii")


def _write_issuer_key(pub_b64: str) -> None:
    chunks = "\n".join(f'    "{pub_b64[i:i + 76]}"' for i in range(0, len(pub_b64), 76))
    ISSUER_KEY_FILE.write_text(
        "# Copyright (C) 2026 xingxerx / CGX\n"
        "#\n"
        "# Licensed under the Elastic License 2.0 (ELv2); you may not use this file\n"
        "# except in compliance with the License. See the LICENSE file in the\n"
        "# repository root for the full terms.\n"
        "\n"
        '"""LIMEN license issuer PUBLIC key (ML-DSA-65, base64 raw bytes).\n'
        "\n"
        "Public by design: it can only verify keys, never issue them. The matching\n"
        "private seed is DPAPI-wrapped outside the repo and never committed or shipped.\n"
        f"Generated {datetime.date.today().isoformat()} by scripts/issue_license.py --init.\n"
        '"""\n'
        "\n"
        f"ISSUER_PUBLIC_KEY_B64 = (\n{chunks}\n)\n",
        encoding="utf-8",
    )


def _load_seed(path: pathlib.Path) -> pqc.PrivateKey:
    if not path.exists():
        raise SystemExit(f"no issuer seed at {path}; run --init or --restore first")
    seed = _dpapi(path.read_bytes(), _entropy(_passphrase("Issuer passphrase: ", False)), protect=False)
    return pqc.private_key_from_bytes(seed)


def _backup_key(pw: bytes, salt: bytes) -> bytes:
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    return Scrypt(salt=salt, length=32, n=2**17, r=8, p=1).derive(pw)


# ---------------------------------------------------------------- commands


def cmd_init(seed_path: pathlib.Path) -> None:
    _check_outside_tree(seed_path)
    if seed_path.exists():
        raise SystemExit(f"{seed_path} already exists; refusing to overwrite")
    pw = _passphrase("New issuer passphrase: ", True)
    key = pqc.generate_signing_key()
    _write_new(seed_path, _dpapi(pqc.private_key_to_bytes(key), _entropy(pw), protect=True))
    pub = _pub_b64(key)
    _write_issuer_key(pub)
    print(f"seed written (DPAPI-wrapped): {seed_path}")
    print(f"public key written: {ISSUER_KEY_FILE}")
    print("next: --backup to an offline drive, then --selftest")


def cmd_backup(seed_path: pathlib.Path, out: pathlib.Path) -> None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = _load_seed(seed_path)
    pw = _passphrase("Backup passphrase: ", True)
    salt, nonce = os.urandom(16), os.urandom(12)
    ct = AESGCM(_backup_key(pw, salt)).encrypt(nonce, pqc.private_key_to_bytes(key), _BACKUP_MAGIC)
    _write_new(out, _BACKUP_MAGIC + salt + nonce + ct)
    print(f"encrypted backup written: {out} (public key {_pub_b64(key)[:16]}...)")


def cmd_restore(seed_path: pathlib.Path, src: pathlib.Path) -> None:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    _check_outside_tree(seed_path)
    raw = src.read_bytes()
    if raw[:8] != _BACKUP_MAGIC:
        raise SystemExit(f"{src} is not a LIMEN issuer backup")
    salt, nonce, ct = raw[8:24], raw[24:36], raw[36:]
    try:
        seed = AESGCM(_backup_key(_passphrase("Backup passphrase: ", False), salt)).decrypt(nonce, ct, _BACKUP_MAGIC)
    except InvalidTag:
        raise SystemExit("wrong backup passphrase or damaged backup") from None
    key = pqc.private_key_from_bytes(seed)
    pw = _passphrase("New issuer passphrase: ", True)
    _write_new(seed_path, _dpapi(seed, _entropy(pw), protect=True))
    print(f"seed restored (DPAPI-wrapped): {seed_path} (public key {_pub_b64(key)[:16]}...)")


def cmd_selftest(seed_path: pathlib.Path) -> None:
    from limen.licensing._issuer_key import ISSUER_PUBLIC_KEY_B64

    key = _load_seed(seed_path)
    if _pub_b64(key) != ISSUER_PUBLIC_KEY_B64:
        raise SystemExit("FAIL: seed does not match limen/licensing/_issuer_key.py")
    now = int(time.time())
    probe = {"lid": "selftest", "sub": "selftest", "tier": "team", "features": [], "iat": now, "exp": now + 3600}
    lic = decode_key(encode_key(key, probe))
    if not lic.valid:
        raise SystemExit(f"FAIL: probe key rejected ({lic.reason})")
    print("OK: seed matches the shipped public key and a fresh key verifies")


def cmd_issue(seed_path: pathlib.Path, sub: str, tier: str, days: int) -> None:
    key = _load_seed(seed_path)
    now = int(time.time())
    payload = {
        "lid": uuid.uuid4().hex,
        "sub": sub,
        "tier": tier,
        "features": [],
        "iat": now,
        "exp": now + days * 86400 if days else 0,
    }
    print(encode_key(key, payload))


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--init", action="store_true")
    mode.add_argument("--backup", type=pathlib.Path, metavar="FILE")
    mode.add_argument("--restore", type=pathlib.Path, metavar="FILE")
    mode.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", type=pathlib.Path, default=DEFAULT_SEED)
    ap.add_argument("--sub")
    ap.add_argument("--tier", choices=sorted(t for t in TIER_FEATURES if t != "free"))
    ap.add_argument("--days", type=int, default=365, help="0 = perpetual")
    a = ap.parse_args()

    if a.init:
        cmd_init(a.seed)
    elif a.backup:
        cmd_backup(a.seed, a.backup)
    elif a.restore:
        cmd_restore(a.seed, a.restore)
    elif a.selftest:
        cmd_selftest(a.seed)
    elif a.sub and a.tier:
        cmd_issue(a.seed, a.sub, a.tier, a.days)
    else:
        raise SystemExit("--sub and --tier are required (or use --init/--backup/--restore/--selftest)")


if __name__ == "__main__":
    main()
