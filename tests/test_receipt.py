"""Tests for limen.receipt: signed run receipts, the verify and keygen CLI,
and the license gate (signing is paid, checking is free)."""

import copy
import json
import pathlib
from unittest import mock

import pytest

pqc = pytest.importorskip("limen.security.pqc", exc_type=ImportError)

from limen.licensing.license import FREE, PRO, License  # noqa: E402
from limen.receipt import (  # noqa: E402
    KEY_FILE,
    PUB_FILE,
    SigningRouterMemory,
    check_against_ledger,
    main,
    run_signed,
    verify_receipt,
)
from limen.router import DEFAULT_FLEET, RouteRequest  # noqa: E402

_LICENSE_TARGET = "limen.licensing.license.current_license"


def _pro():
    return mock.patch(_LICENSE_TARGET, return_value=License(tier=PRO))


def _free():
    return mock.patch(_LICENSE_TARGET, return_value=License(tier=FREE))


def _cycle_maxcut(n):
    qubo = {}
    for i in range(n):
        a, b = f"x{i}", f"x{(i + 1) % n}"
        qubo[(a, b)] = qubo.get((a, b), 0.0) + 2.0
        qubo[(a, a)] = qubo.get((a, a), 0.0) - 1.0
        qubo[(b, b)] = qubo.get((b, b), 0.0) - 1.0
    return qubo


@pytest.fixture()
def signed(tmp_path):
    """One signed receipt on disk plus its key and ledger."""
    key = pqc.generate_signing_key()
    mem = SigningRouterMemory(tmp_path / "m.sqlite3", signing_key=key)
    try:
        with _pro():
            mem.append_certificate({"energy": -3.0, "is_optimal": True}, backend="ibm_fez")
        path = mem.last_receipt_path
    finally:
        mem.close()
    assert path is not None
    return key, path, tmp_path / "m.sqlite3"


def test_roundtrip_verifies(signed):
    key, path, _ = signed
    receipt = json.loads(path.read_text(encoding="utf-8"))
    check = verify_receipt(receipt)
    assert check.ok and check.signature_valid and check.payload_hash_valid
    assert not check.key_pinned
    pinned = verify_receipt(receipt, key.public_key())
    assert pinned.ok and pinned.key_pinned


def test_edited_payload_is_rejected(signed):
    _, path, _ = signed
    receipt = json.loads(path.read_text(encoding="utf-8"))
    forged = copy.deepcopy(receipt)
    forged["payload"]["energy"] = -99.0
    check = verify_receipt(forged)
    assert not check.ok
    assert not check.payload_hash_valid


def test_edited_chain_head_is_rejected(signed):
    _, path, _ = signed
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["chain_sha256"] = "0" * 64
    check = verify_receipt(receipt)
    assert not check.ok
    assert not check.signature_valid


def test_wrong_pinned_key_is_rejected(signed):
    _, path, _ = signed
    receipt = json.loads(path.read_text(encoding="utf-8"))
    other = pqc.generate_signing_key()
    check = verify_receipt(receipt, other.public_key())
    assert not check.ok
    assert any("different key" in r for r in check.reasons)


def test_garbage_input_never_raises():
    assert not verify_receipt("nope").ok
    assert not verify_receipt({"format": "limen-receipt/1"}).ok
    assert not verify_receipt({"format": "limen-receipt/1", "payload": 1, "payload_sha256": 1,
                               "chain_sha256": 1, "signature": 1, "public_key": 1}).ok


def test_free_tier_chains_unsigned_and_writes_no_receipt(tmp_path):
    key = pqc.generate_signing_key()
    mem = SigningRouterMemory(tmp_path / "m.sqlite3", signing_key=key)
    try:
        with _free():
            entry = mem.append_certificate({"e": 1}, backend="b")
        assert entry.signature is None
        assert mem.last_receipt_path is None
        assert mem.signing_skipped and "signed_certificates" in mem.signing_skipped
        assert mem.verify_ledger()
    finally:
        mem.close()


def test_ledger_cross_check(signed):
    _, path, ledger = signed
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert check_against_ledger(receipt, ledger) == []
    bad = dict(receipt, chain_sha256="f" * 64)
    assert check_against_ledger(bad, ledger)
    assert check_against_ledger(receipt, ledger.with_name("missing.sqlite3"))


def test_run_signed_end_to_end(tmp_path):
    key = pqc.generate_signing_key()
    pqc.save_private_key(tmp_path / KEY_FILE, key)  # picked up from results_dir
    request = RouteRequest(_cycle_maxcut(4), fidelity_target=0.9, credit_budget=0.0)
    with _pro():
        cert = run_signed(request, results_dir=tmp_path, fleet=DEFAULT_FLEET)
    assert cert.is_optimal
    receipt_path = pathlib.Path(cert.metadata["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert verify_receipt(receipt, key.public_key()).ok
    assert check_against_ledger(receipt, tmp_path / "router_memory.sqlite3") == []


def test_run_signed_without_key_still_runs_and_says_so(tmp_path):
    request = RouteRequest(_cycle_maxcut(4), fidelity_target=0.9, credit_budget=0.0)
    with mock.patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop("LIMEN_SIGNING_KEY", None)
        cert = run_signed(request, results_dir=tmp_path, fleet=DEFAULT_FLEET)
    assert cert.is_optimal
    assert "receipt_path" not in cert.metadata
    assert any("limen keygen" in n for n in cert.notes)


def test_run_signed_rejects_memory_kwarg(tmp_path):
    request = RouteRequest(_cycle_maxcut(4), fidelity_target=0.9, credit_budget=0.0)
    with pytest.raises(TypeError):
        run_signed(request, results_dir=tmp_path, memory=None)


def test_cli_verify_and_exit_codes(signed, tmp_path, capsys):
    key, path, ledger = signed
    pub = tmp_path / "pin.pub"
    pqc.save_public_key(pub, key.public_key())
    assert main(["verify", str(path), "--pubkey", str(pub), "--ledger", str(ledger)]) == 0
    assert "VALID" in capsys.readouterr().out

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["payload"]["energy"] = 0.0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(tampered), encoding="utf-8")
    assert main(["verify", str(bad)]) == 1
    assert "INVALID" in capsys.readouterr().out

    assert main(["verify", str(tmp_path / "nope.json")]) == 2


def test_cli_keygen_writes_keys_and_refuses_overwrite(tmp_path):
    folder = tmp_path / "keys"
    assert main(["keygen", str(folder)]) == 0
    assert (folder / KEY_FILE).is_file() and (folder / PUB_FILE).is_file()
    assert pqc.load_private_key(folder / KEY_FILE).public_key() is not None
    assert main(["keygen", str(folder)]) == 2
