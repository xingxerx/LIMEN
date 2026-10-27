# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Turn an *accepted* routing-policy proposal into a one-line source edit.

The loop's gate accepts or rejects a claim, but until now landing an
accepted change (or making a rollback permanent) meant a human hand-editing
``budget_router.py``; that is how the 2026-07-24 proposal ended up accepted
and witnessed but not applied for weeks. This module is that edit, with
guard rails:

  * only a proposal with an ``accepted`` verdict on disk can be applied, so
    the harness, not the caller, decides what may change;
  * that verdict file must be byte-for-byte what the append-only proposal
    ledger recorded when the harness judged it (its most recent row for the
    proposal id, and that row must be an acceptance), so a verdict edited or
    dropped into ``accepted/`` by hand is refused. This catches casual edits,
    not a determined attacker who can also rewrite the ledger; the protection
    against that is review of the commit (branch protection, signed commits);
  * only a constant in :data:`~limen.router.proposal.GATED_POLICIES`;
  * the source must still hold the value the verdict was evaluated against
    (``previous_value``), otherwise the edit is refused as stale;
  * exactly one matching assignment, and nothing else in the file changes.

It edits the working tree only. Committing the change (through the normal
PR gate) stays a human step.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import re
import shutil
import sqlite3
import tempfile

from limen.router.memory import _canonical_json
from limen.router.proposal import GATED_POLICIES, Verdict


class PolicyApplyError(ValueError):
    """The edit was refused; the source was not touched."""


@dataclasses.dataclass(frozen=True)
class ApplyResult:
    proposal_id: str
    policy_name: str
    old_value: float
    new_value: float
    changed: bool  # False when the source already held the proposed value
    source: pathlib.Path


def _assignment(name: str) -> re.Pattern[str]:
    # NAME = 10.25   /   NAME: float = 10.25   (trailing comment preserved)
    return re.compile(rf"^({re.escape(name)}\s*(?::[^=\n]+)?=\s*)([^\s#]+)([^\n]*)$", re.MULTILINE)


def set_constant(
    text: str, name: str, new_value: float, *, expected_old: float | None = None
) -> tuple[str, float]:
    """Return ``(new_text, old_value)`` with the one top-level assignment of
    *name* set to *new_value*. Raises :class:`PolicyApplyError` unless there
    is exactly one assignment holding a plain number, and, when *expected_old*
    is given, that number equals it."""
    pattern = _assignment(name)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise PolicyApplyError(f"expected exactly one assignment of {name}, found {len(matches)}")
    try:
        old = float(matches[0].group(2).replace("_", ""))
    except ValueError as exc:
        raise PolicyApplyError(f"{name} is not a plain number in source: {matches[0].group(2)!r}") from exc
    if expected_old is not None and old != expected_old:
        raise PolicyApplyError(
            f"stale: source has {name} = {old!r}, but the verdict was evaluated against {expected_old!r}"
        )
    new_text = pattern.sub(lambda m: f"{m.group(1)}{float(new_value)!r}{m.group(3)}", text, count=1)
    return new_text, old


def _check_witnessed(ledger: pathlib.Path, proposal_id: str, verdict_doc: dict) -> None:
    """Require the verdict file to equal the ledger's latest row for the id,
    and that row to be an acceptance.

    The ledger is read from a scratch copy (main file plus any -wal), so the
    check never writes to the repository: opening a WAL-mode database in place,
    even read-only, creates -wal/-shm files next to it."""
    if not ledger.is_file():
        raise PolicyApplyError(
            f"no proposal ledger at {ledger}; cannot confirm the harness witnessed {proposal_id}"
        )
    try:
        with tempfile.TemporaryDirectory() as scratch:
            copy = pathlib.Path(scratch) / "ledger.sqlite3"
            shutil.copy2(ledger, copy)
            wal = ledger.with_name(ledger.name + "-wal")
            if wal.is_file():
                shutil.copy2(wal, copy.with_name(copy.name + "-wal"))
            conn = sqlite3.connect(copy)
            try:
                row = conn.execute(
                    "SELECT accepted, verdict FROM policy_proposals WHERE proposal_id = ? "
                    "ORDER BY seq DESC LIMIT 1",
                    (proposal_id,),
                ).fetchone()
            finally:
                conn.close()
    except (sqlite3.Error, OSError) as exc:
        raise PolicyApplyError(f"cannot read proposal ledger {ledger}: {exc}") from exc
    if row is None:
        raise PolicyApplyError(f"{proposal_id} has no row in {ledger}; the harness never witnessed it")
    accepted, recorded = row
    if not accepted:
        raise PolicyApplyError(f"the ledger's latest verdict for {proposal_id} is not an acceptance")
    if _canonical_json(json.loads(recorded)) != _canonical_json(verdict_doc):
        raise PolicyApplyError(
            f"the verdict file for {proposal_id} differs from what the ledger recorded; "
            "re-run scripts/run_proposal.py on its claim instead of editing it"
        )


def apply_accepted(
    proposal_id: str,
    proposals_dir: pathlib.Path | str,
    source: pathlib.Path | str,
    *,
    dry_run: bool = False,
    ledger: pathlib.Path | str | None = None,
) -> ApplyResult:
    """Apply the accepted proposal *proposal_id* to *source*.

    Raises :class:`PolicyApplyError` if there is no accepted verdict for it,
    the ledger (default ``<proposals_dir>/ledger/router_memory.sqlite3``) does
    not witness exactly that verdict, its policy is not gated, or the source
    value is stale. A source that already holds the proposed value is a no-op
    (``changed=False``)."""
    verdict_path = pathlib.Path(proposals_dir) / "accepted" / f"{proposal_id}.json"
    if not verdict_path.is_file():
        raise PolicyApplyError(
            f"no accepted verdict at {verdict_path}; only a proposal the CI verdict accepted can be applied"
        )
    try:
        verdict_doc = json.loads(verdict_path.read_text())
        verdict = Verdict.from_dict(verdict_doc)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise PolicyApplyError(f"unreadable verdict {verdict_path}: {exc}") from exc
    if not verdict.accepted:
        raise PolicyApplyError(f"verdict for {proposal_id} is not accepted")
    prop = verdict.proposal
    if prop.policy_name not in GATED_POLICIES:
        raise PolicyApplyError(f"{prop.policy_name!r} is not a gated routing-policy tunable")
    if verdict.previous_value is None:
        raise PolicyApplyError(
            f"verdict for {proposal_id} records no previous_value, so it cannot be checked "
            "against the source it was evaluated for; re-run scripts/run_proposal.py on its claim"
        )

    ledger_path = (
        pathlib.Path(ledger)
        if ledger is not None
        else pathlib.Path(proposals_dir) / "ledger" / "router_memory.sqlite3"
    )
    _check_witnessed(ledger_path, proposal_id, verdict_doc)

    path = pathlib.Path(source)
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")

    # Already applied? Check before the staleness test, which would refuse it.
    _, current = set_constant(text, prop.policy_name, prop.proposed_value)
    if current == prop.proposed_value:
        return ApplyResult(proposal_id, prop.policy_name, current, prop.proposed_value, False, path)

    new_text, old = set_constant(
        text, prop.policy_name, prop.proposed_value, expected_old=verdict.previous_value
    )
    if not dry_run:
        path.write_bytes((new_text.replace("\n", "\r\n") if crlf else new_text).encode("utf-8"))
    return ApplyResult(proposal_id, prop.policy_name, old, prop.proposed_value, True, path)
