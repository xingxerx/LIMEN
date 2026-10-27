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


def apply_accepted(
    proposal_id: str,
    proposals_dir: pathlib.Path | str,
    source: pathlib.Path | str,
    *,
    dry_run: bool = False,
) -> ApplyResult:
    """Apply the accepted proposal *proposal_id* to *source*.

    Raises :class:`PolicyApplyError` if there is no accepted verdict for it,
    its policy is not gated, or the source value is stale. A source that
    already holds the proposed value is a no-op (``changed=False``)."""
    verdict_path = pathlib.Path(proposals_dir) / "accepted" / f"{proposal_id}.json"
    if not verdict_path.is_file():
        raise PolicyApplyError(
            f"no accepted verdict at {verdict_path}; only a proposal the CI verdict accepted can be applied"
        )
    try:
        verdict = Verdict.from_dict(json.loads(verdict_path.read_text()))
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
