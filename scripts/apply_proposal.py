#!/usr/bin/env python3
# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Apply an accepted routing-policy proposal to the source (harness-roadmap/03).

Takes a proposal id, requires an ``accepted`` verdict under
``policy_proposals/accepted/``, and edits the one matching constant in
``limen/router/budget_router.py``. This is also how a rollback becomes
permanent: when limend rolls a landed change back it writes a ``-revert-``
claim to ``policy_proposals/pending/``; once CI accepts that claim, run this
on its id.

It edits the working tree only. Review the diff and commit through the
normal PR gate yourself.

Usage:
    python scripts/apply_proposal.py <proposal-id> [--dry-run]
        [--proposals-dir DIR] [--source FILE]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from limen.router.policy_source import PolicyApplyError, apply_accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("proposal_id")
    parser.add_argument("--proposals-dir", type=pathlib.Path, default=ROOT / "policy_proposals")
    parser.add_argument("--source", type=pathlib.Path, default=ROOT / "limen/router/budget_router.py")
    parser.add_argument("--dry-run", action="store_true", help="report the edit without writing it")
    args = parser.parse_args()

    try:
        result = apply_accepted(args.proposal_id, args.proposals_dir, args.source, dry_run=args.dry_run)
    except PolicyApplyError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    if not result.changed:
        print(f"already applied: {result.policy_name} is already {result.new_value!r} in {result.source}")
        return 0
    verb = "would set" if args.dry_run else "set"
    print(f"{verb} {result.policy_name}: {result.old_value!r} -> {result.new_value!r} in {result.source}")
    if not args.dry_run:
        print("review the diff, update the comment above the constant if it quotes the old value, "
              "then commit through the normal PR gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
