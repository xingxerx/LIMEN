#!/usr/bin/env python3
# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Proposer for the self-improvement loop (harness-roadmap/03).

Reads the certificate ledger, sweeps each gated routing-policy knob, and
reports whether the visible scenarios support a change. With ``--write`` a
qualifying claim is written to ``policy_proposals/pending/`` for the CI
verdict job (``scripts/run_proposal.py``) to judge, held-out set included.
"Agents propose; the harness disposes": this script never accepts anything.

The committed ledger (``policy_proposals/ledger/router_memory.sqlite3``) is
never modified: it is copied to a scratch directory, ``results/`` is ingested
into the copy, and the search runs against that.

Usage:
    python scripts/propose.py [--policy NAME] [--results-dir DIR]
        [--ledger-dir DIR] [--proposals-dir DIR] [--write]
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from limen.router.memory import DEFAULT_FILENAME, RouterMemory
from limen.router.proposal import GATED_POLICIES
from limen.router.proposer import propose


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=sorted(GATED_POLICIES), default=None,
                        help="one gated policy (default: all of them)")
    parser.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--ledger-dir", type=pathlib.Path, default=pathlib.Path("policy_proposals/ledger"))
    parser.add_argument("--proposals-dir", type=pathlib.Path, default=pathlib.Path("policy_proposals"))
    parser.add_argument("--write", action="store_true",
                        help="write a qualifying claim to <proposals-dir>/pending/ (default: report only)")
    args = parser.parse_args()

    policies = [args.policy] if args.policy else sorted(GATED_POLICIES)

    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = pathlib.Path(scratch)
        committed = args.ledger_dir / "router_memory.sqlite3"
        if committed.exists():
            shutil.copy2(committed, scratch_dir / DEFAULT_FILENAME)
        with RouterMemory.in_results_dir(scratch_dir) as mem:
            added = mem.ingest_results(args.results_dir)
            print(f"ledger: ingested {added} new sample(s) from {args.results_dir} (scratch copy)")

            for policy in policies:
                search, proposal, path = propose(
                    mem, args.proposals_dir, policy_name=policy, write=args.write
                )
                print(f"\n{policy} (live {search.live_value:g})")
                print(f"  swept {search.tested} values; {search.passing} passed the non-regression gates")
                if proposal is None:
                    print("  nothing to propose: no value passing the gates strictly improves cost or "
                          "physical-error exposure (fixed point for the visible scenarios)")
                    continue
                cand = search.candidate
                print(f"  candidate {cand.value:g}: cost delta {cand.cost_delta:+.4g}, "
                      f"exposure delta {cand.exposure_delta:+.4g}")
                if path is not None:
                    print(f"  wrote {path}")
                elif args.write:
                    print("  not written: a claim with this id already exists (pending/accepted/failed)")
                else:
                    print(f"  dry run; claim id {proposal.id!r} (re-run with --write to file it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
