#!/usr/bin/env python3
# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Self-improvement loop CI verdict (harness-roadmap/03).

Loads a routing-policy proposal claim file, replays it against the
ledger built from real results/ certificates, records the verdict in the
append-only proposal ledger, writes a plain-English report, and exits
non-zero on rejection so CI blocks the merge -- "agents propose, the
harness disposes" (ATRIUM AGENTS.md section 5).

Rejected proposals are witnessed, not punished: the report and verdict
are written to policy_proposals/failed/ and recorded in the ledger
exactly like an accepted one, never deleted (ATRIUM AGENTS.md section 4,
"no punishment realms").

Usage:
    python scripts/run_proposal.py <claim_file.json>
        [--results-dir DIR] [--ledger-dir DIR] [--proposals-dir DIR]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from limen.router.memory import RouterMemory
from limen.router.proposal import Proposal, evaluate_proposal
from limen.router.report import build_proposal_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("claim_file", type=pathlib.Path)
    parser.add_argument("--results-dir", type=pathlib.Path, default=pathlib.Path("results"))
    parser.add_argument("--ledger-dir", type=pathlib.Path, default=pathlib.Path("policy_proposals/ledger"))
    parser.add_argument("--proposals-dir", type=pathlib.Path, default=pathlib.Path("policy_proposals"))
    args = parser.parse_args()

    proposal = Proposal.from_claim_file(args.claim_file)
    print(f"proposal {proposal.id!r}: {proposal.title}")
    print(f"  policy: {proposal.policy_name} -> {proposal.proposed_value!r}")

    args.ledger_dir.mkdir(parents=True, exist_ok=True)
    with RouterMemory(args.ledger_dir / "router_memory.sqlite3") as mem:
        added = mem.ingest_results(args.results_dir)
        print(f"  ledger baseline: ingested {added} new sample(s) from {args.results_dir}")

        verdict = evaluate_proposal(proposal, mem)
        mem.record_proposal(
            proposal.id,
            proposal.policy_name,
            verdict.accepted,
            verdict.to_dict(),
        )

        outcome_dir = args.proposals_dir / ("accepted" if verdict.accepted else "failed")
        outcome_dir.mkdir(parents=True, exist_ok=True)
        (outcome_dir / f"{proposal.id}.json").write_text(
            __import__("json").dumps(verdict.to_dict(), indent=2, sort_keys=True)
        )
        report = build_proposal_report(verdict)
        (outcome_dir / f"{proposal.id}.md").write_text(report)

        print()
        print(report)

    if not verdict.accepted:
        print(f"VERDICT: REJECTED -- {verdict.reason}", file=sys.stderr)
        return 1
    print(f"VERDICT: ACCEPTED -- {verdict.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
