# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""CLI entry point: ``python -m limen.limend <spool_dir>``."""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from limen.limend.daemon import run_forever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="limend", description=__doc__)
    parser.add_argument("spool_dir", type=pathlib.Path, help="spool root (contains pending/done/certs/failed)")
    parser.add_argument("--results-dir", type=pathlib.Path, default=None,
                         help="results dir for run history/calibration snapshots and QPU job-state persistence")
    parser.add_argument("--memory", type=pathlib.Path, default=None,
                         help="path to a RouterMemory sqlite3 ledger (created if missing)")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--memory-ceiling-mb", type=float, default=None,
                         help="preemptively exit (for supervisor restart) once RSS crosses this")
    parser.add_argument("--once", action="store_true", help="process whatever is pending once, then exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_forever(
        args.spool_dir,
        poll_interval=args.poll_interval,
        results_dir=args.results_dir,
        memory=args.memory,
        memory_ceiling_mb=args.memory_ceiling_mb,
        once=args.once,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
