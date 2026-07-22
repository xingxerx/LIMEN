# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Spool directory layout shared by ``limend`` and (on the DUCTEI side)
``ductei-limen-relay``.

::

    spool/
      pending/   job requests waiting to run  -- {job_id}.json
      done/      original request, moved here after limend finishes it
      certs/     {job_id}.json written by limend, matches Rust
                 CertSummary exactly -- tailed by the relay
      failed/    anything that errored at any stage, with an "error" field

Plain directories, not a queue library, so the whole pipeline stays
inspectable with ``ls``/``cat`` at any point and survives a daemon
restart with no data loss -- whatever is still in ``pending/`` just gets
picked up again.
"""

from __future__ import annotations

import pathlib

PENDING = "pending"
DONE = "done"
CERTS = "certs"
FAILED = "failed"

SUBDIRS = (PENDING, DONE, CERTS, FAILED)


def ensure_spool_dirs(root: pathlib.Path) -> dict[str, pathlib.Path]:
    """Create (if missing) and return the four spool subdirectories."""
    root = pathlib.Path(root)
    dirs = {name: root / name for name in SUBDIRS}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
