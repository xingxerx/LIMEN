# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""``limend``: daemonized routing loop, DUCTEI's first real producer.

See daemon.py for the watch/route/certify loop and spool.py for the
spool directory contract shared with ductei-limen-relay.
"""

from limen.limend.daemon import process_one, run_forever
from limen.limend.spool import ensure_spool_dirs

__all__ = ["run_forever", "process_one", "ensure_spool_dirs"]
