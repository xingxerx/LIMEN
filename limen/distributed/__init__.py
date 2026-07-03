# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Multi-node coordination layer: node discovery and calibration sync.

Requires the `distributed` extra (grpcio, grpcio-tools, protobuf):

    pip install -e .[distributed]
"""

from __future__ import annotations

from limen.distributed.config import NodeConfig
from limen.distributed.node import NodeInfo
from limen.distributed.registry import NodeRegistry

__all__ = ["NodeConfig", "NodeInfo", "NodeRegistry"]
