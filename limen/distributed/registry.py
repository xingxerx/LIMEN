# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Peer-aware node registry for LIMEN multi-node deployments.

Wraps the existing single-process DeltaModelRegistry rather than replacing
it: local devices resolve exactly as they do today. Device IDs not known
locally are looked up against registered peers and cached with a TTL.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from limen.analog.delta_model import DeltaModelRegistry, HardwareDeltaModel
from limen.distributed.node import NodeInfo

logger = logging.getLogger("limen.distributed")

DEFAULT_PEER_TTL_SECONDS = 30.0
DEFAULT_CACHE_TTL_SECONDS = 60.0


@dataclass
class _PeerEntry:
    info: NodeInfo
    last_seen: float


@dataclass
class _CacheEntry:
    model: HardwareDeltaModel
    cached_at: float


class NodeRegistry:
    """Tracks peer LIMEN nodes and caches their calibration data.

    Args:
        local_delta_registry: The process's existing DeltaModelRegistry for
            devices attached directly to this node.
        peer_ttl: Seconds of silence after which a peer is considered stale
            and evicted by evict_stale().
        cache_ttl: Seconds a remotely-fetched HardwareDeltaModel stays valid
            in the local cache before it must be re-synced.
    """

    def __init__(
        self,
        local_delta_registry: DeltaModelRegistry | None = None,
        peer_ttl: float = DEFAULT_PEER_TTL_SECONDS,
        cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self.local = local_delta_registry or DeltaModelRegistry()
        self.peer_ttl = peer_ttl
        self.cache_ttl = cache_ttl
        self._peers: dict[str, _PeerEntry] = {}
        self._device_cache: dict[str, _CacheEntry] = {}

    # -- peer table -----------------------------------------------------

    def add_peer(self, info: NodeInfo) -> None:
        """Register or refresh a peer node."""
        self._peers[info.node_id] = _PeerEntry(info=info, last_seen=time.monotonic())
        logger.info("registered peer %s at %s", info.node_id, info.address())

    def heartbeat(self, node_id: str) -> bool:
        """Refresh a peer's last-seen time. Returns False if unknown."""
        entry = self._peers.get(node_id)
        if entry is None:
            return False
        entry.last_seen = time.monotonic()
        return True

    def evict_stale(self, now: float | None = None) -> list[str]:
        """Remove peers not heard from within peer_ttl. Returns evicted IDs."""
        now = now if now is not None else time.monotonic()
        stale = [nid for nid, e in self._peers.items() if now - e.last_seen > self.peer_ttl]
        for nid in stale:
            del self._peers[nid]
            logger.info("evicted stale peer %s", nid)
        return stale

    def list_peers(self) -> list[NodeInfo]:
        """Return known peers, sorted by node_id."""
        return [e.info for _, e in sorted(self._peers.items())]

    def peer_for_device(self, device_id: str) -> NodeInfo | None:
        """Return the first known peer that serves device_id, if any."""
        for entry in self._peers.values():
            if device_id in entry.info.device_ids:
                return entry.info
        return None

    # -- calibration cache ------------------------------------------------

    def cache_calibration(self, model: HardwareDeltaModel) -> None:
        """Store a remotely-fetched HardwareDeltaModel with a fresh timestamp."""
        self._device_cache[model.device_id] = _CacheEntry(model=model, cached_at=time.monotonic())

    def cached_calibration(self, device_id: str) -> HardwareDeltaModel | None:
        """Return a cached remote model if present and not expired."""
        entry = self._device_cache.get(device_id)
        if entry is None:
            return None
        if time.monotonic() - entry.cached_at > self.cache_ttl:
            del self._device_cache[device_id]
            return None
        return entry.model

    def resolve(self, device_id: str) -> HardwareDeltaModel | None:
        """Resolve a device's calibration model: local store, then cache.

        Does not perform network I/O itself — callers (e.g. the gRPC
        client) are responsible for populating the cache via
        cache_calibration() after a SyncCalibration round trip.
        """
        local = self.local.get(device_id)
        if local is not None:
            return local
        return self.cached_calibration(device_id)
