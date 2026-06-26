# Copyright 2026 LIMEN Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Distributed QUBO partitioning: split, dispatch, and merge.

Splits a LogicalGraph into balanced sub-graphs, compiles each one
independently (locally or dispatched to a peer node over the
Coordination service's CompilePartition RPC), and merges the resulting
PhysicalEncodings back into a single encoding equivalent to compiling
the original graph in one shot.

The partitioning strategy is a deterministic lexicographic split, not a
min-cut or balanced-weight algorithm - naive by design, matching the
existing lexicographic compiler's own philosophy (limen/core/compiler.py).
A cross-partition interaction is owned by the lower-indexed partition
exactly once; it is never duplicated or dropped.

Compiling each partition against its own namespaced hardware-qubit
labels (e.g. "p0:q3" vs "p1:q3") is what makes merging the resulting
QUBO dicts safe: compile_lexicographic always allocates from "q0",
"q1", ... regardless of partition, so without namespacing two
partitions would silently collide on the same physical labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from limen.core.compiler import PhysicalEncoding, compile_lexicographic
from limen.core.ir import Interaction, LogicalGraph, Variable


@dataclass
class GraphPartition:
    """One slice of a partitioned LogicalGraph.

    Attributes:
        partition_id: Index of this partition (0-based).
        graph: Local sub-graph: local_vars + boundary_refs as Variables,
            plus the interactions this partition owns.
        local_vars: Names of variables whose canonical home is this partition.
        boundary_refs: Names of foreign variables referenced by owned
            cross-partition interactions.
    """

    partition_id: int
    graph: LogicalGraph
    local_vars: set[str] = field(default_factory=set)
    boundary_refs: set[str] = field(default_factory=set)


def partition_graph(graph: LogicalGraph, num_partitions: int) -> list[GraphPartition]:
    """Split a LogicalGraph into num_partitions balanced, valid sub-graphs.

    Variables are sorted lexicographically and divided into contiguous
    chunks of nearly equal size (same ordering convention as
    compile_lexicographic). For an interaction whose two variables fall
    in different partitions, the lower-indexed partition owns it: that
    interaction is included in its local graph, and the foreign
    variable is added there as a boundary reference so the partition's
    LogicalGraph.validate() passes on its own.

    Args:
        graph: The LogicalGraph to partition.
        num_partitions: Number of partitions to produce. Must be >= 1
            and <= the number of variables in the graph.

    Returns:
        A list of GraphPartition, one per partition index.

    Raises:
        ValueError: If num_partitions is not between 1 and len(graph.variables).
    """
    var_names = sorted(v.name for v in graph.variables)
    if num_partitions < 1 or num_partitions > max(len(var_names), 1):
        raise ValueError(
            f"num_partitions must be between 1 and {len(var_names)}, got {num_partitions}"
        )

    chunks: list[list[str]] = [[] for _ in range(num_partitions)]
    base, extra = divmod(len(var_names), num_partitions)
    cursor = 0
    for idx in range(num_partitions):
        size = base + (1 if idx < extra else 0)
        chunks[idx] = var_names[cursor : cursor + size]
        cursor += size

    owner: dict[str, int] = {}
    for idx, chunk in enumerate(chunks):
        for name in chunk:
            owner[name] = idx

    var_lookup = {v.name: v for v in graph.variables}
    local_vars: list[set[str]] = [set(chunk) for chunk in chunks]
    boundary_refs: list[set[str]] = [set() for _ in range(num_partitions)]
    owned_interactions: list[list[Interaction]] = [[] for _ in range(num_partitions)]

    for ix in graph.interactions:
        owner_i, owner_j = owner[ix.i], owner[ix.j]
        owning = min(owner_i, owner_j)
        owned_interactions[owning].append(ix)
        if owner_i != owning:
            boundary_refs[owning].add(ix.i)
        if owner_j != owning:
            boundary_refs[owning].add(ix.j)

    partitions: list[GraphPartition] = []
    for idx in range(num_partitions):
        names = local_vars[idx] | boundary_refs[idx]
        sub_graph = LogicalGraph(
            variables=[var_lookup[n] for n in sorted(names)],
            interactions=list(owned_interactions[idx]),
            metadata={"partition_id": idx, "source_metadata": dict(graph.metadata)},
        )
        partitions.append(
            GraphPartition(
                partition_id=idx,
                graph=sub_graph,
                local_vars=local_vars[idx],
                boundary_refs=boundary_refs[idx],
            )
        )
    return partitions


def namespaced_hardware_graph(n: int, prefix: str) -> dict[str, list[str]]:
    """Return a complete graph over n nodes labelled '{prefix}:q0'..'{prefix}:q{n-1}'.

    Mirrors limen.core.compiler.default_hardware_graph, but namespaces
    labels per partition so two partitions' physical qubit labels never
    collide when their QUBO dicts are later merged.

    Args:
        n: Number of physical qubits.
        prefix: Namespace prefix unique to one partition.

    Returns:
        Adjacency dict where every node is connected to every other node.
    """
    nodes = [f"{prefix}:q{i}" for i in range(n)]
    return {node: [other for other in nodes if other != node] for node in nodes}


def merge_partition_results(
    partitions: list[GraphPartition], encodings: list[PhysicalEncoding], graph: LogicalGraph
) -> PhysicalEncoding:
    """Merge per-partition PhysicalEncodings into one encoding over the original graph.

    Rewrites any QUBO key referencing a boundary-reference variable's
    (throwaway, partition-local) physical label to that variable's
    canonical physical label, taken from its home partition's
    embedding. After rewriting, the union of all partitions' QUBO dicts
    has no key collisions: non-boundary labels are unique per
    partition's namespace prefix, and rewritten boundary labels all
    point at a single canonical owner.

    Args:
        partitions: The GraphPartitions produced by partition_graph().
        encodings: The PhysicalEncoding for each partition, in the same
            order as partitions (encodings[i] compiled partitions[i].graph).
        graph: The original, unpartitioned LogicalGraph.

    Returns:
        A PhysicalEncoding over `graph`, energetically equivalent to
        compiling it directly with compile_lexicographic in one shot.
    """
    canonical_label: dict[str, str] = {}
    canonical_embedding: dict[str, list[str]] = {}
    for partition, encoding in zip(partitions, encodings):
        for var in partition.local_vars:
            label = encoding.embedding[var][0]
            canonical_label[var] = label
            canonical_embedding[var] = [label]

    def _rewrite(label: str, owning_partition_id: int, encoding: PhysicalEncoding) -> str:
        for var, embedded in encoding.embedding.items():
            if embedded == [label] and var not in partitions[owning_partition_id].local_vars:
                return canonical_label[var]
        return label

    merged_qubo: dict[tuple[str, str], float] = {}
    for partition, encoding in zip(partitions, encodings):
        for (a, b), weight in encoding.qubo.items():
            ra = _rewrite(a, partition.partition_id, encoding)
            rb = _rewrite(b, partition.partition_id, encoding)
            merged_qubo[(ra, rb)] = weight

    chain_strength = max((e.chain_strength for e in encodings), default=1.0)
    metadata = {
        "compiler": "distributed_partition_merge",
        "num_partitions": len(partitions),
    }

    return PhysicalEncoding(
        graph=graph,
        qubo=merged_qubo,
        embedding=canonical_embedding,
        chain_strength=chain_strength,
        metadata=metadata,
    )


def dispatch_partitions(
    partitions: list[GraphPartition], peer_addresses: list[str] | None = None
) -> list[PhysicalEncoding]:
    """Compile each partition, locally or dispatched round-robin to peers.

    If peer_addresses is empty or None, every partition is compiled
    in-process via compile_lexicographic. Otherwise, partitions are
    round-robined across the given peer addresses and compiled remotely
    via CoordinationClient.compile_partition.

    Args:
        partitions: The GraphPartitions to compile, in partition order.
        peer_addresses: "host:port" addresses of peers to dispatch to.

    Returns:
        A list of PhysicalEncoding, one per partition, in the same
        order as `partitions`.
    """
    if not peer_addresses:
        return [
            compile_lexicographic(
                p.graph, namespaced_hardware_graph(len(p.graph.variables), f"p{p.partition_id}")
            )
            for p in partitions
        ]

    from limen.distributed.client import CoordinationClient

    results: list[PhysicalEncoding] = []
    clients = [CoordinationClient(addr) for addr in peer_addresses]
    try:
        for p in partitions:
            client = clients[p.partition_id % len(clients)]
            prefix = f"p{p.partition_id}"
            results.append(client.compile_partition(str(p.partition_id), p.graph, prefix))
    finally:
        for client in clients:
            client.close()
    return results
