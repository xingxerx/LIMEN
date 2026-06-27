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
"""Two-node distributed compilation demo: the qubit-doubling story, end to end.

run_pipeline() already supports splitting a logical graph across peer LIMEN
nodes (see limen/pipeline.py:_distributed_compile and
limen/distributed/partition.py) — this script is the example invocation that
was missing to actually exercise it:

    cert = limen.run_pipeline(
        qubo,
        server_addresses=["node-b:50051"],
        num_partitions=2,
        physical_error_rate=0.01,
    )
    print(cert.distributed_compilation)

By default this script stands up "LIMEN-B" itself, in-process, on an
OS-assigned port, so it runs end to end with zero setup:

    python examples/distributed_two_node.py

To point at a real second node (e.g. one started with
scripts/deploy_node.sh on a remote machine), pass its address instead and
the script skips the local spin-up entirely:

    python examples/distributed_two_node.py --peer node-b.example.com:50051
"""

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Windows consoles often default to cp1252, which cannot print the box-drawing
# characters used in the report below.
_reconfigure = getattr(sys.stdout, "reconfigure", None)
if _reconfigure and (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
    _reconfigure(encoding="utf-8", errors="replace")

import grpc
import limen
from limen.distributed.config import NodeConfig
from limen.distributed.registry import NodeRegistry
from limen.distributed.server import serve

# An 8-variable Max-Cut QUBO: large enough that partition_graph() splits it
# into two genuinely different halves (x0..x3 on partition 0, x4..x7 on
# partition 1), with cross-partition edges forcing the merge step to do real
# boundary-variable rewriting rather than trivially no-op.
_EDGES = [
    ("x0", "x1"), ("x1", "x2"), ("x2", "x3"), ("x3", "x0"),
    ("x4", "x5"), ("x5", "x6"), ("x6", "x7"), ("x7", "x4"),
    ("x3", "x4"),  # the cross-partition edge
]


def _build_qubo() -> dict[tuple[str, str], float]:
    qubo: dict[tuple[str, str], float] = {}
    for i, j in _EDGES:
        qubo[(i, j)] = qubo.get((i, j), 0.0) + 1.0
        qubo[(i, i)] = qubo.get((i, i), 0.0) - 1.0
        qubo[(j, j)] = qubo.get((j, j), 0.0) - 1.0
    return qubo


def _start_local_peer() -> tuple[grpc.Server, str]:
    """Stand in for LIMEN-B: a real CompilePartition server on 127.0.0.1."""
    registry = NodeRegistry()
    config = NodeConfig(node_id="limen-b-demo", host="127.0.0.1", port=0)
    server, port = serve(config, registry, port=0)
    return server, f"127.0.0.1:{port}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--peer",
        metavar="HOST:PORT",
        default=None,
        help="Address of an already-running LIMEN-B node. If omitted, a "
        "local stand-in server is started on this machine instead.",
    )
    parser.add_argument(
        "--num-partitions", type=int, default=2,
        help="Number of graph partitions to dispatch across peers (default: 2).",
    )
    args = parser.parse_args()

    qubo = _build_qubo()

    server = None
    if args.peer:
        peer_address = args.peer
        print(f"[1/2] Using existing peer at {peer_address} (LIMEN-B) ...")
    else:
        server, peer_address = _start_local_peer()
        print(f"[1/2] Started local stand-in peer at {peer_address} (LIMEN-B) ...")
        print(
            "      (pass --peer host:port to dispatch to a real remote "
            "LIMEN-B node instead)"
        )

    try:
        print(
            f"[2/2] Running pipeline: {len(qubo)} QUBO terms, "
            f"{args.num_partitions} partitions, server_addresses=[{peer_address!r}] ..."
        )
        cert = limen.run_pipeline(
            qubo,
            qaoa_layers=2,
            grid_size=16,
            server_addresses=[peer_address],
            num_partitions=args.num_partitions,
            physical_error_rate=0.01,
        )
    finally:
        if server is not None:
            server.stop(grace=0)

    dc = cert.distributed_compilation
    print()
    print("── Distributed compilation ─────────────────────────────")
    print(json.dumps(dc, indent=2))
    print()
    print("── Certificate ──────────────────────────────────────────")
    print(f"  is_optimal              : {cert.is_optimal}")
    print(f"  energy / classical      : {cert.energy} / {cert.classical_energy}")
    print(f"  success_probability     : {cert.success_probability:.4f}")
    print(f"  logical_error_rate      : {cert.logical_error_rate}")
    for note in cert.notes:
        print(f"  note: {note}")

    out_dir = pathlib.Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"distributed_two_node_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(cert.to_dict(), indent=2), encoding="utf-8")
    print(f"\nFull certificate saved to {out_path}")


if __name__ == "__main__":
    main()
