"""Regenerate limen/distributed/proto/*_pb2*.py from coordination.proto.

Run after editing coordination.proto:

    python scripts/gen_proto.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROTO_DIR = ROOT / "limen" / "distributed" / "proto"


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_DIR}",
        f"--python_out={PROTO_DIR}",
        f"--grpc_python_out={PROTO_DIR}",
        str(PROTO_DIR / "coordination.proto"),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)

    # protoc emits an absolute "import coordination_pb2 as ..." in the
    # _grpc.py file, which breaks when this directory is imported as a
    # package. Rewrite it to a relative import.
    grpc_file = PROTO_DIR / "coordination_pb2_grpc.py"
    text = grpc_file.read_text()
    text = text.replace(
        "import coordination_pb2 as coordination__pb2",
        "from . import coordination_pb2 as coordination__pb2",
    )
    grpc_file.write_text(text)

    print(f"Generated coordination_pb2.py / coordination_pb2_grpc.py in {PROTO_DIR}")


if __name__ == "__main__":
    main()
