"""End-to-end test of run_pipeline dispatching compilation over real gRPC.

Spins up a live CoordinationServer and runs the full pipeline with
server_addresses pointing at it, exercising the CompilePartition RPC path
(partition -> dispatch -> merge -> verify) rather than the local-only path.
Skipped cleanly when grpcio is not installed.
"""

import unittest

import pytest

grpc = pytest.importorskip("grpc")

from limen.distributed.config import NodeConfig
from limen.distributed.registry import NodeRegistry
from limen.distributed.server import serve
from limen.pipeline import run_pipeline

pytestmark = pytest.mark.network

# A separable 4-variable QUBO: optimum is x0=x1? No coupling forces splits,
# so each pair {x0,x1} and {x2,x3} prefers exactly one of the two ON.
_QUBO = {
    ("x0", "x0"): -1.0,
    ("x1", "x1"): -1.0,
    ("x2", "x2"): -1.0,
    ("x3", "x3"): -1.0,
    ("x0", "x1"): 2.0,
    ("x2", "x3"): 2.0,
}


class TestPipelineDistributed(unittest.TestCase):

    def setUp(self):
        self.registry = NodeRegistry()
        config = NodeConfig(node_id="node-a", host="127.0.0.1", port=0)
        self.server, self.port = serve(config, self.registry, port=0)
        self.addCleanup(lambda: self.server.stop(grace=0))
        self.address = f"127.0.0.1:{self.port}"

    def test_pipeline_compiles_over_grpc(self):
        cert = run_pipeline(
            _QUBO,
            qaoa_layers=2,
            grid_size=16,
            encode_logical=False,
            server_addresses=[self.address],
            num_partitions=2,
        )

        dc = cert.distributed_compilation
        self.assertIsNotNone(dc)
        self.assertEqual(dc["num_partitions"], 2)
        self.assertEqual(dc["server_addresses"], [self.address])
        self.assertEqual(dc["n_physical_qubits"], 4)
        self.assertTrue(dc["verified_equivalent_to_single_shot"])
        self.assertTrue(
            any("Distributed compilation" in n for n in cert.notes)
        )

    def test_local_solution_still_certified_when_distributed(self):
        cert = run_pipeline(
            _QUBO,
            qaoa_layers=2,
            grid_size=16,
            encode_logical=False,
            server_addresses=[self.address],
        )
        # The gate-model solution path is unaffected by remote compilation.
        self.assertTrue(cert.is_optimal)
        self.assertEqual(cert.energy, cert.classical_energy)

    def test_distributed_compilation_is_serializable(self):
        cert = run_pipeline(
            _QUBO,
            encode_logical=False,
            server_addresses=[self.address],
            num_partitions=2,
        )
        d = cert.to_dict()
        self.assertIn("distributed_compilation", d)
        self.assertEqual(d["distributed_compilation"]["num_partitions"], 2)


if __name__ == "__main__":
    unittest.main()
