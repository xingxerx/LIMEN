"""End-to-end test of CircuitIR execution via the Qiskit backend."""

import unittest

import pytest

qiskit = pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from limen.gates.ir import CircuitIR, GateInstruction
from limen.gates.qiskit_exec import run_circuit, to_qiskit_circuit


class TestGateExec(unittest.TestCase):

    def _bell_circuit(self) -> CircuitIR:
        return CircuitIR(
            n_qubits=2,
            instructions=[GateInstruction("h", [0]), GateInstruction("cx", [0, 1])],
        )

    def test_to_qiskit_circuit_matches_hand_written(self):
        from qiskit import QuantumCircuit

        built = to_qiskit_circuit(self._bell_circuit())

        expected = QuantumCircuit(2, 2)
        expected.h(0)
        expected.cx(0, 1)
        expected.measure(range(2), range(2))

        self.assertEqual(built.count_ops(), expected.count_ops())
        self.assertEqual(built.num_qubits, expected.num_qubits)

    def test_bell_state_measurement_distribution(self):
        result = run_circuit(self._bell_circuit(), shots=2000)
        total = sum(result.counts.values())
        correlated = sum(c for bitstring, c in result.counts.items() if bitstring in ("00", "11"))
        self.assertGreater(correlated / total, 0.95)

    def test_invalid_circuit_raises(self):
        bad = CircuitIR(n_qubits=1, instructions=[GateInstruction("cx", [0, 1])])
        with self.assertRaises(ValueError):
            to_qiskit_circuit(bad)


if __name__ == "__main__":
    unittest.main()
