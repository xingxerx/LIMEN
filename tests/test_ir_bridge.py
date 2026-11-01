import json

import pytest

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

from limen import ir  # noqa: E402


def bell():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.rz(1.5, 1)
    qc.measure([0, 1], [0, 1])
    return qc


def test_roundtrip_through_ir_is_lossless():
    j = ir.from_qiskit(bell())
    assert json.loads(j)["schema_ver"] == ir.SCHEMA_VER
    back = ir.to_qiskit(j)
    assert ir.hash(ir.from_qiskit(back)) == ir.hash(j)


def test_hash_is_stable_and_canonical():
    j = ir.from_qiskit(bell())
    assert ir.hash(j) == ir.hash(ir.canonical(j))
    assert len(ir.hash(j)) == 64


def test_optimize_preserves_state():
    qc = QuantumCircuit(2)
    qc.h(0); qc.h(0); qc.cx(0, 1); qc.x(1); qc.x(1); qc.ry(0.7, 0)
    j = ir.from_qiskit(qc)
    opt = ir.optimize(j)
    assert len(json.loads(opt)["ops"]) == 2
    assert Statevector(ir.to_qiskit(opt)).equiv(Statevector(qc))


def test_lower_sets_physical_level():
    low = json.loads(ir.lower(ir.from_qiskit(bell())))
    assert low["level"] == "physical"
    assert low["meta"]["layout"] == "0,1"


def test_bad_input_raises_not_crashes():
    with pytest.raises(ValueError):
        ir.from_qasm3("qubit[2] q;")
    with pytest.raises(ValueError):
        ir.hash("{not json")
