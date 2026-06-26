# tests/test_quantum_channel.py
import math
import pytest
from limen.quantum_channel.channel_delta import ChannelDeltaModel
from limen.quantum_channel.qkd import sift_and_evaluate


def test_within_coherence_true():
    m = ChannelDeltaModel(latency_ms=0.01, t2_us=100.0)
    assert m.within_coherence()


def test_within_coherence_false():
    m = ChannelDeltaModel(latency_ms=10.0, t2_us=50.0)
    assert not m.within_coherence()


def test_fidelity_penalty_range():
    m = ChannelDeltaModel(latency_ms=0.01, t2_us=100.0)
    p = m.fidelity_penalty()
    assert 0.0 < p <= 1.0


def test_fidelity_penalty_formula():
    m = ChannelDeltaModel(latency_ms=1.0, t2_us=1000.0)
    expected = math.exp(-1.0)
    assert abs(m.fidelity_penalty() - expected) < 1e-9


def test_channel_delta_to_dict():
    m = ChannelDeltaModel(latency_ms=0.5, t2_us=200.0)
    d = m.to_dict()
    assert "fidelity_penalty" in d
    assert "within_coherence" in d


def test_sift_perfect_channel():
    bits  = [0, 1, 0, 1, 1, 0, 1, 0]
    bases = [0, 0, 1, 1, 0, 1, 0, 1]
    result = sift_and_evaluate(bits, bases, bases, bits)
    assert result.qber == 0.0
    assert result.secure
    assert len(result.sifted_key) == len(bits)


def test_sift_noisy_channel():
    bits     = [0, 1, 0, 1]
    bases    = [0, 0, 1, 1]
    bad_bits = [1, 0, 1, 0]
    result = sift_and_evaluate(bits, bases, bases, bad_bits)
    assert result.qber > 0.11
    assert not result.secure


def test_sift_no_matching_bases():
    alice_bases = [0, 0, 0, 0]
    bob_bases   = [1, 1, 1, 1]
    result = sift_and_evaluate([0,1,0,1], alice_bases, bob_bases, [0,1,0,1])
    assert result.qber == 1.0
    assert not result.secure
    assert result.sifted_key == []


def test_qkd_result_to_dict():
    result = sift_and_evaluate([0,1],[0,1],[0,1],[0,1], backend="ibm")
    d = result.to_dict()
    assert d["backend"] == "ibm"
    assert "qber" in d


@pytest.mark.skipif(
    True,  # flip to False once qiskit is installed
    reason="requires qiskit"
)
def test_bb84_circuit_shape():
    from limen.quantum_channel.qkd import bb84_circuit
    qc, _, _, _ = bb84_circuit(8)
    assert qc.num_qubits == 8


@pytest.mark.skipif(
    True,
    reason="requires qiskit"
)
def test_teleport_circuit_shape():
    from limen.quantum_channel.teleport import teleport_circuit
    qc = teleport_circuit()
    assert qc.num_qubits == 3
