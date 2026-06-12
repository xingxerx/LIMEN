# Copyright 2026 LIMEN Contributors. Apache 2.0.
"""Tests for calibration loaders (QuEra/Pasqal and IBMQ formats)."""

import json
import tempfile
from pathlib import Path

import pytest

from limen.analog.calibration_loader import load_ibmq_calibration, load_quera_calibration
from limen.analog.hamiltonian import SubstrateType


QUERA_DICT = {
    "device_id": "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
    "substrate": "neutral_atom",
    "n_sites": 256,
    "calibration_timestamp": 1704067200.0,
    "site_detuning_offsets_mhz": {"0": 0.015, "3": -0.008},
    "coupling_scale_errors": {"[0, 1]": 0.02, "[2, 3]": -0.01},
    "global_rabi_error": 0.005,
    "metadata": {"vendor": "QuEra"},
}

IBMQ_DICT = {
    "backend_name": "ibm_perth",
    "n_qubits": 7,
    "properties_timestamp": "2024-01-01T00:00:00+00:00",
    "qubits": [
        [{"name": "frequency_error", "value": 0.001, "unit": "GHz"}],
        [{"name": "frequency_error", "value": -0.0005, "unit": "GHz"}],
        [{"name": "T1", "value": 100.0, "unit": "us"}],
    ],
    "gates": [
        {"gate": "cx", "qubits": [0, 1], "parameters": [{"name": "gate_error", "value": 0.01}]},
        {"gate": "ecr", "qubits": [3, 4], "parameters": [{"name": "gate_error", "value": 0.008}]},
        {"gate": "u3", "qubits": [0],    "parameters": [{"name": "gate_error", "value": 0.001}]},
    ],
}


# ---------------------------------------------------------------------------
# QuEra loader
# ---------------------------------------------------------------------------

def test_quera_fields():
    m = load_quera_calibration(QUERA_DICT)
    assert m.device_id == QUERA_DICT["device_id"]
    assert m.n_sites == 256
    assert m.substrate == SubstrateType.NEUTRAL_ATOM
    assert m.drift.global_rabi_error == pytest.approx(0.005)
    assert m.drift.timestamp == pytest.approx(1704067200.0)


def test_quera_detuning_and_coupling_errors():
    m = load_quera_calibration(QUERA_DICT)
    assert m.drift.site_detuning_offsets[0] == pytest.approx(0.015)
    assert m.drift.site_detuning_offsets[3] == pytest.approx(-0.008)
    assert m.drift.coupling_scale_errors[(0, 1)] == pytest.approx(0.02)
    assert m.drift.coupling_scale_errors[(2, 3)] == pytest.approx(-0.01)


def test_quera_coupling_key_formats():
    # Array notation and comma-separated notation both parse correctly.
    m_array = load_quera_calibration({**QUERA_DICT, "coupling_scale_errors": {"[4, 5]": 0.03}})
    m_comma = load_quera_calibration({**QUERA_DICT, "coupling_scale_errors": {"4,5": 0.03}})
    assert m_array.drift.coupling_scale_errors[(4, 5)] == pytest.approx(0.03)
    assert m_comma.drift.coupling_scale_errors[(4, 5)] == pytest.approx(0.03)


def test_quera_unknown_substrate_defaults_to_neutral_atom():
    m = load_quera_calibration({**QUERA_DICT, "substrate": "unknown_xyz"})
    assert m.substrate == SubstrateType.NEUTRAL_ATOM


def test_quera_from_json_string_and_file():
    m_str = load_quera_calibration(json.dumps(QUERA_DICT))
    assert m_str.device_id == QUERA_DICT["device_id"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(QUERA_DICT, f)
        tmp = Path(f.name)
    try:
        assert load_quera_calibration(tmp).n_sites == 256
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# IBMQ loader
# ---------------------------------------------------------------------------

def test_ibmq_fields():
    m = load_ibmq_calibration(IBMQ_DICT)
    assert m.device_id == "ibm_perth"
    assert m.n_sites == 7
    assert m.substrate == SubstrateType.UNSPECIFIED


def test_ibmq_qubit_frequency_to_detuning():
    m = load_ibmq_calibration(IBMQ_DICT)
    assert m.drift.site_detuning_offsets[0] == pytest.approx(1.0)   # 0.001 GHz × 1000
    assert m.drift.site_detuning_offsets[1] == pytest.approx(-0.5)  # -0.0005 GHz × 1000
    assert 2 not in m.drift.site_detuning_offsets                    # no frequency_error entry


def test_ibmq_gate_errors_to_coupling_scale():
    m = load_ibmq_calibration(IBMQ_DICT)
    assert m.drift.coupling_scale_errors[(0, 1)] == pytest.approx(0.01)
    assert m.drift.coupling_scale_errors[(3, 4)] == pytest.approx(0.008)  # ecr gate
    assert (0, 0) not in m.drift.coupling_scale_errors                    # u3 ignored


def test_ibmq_timestamp_from_iso():
    m = load_ibmq_calibration(IBMQ_DICT)
    assert m.drift.timestamp == pytest.approx(1704067200.0)


def test_ibmq_custom_freq_conversion():
    m = load_ibmq_calibration(IBMQ_DICT, freq_error_to_detuning_mhz=500.0)
    assert m.drift.site_detuning_offsets[0] == pytest.approx(0.5)


def test_ibmq_from_json_string_and_file():
    m_str = load_ibmq_calibration(json.dumps(IBMQ_DICT))
    assert m_str.device_id == "ibm_perth"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(IBMQ_DICT, f)
        tmp = Path(f.name)
    try:
        assert load_ibmq_calibration(tmp).n_sites == 7
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Live IBMQ calibration & QEM mock tests
# ---------------------------------------------------------------------------

def test_load_live_ibmq_calibration(monkeypatch):
    class MockBackend:
        def properties(self):
            class MockProperties:
                def to_dict(self):
                    return IBMQ_DICT
            return MockProperties()

    class MockService:
        def __init__(self, channel, token, instance):
            pass
        def backend(self, name):
            return MockBackend()

    import sys
    from types import ModuleType

    mock_runtime = ModuleType("qiskit_ibm_runtime")
    mock_runtime.QiskitRuntimeService = MockService
    sys.modules["qiskit_ibm_runtime"] = mock_runtime

    try:
        from limen.analog.calibration_loader import load_live_ibmq_calibration
        m = load_live_ibmq_calibration(token="mock_token", crn="mock_crn", backend_name="ibm_perth")
        assert m.device_id == "ibm_perth"
        assert m.n_sites == 7
    finally:
        del sys.modules["qiskit_ibm_runtime"]


def test_run_qiskit_qpu_error_mitigation(monkeypatch):
    class MockJob:
        def job_id(self):
            return "mock_job_id"
        def result(self, timeout=None):
            class MockPubResult:
                class MockData:
                    class MockMeas:
                        def get_counts(self):
                            return {"00": 50, "11": 50}
                    meas = MockMeas()
                data = MockData()
            return [MockPubResult()]

    class MockSampler:
        def __init__(self, mode):
            class MockOptions:
                class MockDD:
                    enable = False
                    sequence_type = None
                class MockTwirling:
                    enable_gates = False
                    enable_measure = False
                dynamical_decoupling = MockDD()
                twirling = MockTwirling()
            self.options = MockOptions()
        def run(self, pubs, shots=1000):
            assert self.options.dynamical_decoupling.enable is True
            assert self.options.dynamical_decoupling.sequence_type == "XY4"
            assert self.options.twirling.enable_gates is True
            assert self.options.twirling.enable_measure is True
            return MockJob()

    class MockBackend:
        pass

    class MockService:
        def __init__(self, channel, token, instance):
            pass
        def backend(self, name):
            return MockBackend()

    import sys
    from types import ModuleType

    mock_runtime = ModuleType("qiskit_ibm_runtime")
    mock_runtime.QiskitRuntimeService = MockService
    mock_runtime.SamplerV2 = MockSampler
    sys.modules["qiskit_ibm_runtime"] = mock_runtime

    mock_passmanagers = ModuleType("qiskit.transpiler.preset_passmanagers")
    class MockPM:
        def run(self, circuit):
            return circuit
    def generate_preset_pass_manager(optimization_level, backend):
        return MockPM()
    mock_passmanagers.generate_preset_pass_manager = generate_preset_pass_manager
    sys.modules["qiskit.transpiler.preset_passmanagers"] = mock_passmanagers

    try:
        from limen.backends.qiskit_backend import run_qiskit_qpu
        from limen import compile_lexicographic, default_hardware_graph, from_qubo_dict
        
        encoding = compile_lexicographic(
            from_qubo_dict({("q0", "q0"): -1.0, ("q1", "q1"): -1.0}),
            default_hardware_graph(8)
        )
        
        result = run_qiskit_qpu(
            encoding,
            token="mock_token",
            crn="mock_crn",
            backend_name="ibm_perth",
            shots=100,
            dynamical_decoupling=True,
            twirling=True,
        )
        assert result.metadata["backend"] == "ibm_perth"
        assert result.metadata["job_id"] == "mock_job_id"
    finally:
        del sys.modules["qiskit_ibm_runtime"]
        del sys.modules["qiskit.transpiler.preset_passmanagers"]

