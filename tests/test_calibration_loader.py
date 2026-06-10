# Copyright 2026 LIMEN Contributors. Apache 2.0.
"""Tests for calibration loaders (QuEra/Pasqal and IBMQ formats)."""

import json
import tempfile
from pathlib import Path

import pytest

from limen.analog.calibration_loader import load_ibmq_calibration, load_quera_calibration
from limen.analog.hamiltonian import SubstrateType


# ---------------------------------------------------------------------------
# Fixtures: minimal calibration dicts
# ---------------------------------------------------------------------------

QUERA_DICT = {
    "device_id": "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
    "substrate": "neutral_atom",
    "n_sites": 256,
    "calibration_timestamp": 1704067200.0,
    "site_detuning_offsets_mhz": {"0": 0.015, "3": -0.008},
    "coupling_scale_errors": {"[0, 1]": 0.02, "[2, 3]": -0.01},
    "global_rabi_error": 0.005,
    "metadata": {"vendor": "QuEra", "device_name": "Aquila"},
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
        {
            "gate": "cx",
            "qubits": [0, 1],
            "parameters": [{"name": "gate_error", "value": 0.01}],
        },
        {
            "gate": "cx",
            "qubits": [1, 2],
            "parameters": [{"name": "gate_error", "value": 0.015}],
        },
        {
            "gate": "u3",
            "qubits": [0],
            "parameters": [{"name": "gate_error", "value": 0.001}],
        },
    ],
}


# ---------------------------------------------------------------------------
# QuEra loader — basic fields
# ---------------------------------------------------------------------------

def test_quera_device_id():
    model = load_quera_calibration(QUERA_DICT)
    assert model.device_id == QUERA_DICT["device_id"]


def test_quera_n_sites():
    model = load_quera_calibration(QUERA_DICT)
    assert model.n_sites == 256


def test_quera_substrate():
    model = load_quera_calibration(QUERA_DICT)
    assert model.substrate == SubstrateType.NEUTRAL_ATOM


def test_quera_site_detuning_offsets():
    model = load_quera_calibration(QUERA_DICT)
    assert model.drift.site_detuning_offsets[0] == pytest.approx(0.015)
    assert model.drift.site_detuning_offsets[3] == pytest.approx(-0.008)


def test_quera_coupling_scale_errors_array_key():
    model = load_quera_calibration(QUERA_DICT)
    assert model.drift.coupling_scale_errors[(0, 1)] == pytest.approx(0.02)
    assert model.drift.coupling_scale_errors[(2, 3)] == pytest.approx(-0.01)


def test_quera_global_rabi_error():
    model = load_quera_calibration(QUERA_DICT)
    assert model.drift.global_rabi_error == pytest.approx(0.005)


def test_quera_timestamp():
    model = load_quera_calibration(QUERA_DICT)
    assert model.drift.timestamp == pytest.approx(1704067200.0)


def test_quera_loader_tag_in_metadata():
    model = load_quera_calibration(QUERA_DICT)
    assert model.metadata.get("loader") == "load_quera_calibration"


# ---------------------------------------------------------------------------
# QuEra loader — coupling key formats
# ---------------------------------------------------------------------------

def test_quera_comma_separated_coupling_key():
    d = dict(QUERA_DICT)
    d["coupling_scale_errors"] = {"4,5": 0.03}
    model = load_quera_calibration(d)
    assert model.drift.coupling_scale_errors[(4, 5)] == pytest.approx(0.03)


def test_quera_empty_offsets_and_errors():
    d = {
        "device_id": "test-device",
        "n_sites": 10,
    }
    model = load_quera_calibration(d)
    assert model.drift.site_detuning_offsets == {}
    assert model.drift.coupling_scale_errors == {}
    assert model.drift.global_rabi_error == pytest.approx(0.0)


def test_quera_unknown_substrate_defaults_to_neutral_atom():
    d = dict(QUERA_DICT)
    d["substrate"] = "unknown_substrate_xyz"
    model = load_quera_calibration(d)
    assert model.substrate == SubstrateType.NEUTRAL_ATOM


# ---------------------------------------------------------------------------
# QuEra loader — file path and JSON string inputs
# ---------------------------------------------------------------------------

def test_quera_from_json_string():
    model = load_quera_calibration(json.dumps(QUERA_DICT))
    assert model.device_id == QUERA_DICT["device_id"]


def test_quera_from_file_path():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(QUERA_DICT, f)
        tmppath = f.name
    try:
        model = load_quera_calibration(tmppath)
        assert model.n_sites == 256
    finally:
        Path(tmppath).unlink(missing_ok=True)


def test_quera_from_path_object():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(QUERA_DICT, f)
        tmppath = Path(f.name)
    try:
        model = load_quera_calibration(tmppath)
        assert model.device_id == QUERA_DICT["device_id"]
    finally:
        tmppath.unlink(missing_ok=True)


def test_quera_invalid_source_type():
    with pytest.raises(TypeError):
        load_quera_calibration(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# QuEra loader — apply_detuning_correction round-trip
# ---------------------------------------------------------------------------

def test_quera_apply_detuning_correction():
    model = load_quera_calibration(QUERA_DICT)
    detunings = [0.0] * 10
    corrected = model.apply_detuning_correction(detunings)
    assert corrected[0] == pytest.approx(-0.015)
    assert corrected[3] == pytest.approx(0.008)
    assert corrected[1] == pytest.approx(0.0)


def test_quera_apply_coupling_correction():
    model = load_quera_calibration(QUERA_DICT)
    couplings = {(0, 1): 1.0, (2, 3): 2.0, (4, 5): 3.0}
    corrected = model.apply_coupling_correction(couplings)
    # J_corrected = J / (1 + error)
    assert corrected[(0, 1)] == pytest.approx(1.0 / 1.02)
    assert corrected[(2, 3)] == pytest.approx(2.0 / 0.99)
    assert corrected[(4, 5)] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# IBMQ loader — basic fields
# ---------------------------------------------------------------------------

def test_ibmq_device_id():
    model = load_ibmq_calibration(IBMQ_DICT)
    assert model.device_id == "ibm_perth"


def test_ibmq_n_sites():
    model = load_ibmq_calibration(IBMQ_DICT)
    assert model.n_sites == 7


def test_ibmq_substrate_unspecified():
    model = load_ibmq_calibration(IBMQ_DICT)
    assert model.substrate == SubstrateType.UNSPECIFIED


def test_ibmq_qubit_frequency_errors_to_detuning():
    model = load_ibmq_calibration(IBMQ_DICT)
    # 0.001 GHz × 1000 = 1.0 MHz
    assert model.drift.site_detuning_offsets[0] == pytest.approx(1.0)
    # -0.0005 GHz × 1000 = -0.5 MHz
    assert model.drift.site_detuning_offsets[1] == pytest.approx(-0.5)
    # Qubit 2 has no frequency_error entry
    assert 2 not in model.drift.site_detuning_offsets


def test_ibmq_cx_gate_errors_to_coupling_scale():
    model = load_ibmq_calibration(IBMQ_DICT)
    assert model.drift.coupling_scale_errors[(0, 1)] == pytest.approx(0.01)
    assert model.drift.coupling_scale_errors[(1, 2)] == pytest.approx(0.015)


def test_ibmq_single_qubit_gates_ignored():
    model = load_ibmq_calibration(IBMQ_DICT)
    # u3 on qubit 0 should not appear in coupling_scale_errors
    assert (0, 0) not in model.drift.coupling_scale_errors


def test_ibmq_timestamp_parsed_from_iso():
    model = load_ibmq_calibration(IBMQ_DICT)
    assert model.drift.timestamp > 0.0
    # 2024-01-01T00:00:00+00:00 → unix 1704067200.0
    assert model.drift.timestamp == pytest.approx(1704067200.0)


def test_ibmq_loader_tag_in_metadata():
    model = load_ibmq_calibration(IBMQ_DICT)
    assert model.metadata.get("loader") == "load_ibmq_calibration"


# ---------------------------------------------------------------------------
# IBMQ loader — custom conversion factor
# ---------------------------------------------------------------------------

def test_ibmq_custom_freq_conversion():
    model = load_ibmq_calibration(IBMQ_DICT, freq_error_to_detuning_mhz=500.0)
    assert model.drift.site_detuning_offsets[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# IBMQ loader — ecr gate variant
# ---------------------------------------------------------------------------

def test_ibmq_ecr_gate_parsed():
    d = dict(IBMQ_DICT)
    d["gates"] = [
        {
            "gate": "ecr",
            "qubits": [3, 4],
            "parameters": [{"name": "gate_error", "value": 0.008}],
        }
    ]
    model = load_ibmq_calibration(d)
    assert model.drift.coupling_scale_errors[(3, 4)] == pytest.approx(0.008)


# ---------------------------------------------------------------------------
# IBMQ loader — file/string inputs
# ---------------------------------------------------------------------------

def test_ibmq_from_json_string():
    model = load_ibmq_calibration(json.dumps(IBMQ_DICT))
    assert model.device_id == "ibm_perth"


def test_ibmq_from_file_path():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(IBMQ_DICT, f)
        tmppath = f.name
    try:
        model = load_ibmq_calibration(tmppath)
        assert model.n_sites == 7
    finally:
        Path(tmppath).unlink(missing_ok=True)


def test_ibmq_invalid_source_type():
    with pytest.raises(TypeError):
        load_ibmq_calibration(42)  # type: ignore[arg-type]
