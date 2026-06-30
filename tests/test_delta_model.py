# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.
"""Tests for limen.analog.delta_model."""

import pytest

from limen.analog.delta_model import (
    DeviceDrift,
    DeltaModelRegistry,
    HardwareDeltaModel,
)
from limen.analog.hamiltonian import SubstrateType


# ---------------------------------------------------------------------------
# Test 0: DeviceDrift roundtrip
# ---------------------------------------------------------------------------

def test_device_drift_roundtrip():
    drift = DeviceDrift(
        site_detuning_offsets={0: 0.5, 1: -0.3},
        coupling_scale_errors={(0, 1): 0.1, (1, 2): -0.05},
        global_rabi_error=0.02,
        timestamp=1234567890.0,
        metadata={"source": "test"},
    )
    restored = DeviceDrift.from_dict(drift.to_dict())
    assert restored.site_detuning_offsets == drift.site_detuning_offsets
    assert restored.coupling_scale_errors == drift.coupling_scale_errors
    assert restored.global_rabi_error == pytest.approx(drift.global_rabi_error)
    assert restored.timestamp == pytest.approx(drift.timestamp)
    assert restored.metadata == drift.metadata


# ---------------------------------------------------------------------------
# Test 1: identity returns zero-drift model
# ---------------------------------------------------------------------------

def test_identity_zero_drift():
    model = HardwareDeltaModel.identity("dev-0", SubstrateType.NEUTRAL_ATOM, 4)
    assert model.device_id == "dev-0"
    assert model.substrate == SubstrateType.NEUTRAL_ATOM
    assert model.n_sites == 4
    assert model.drift.site_detuning_offsets == {}
    assert model.drift.coupling_scale_errors == {}
    assert model.drift.global_rabi_error == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 2: apply_detuning_correction subtracts offsets
# ---------------------------------------------------------------------------

def test_apply_detuning_correction():
    drift = DeviceDrift(site_detuning_offsets={0: 0.5, 1: -0.3})
    model = HardwareDeltaModel(
        device_id="dev-1",
        substrate=SubstrateType.NEUTRAL_ATOM,
        drift=drift,
        n_sites=2,
    )
    corrected = model.apply_detuning_correction([1.0, 2.0])
    assert corrected[0] == pytest.approx(0.5)   # 1.0 - 0.5
    assert corrected[1] == pytest.approx(2.3)   # 2.0 - (-0.3)


# ---------------------------------------------------------------------------
# Test 3: apply_coupling_correction divides by (1 + error)
# ---------------------------------------------------------------------------

def test_apply_coupling_correction():
    drift = DeviceDrift(coupling_scale_errors={(0, 1): 0.1})
    model = HardwareDeltaModel(
        device_id="dev-2",
        substrate=SubstrateType.NEUTRAL_ATOM,
        drift=drift,
        n_sites=2,
    )
    corrected = model.apply_coupling_correction({(0, 1): 1.1})
    assert corrected[(0, 1)] == pytest.approx(1.0)  # 1.1 / 1.1


# ---------------------------------------------------------------------------
# Test 4: detuning correction on site with no offset entry is unchanged
# ---------------------------------------------------------------------------

def test_apply_detuning_correction_missing_site():
    model = HardwareDeltaModel.identity("dev-3", SubstrateType.PHOTONIC, 3)
    corrected = model.apply_detuning_correction([1.0, 2.0, 3.0])
    assert corrected == pytest.approx([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Test 5: coupling correction on pair with no error entry is unchanged
# ---------------------------------------------------------------------------

def test_apply_coupling_correction_missing_pair():
    model = HardwareDeltaModel.identity("dev-4", SubstrateType.PHOTONIC, 2)
    corrected = model.apply_coupling_correction({(0, 1): 2.5})
    assert corrected[(0, 1)] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Test 6: HardwareDeltaModel roundtrip
# ---------------------------------------------------------------------------

def test_hardware_delta_model_roundtrip():
    drift = DeviceDrift(
        site_detuning_offsets={0: 0.1},
        coupling_scale_errors={(0, 1): 0.05},
        global_rabi_error=0.01,
        timestamp=9999.0,
    )
    model = HardwareDeltaModel(
        device_id="dev-5",
        substrate=SubstrateType.BEC,
        drift=drift,
        n_sites=5,
        metadata={"cal_run": 42},
    )
    restored = HardwareDeltaModel.from_dict(model.to_dict())
    assert restored.device_id == model.device_id
    assert restored.substrate == model.substrate
    assert restored.n_sites == model.n_sites
    assert restored.metadata == model.metadata
    assert restored.drift.site_detuning_offsets == drift.site_detuning_offsets
    assert restored.drift.coupling_scale_errors == drift.coupling_scale_errors
    assert restored.drift.global_rabi_error == pytest.approx(drift.global_rabi_error)


# ---------------------------------------------------------------------------
# Test 7: registry get returns None for unknown device
# ---------------------------------------------------------------------------

def test_registry_get_unknown():
    registry = DeltaModelRegistry()
    assert registry.get("no-such-device") is None


# ---------------------------------------------------------------------------
# Test 8: registry get_or_identity returns identity for unknown device
# ---------------------------------------------------------------------------

def test_registry_get_or_identity_unknown():
    registry = DeltaModelRegistry()
    model = registry.get_or_identity("ghost", SubstrateType.NEUTRAL_ATOM, 6)
    assert model.device_id == "ghost"
    assert model.drift.site_detuning_offsets == {}
    assert model.n_sites == 6


# ---------------------------------------------------------------------------
# Test 9: registry register and get roundtrip
# ---------------------------------------------------------------------------

def test_registry_register_and_get():
    registry = DeltaModelRegistry()
    model = HardwareDeltaModel.identity("dev-reg", SubstrateType.PHOTONIC, 3)
    registry.register(model)
    retrieved = registry.get("dev-reg")
    assert retrieved is model


# ---------------------------------------------------------------------------
# Test 10: list_devices returns sorted list
# ---------------------------------------------------------------------------

def test_registry_list_devices_sorted():
    registry = DeltaModelRegistry()
    for dev_id in ("zeta", "alpha", "mu"):
        registry.register(HardwareDeltaModel.identity(dev_id, SubstrateType.BEC, 2))
    assert registry.list_devices() == ["alpha", "mu", "zeta"]


# ---------------------------------------------------------------------------
# Test 11: apply_detuning_correction matches pure Python fallback
# ---------------------------------------------------------------------------

def test_apply_detuning_correction_matches_fallback():
    drift = DeviceDrift(site_detuning_offsets={0: 0.5, 1: -0.3, 2: 0.0})
    model = HardwareDeltaModel(
        device_id="dev-rust-det",
        substrate=SubstrateType.NEUTRAL_ATOM,
        drift=drift,
        n_sites=3,
    )
    detunings = [1.0, 2.0, 3.0]
    rust_result = model.apply_detuning_correction(detunings)
    fallback = [
        d - drift.site_detuning_offsets.get(i, 0.0)
        for i, d in enumerate(detunings)
    ]
    assert rust_result == pytest.approx(fallback)


# ---------------------------------------------------------------------------
# Test 12: apply_coupling_correction matches pure Python fallback
# ---------------------------------------------------------------------------

def test_apply_coupling_correction_matches_fallback():
    drift = DeviceDrift(coupling_scale_errors={(0, 1): 0.1, (1, 2): -0.05})
    model = HardwareDeltaModel(
        device_id="dev-rust-coup",
        substrate=SubstrateType.NEUTRAL_ATOM,
        drift=drift,
        n_sites=3,
    )
    couplings = {(0, 1): 1.1, (1, 2): 2.0, (0, 2): 0.5}
    rust_result = model.apply_coupling_correction(couplings)
    fallback = {
        key: J / max(1.0 + drift.coupling_scale_errors.get(key, 0.0), 0.01)
        for key, J in couplings.items()
    }
    for key in couplings:
        assert rust_result[key] == pytest.approx(fallback[key])
