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
"""Calibration loaders for QuEra/Pasqal and IBMQ hardware formats."""

from __future__ import annotations


def _parse_source(source):
    import json
    from pathlib import Path

    if isinstance(source, dict):
        return source
    if isinstance(source, Path):
        return json.loads(source.read_text(encoding="utf-8"))
    if isinstance(source, str):
        try:
            return json.loads(source)
        except json.JSONDecodeError:
            return json.loads(Path(source).read_text(encoding="utf-8"))
    raise TypeError(f"source must be dict, str, or Path; got {type(source).__name__}")


def load_quera_calibration(source):
    """Load a QuEra/Pasqal calibration file and return a HardwareDeltaModel.

    Args:
        source: dict, JSON string, str path, or Path object.

    Returns:
        HardwareDeltaModel populated from the calibration data.
    """
    import ast

    from limen.analog.delta_model import DeviceDrift, HardwareDeltaModel
    from limen.analog.hamiltonian import SubstrateType

    data = _parse_source(source)

    device_id = str(data.get("device_id", "unknown"))
    n_sites = int(data.get("n_sites", 0))

    substrate_str = data.get("substrate", "neutral_atom")
    try:
        substrate = SubstrateType(substrate_str)
    except ValueError:
        substrate = SubstrateType.NEUTRAL_ATOM

    timestamp = float(data.get("calibration_timestamp", 0.0))

    raw_offsets = data.get("site_detuning_offsets_mhz", {})
    site_detuning_offsets = {int(k): float(v) for k, v in raw_offsets.items()}

    raw_coupling = data.get("coupling_scale_errors", {})
    coupling_scale_errors: dict = {}
    for key_str, val in raw_coupling.items():
        key_str = key_str.strip()
        if key_str.startswith("["):
            pair = ast.literal_eval(key_str)
        else:
            parts = key_str.split(",")
            pair = (int(parts[0].strip()), int(parts[1].strip()))
        coupling_scale_errors[(int(pair[0]), int(pair[1]))] = float(val)

    global_rabi_error = float(data.get("global_rabi_error", 0.0))

    metadata: dict = {}
    if "metadata" in data:
        metadata.update(data["metadata"])
    metadata["loader"] = "load_quera_calibration"

    drift = DeviceDrift(
        site_detuning_offsets=site_detuning_offsets,
        coupling_scale_errors=coupling_scale_errors,
        global_rabi_error=global_rabi_error,
        timestamp=timestamp,
    )

    return HardwareDeltaModel(
        device_id=device_id,
        substrate=substrate,
        drift=drift,
        n_sites=n_sites,
        metadata=metadata,
    )


def load_ibmq_calibration(source, freq_error_to_detuning_mhz=1000.0):
    """Load an IBMQ backend properties file and return a HardwareDeltaModel.

    Args:
        source: dict, JSON string, str path, or Path object.
        freq_error_to_detuning_mhz: Conversion factor from GHz frequency error
            to MHz detuning offset.

    Returns:
        HardwareDeltaModel populated from the backend properties.
    """
    from datetime import datetime

    from limen.analog.delta_model import DeviceDrift, HardwareDeltaModel
    from limen.analog.hamiltonian import SubstrateType

    data = _parse_source(source)

    device_id = str(data["backend_name"])
    n_sites = int(data["n_qubits"])
    substrate = SubstrateType.UNSPECIFIED

    try:
        timestamp = datetime.fromisoformat(data["properties_timestamp"]).timestamp()
    except (KeyError, ValueError):
        timestamp = 0.0

    site_detuning_offsets: dict = {}
    for i, qubit_props in enumerate(data.get("qubits", [])):
        for prop in qubit_props:
            if prop.get("name") == "frequency_error":
                site_detuning_offsets[i] = float(prop["value"]) * freq_error_to_detuning_mhz
                break

    coupling_scale_errors: dict = {}
    for gate in data.get("gates", []):
        qubits = gate.get("qubits", [])
        if len(qubits) == 2:
            for param in gate.get("parameters", []):
                if param.get("name") == "gate_error":
                    q0, q1 = int(qubits[0]), int(qubits[1])
                    coupling_scale_errors[(q0, q1)] = float(param["value"])
                    break

    metadata = {"loader": "load_ibmq_calibration"}

    drift = DeviceDrift(
        site_detuning_offsets=site_detuning_offsets,
        coupling_scale_errors=coupling_scale_errors,
        timestamp=timestamp,
    )

    return HardwareDeltaModel(
        device_id=device_id,
        substrate=substrate,
        drift=drift,
        n_sites=n_sites,
        metadata=metadata,
    )
