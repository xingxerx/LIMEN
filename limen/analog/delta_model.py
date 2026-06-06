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
"""Live calibration abstraction layer for analog hardware.

Captures per-device drift, coupling errors, and detuning offsets so that
analog backend compilers can pre-distort their outputs to cancel hardware
imperfections. This is the engineering prerequisite for the
HardwareDeltaModel parameter described in the AnalogTargetAdapter interface
specification in limen/docs/architecture.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from limen.analog.hamiltonian import SubstrateType


@dataclass
class DeviceDrift:
    """Measured drift profile for a single analog device.

    Attributes:
        site_detuning_offsets: Per-site detuning error in MHz.
            Keys are site indices; missing keys imply zero error.
        coupling_scale_errors: Fractional error on each pairwise coupling.
            0.0 means perfect; 0.1 means 10% over-coupling.
            Keys are (site_i, site_j) tuples; missing keys imply zero error.
        global_rabi_error: Fractional error on the global Rabi drive.
        timestamp: Unix timestamp of when this drift snapshot was measured.
        metadata: Arbitrary annotations.
    """

    site_detuning_offsets: dict[int, float] = field(default_factory=dict)
    coupling_scale_errors: dict[tuple[int, int], float] = field(default_factory=dict)
    global_rabi_error: float = 0.0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe plain Python dict.

        Tuple keys in coupling_scale_errors are stored as two-element lists.
        """
        return {
            "site_detuning_offsets": {str(k): v for k, v in self.site_detuning_offsets.items()},
            "coupling_scale_errors": {
                str(list(k)): v for k, v in self.coupling_scale_errors.items()
            },
            "global_rabi_error": self.global_rabi_error,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DeviceDrift:
        """Deserialize from a plain Python dict produced by to_dict().

        Coupling keys stored as two-element lists are restored as tuples.
        """
        import ast

        site_detuning_offsets = {
            int(k): float(v) for k, v in d.get("site_detuning_offsets", {}).items()
        }
        coupling_scale_errors: dict[tuple[int, int], float] = {}
        for raw_key, v in d.get("coupling_scale_errors", {}).items():
            pair = ast.literal_eval(raw_key)
            coupling_scale_errors[(int(pair[0]), int(pair[1]))] = float(v)
        return cls(
            site_detuning_offsets=site_detuning_offsets,
            coupling_scale_errors=coupling_scale_errors,
            global_rabi_error=float(d.get("global_rabi_error", 0.0)),
            timestamp=float(d.get("timestamp", 0.0)),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class HardwareDeltaModel:
    """Calibration model that maps hardware imperfections to correction vectors.

    Analog backend compilers use this model to pre-distort detunings and
    couplings before submission so that the as-executed Hamiltonian matches
    the intended one.

    Attributes:
        device_id: Unique identifier for the physical device.
        substrate: Which substrate type this model covers.
        drift: The current measured drift profile.
        n_sites: Number of sites this model covers.
        metadata: Arbitrary annotations.
    """

    device_id: str
    substrate: SubstrateType
    drift: DeviceDrift
    n_sites: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply_detuning_correction(self, detunings: list[float]) -> list[float]:
        """Return corrected detunings with hardware offsets subtracted.

        Delegates to the limen_core Rust extension when available; falls back
        to a pure Python implementation otherwise.

        For each site i, the corrected value is detunings[i] minus
        drift.site_detuning_offsets.get(i, 0.0). Sites with no recorded
        offset are returned unchanged.

        Args:
            detunings: List of target detuning values (one per site).

        Returns:
            New list of pre-distorted detunings.
        """
        try:
            from limen_core import apply_detuning_correction as _rust_fn
            offsets = list(self.drift.site_detuning_offsets.items())
            return _rust_fn(detunings, offsets)
        except ImportError:
            # Pure Python fallback for environments without the Rust extension.
            return [
                d - self.drift.site_detuning_offsets.get(i, 0.0)
                for i, d in enumerate(detunings)
            ]

    def apply_coupling_correction(
        self, couplings: dict[tuple[int, int], float]
    ) -> dict[tuple[int, int], float]:
        """Return corrected couplings with scale errors divided out.

        Delegates to the limen_core Rust extension when available; falls back
        to a pure Python implementation otherwise.

        For each pair key, the corrected value is J / (1 + error), where
        error = drift.coupling_scale_errors.get(key, 0.0). The denominator
        is clamped to a minimum of 0.01 to prevent division by zero.

        Args:
            couplings: Dict mapping (site_i, site_j) pairs to coupling strengths.

        Returns:
            New dict of pre-distorted couplings.
        """
        try:
            from limen_core import apply_coupling_correction as _rust_fn
            errors = list(self.drift.coupling_scale_errors.items())
            result_list = _rust_fn(list(couplings.items()), errors)
            return dict(result_list)
        except ImportError:
            # Pure Python fallback.
            return {
                key: J / max(1.0 + self.drift.coupling_scale_errors.get(key, 0.0), 0.01)
                for key, J in couplings.items()
            }

    @classmethod
    def identity(
        cls, device_id: str, substrate: SubstrateType, n_sites: int
    ) -> HardwareDeltaModel:
        """Return a zero-drift model suitable for uncalibrated devices.

        All offset and error dicts are empty; global_rabi_error is 0.0.
        This is the safe default when no calibration data is available.

        Args:
            device_id: Unique identifier for the physical device.
            substrate: Substrate type for this device.
            n_sites: Number of sites this model should cover.

        Returns:
            A HardwareDeltaModel with no corrections applied.
        """
        return cls(
            device_id=device_id,
            substrate=substrate,
            drift=DeviceDrift(),
            n_sites=n_sites,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe plain Python dict."""
        return {
            "device_id": self.device_id,
            "substrate": self.substrate.value,
            "drift": self.drift.to_dict(),
            "n_sites": self.n_sites,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HardwareDeltaModel:
        """Deserialize from a plain Python dict produced by to_dict()."""
        return cls(
            device_id=str(d["device_id"]),
            substrate=SubstrateType(d["substrate"]),
            drift=DeviceDrift.from_dict(d["drift"]),
            n_sites=int(d["n_sites"]),
            metadata=dict(d.get("metadata", {})),
        )


class DeltaModelRegistry:
    """Registry mapping device IDs to their HardwareDeltaModel instances.

    Provides safe lookup with identity-model fallback so callers never
    need to guard against missing calibration data.
    """

    def __init__(self) -> None:
        self._store: dict[str, HardwareDeltaModel] = {}

    def register(self, model: HardwareDeltaModel) -> None:
        """Store a HardwareDeltaModel, keyed by its device_id.

        Args:
            model: The calibration model to register.
        """
        self._store[model.device_id] = model

    def get(self, device_id: str) -> HardwareDeltaModel | None:
        """Return the registered model for device_id, or None if absent.

        Args:
            device_id: The device identifier to look up.

        Returns:
            The registered HardwareDeltaModel, or None.
        """
        return self._store.get(device_id)

    def get_or_identity(
        self, device_id: str, substrate: SubstrateType, n_sites: int
    ) -> HardwareDeltaModel:
        """Return the registered model, or a zero-drift identity model if absent.

        Args:
            device_id: The device identifier to look up.
            substrate: Substrate type used when constructing the identity model.
            n_sites: Site count used when constructing the identity model.

        Returns:
            A HardwareDeltaModel — registered or identity.
        """
        model = self._store.get(device_id)
        if model is not None:
            return model
        return HardwareDeltaModel.identity(device_id, substrate, n_sites)

    def list_devices(self) -> list[str]:
        """Return a sorted list of all registered device IDs."""
        return sorted(self._store)


default_registry = DeltaModelRegistry()
"""Module-level registry instance for convenience use across the codebase."""
