# Copyright (C) 2026 xingxerx / CGX
#
# Licensed under the Elastic License 2.0 (ELv2); you may not use this file
# except in compliance with the License. See the LICENSE file in the
# repository root for the full terms.

"""Thin wrapper isolating the limen_core (Rust) import in one place.

The real reconstruction logic lives in limen_core::cutting::reconstruct_expectation
(src/cutting/reconstruct.rs); this module only converts CutDispatchResult's
plain Python data into the pyo3 types it expects.
"""

from __future__ import annotations

from limen.cutting.dispatch import CutDispatchResult

_INSTALL_MSG = (
    "limen_core (the Rust extension) is required for cut-circuit "
    "reconstruction. Build it with: maturin develop"
)


def reconstruct_from_results(result: CutDispatchResult) -> float:
    """Reconstruct the original circuit's expectation value from real counts.

    Args:
        result: A CutDispatchResult with real per-sample, per-subcircuit
            measurement counts and their real QPD coefficients.

    Returns:
        The reconstructed expectation value of the original (uncut) circuit.

    Raises:
        ImportError: If limen_core is not built (run `maturin develop`).
        ValueError: If result has missing or duplicate (sample, subcircuit)
            data -- propagated from limen_core::cutting::reconstruct_expectation.
    """
    try:
        from limen import limen_core
    except ModuleNotFoundError as exc:
        raise ImportError(_INSTALL_MSG) from exc

    counts = [
        limen_core.cutting.SubcircuitSampleCounts(
            sample_index, label, counts_dict, shots
        )
        for sample_index, label, counts_dict, shots in result.counts
    ]
    coefficients = [
        limen_core.cutting.SampleCoefficient(sample_index, coefficient)
        for sample_index, coefficient in result.coefficients
    ]
    return limen_core.cutting.reconstruct_expectation(
        counts, coefficients, result.subcircuit_labels
    )
