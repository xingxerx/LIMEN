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

from typing import Any

from limen.cutting.dispatch import CutDispatchResult

_INSTALL_MSG = (
    "limen_core (the Rust extension) is required for cut-circuit "
    "reconstruction. Build it with: maturin develop"
)


def reconstruct_from_results(result: CutDispatchResult) -> float:
    """Reconstruct the original circuit's expectation value from real counts.

    When *result* carries the sampler's raw PrimitiveResults and the plan's
    sub-observables (every dispatcher in this package sets them), the
    reconstruction is qiskit-addon-cutting's own
    ``reconstruct_expectation_values``, which applies each sub-observable's
    Pauli bitmask to the ``observable_measurements`` register. The Rust
    kernel below instead takes the parity of the whole concatenated
    register; that equals the reference only when every measured
    observable bit belongs to the sub-observable, which wire-cut plans
    violate. Measured on an 8-qubit random-field ring cut into 4-qubit
    pieces (3 wire cuts): exact <Z_0> = +0.836, reference +0.837, parity
    kernel +0.62 / -0.01 across two runs. The kernel remains the fallback
    for results rebuilt from plain counts.

    Args:
        result: A CutDispatchResult from limen.cutting.dispatch or
            limen.cutting.local_dispatch.

    Returns:
        The reconstructed expectation value of the original (uncut) circuit.

    Raises:
        ImportError: If neither qiskit-addon-cutting (reference path) nor
            limen_core (fallback) is available.
        ValueError: If result has missing or duplicate (sample, subcircuit)
            data -- propagated from limen_core::cutting::reconstruct_expectation.
    """
    if result.raw_results and result.subobservables:
        return _reconstruct_reference(result)

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


def _reconstruct_reference(result: CutDispatchResult) -> float:
    """qiskit-addon-cutting's reconstruction, computed from counts.

    Same arithmetic as ``qiskit_addon_cutting.reconstruct_expectation_values``
    (``_process_outcome_v2``: ``(-1)^popcount(qpd) * (-1)^popcount(obs & mask)``
    per shot, averaged per (sample, partition), multiplied across
    partitions, summed with the QPD coefficients), but driven from each
    pub's integer counts instead of a Python loop over every shot. The
    library loop costs ~1 us per shot per sample; at 512 QPD samples x
    4000 shots x 3 partitions x n marginals that is minutes per marginal,
    which is why this exists.
    """
    from qiskit_addon_cutting.utils.observable_grouping import ObservableCollection

    coeffs = [c for _, c in sorted(result.coefficients)]
    n_samples = len(coeffs)
    per_label_expvals: dict[str, list[float]] = {}
    for label, pub_results in result.raw_results.items():
        observables = result.subobservables[label]
        so = ObservableCollection(observables)
        if len(so.groups) != 1 or len(observables) != 1:
            raise ValueError(
                f"partition {label!r}: expected one observable in one commuting group, "
                f"got {len(observables)} observable(s) in {len(so.groups)} group(s)"
            )
        mask = so.groups[0].pauli_bitmasks[0]
        if len(pub_results) != n_samples:
            raise ValueError(
                f"partition {label!r} has {len(pub_results)} sub-experiment results for "
                f"{n_samples} QPD coefficients"
            )
        expvals: list[float] = []
        for pub in pub_results:
            data = pub.data
            obs_ints = _bitarray_ints(data.observable_measurements)
            qpd_ints = _bitarray_ints(data.qpd_measurements)
            shots = obs_ints.shape[0]
            # (-1)^popcount(qpd) * (-1)^popcount(obs & mask), as integers
            parity = _popcount(qpd_ints) + _popcount(obs_ints & mask)
            signs = 1.0 - 2.0 * (parity & 1)
            expvals.append(float(signs.sum()) / shots)
        per_label_expvals[label] = expvals

    labels = list(per_label_expvals)
    value = 0.0
    for i, coeff in enumerate(coeffs):
        prod = 1.0
        for label in labels:
            prod *= per_label_expvals[label][i]
        value += coeff * prod
    return float(value)


def _bitarray_ints(bits: Any) -> Any:
    """One big-endian integer per shot from a qiskit BitArray (same byte
    order as ``int.from_bytes(array[j], "big")`` in the reference)."""
    import numpy as np

    arr = np.asarray(bits.array, dtype=np.uint64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    arr = arr.reshape(-1, arr.shape[-1])  # (shots, nbytes)
    if arr.shape[1] == 0:
        return np.zeros(arr.shape[0], dtype=np.uint64)
    weights = (np.uint64(256) ** np.arange(arr.shape[1], dtype=np.uint64)[::-1]).astype(np.uint64)
    return (arr * weights).sum(axis=1).astype(np.uint64)


def _popcount(values: Any) -> Any:
    import numpy as np

    v = values.astype(np.uint64)
    out = np.zeros_like(v)
    while v.any():
        out += v & np.uint64(1)
        v >>= np.uint64(1)
    return out
