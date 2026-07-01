// Copyright (C) 2026 xingxerx / CGX
//
// Licensed under the Elastic License 2.0 (ELv2); you may not use this file
// except in compliance with the License. See the LICENSE file in the
// repository root for the full terms.

use pyo3::prelude::*;
use std::collections::HashMap;

/// Apply per-site detuning corrections to a list of detuning values.
///
/// For each index `i` in `detunings`, subtracts the corresponding offset
/// from `offsets` if one is present. Returns a new Vec; pure with no side
/// effects.
///
/// # Arguments
/// * `detunings` - Target detuning values, one per site.
/// * `offsets`   - List of `(site_index, offset_mhz)` pairs recording
///                 measured hardware detuning errors.
#[pyfunction]
pub fn apply_detuning_correction(
    detunings: Vec<f64>,
    offsets: Vec<(usize, f64)>,
) -> Vec<f64> {
    let offset_map: HashMap<usize, f64> = offsets.into_iter().collect();
    detunings
        .into_iter()
        .enumerate()
        .map(|(i, d)| d - offset_map.get(&i).copied().unwrap_or(0.0))
        .collect()
}

/// Apply per-pair coupling scale corrections to a list of couplings.
///
/// For each `(key, J)` in `couplings`, divides `J` by `(1.0 + error)` where
/// `error` is the matching entry in `errors` for that key (0.0 if absent).
/// The denominator is clamped to a minimum of 0.01 to prevent division by
/// zero. Returns a new Vec; pure with no side effects.
///
/// # Arguments
/// * `couplings` - List of `((site_i, site_j), J)` pairs.
/// * `errors`    - List of `((site_i, site_j), fractional_error)` pairs.
#[pyfunction]
pub fn apply_coupling_correction(
    couplings: Vec<((usize, usize), f64)>,
    errors: Vec<((usize, usize), f64)>,
) -> Vec<((usize, usize), f64)> {
    let error_map: HashMap<(usize, usize), f64> = errors.into_iter().collect();
    couplings
        .into_iter()
        .map(|(key, j)| {
            let error = error_map.get(&key).copied().unwrap_or(0.0);
            let denom = (1.0 + error).max(0.01);
            (key, j / denom)
        })
        .collect()
}

/// Apply a global Rabi-drive scale correction to a target Rabi frequency.
///
/// Pre-distorts the requested Rabi frequency so that the as-executed drive
/// (which hardware multiplies by `(1.0 + global_rabi_error)`) lands on the
/// target value: `corrected = target / (1.0 + global_rabi_error)`. The
/// denominator is clamped to a minimum of 0.01 to prevent division by zero,
/// mirroring `apply_coupling_correction`.
///
/// # Arguments
/// * `rabi_frequency`   - Target global Rabi frequency (e.g. MHz).
/// * `global_rabi_error` - Measured fractional error on the drive (0.0 = perfect).
#[pyfunction]
pub fn apply_rabi_correction(rabi_frequency: f64, global_rabi_error: f64) -> f64 {
    let denom = (1.0 + global_rabi_error).max(0.01);
    rabi_frequency / denom
}
