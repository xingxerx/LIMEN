// Copyright 2026 LIMEN Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
