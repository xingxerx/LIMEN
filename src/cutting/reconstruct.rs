// Copyright (C) 2026 xingxerx / CGX
//
// Licensed under the Elastic License 2.0 (ELv2); you may not use this file
// except in compliance with the License. See the LICENSE file in the
// repository root for the full terms.

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// One real measurement-count histogram from one sub-circuit run on real
/// hardware, for one joint QPD sample.
///
/// `counts` maps a bitstring (the sub-circuit's classical registers,
/// `observable_measurements` concatenated with `qpd_measurements`,
/// concatenation order does not matter) to its real shot count. Per-shot
/// parity of the *combined* register reproduces
/// `qpd_factor * obs_outcome` from qiskit-addon-cutting's
/// `reconstruct_expectation_values`, since
/// `(-1)^popcount(qpd) * (-1)^popcount(obs) == (-1)^popcount(qpd ++ obs)`.
#[pyclass]
#[derive(Clone)]
pub struct SubcircuitSampleCounts {
    #[pyo3(get)]
    pub sample_index: usize,
    #[pyo3(get)]
    pub subcircuit_label: String,
    #[pyo3(get)]
    pub counts: HashMap<String, u64>,
    #[pyo3(get)]
    pub shots: u64,
}

#[pymethods]
impl SubcircuitSampleCounts {
    #[new]
    fn new(
        sample_index: usize,
        subcircuit_label: String,
        counts: HashMap<String, u64>,
        shots: u64,
    ) -> Self {
        Self {
            sample_index,
            subcircuit_label,
            counts,
            shots,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SubcircuitSampleCounts(sample_index={}, subcircuit_label={:?}, shots={})",
            self.sample_index, self.subcircuit_label, self.shots
        )
    }
}

/// The real-valued coefficient for one joint QPD sample, as produced by
/// `qiskit_addon_cutting.generate_cutting_experiments` (the first element
/// of each `(coefficient, WeightType)` tuple it returns).
#[pyclass]
#[derive(Clone)]
pub struct SampleCoefficient {
    #[pyo3(get)]
    pub sample_index: usize,
    #[pyo3(get)]
    pub coefficient: f64,
}

#[pymethods]
impl SampleCoefficient {
    #[new]
    fn new(sample_index: usize, coefficient: f64) -> Self {
        Self {
            sample_index,
            coefficient,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SampleCoefficient(sample_index={}, coefficient={})",
            self.sample_index, self.coefficient
        )
    }
}

fn empirical_expectation(entry: &SubcircuitSampleCounts) -> PyResult<f64> {
    if entry.shots == 0 {
        return Err(PyValueError::new_err(format!(
            "SubcircuitSampleCounts for sample {} / subcircuit {:?} has shots == 0",
            entry.sample_index, entry.subcircuit_label
        )));
    }
    let mut acc = 0f64;
    for (bitstring, count) in &entry.counts {
        let ones = bitstring.chars().filter(|c| *c == '1').count();
        let parity = if ones % 2 == 0 { 1.0 } else { -1.0 };
        acc += parity * (*count as f64);
    }
    Ok(acc / entry.shots as f64)
}

/// Reconstruct the original circuit's expectation value from real
/// per-sub-circuit, per-sample measurement counts and their real QPD
/// sample coefficients, following the cutting identity used by
/// `qiskit_addon_cutting.reconstruct_expectation_values`:
///
/// ⟨O⟩ = Σ_sample  coefficient[sample] · Π_subcircuit ⟨O⟩_{subcircuit,sample}
///
/// `subcircuit_labels` must list every sub-circuit label that participates
/// in the cut circuit. For every `(coefficient.sample_index, label)` pair
/// implied by `coefficients × subcircuit_labels`, there must be exactly one
/// matching `SubcircuitSampleCounts` entry in `counts` -- missing or
/// duplicate entries raise `ValueError` rather than silently contributing
/// zero. The outer sum over samples is parallelized with rayon.
#[pyfunction]
pub fn reconstruct_expectation(
    counts: Vec<SubcircuitSampleCounts>,
    coefficients: Vec<SampleCoefficient>,
    subcircuit_labels: Vec<String>,
) -> PyResult<f64> {
    if subcircuit_labels.is_empty() {
        return Err(PyValueError::new_err(
            "subcircuit_labels must not be empty",
        ));
    }

    let mut index: HashMap<(usize, &str), &SubcircuitSampleCounts> = HashMap::new();
    for entry in &counts {
        let key = (entry.sample_index, entry.subcircuit_label.as_str());
        if index.insert(key, entry).is_some() {
            return Err(PyValueError::new_err(format!(
                "duplicate SubcircuitSampleCounts for sample {} / subcircuit {:?}",
                entry.sample_index, entry.subcircuit_label
            )));
        }
    }

    // Validate every (sample, label) pair implied by the coefficients is present
    // before doing any parallel work, so missing data fails loudly up front.
    for coeff in &coefficients {
        for label in &subcircuit_labels {
            if !index.contains_key(&(coeff.sample_index, label.as_str())) {
                return Err(PyValueError::new_err(format!(
                    "missing SubcircuitSampleCounts for sample {} / subcircuit {:?}",
                    coeff.sample_index, label
                )));
            }
        }
    }

    coefficients
        .par_iter()
        .map(|coeff| -> PyResult<f64> {
            let mut product = 1.0f64;
            for label in &subcircuit_labels {
                let entry = index
                    .get(&(coeff.sample_index, label.as_str()))
                    .expect("presence already validated above");
                product *= empirical_expectation(entry)?;
            }
            Ok(coeff.coefficient * product)
        })
        .collect::<PyResult<Vec<f64>>>()
        .map(|terms| terms.into_iter().sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts_from(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
        pairs
            .iter()
            .map(|(bs, c)| (bs.to_string(), *c))
            .collect()
    }

    #[test]
    fn single_subcircuit_single_sample_is_plain_expectation() {
        // 70% '0' (parity +1), 30% '1' (parity -1) -> expectation 0.4
        let counts = vec![SubcircuitSampleCounts {
            sample_index: 0,
            subcircuit_label: "A".to_string(),
            counts: counts_from(&[("0", 700), ("1", 300)]),
            shots: 1000,
        }];
        let coefficients = vec![SampleCoefficient {
            sample_index: 0,
            coefficient: 1.0,
        }];
        let result =
            reconstruct_expectation(counts, coefficients, vec!["A".to_string()]).unwrap();
        assert!((result - 0.4).abs() < 1e-9);
    }

    #[test]
    fn two_subcircuits_product_matches_hand_computation() {
        // Subcircuit A: expectation 0.4 (as above)
        // Subcircuit B: 100% '0' -> expectation 1.0
        // coefficient 2.0 -> expected 2.0 * 0.4 * 1.0 = 0.8
        let counts = vec![
            SubcircuitSampleCounts {
                sample_index: 0,
                subcircuit_label: "A".to_string(),
                counts: counts_from(&[("0", 700), ("1", 300)]),
                shots: 1000,
            },
            SubcircuitSampleCounts {
                sample_index: 0,
                subcircuit_label: "B".to_string(),
                counts: counts_from(&[("0", 1000)]),
                shots: 1000,
            },
        ];
        let coefficients = vec![SampleCoefficient {
            sample_index: 0,
            coefficient: 2.0,
        }];
        let result = reconstruct_expectation(
            counts,
            coefficients,
            vec!["A".to_string(), "B".to_string()],
        )
        .unwrap();
        assert!((result - 0.8).abs() < 1e-9);
    }

    #[test]
    fn missing_subcircuit_data_errors_instead_of_defaulting_to_zero() {
        // Only subcircuit "A" has data, but the cut has two subcircuits.
        let counts = vec![SubcircuitSampleCounts {
            sample_index: 0,
            subcircuit_label: "A".to_string(),
            counts: counts_from(&[("0", 1000)]),
            shots: 1000,
        }];
        let coefficients = vec![SampleCoefficient {
            sample_index: 0,
            coefficient: 1.0,
        }];
        let result = reconstruct_expectation(
            counts,
            coefficients,
            vec!["A".to_string(), "B".to_string()],
        );
        assert!(result.is_err());
    }
}
