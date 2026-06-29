use pyo3::prelude::*;
use std::collections::HashMap;

/// Equilibrium score for a single Stackelberg iteration.
#[pyclass]
#[derive(Clone)]
pub struct EquilibriumScore {
    /// Calibration margin κ ∈ [0.0, 1.0]. Higher is more trustworthy.
    #[pyo3(get)]
    pub kappa: f64,
    /// Raw confidence from the validator.
    #[pyo3(get)]
    pub confidence: f64,
    /// Absolute energy gap between best and second-best solution.
    #[pyo3(get)]
    pub energy_gap: f64,
    /// Number of solver iterations taken to reach this score.
    #[pyo3(get)]
    pub iterations: usize,
    /// Standard deviation of κ across all scored iterations (0.0 if < 2).
    #[pyo3(get)]
    pub kappa_std: f64,
}

#[pymethods]
impl EquilibriumScore {
    fn __repr__(&self) -> String {
        format!(
            "EquilibriumScore(kappa={:.4}, confidence={:.4}, energy_gap={:.4}, kappa_std={:.4}, iterations={})",
            self.kappa, self.confidence, self.energy_gap, self.kappa_std, self.iterations
        )
    }
}

/// Compute the standard deviation of a slice of κ values.
///
/// Returns 0.0 if fewer than 2 observations are provided.
pub fn compute_stability(scores: &[f64]) -> f64 {
    let n = scores.len();
    if n < 2 {
        return 0.0;
    }
    let mean = scores.iter().sum::<f64>() / n as f64;
    let variance = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    variance.sqrt()
}

/// Compute an EquilibriumScore from raw validator statistics.
///
/// kappa = 0.5 * confidence + 0.3 * gap_term + 0.2 * cbf_penalty
/// where:
///   gap_term    = |second_best_energy - best_energy|.min(10.0) / 10.0
///   cbf_penalty = 1.0 - chain_break_fraction.clamp(0.0, 1.0)
pub fn compute(
    confidence: f64,
    best_energy: f64,
    second_best_energy: f64,
    chain_break_fraction: f64,
    iterations: usize,
    kappa_std: f64,
) -> EquilibriumScore {
    let gap_term = (second_best_energy - best_energy).abs().min(10.0) / 10.0;
    let cbf_penalty = 1.0 - chain_break_fraction.clamp(0.0, 1.0);
    let kappa = (0.5 * confidence + 0.3 * gap_term + 0.2 * cbf_penalty).clamp(0.0, 1.0);
    let energy_gap = (second_best_energy - best_energy).abs();

    EquilibriumScore {
        kappa,
        confidence,
        energy_gap,
        iterations,
        kappa_std,
    }
}

/// Rank QUBO variables by criticality for QEC patch allocation.
///
/// Criticality is the sum of absolute penalty weights of every term
/// touching a variable (its own linear bias plus all quadratic couplings
/// it participates in). Higher criticality means the variable's value is
/// more consequential to the objective, and is therefore a better
/// candidate for error-corrected protection under a limited physical
/// qubit budget.
///
/// # Arguments
/// * `linear` - `(var_index, bias)` pairs.
/// * `quadratic` - `(var_i, var_j, coupling)` pairs.
///
/// Returns `(var_index, criticality)` pairs sorted by descending
/// criticality.
#[pyfunction]
pub fn qubo_criticality(
    linear: Vec<(usize, f64)>,
    quadratic: Vec<(usize, usize, f64)>,
) -> Vec<(usize, f64)> {
    let mut weights: HashMap<usize, f64> = HashMap::new();
    for (i, bias) in linear {
        *weights.entry(i).or_insert(0.0) += bias.abs();
    }
    for (i, j, coupling) in quadratic {
        let w = coupling.abs();
        *weights.entry(i).or_insert(0.0) += w;
        *weights.entry(j).or_insert(0.0) += w;
    }
    let mut ranked: Vec<(usize, f64)> = weights.into_iter().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    ranked
}
