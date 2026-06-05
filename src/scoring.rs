use pyo3::prelude::*;

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
}

#[pymethods]
impl EquilibriumScore {
    fn __repr__(&self) -> String {
        format!(
            "EquilibriumScore(kappa={:.4}, confidence={:.4}, energy_gap={:.4}, iterations={})",
            self.kappa, self.confidence, self.energy_gap, self.iterations
        )
    }
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
    }
}
