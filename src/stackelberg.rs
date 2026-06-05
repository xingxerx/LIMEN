use pyo3::prelude::*;
use crate::scoring::{self, EquilibriumScore};

/// Stackelberg co-design solver.
///
/// Scores a history of validation runs and recommends a chain strength
/// adjustment to drive the encoding toward a target calibration margin κ.
/// When κ is oscillating the effective learning rate is reduced proportionally
/// to the κ standard deviation, preventing overshooting.
#[pyclass]
pub struct StackelbergSolver {
    /// Convergence target for κ.
    #[pyo3(get)]
    pub target_kappa: f64,
    /// Maximum number of iterations before stopping.
    #[pyo3(get)]
    pub max_iterations: usize,
    /// Base multiplicative step size for chain-strength adjustment.
    #[pyo3(get)]
    pub learning_rate: f64,
}

#[pymethods]
impl StackelbergSolver {
    #[new]
    pub fn new(target_kappa: f64, max_iterations: usize, learning_rate: f64) -> Self {
        StackelbergSolver {
            target_kappa,
            max_iterations,
            learning_rate,
        }
    }

    /// Score each iteration and return (recommended_chain_strength, best_score).
    ///
    /// The effective learning rate is reduced when κ oscillates:
    ///   stability_penalty = (kappa_std * 5.0).clamp(0.0, 0.9)
    ///   effective_lr      = learning_rate * (1.0 - stability_penalty)
    ///
    /// If the best κ already meets target_kappa the chain strength is returned
    /// unchanged. Otherwise it is scaled up by `effective_lr * (1 - best_kappa)`.
    pub fn solve(
        &self,
        confidences: Vec<f64>,
        best_energies: Vec<f64>,
        second_best_energies: Vec<f64>,
        chain_break_fractions: Vec<f64>,
        current_chain_strength: f64,
    ) -> (f64, EquilibriumScore) {
        let n = confidences.len();

        // First pass: compute all per-iteration scores (kappa_std not yet known).
        let mut scores: Vec<EquilibriumScore> = (0..n)
            .map(|i| {
                scoring::compute(
                    confidences[i],
                    best_energies[i],
                    second_best_energies[i],
                    chain_break_fractions[i],
                    i,
                    0.0, // placeholder; overwritten below
                )
            })
            .collect();

        // Compute κ stability across all iterations.
        let kappa_values: Vec<f64> = scores.iter().map(|s| s.kappa).collect();
        let kappa_std = scoring::compute_stability(&kappa_values);

        // Find index of best kappa, then stamp kappa_std on all scores.
        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.kappa.partial_cmp(&b.1.kappa).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        for score in scores.iter_mut() {
            score.kappa_std = kappa_std;
        }

        let best_score = scores[best_idx].clone();

        // Stability-penalised learning rate.
        let stability_penalty = (kappa_std * 5.0).clamp(0.0, 0.9);
        let effective_lr = self.learning_rate * (1.0 - stability_penalty);

        let recommended = if best_score.kappa >= self.target_kappa {
            current_chain_strength
        } else {
            let adjustment = effective_lr * (1.0 - best_score.kappa);
            current_chain_strength * (1.0 + adjustment)
        };

        (recommended, best_score)
    }
}
