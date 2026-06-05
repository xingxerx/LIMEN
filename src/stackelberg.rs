use pyo3::prelude::*;
use crate::scoring::{self, EquilibriumScore};

/// Stackelberg co-design solver.
///
/// Scores a history of validation runs and recommends a chain strength
/// adjustment to drive the encoding toward a target calibration margin κ.
#[pyclass]
pub struct StackelbergSolver {
    /// Convergence target for κ.
    #[pyo3(get)]
    pub target_kappa: f64,
    /// Maximum number of iterations before stopping.
    #[pyo3(get)]
    pub max_iterations: usize,
    /// Multiplicative step size for chain-strength adjustment.
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
    /// If the best κ already meets target_kappa the chain strength is returned
    /// unchanged. Otherwise it is scaled up by `learning_rate * (1 - best_kappa)`.
    pub fn solve(
        &self,
        confidences: Vec<f64>,
        best_energies: Vec<f64>,
        second_best_energies: Vec<f64>,
        chain_break_fractions: Vec<f64>,
        current_chain_strength: f64,
    ) -> (f64, EquilibriumScore) {
        let n = confidences.len();

        let mut best_score = scoring::compute(
            *confidences.first().unwrap_or(&0.0),
            *best_energies.first().unwrap_or(&0.0),
            *second_best_energies.first().unwrap_or(&0.0),
            *chain_break_fractions.first().unwrap_or(&0.0),
            0,
        );

        for i in 1..n {
            let score = scoring::compute(
                confidences[i],
                best_energies[i],
                second_best_energies[i],
                chain_break_fractions[i],
                i,
            );
            if score.kappa > best_score.kappa {
                best_score = score;
            }
        }

        let recommended = if best_score.kappa >= self.target_kappa {
            current_chain_strength
        } else {
            let adjustment = self.learning_rate * (1.0 - best_score.kappa);
            current_chain_strength * (1.0 + adjustment)
        };

        (recommended, best_score)
    }
}
