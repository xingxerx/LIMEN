use pyo3::prelude::*;
use std::collections::HashMap;

/// Compute the exact operator-norm of an Ising error Hamiltonian.
///
/// Enumerates all 2^n spin configurations and returns
/// `max_{s ∈ {-1,+1}^n} |ΔE(s)|`, where
/// `ΔE(s) = Σ_i dh_i·s_i + Σ_{i<j} dJ_ij·s_i·s_j`.
///
/// This is the equality part of Theorem 1 in
/// `limen/docs/universality_theorem.md`. The Python fallback in
/// `limen.analog.certificate.certify_ising` calls this function when
/// `limen_core` is available and falls back to a pure-Python loop otherwise.
///
/// # Arguments
/// * `dh`      - Linear error coefficients as `[(site, Δh_i)]`.
/// * `dJ`      - Quadratic error coefficients as `[((site_i, site_j), ΔJ_ij)]`.
/// * `n_sites` - Number of sites. Must be ≤ 20.
///
/// # Errors
/// Returns `SizeViolation` when `n_sites > 20`.
#[pyfunction]
pub fn exact_ising_norm(
    dh: Vec<(usize, f64)>,
    dj: Vec<((usize, usize), f64)>,
    n_sites: usize,
) -> PyResult<f64> {
    if n_sites > 20 {
        return Err(crate::SizeViolation::new_err(
            "SizeViolation: exact_ising_norm requires n_sites <= 20",
        ));
    }
    if n_sites == 0 || (dh.is_empty() && dj.is_empty()) {
        return Ok(0.0);
    }

    let num_states: usize = 1 << n_sites;
    let mut max_abs: f64 = 0.0;

    for state_idx in 0..num_states {
        // Map bit i → spin +1 if bit set, −1 otherwise.
        let spins: Vec<f64> = (0..n_sites)
            .map(|i| if (state_idx >> i) & 1 == 1 { 1.0 } else { -1.0 })
            .collect();

        let mut energy: f64 = 0.0;
        for (i, v) in &dh {
            energy += v * spins[*i];
        }
        for ((i, j), v) in &dj {
            energy += v * spins[*i] * spins[*j];
        }

        let abs_e = energy.abs();
        if abs_e > max_abs {
            max_abs = abs_e;
        }
    }

    Ok(max_abs)
}

/// Exhaustively enumerate a QUBO's energy spectrum.
///
/// Enumerates all `2^n_vars` binary assignments and computes
/// `E(x) = Σ (i,j,w) w·x_i·x_j` for each (with `i == j` encoding a linear
/// term `w·x_i`, matching the `(var, var) -> weight` QUBO dict convention
/// used throughout `limen.validator` / `limen.codesign` / `limen.backends`).
///
/// This consolidates the O(2^n) brute-force enumeration that previously
/// existed independently in `limen/validator/validator.py::brute_force_solve`,
/// `limen/codesign/solver.py::_second_best_energy`, and
/// `limen/backends/qiskit_backend.py::_enumerate_assignments` into a single
/// Rust pass over the state space.
///
/// # Arguments
/// * `qubo`    - QUBO terms as `[((var_i, var_j), weight)]` using 0-based
///   variable indices. `i == j` encodes a linear term.
/// * `n_vars`  - Number of variables. Must be ≤ 20.
///
/// # Returns
/// A tuple `(best_assignment, best_energy, sorted_distinct_energies)` where
/// `best_assignment` is a `Vec<u8>` of 0/1 bits (index = variable index),
/// `best_energy` is the minimum energy found, and `sorted_distinct_energies`
/// is the ascending list of unique energies seen across all `2^n_vars`
/// assignments.
///
/// # Errors
/// Returns `SizeViolation` when `n_vars > 20`.
#[pyfunction]
pub fn qubo_energy_spectrum(
    qubo: Vec<((usize, usize), f64)>,
    n_vars: usize,
) -> PyResult<(Vec<u8>, f64, Vec<f64>)> {
    if n_vars > 20 {
        return Err(crate::SizeViolation::new_err(
            "SizeViolation: qubo_energy_spectrum requires n_vars <= 20",
        ));
    }
    if n_vars == 0 {
        return Ok((Vec::new(), 0.0, vec![0.0]));
    }

    let num_states: usize = 1 << n_vars;
    let mut best_energy = f64::INFINITY;
    let mut best_state = vec![0u8; n_vars];

    // Quantize energies to a fixed tolerance (1e-9) and dedup via a hash set
    // keyed on the quantized integer value. A linear Vec scan-per-state here
    // would be O(2^n * distinct_count) — quadratic blowup for n ~ 18-20 with
    // many distinct energies (minutes instead of milliseconds). The hash set
    // keeps the whole enumeration O(2^n).
    const TOL_INV: f64 = 1e9;
    let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();

    for state_idx in 0..num_states {
        let bits: Vec<u8> = (0..n_vars)
            .map(|i| ((state_idx >> i) & 1) as u8)
            .collect();

        let mut energy: f64 = 0.0;
        for ((i, j), w) in &qubo {
            energy += w * (bits[*i] as f64) * (bits[*j] as f64);
        }

        if energy < best_energy {
            best_energy = energy;
            best_state = bits;
        }

        seen.insert((energy * TOL_INV).round() as i64);
    }

    let mut distinct_energies: Vec<f64> = seen.into_iter().map(|q| q as f64 / TOL_INV).collect();
    distinct_energies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    Ok((best_state, best_energy, distinct_energies))
}

#[pyclass]
#[derive(Clone)]
pub struct Variable {
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub domain: String,
}

#[pymethods]
impl Variable {
    #[new]
    pub fn new(name: String, domain: String) -> Self {
        Variable { name, domain }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Interaction {
    #[pyo3(get, set)]
    pub i: String,
    #[pyo3(get, set)]
    pub j: String,
    #[pyo3(get, set)]
    pub weight: f64,
}

#[pymethods]
impl Interaction {
    #[new]
    pub fn new(i: String, j: String, weight: f64) -> Self {
        Interaction { i, j, weight }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct LogicalGraph {
    #[pyo3(get, set)]
    pub variables: Vec<Variable>,
    #[pyo3(get, set)]
    pub interactions: Vec<Interaction>,
}

#[pymethods]
impl LogicalGraph {
    #[new]
    pub fn new(variables: Vec<Variable>, interactions: Vec<Interaction>) -> Self {
        LogicalGraph { variables, interactions }
    }
}

#[pyclass]
pub struct IsingSimulator {}

#[pymethods]
impl IsingSimulator {
    #[new]
    pub fn new() -> Self {
        IsingSimulator {}
    }

    pub fn solve_exact(&self, graph: &LogicalGraph) -> PyResult<Vec<i8>> {
        let n = graph.variables.len();
        if n > 20 {
            return Err(crate::SizeViolation::new_err("SizeViolation: N > 20 is not supported for exact solver"));
        }

        if n == 0 {
            return Ok(Vec::new());
        }

        let mut min_energy = f64::INFINITY;
        let mut best_state = vec![-1i8; n];

        let name_to_idx: HashMap<String, usize> = graph.variables.iter().enumerate()
            .map(|(i, v)| (v.name.clone(), i))
            .collect();

        // Ising Hamiltonian: H = sum J_ij s_i s_j + sum h_i s_i
        // For our LogicalGraph, interactions with i == j are h_i, i != j are J_ij

        let num_states = 1 << n;
        for state_idx in 0..num_states {
            let mut current_energy = 0.0;
            let mut state = vec![1i8; n];
            for i in 0..n {
                if (state_idx >> i) & 1 == 1 {
                    state[i] = 1;
                } else {
                    state[i] = -1;
                }
            }

            for interaction in &graph.interactions {
                let i_idx = name_to_idx.get(&interaction.i);
                let j_idx = name_to_idx.get(&interaction.j);

                if let (Some(&ii), Some(&jj)) = (i_idx, j_idx) {
                    if ii == jj {
                        // Linear term
                        current_energy += interaction.weight * (state[ii] as f64);
                    } else {
                        // Quadratic term
                        current_energy += interaction.weight * (state[ii] as f64) * (state[jj] as f64);
                    }
                }
            }

            if current_energy < min_energy {
                min_energy = current_energy;
                best_state = state;
            }
        }

        Ok(best_state)
    }
}
