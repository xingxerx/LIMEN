use pyo3::prelude::*;
use std::collections::HashMap;

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
