use crate::sim::ising_backend::LogicalGraph;
use crate::analog::interface::{AnalogTargetAdapter, ContinuousFieldIR, HardwareDeltaModel, AnalogError, ScalingError};
use std::collections::HashMap;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct PhotonicGBSParameters {
    #[pyo3(get)]
    pub adjacency_matrix: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub phases: Vec<f64>,
}

#[pymethods]
impl PhotonicGBSParameters {
    #[new]
    pub fn new(adjacency_matrix: Vec<Vec<f64>>, phases: Vec<f64>) -> Self {
        PhotonicGBSParameters { adjacency_matrix, phases }
    }
}

#[pyclass]
pub struct PhotonicCompiler;

#[pymethods]
impl PhotonicCompiler {
    #[new]
    pub fn new() -> Self {
        PhotonicCompiler
    }

    pub fn build_arrazola_bromley_encoding(&self, graph: &LogicalGraph) -> PhotonicGBSParameters {
        build_arrazola_bromley_encoding(graph)
    }
}

impl AnalogTargetAdapter for PhotonicCompiler {
    fn compile_to_field(&self, _logical_graph: &LogicalGraph, _delta_model: &HardwareDeltaModel) -> Result<ContinuousFieldIR, AnalogError> {
        Ok(ContinuousFieldIR { data: vec![] })
    }

    fn compute_convex_scaling(&self, _field: &mut ContinuousFieldIR) -> Result<(), ScalingError> {
        Ok(())
    }
}

pub fn build_arrazola_bromley_encoding(graph: &LogicalGraph) -> PhotonicGBSParameters {
    let n = graph.variables.len();
    let mut adj = vec![vec![0.0; n]; n];
    let mut phases = vec![0.0; n];

    let name_to_idx: HashMap<String, usize> = graph.variables.iter().enumerate()
        .map(|(i, v)| (v.name.clone(), i))
        .collect();

    for interaction in &graph.interactions {
        if let (Some(&i), Some(&j)) = (name_to_idx.get(&interaction.i), name_to_idx.get(&interaction.j)) {
            if i == j {
                // Map discrete binary nodes to continuous phases on unit circle
                // e^{i * phi}
                phases[i] = interaction.weight.atan();
            } else {
                adj[i][j] = interaction.weight;
                adj[j][i] = interaction.weight;
            }
        }
    }

    PhotonicGBSParameters {
        adjacency_matrix: adj,
        phases,
    }
}
