use crate::sim::ising_backend::LogicalGraph;
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;

#[pyclass]
#[derive(Clone, Debug)]
pub struct ContinuousFieldIR {
    #[pyo3(get, set)]
    pub data: Vec<f64>,
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct HardwareDeltaModel {
    #[pyo3(get, set)]
    pub calibration_data: Vec<f64>,
}

pub enum AnalogError {
    CompilationFailed(String),
}

pub enum ScalingError {
    OptimizationFailed(String),
}

impl From<AnalogError> for PyErr {
    fn from(err: AnalogError) -> PyErr {
        match err {
            AnalogError::CompilationFailed(s) => PyRuntimeError::new_err(s),
        }
    }
}

impl From<ScalingError> for PyErr {
    fn from(err: ScalingError) -> PyErr {
        match err {
            ScalingError::OptimizationFailed(s) => PyRuntimeError::new_err(s),
        }
    }
}

pub trait AnalogTargetAdapter {
    fn compile_to_field(&self, logical_graph: &LogicalGraph, delta_model: &HardwareDeltaModel) -> Result<ContinuousFieldIR, AnalogError>;
    fn compute_convex_scaling(&self, field: &mut ContinuousFieldIR) -> Result<(), ScalingError>;
}
