use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct QuditSimulator {
    pub levels: usize,
    pub num_sites: usize,
}

#[pymethods]
impl QuditSimulator {
    #[new]
    pub fn new(levels: usize, num_sites: usize) -> Self {
        QuditSimulator { levels, num_sites }
    }

    pub fn simulate(&self) -> PyResult<Vec<usize>> {
        // Mock fallback simulator for qudits.
        // Returns a vector of states [0..levels-1] for each site.
        Ok(vec![0; self.num_sites])
    }
}
