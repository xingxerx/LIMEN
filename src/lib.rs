use pyo3::prelude::*;

mod scoring;
mod stackelberg;

pub use scoring::EquilibriumScore;
pub use stackelberg::StackelbergSolver;

#[pymodule]
fn limen_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<stackelberg::StackelbergSolver>()?;
    m.add_class::<scoring::EquilibriumScore>()?;
    Ok(())
}
