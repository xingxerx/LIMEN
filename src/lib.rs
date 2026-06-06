use pyo3::prelude::*;

mod delta;
mod scoring;
mod stackelberg;

pub use scoring::EquilibriumScore;
pub use stackelberg::StackelbergSolver;

#[pymodule]
fn limen_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<stackelberg::StackelbergSolver>()?;
    m.add_class::<scoring::EquilibriumScore>()?;
    m.add_function(wrap_pyfunction!(delta::apply_detuning_correction, m)?)?;
    m.add_function(wrap_pyfunction!(delta::apply_coupling_correction, m)?)?;
    Ok(())
}
