use pyo3::prelude::*;

mod delta;
mod scoring;
mod stackelberg;
pub mod sim;
pub mod analog;

pub use scoring::EquilibriumScore;
pub use stackelberg::StackelbergSolver;

pyo3::import_exception!(limen.exceptions, SizeViolation);

#[pymodule]
fn limen_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<stackelberg::StackelbergSolver>()?;
    m.add_class::<scoring::EquilibriumScore>()?;
    m.add_function(wrap_pyfunction!(delta::apply_detuning_correction, m)?)?;
    m.add_function(wrap_pyfunction!(delta::apply_coupling_correction, m)?)?;

    let sim = PyModule::new_bound(m.py(), "sim")?;
    sim.add_class::<sim::ising_backend::IsingSimulator>()?;
    sim.add_class::<sim::ising_backend::LogicalGraph>()?;
    sim.add_class::<sim::ising_backend::Variable>()?;
    sim.add_class::<sim::ising_backend::Interaction>()?;
    m.add_submodule(&sim)?;

    let analog = PyModule::new_bound(m.py(), "analog")?;
    analog.add_class::<analog::interface::ContinuousFieldIR>()?;
    analog.add_class::<analog::interface::HardwareDeltaModel>()?;
    analog.add_class::<analog::neutral_atom::RydbergControlParameters>()?;
    analog.add_class::<analog::neutral_atom::SpatialCoordinates>()?;
    analog.add_class::<analog::neutral_atom::NeutralAtomCompiler>()?;
    analog.add_class::<analog::photonic::PhotonicGBSParameters>()?;
    analog.add_class::<analog::photonic::PhotonicCompiler>()?;
    m.add_submodule(&analog)?;

    Ok(())
}
