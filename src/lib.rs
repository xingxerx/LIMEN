use pyo3::prelude::*;

mod cutting;
mod delta;
mod ecc;
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
    m.add_function(wrap_pyfunction!(delta::apply_rabi_correction, m)?)?;
    m.add_function(wrap_pyfunction!(scoring::qubo_criticality, m)?)?;

    m.add_function(wrap_pyfunction!(sim::ising_backend::exact_ising_norm, m)?)?;
    m.add_function(wrap_pyfunction!(sim::ising_backend::qubo_energy_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(sim::statevector_backend::run_statevector, m)?)?;

    let sim = PyModule::new_bound(m.py(), "sim")?;
    sim.add_class::<sim::ising_backend::IsingSimulator>()?;
    sim.add_class::<sim::ising_backend::LogicalGraph>()?;
    sim.add_class::<sim::ising_backend::Variable>()?;
    sim.add_class::<sim::ising_backend::Interaction>()?;
    sim.add_class::<sim::qudit::QuditSimulator>()?;
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

    let ecc = PyModule::new_bound(m.py(), "ecc")?;
    ecc.add_class::<ecc::surface_code::SurfaceCodePatch>()?;
    ecc.add_function(wrap_pyfunction!(ecc::surface_code::build_surface_code, &ecc)?)?;
    ecc.add_class::<ecc::selector::PatchAssignment>()?;
    ecc.add_function(wrap_pyfunction!(ecc::selector::select_patches, &ecc)?)?;
    ecc.add_function(wrap_pyfunction!(ecc::remapper::remap_circuit, &ecc)?)?;
    m.add_submodule(&ecc)?;

    let cutting = PyModule::new_bound(m.py(), "cutting")?;
    cutting.add_class::<cutting::reconstruct::SubcircuitSampleCounts>()?;
    cutting.add_class::<cutting::reconstruct::SampleCoefficient>()?;
    cutting.add_function(wrap_pyfunction!(
        cutting::reconstruct::reconstruct_expectation,
        &cutting
    )?)?;
    m.add_submodule(&cutting)?;

    Ok(())
}
