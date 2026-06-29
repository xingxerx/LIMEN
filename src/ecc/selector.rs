use pyo3::prelude::*;

/// Assignment of one logical variable to a surface-code patch, or to
/// an unprotected physical qubit once the protection budget runs out.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PatchAssignment {
    #[pyo3(get)]
    pub logical_var: usize,
    /// Code distance. `1` means "unprotected" — the variable maps
    /// straight to a single physical qubit.
    #[pyo3(get)]
    pub distance: usize,
    /// Physical qubit indices backing this assignment (`distance^2`
    /// of them for a protected patch, exactly one for `distance == 1`).
    #[pyo3(get)]
    pub physical_qubits: Vec<usize>,
}

/// Select which logical variables get ECC protection under a fixed
/// physical-qubit budget, given the criticality ranking produced by
/// `scoring::qubo_criticality`.
///
/// TODO(ecc-budget): consume `qubit_budget` greedily in ranked order —
/// allocate a distance-3 patch (9 physical qubits) to each variable
/// while the budget allows it, then fall back to a 1:1 unprotected
/// mapping for whatever remains. See the LIMEN build plan, step 2.
pub fn select_patches(_ranked: &[(usize, f64)], _qubit_budget: usize) -> Vec<PatchAssignment> {
    todo!("port the QEC budget selector — see build plan step 2 (src/ecc/selector.rs)")
}
