use pyo3::prelude::*;

/// A QEC patch allocation for one logical variable.
///
/// `physical_start..physical_end` is the contiguous block of physical
/// qubits reserved for this variable's surface code patch.
#[pyclass]
#[derive(Clone)]
pub struct PatchAssignment {
    #[pyo3(get)]
    pub logical_var: usize,
    #[pyo3(get)]
    pub distance: usize,
    #[pyo3(get)]
    pub physical_start: usize,
    #[pyo3(get)]
    pub physical_end: usize,
}

#[pymethods]
impl PatchAssignment {
    fn __repr__(&self) -> String {
        format!(
            "PatchAssignment(logical_var={}, distance={}, physical_qubits={}..{})",
            self.logical_var, self.distance, self.physical_start, self.physical_end
        )
    }
}

/// Greedily assign surface code patches to the most critical variables
/// until the physical qubit budget runs out.
///
/// `ranked` is expected sorted by descending criticality (the output of
/// `qubo_criticality`); ties or unsorted input are not re-sorted here.
/// Each patch costs `distance * distance` physical qubits. Variables that
/// don't fit in the remaining budget are skipped (lower-criticality
/// variables later in the list may still fit and will be tried).
#[pyfunction]
pub fn select_patches(
    ranked: Vec<(usize, f64)>,
    physical_qubit_budget: usize,
    distance: usize,
) -> Vec<PatchAssignment> {
    let patch_cost = distance * distance;
    let mut assignments = Vec::new();
    let mut used = 0usize;

    if patch_cost == 0 {
        return assignments;
    }

    for (var, _criticality) in ranked {
        if used + patch_cost > physical_qubit_budget {
            continue;
        }
        assignments.push(PatchAssignment {
            logical_var: var,
            distance,
            physical_start: used,
            physical_end: used + patch_cost,
        });
        used += patch_cost;
    }

    assignments
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
