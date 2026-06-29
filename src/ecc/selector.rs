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
}
