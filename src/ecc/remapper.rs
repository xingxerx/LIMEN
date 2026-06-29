use pyo3::prelude::*;
use std::collections::HashMap;

use crate::ecc::selector::PatchAssignment;

/// Build a logical-to-physical qubit index map from patch assignments.
///
/// Each logical variable maps to the first physical qubit of its patch
/// (`physical_start`), which serves as that patch's representative index
/// for gate operand rewriting.
fn index_map(assignments: &[PatchAssignment]) -> HashMap<usize, usize> {
    assignments
        .iter()
        .map(|a| (a.logical_var, a.physical_start))
        .collect()
}

/// Rewrite gate operand indices from logical to physical qubits.
///
/// `circuit` is a list of `(gate_name, operand_indices)` pairs. Operands
/// referencing a logical variable with a patch assignment are rewritten
/// to that patch's representative physical qubit; operands with no
/// assignment pass through unchanged. This runs on the hot dispatch path,
/// so it's a single linear pass with no allocation beyond the output.
#[pyfunction]
pub fn remap_circuit(
    circuit: Vec<(String, Vec<usize>)>,
    assignments: Vec<PatchAssignment>,
) -> Vec<(String, Vec<usize>)> {
    let map = index_map(&assignments);
    circuit
        .into_iter()
        .map(|(gate, operands)| {
            let remapped = operands
                .into_iter()
                .map(|q| map.get(&q).copied().unwrap_or(q))
                .collect();
            (gate, remapped)
        })
        .collect()
use super::selector::PatchAssignment;

/// Rewrite a circuit IR's logical qubit indices to the physical qubit
/// ranges chosen by `selector::select_patches`.
///
/// This is the hot path: remapping runs before every job dispatch, so
/// it is the part of the ECC pipeline most worth keeping in Rust.
///
/// TODO(ecc-remap): take the compiler's circuit IR, look up each
/// logical index's `PatchAssignment`, and emit a new IR with physical
/// indices substituted in. See the LIMEN build plan, step 4.
pub fn remap_indices(_assignments: &[PatchAssignment]) {
    todo!("port the index remapper — see build plan step 4 (src/ecc/remapper.rs)")
}
