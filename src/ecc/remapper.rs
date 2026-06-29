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
