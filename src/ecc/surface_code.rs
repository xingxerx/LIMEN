use pyo3::prelude::*;

/// One logical qubit encoded in a rotated surface-code patch.
///
/// Mirrors `limen.ecc.surface_code.SurfaceCodePatch`. Porting the
/// stabilizer construction here lets syndrome enumeration run in
/// parallel across many patches with rayon, instead of the pure-Python
/// per-patch loop.
#[pyclass]
#[derive(Clone, Debug)]
pub struct SurfaceCodePatch {
    #[pyo3(get)]
    pub distance: usize,
    #[pyo3(get)]
    pub data_qubits: Vec<usize>,
    #[pyo3(get)]
    pub x_stabilizers: Vec<Vec<usize>>,
    #[pyo3(get)]
    pub z_stabilizers: Vec<Vec<usize>>,
    #[pyo3(get)]
    pub logical_x: Vec<usize>,
    #[pyo3(get)]
    pub logical_z: Vec<usize>,
}

/// TODO(ecc-patch-math): port `limen.ecc.surface_code.build_surface_code`
/// (bulk/boundary stabilizer construction on a `distance x distance`
/// grid). See the LIMEN build plan, step 3.
pub fn build_surface_code(_distance: usize) -> SurfaceCodePatch {
    todo!("port limen/ecc/surface_code.py::build_surface_code — see build plan step 3")
}
