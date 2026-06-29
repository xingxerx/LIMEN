use pyo3::prelude::*;

/// One logical qubit encoded in a rotated surface code patch.
///
/// Port of `limen.ecc.surface_code.SurfaceCodePatch`; see that module's
/// docstring for the construction rule. Correctness is established by
/// the existing Python test suite, not re-derived here.
#[pyclass]
#[derive(Clone)]
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

#[pymethods]
impl SurfaceCodePatch {
    fn __repr__(&self) -> String {
        format!(
            "SurfaceCodePatch(distance={}, data_qubits={})",
            self.distance,
            self.data_qubits.len()
        )
    }
}

/// Build a rotated surface code patch encoding one logical qubit.
///
/// Direct port of `limen.ecc.surface_code.build_surface_code`. Only
/// `distance=3` is tested/supported upstream.
#[pyfunction]
#[pyo3(signature = (distance=3))]
pub fn build_surface_code(distance: usize) -> SurfaceCodePatch {
    let d = distance as isize;
    let qubit_index = |r: isize, c: isize| -> usize { (r * d + c) as usize };
    let neighbors = |r: isize, c: isize| -> Vec<(isize, isize)> {
        [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]
            .into_iter()
            .filter(|&(rr, cc)| rr >= 0 && rr < d && cc >= 0 && cc < d)
            .collect()
    };

    let mut x_stabilizers: Vec<Vec<usize>> = Vec::new();
    let mut z_stabilizers: Vec<Vec<usize>> = Vec::new();

    for r in -1..d {
        for c in -1..d {
            let support = neighbors(r, c);
            if support.len() == 4 {
                let qubits: Vec<usize> = support.iter().map(|&(rr, cc)| qubit_index(rr, cc)).collect();
                if (r + c).rem_euclid(2) == 0 {
                    z_stabilizers.push(qubits);
                } else {
                    x_stabilizers.push(qubits);
                }
            } else if support.len() == 2 {
                let is_horizontal_edge = r == -1 || r == d - 1;
                let qubits: Vec<usize> = support.iter().map(|&(rr, cc)| qubit_index(rr, cc)).collect();
                let edge_parity = (r + c).rem_euclid(2);
                if is_horizontal_edge {
                    if edge_parity == 0 {
                        z_stabilizers.push(qubits);
                    }
                } else if edge_parity == 1 {
                    x_stabilizers.push(qubits);
                }
            }
        }
    }

    let data_qubits: Vec<usize> = (0..d).flat_map(|r| (0..d).map(move |c| qubit_index(r, c))).collect();
    let logical_x: Vec<usize> = (0..d).map(|c| qubit_index(0, c)).collect();
    let logical_z: Vec<usize> = (0..d).map(|r| qubit_index(r, 0)).collect();

    SurfaceCodePatch {
        distance,
        data_qubits,
        x_stabilizers,
        z_stabilizers,
        logical_x,
        logical_z,
    }
}
