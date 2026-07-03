use pyo3::prelude::*;

/// A complex amplitude as a (real, imaginary) pair.
type C = (f64, f64);

const ZERO: C = (0.0, 0.0);
const ONE: C = (1.0, 0.0);

fn cadd(a: C, b: C) -> C {
    (a.0 + b.0, a.1 + b.1)
}

fn cmul(a: C, b: C) -> C {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn cneg(a: C) -> C {
    (-a.0, -a.1)
}

/// Return the 2x2 matrix [a, b, c, d] (row-major) for a single-qubit gate
/// in `limen.gates.ir.KNOWN_GATES`. Mirrors
/// `limen.gates.simulator._single_qubit_matrix` exactly.
fn single_qubit_matrix(name: &str, params: &[f64]) -> PyResult<[C; 4]> {
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let m = match name {
        "h" => [
            (inv_sqrt2, 0.0),
            (inv_sqrt2, 0.0),
            (inv_sqrt2, 0.0),
            (-inv_sqrt2, 0.0),
        ],
        "x" => [ZERO, ONE, ONE, ZERO],
        "y" => [ZERO, (0.0, -1.0), (0.0, 1.0), ZERO],
        "z" => [ONE, ZERO, ZERO, (-1.0, 0.0)],
        "s" => [ONE, ZERO, ZERO, (0.0, 1.0)],
        "t" => {
            let phase = std::f64::consts::FRAC_PI_4;
            [ONE, ZERO, ZERO, (phase.cos(), phase.sin())]
        }
        "rx" => {
            let c = (params[0] / 2.0).cos();
            let s = (params[0] / 2.0).sin();
            [(c, 0.0), (0.0, -s), (0.0, -s), (c, 0.0)]
        }
        "ry" => {
            let c = (params[0] / 2.0).cos();
            let s = (params[0] / 2.0).sin();
            [(c, 0.0), (-s, 0.0), (s, 0.0), (c, 0.0)]
        }
        "rz" => {
            let half = params[0] / 2.0;
            [
                (half.cos(), -half.sin()),
                ZERO,
                ZERO,
                (half.cos(), half.sin()),
            ]
        }
        "u" => {
            let (theta, phi, lam) = (params[0], params[1], params[2]);
            let cos_half = (theta / 2.0).cos();
            let sin_half = (theta / 2.0).sin();
            let exp_i_lam = (lam.cos(), lam.sin());
            let exp_i_phi = (phi.cos(), phi.sin());
            let exp_i_phi_lam = ((phi + lam).cos(), (phi + lam).sin());
            [
                (cos_half, 0.0),
                cneg(cmul(exp_i_lam, (sin_half, 0.0))),
                cmul(exp_i_phi, (sin_half, 0.0)),
                cmul(exp_i_phi_lam, (cos_half, 0.0)),
            ]
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "'{}' is not a single-qubit gate",
                other
            )))
        }
    };
    Ok(m)
}

/// Apply a 2x2 gate to qubit `q` of an `n`-qubit state, in place. Mirrors
/// `limen.gates.simulator._apply_1q`.
fn apply_1q(state: &mut [C], n: usize, q: usize, m: [C; 4]) {
    let (a, b, c, d) = (m[0], m[1], m[2], m[3]);
    let step = 1usize << q;
    let mut base = 0usize;
    while base < (1usize << n) {
        for s0 in base..base + step {
            let s1 = s0 | step;
            let v0 = state[s0];
            let v1 = state[s1];
            state[s0] = cadd(cmul(a, v0), cmul(b, v1));
            state[s1] = cadd(cmul(c, v0), cmul(d, v1));
        }
        base += step << 1;
    }
}

/// Apply a two-qubit gate (cx, cz, swap) to the state, in place. Mirrors
/// `limen.gates.simulator._apply_2q`.
fn apply_2q(state: &mut [C], n: usize, name: &str, qubits: &[usize]) -> PyResult<()> {
    let (a, b) = (qubits[0], qubits[1]);
    let abit = 1usize << a;
    let bbit = 1usize << b;
    match name {
        "cx" => {
            for s in 0..(1usize << n) {
                if (s & abit != 0) && (s & bbit == 0) {
                    let t = s | bbit;
                    state.swap(s, t);
                }
            }
        }
        "cz" => {
            for s in 0..(1usize << n) {
                if (s & abit != 0) && (s & bbit != 0) {
                    state[s] = cneg(state[s]);
                }
            }
        }
        "swap" => {
            for s in 0..(1usize << n) {
                if (s & abit != 0) && (s & bbit == 0) {
                    let t = (s & !abit) | bbit;
                    state.swap(s, t);
                }
            }
        }
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "'{}' is not a supported two-qubit gate",
                other
            )))
        }
    }
    Ok(())
}

/// Run a `CircuitIR`'s gate sequence on a Rust-backed statevector and
/// return the final amplitudes as `(real, imag)` pairs.
///
/// Mirrors `limen.gates.simulator.statevector` exactly, including its
/// basis-index convention (qubit `q` is bit `(index >> q) & 1`); the
/// Python wrapper falls back to its pure-Python loop when this extension
/// isn't built. This is the hot path for QAOA/gate-model testing, where
/// the pure-Python version re-walks all `2^n` amplitudes per gate in
/// interpreted bytecode.
///
/// # Arguments
/// * `instructions` - `[(gate_name, qubit_indices, params)]` in execution order.
/// * `n_qubits`     - Number of qubits; the returned vector has length `2^n_qubits`.
///
/// # Errors
/// Returns `SizeViolation` when `n_qubits > 24` (statevector memory grows
/// as `2^n_qubits` complex amplitudes).
#[pyfunction]
pub fn run_statevector(
    instructions: Vec<(String, Vec<usize>, Vec<f64>)>,
    n_qubits: usize,
) -> PyResult<Vec<(f64, f64)>> {
    if n_qubits > 24 {
        return Err(crate::SizeViolation::new_err(
            "SizeViolation: run_statevector requires n_qubits <= 24",
        ));
    }

    let mut state: Vec<C> = vec![ZERO; 1usize << n_qubits];
    state[0] = ONE;

    for (name, qubits, params) in &instructions {
        if qubits.len() == 1 {
            let m = single_qubit_matrix(name, params)?;
            apply_1q(&mut state, n_qubits, qubits[0], m);
        } else {
            apply_2q(&mut state, n_qubits, name, qubits)?;
        }
    }

    Ok(state)
}
