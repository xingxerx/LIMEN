use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};
use std::collections::HashMap;

/// Build the one-hot tour QUBO for a VRP/TSP instance.
///
/// This is the Rust port of the term-construction loops in
/// `limen.frontends.vrp.vrp_qubo`: given an (already depot-augmented)
/// `n x n` distance matrix it emits, over variables `x_{i}_{t}`
/// ("augmented node i occupies tour position t"),
///
///   H = A · Σ_i (1 − Σ_t x_it)²        (each node once)
///     + A · Σ_t (1 − Σ_i x_it)²        (each position once)
///     + B · Σ_{u≠v,t} d_uv x_ut x_v,t+1 (tour length, cyclic)
///
/// dropping the constant offsets, exactly as the Python reference does.
///
/// The accumulation runs over integer `(node, position)` index keys and
/// only crosses the Python boundary once at the end: the returned dict is
/// built directly with one cached `PyString` per variable name, because
/// at benchmark scale (~300k terms for 50 customers) allocating fresh
/// key strings per term costs more than the arithmetic being ported.
/// Keys are `("x_i_t", "x_j_s")` name pairs ordered by string comparison
/// exactly like the Python `add()` helper, inserted in sorted index
/// order so the result is deterministic across runs.
///
/// # Arguments
/// * `dist`      - Augmented distance matrix (square, depot copies included).
/// * `penalty_a` - Constraint penalty weight A.
/// * `penalty_b` - Objective (distance) weight B.
///
/// # Returns
/// The QUBO as a dict mapping `(name, name)` tuples to accumulated weights,
/// interchangeable with the pure-Python implementation's output.
#[pyfunction]
pub fn vrp_qubo_terms<'py>(
    py: Python<'py>,
    dist: Vec<Vec<f64>>,
    penalty_a: f64,
    penalty_b: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let n = dist.len();
    for row in &dist {
        if row.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "distance matrix must be square",
            ));
        }
    }

    // names[flat] = "x_{i}_{t}" with flat = i * n + t.
    let names: Vec<String> = (0..n)
        .flat_map(|i| (0..n).map(move |t| format!("x_{i}_{t}")))
        .collect();

    // rank[flat] = position of names[flat] in string-sorted order, so the
    // canonical key ordering below matches the Python reference's string
    // comparison without comparing strings per term.
    let mut by_name: Vec<usize> = (0..n * n).collect();
    by_name.sort_unstable_by(|&a, &b| names[a].cmp(&names[b]));
    let mut rank = vec![0usize; n * n];
    for (pos, &flat) in by_name.iter().enumerate() {
        rank[flat] = pos;
    }

    let mut qubo: HashMap<u64, f64> = HashMap::new();
    let mut add = |u: (usize, usize), v: (usize, usize), w: f64| {
        let (fu, fv) = (u.0 * n + u.1, v.0 * n + v.1);
        let key = if rank[fu] <= rank[fv] {
            ((fu as u64) << 32) | fv as u64
        } else {
            ((fv as u64) << 32) | fu as u64
        };
        *qubo.entry(key).or_insert(0.0) += w;
    };

    // Each node visited exactly once: A·(1 − Σ_t x_it)²
    for i in 0..n {
        for t in 0..n {
            add((i, t), (i, t), -penalty_a);
            for s in (t + 1)..n {
                add((i, t), (i, s), 2.0 * penalty_a);
            }
        }
    }

    // Each position filled exactly once: A·(1 − Σ_i x_it)²
    for t in 0..n {
        for i in 0..n {
            add((i, t), (i, t), -penalty_a);
            for j in (i + 1)..n {
                add((i, t), (j, t), 2.0 * penalty_a);
            }
        }
    }

    // Tour length objective: B · Σ_{u≠v,t} d_uv · x_ut · x_v,t+1
    for u in 0..n {
        for v in 0..n {
            if u == v {
                continue;
            }
            for t in 0..n {
                let s = (t + 1) % n;
                add((u, t), (v, s), penalty_b * dist[u][v]);
            }
        }
    }

    let mut terms: Vec<(u64, f64)> = qubo.into_iter().collect();
    terms.sort_unstable_by_key(|(k, _)| *k);

    let py_names: Vec<Bound<'py, PyString>> = names
        .iter()
        .map(|s| PyString::new_bound(py, s))
        .collect();

    let out = PyDict::new_bound(py);
    for (key, w) in terms {
        let (fu, fv) = ((key >> 32) as usize, (key & 0xFFFF_FFFF) as usize);
        let key = PyTuple::new_bound(py, [&py_names[fu], &py_names[fv]]);
        out.set_item(key, w)?;
    }
    Ok(out)
}
