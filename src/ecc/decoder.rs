use pyo3::prelude::*;

/// Pack the Z-stabilizer supports into per-stabilizer bitmasks over the
/// data qubits, validating index bounds.
fn stabilizer_masks(n_data: usize, z_stabilizers: &[Vec<usize>]) -> PyResult<Vec<u64>> {
    if n_data > 26 {
        return Err(crate::SizeViolation::new_err(
            "SizeViolation: lookup decoding requires n_data <= 26",
        ));
    }
    if z_stabilizers.len() > 32 {
        return Err(crate::SizeViolation::new_err(
            "SizeViolation: lookup decoding requires at most 32 Z-stabilizers",
        ));
    }
    z_stabilizers
        .iter()
        .map(|support| {
            let mut mask = 0u64;
            for &q in support {
                if q >= n_data {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "stabilizer references data qubit {q} but n_data is {n_data}"
                    )));
                }
                mask |= 1 << q;
            }
            Ok(mask)
        })
        .collect()
}

/// Enumerate all X-error patterns in the same order as
/// `itertools.product((0, 1), repeat=n)` (bits[n-1] varies fastest) and
/// keep the first — i.e. minimum-weight, earliest-in-product-order —
/// error pattern per syndrome. Errors are bitmasks with bit q = data
/// qubit q; syndromes are packed with bit k = stabilizer k's parity.
fn build_table(n_data: usize, masks: &[u64]) -> Vec<Option<u64>> {
    let n_syndromes = 1usize << masks.len();
    let mut table: Vec<Option<(u32, u64)>> = vec![None; n_syndromes];

    for counter in 0..(1u64 << n_data) {
        // Reverse the counter's bits over n_data positions so the
        // enumeration order (and therefore equal-weight tie-breaking)
        // matches the Python itertools.product reference exactly.
        let mut error = 0u64;
        for q in 0..n_data {
            if (counter >> (n_data - 1 - q)) & 1 == 1 {
                error |= 1 << q;
            }
        }
        let weight = error.count_ones();

        let mut syndrome = 0usize;
        for (k, mask) in masks.iter().enumerate() {
            if (error & mask).count_ones() & 1 == 1 {
                syndrome |= 1 << k;
            }
        }

        match table[syndrome] {
            Some((w, _)) if w <= weight => {}
            _ => table[syndrome] = Some((weight, error)),
        }
    }

    table.into_iter().map(|e| e.map(|(_, m)| m)).collect()
}

/// Build the minimum-weight syndrome-to-correction lookup table for a
/// surface-code patch, by exhaustive enumeration over all `2^n_data`
/// X-error patterns.
///
/// This is the Rust port of the table construction inside
/// `limen.ecc.decoder.LookupDecoder.__init__`, which walks the same
/// `2^n` space in interpreted Python. Enumeration order (and therefore
/// equal-weight tie-breaking) matches the Python reference exactly.
///
/// # Arguments
/// * `n_data`        - Number of data qubits. Must be ≤ 26.
/// * `z_stabilizers` - Z-stabilizer supports as lists of data-qubit indices.
///
/// # Returns
/// A list of `(syndrome_bits, error_bits)` pairs, where `syndrome_bits`
/// has one 0/1 entry per stabilizer (in `z_stabilizers` order) and
/// `error_bits` has one 0/1 entry per data qubit. Only realizable
/// syndromes are present.
///
/// # Errors
/// Returns `SizeViolation` when `n_data > 26` or there are more than 32
/// stabilizers.
#[pyfunction]
pub fn build_ecc_lookup_table(
    n_data: usize,
    z_stabilizers: Vec<Vec<usize>>,
) -> PyResult<Vec<(Vec<u8>, Vec<u8>)>> {
    let masks = stabilizer_masks(n_data, &z_stabilizers)?;
    let table = build_table(n_data, &masks);

    let n_stabs = masks.len();
    let mut pairs = Vec::new();
    for (syndrome, entry) in table.iter().enumerate() {
        if let Some(error) = entry {
            let syndrome_bits: Vec<u8> =
                (0..n_stabs).map(|k| ((syndrome >> k) & 1) as u8).collect();
            let error_bits: Vec<u8> =
                (0..n_data).map(|q| ((error >> q) & 1) as u8).collect();
            pairs.push((syndrome_bits, error_bits));
        }
    }
    Ok(pairs)
}

/// Compute the exact logical X-error rate of a surface-code patch under
/// independent per-qubit bit-flip noise, decoded with the minimum-weight
/// lookup decoder.
///
/// Brute-forces all `2^n_data` error patterns, weights each by its exact
/// binomial probability, decodes its syndrome, and sums the probability
/// mass of patterns where `error XOR correction` overlaps `logical_z` on
/// an odd number of qubits — the Rust port of the enumeration loop in
/// `limen.ecc.certificate.certify_logical_qubit`, including the decoder
/// table build it depends on.
///
/// # Arguments
/// * `n_data`              - Number of data qubits. Must be ≤ 26.
/// * `z_stabilizers`       - Z-stabilizer supports (data-qubit indices).
/// * `logical_z`           - Support of the logical-Z operator.
/// * `physical_error_rate` - Independent per-qubit bit-flip probability.
///
/// # Errors
/// Returns `SizeViolation` when `n_data > 26` or there are more than 32
/// stabilizers.
#[pyfunction]
pub fn logical_failure_probability(
    n_data: usize,
    z_stabilizers: Vec<Vec<usize>>,
    logical_z: Vec<usize>,
    physical_error_rate: f64,
) -> PyResult<f64> {
    let masks = stabilizer_masks(n_data, &z_stabilizers)?;
    let table = build_table(n_data, &masks);

    let mut logical_mask = 0u64;
    for &q in &logical_z {
        if q >= n_data {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "logical_z references data qubit {q} but n_data is {n_data}"
            )));
        }
        logical_mask |= 1 << q;
    }

    // Precompute p^w * (1-p)^(n-w) for every error weight w.
    let p = physical_error_rate;
    let probs: Vec<f64> = (0..=n_data)
        .map(|w| p.powi(w as i32) * (1.0 - p).powi((n_data - w) as i32))
        .collect();

    let mut failure_probability = 0.0;
    for error in 0..(1u64 << n_data) {
        let mut syndrome = 0usize;
        for (k, mask) in masks.iter().enumerate() {
            if (error & mask).count_ones() & 1 == 1 {
                syndrome |= 1 << k;
            }
        }
        let correction = table[syndrome].unwrap_or(0);
        let residual = error ^ correction;
        if (residual & logical_mask).count_ones() & 1 == 1 {
            failure_probability += probs[error.count_ones() as usize];
        }
    }

    Ok(failure_probability)
}
