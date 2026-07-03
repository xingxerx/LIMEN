use pyo3::prelude::*;
use rayon::prelude::*;

/// SplitMix64 — a tiny deterministic RNG with a platform-independent
/// sequence for a given seed. The validator's simulated runs only need
/// reproducibility per (seed, backend); they do not need to match
/// CPython's Mersenne Twister stream bit-for-bit.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f64 in [0, 1) with 53 bits of precision.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Simulate noisy hardware runs over a QUBO.
///
/// Each run starts from `base_bits` (the best known assignment, indexed by
/// variable) and independently flips each bit with probability
/// `noise_level`, then evaluates the QUBO energy
/// `E(x) = Σ (i,j,w) w·x_i·x_j` of the noisy assignment (with `i == j`
/// encoding a linear term).
///
/// This is the hot loop behind `limen.validator.validator.simulate_runs`:
/// the Stackelberg co-design loop calls it with hundreds of runs per
/// iteration, and each run walks every QUBO term. Bit flips are drawn
/// serially from a seeded SplitMix64 stream (deterministic per seed);
/// energies are evaluated in parallel with rayon.
///
/// # Arguments
/// * `qubo`        - QUBO terms as `[((var_i, var_j), weight)]` with 0-based
///   variable indices into `base_bits`.
/// * `base_bits`   - Starting 0/1 assignment, one entry per variable.
/// * `n_runs`      - Number of simulated runs.
/// * `noise_level` - Per-variable bit-flip probability (0.0–1.0).
/// * `seed`        - RNG seed; the output is deterministic per seed.
///
/// # Returns
/// A tuple `(assignments, energies)` where `assignments[k]` is the noisy
/// 0/1 assignment of run k and `energies[k]` its QUBO energy.
#[pyfunction]
pub fn simulate_qubo_runs(
    qubo: Vec<((usize, usize), f64)>,
    base_bits: Vec<u8>,
    n_runs: usize,
    noise_level: f64,
    seed: u64,
) -> PyResult<(Vec<Vec<u8>>, Vec<f64>)> {
    let mut rng = SplitMix64::new(seed);

    let assignments: Vec<Vec<u8>> = (0..n_runs)
        .map(|_| {
            base_bits
                .iter()
                .map(|&b| {
                    if rng.next_f64() < noise_level {
                        1 - b
                    } else {
                        b
                    }
                })
                .collect()
        })
        .collect();

    let energies: Vec<f64> = assignments
        .par_iter()
        .map(|bits| {
            qubo.iter()
                .map(|((i, j), w)| w * (bits[*i] as f64) * (bits[*j] as f64))
                .sum()
        })
        .collect();

    Ok((assignments, energies))
}
