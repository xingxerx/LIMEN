//! Backend trait. Hardware details live here and in Target, never in the IR.
//! AQRK (bare-metal drivers) and a PyO3 Qiskit adapter both implement `Backend`.

use limen_ir::{Circuit, Level, Op};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// What a device accepts. Gate names match `Op::name()`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Target {
    pub name: String,
    pub n_qubits: u32,
    pub gates: BTreeSet<String>,
    /// Undirected coupling pairs; empty means all-to-all.
    pub coupling: Vec<(u32, u32)>,
    /// Optional per-gate error rates for the fidelity tiers.
    #[serde(default)]
    pub gate_error: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Health { Ready, Degraded, Offline }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Job { pub id: String, pub circuit_hash: String, pub shots: u32 }

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Result_ { pub job: Job, pub counts: BTreeMap<String, u64> }

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum BackendError {
    #[error("circuit must be at Physical level, got {0:?}")]
    NotPhysical(Level),
    #[error("circuit needs {needed} qubits, target {target} has {have}")]
    TooManyQubits { needed: u32, have: u32, target: String },
    #[error("gate {gate} not in target {target}")]
    UnsupportedGate { gate: String, target: String },
    #[error("qubits {a},{b} are not coupled on {target}")]
    NotCoupled { a: u32, b: u32, target: String },
    #[error("backend {0} is offline")]
    Offline(String),
    #[error("{0}")]
    Other(String),
}

impl Target {
    /// Static admission check; the router calls this before spending budget.
    pub fn accepts(&self, c: &Circuit) -> Result<(), BackendError> {
        if c.level != Level::Physical { return Err(BackendError::NotPhysical(c.level)); }
        if c.n_qubits > self.n_qubits { return Err(BackendError::TooManyQubits { needed: c.n_qubits, have: self.n_qubits, target: self.name.clone() }); }
        for op in &c.ops {
            if matches!(op, Op::Barrier { .. }) { continue; }
            if !self.gates.contains(op.name()) { return Err(BackendError::UnsupportedGate { gate: op.name().into(), target: self.name.clone() }); }
            let qs = op.qubits();
            if qs.len() == 2 && !self.coupling.is_empty() {
                let (a, b) = (qs[0].0, qs[1].0);
                if !self.coupling.iter().any(|&(x, y)| (x, y) == (a, b) || (y, x) == (a, b)) {
                    return Err(BackendError::NotCoupled { a, b, target: self.name.clone() });
                }
            }
        }
        Ok(())
    }
}

pub trait Backend {
    fn target(&self) -> &Target;
    fn health(&self) -> Health;
    fn submit(&mut self, c: &Circuit, shots: u32) -> Result<Job, BackendError>;
    fn result(&mut self, job: &Job) -> Result<Result_, BackendError>;
}

/// Zero-shot stub so pipelines can be tested end to end without hardware.
pub struct NullBackend { pub target: Target, next: u64 }

impl NullBackend {
    pub fn new(target: Target) -> Self { Self { target, next: 0 } }
}

impl Backend for NullBackend {
    fn target(&self) -> &Target { &self.target }
    fn health(&self) -> Health { Health::Ready }
    fn submit(&mut self, c: &Circuit, shots: u32) -> Result<Job, BackendError> {
        self.target.accepts(c)?; self.next += 1;
        Ok(Job { id: format!("null-{}", self.next), circuit_hash: c.hash(), shots })
    }
    fn result(&mut self, job: &Job) -> Result<Result_, BackendError> {
        let mut counts = BTreeMap::new(); counts.insert("0".repeat(1), job.shots as u64);
        Ok(Result_ { job: job.clone(), counts })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use limen_ir::Qubit;

    fn line3() -> Target {
        Target { name: "line3".into(), n_qubits: 3, gates: ["h", "cx", "rz", "measure"].iter().map(|s| s.to_string()).collect(), coupling: vec![(0, 1), (1, 2)], gate_error: BTreeMap::new() }
    }
    #[test] fn admission_rules() {
        let t = line3();
        let mut c = Circuit::new(Level::Physical, 3, 0); c.push(Op::Cx { c: Qubit(0), t: Qubit(2) });
        assert!(matches!(t.accepts(&c), Err(BackendError::NotCoupled { .. })));
        let mut d = Circuit::new(Level::Physical, 3, 0); d.push(Op::Swap { a: Qubit(0), b: Qubit(1) });
        assert!(matches!(t.accepts(&d), Err(BackendError::UnsupportedGate { .. })));
        assert!(matches!(t.accepts(&Circuit::new(Level::Logical, 1, 0)), Err(BackendError::NotPhysical(_))));
        let mut e = Circuit::new(Level::Physical, 2, 0); e.push(Op::Cx { c: Qubit(1), t: Qubit(0) });
        assert_eq!(t.accepts(&e), Ok(()));
    }
    #[test] fn null_backend_roundtrip() {
        let mut b = NullBackend::new(line3());
        let mut c = Circuit::new(Level::Physical, 2, 0); c.push(Op::H { q: Qubit(0) });
        let job = b.submit(&c, 100).unwrap();
        assert_eq!(job.circuit_hash, c.hash());
        assert_eq!(b.result(&job).unwrap().counts.values().sum::<u64>(), 100);
    }
}
