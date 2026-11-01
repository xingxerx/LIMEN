//! limen-ir: the LIMEN intermediate representation.
//!
//! Rules (see README): versioned, additive only; no backend details here;
//! three levels (Logical -> Physical -> Pulse) that lower one step at a time;
//! canonical serialization so equal circuits hash equal.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Bump only when a reader of the previous version could misread a file.
pub const SCHEMA_VER: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Level { Logical, Physical, Pulse }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Qubit(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Clbit(pub u32);

/// Gate parameter: a literal angle, or a symbol bound at execution time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "p", content = "v")]
pub enum Param { Lit(f64), Sym(String) }

/// Closed gate set at the logical level. Backends declare which subset they accept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "op")]
pub enum Op {
    H { q: Qubit },
    X { q: Qubit },
    Y { q: Qubit },
    Z { q: Qubit },
    S { q: Qubit },
    T { q: Qubit },
    Rx { q: Qubit, theta: Param },
    Ry { q: Qubit, theta: Param },
    Rz { q: Qubit, theta: Param },
    Cx { c: Qubit, t: Qubit },
    Cz { c: Qubit, t: Qubit },
    Swap { a: Qubit, b: Qubit },
    Measure { q: Qubit, c: Clbit },
    Reset { q: Qubit },
    Barrier { qs: Vec<Qubit> },
}

impl Op {
    pub fn qubits(&self) -> Vec<Qubit> {
        match self {
            Op::H { q } | Op::X { q } | Op::Y { q } | Op::Z { q } | Op::S { q } | Op::T { q }
            | Op::Rx { q, .. } | Op::Ry { q, .. } | Op::Rz { q, .. } | Op::Measure { q, .. } | Op::Reset { q } => vec![*q],
            Op::Cx { c, t } | Op::Cz { c, t } => vec![*c, *t],
            Op::Swap { a, b } => vec![*a, *b],
            Op::Barrier { qs } => qs.clone(),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Op::H { .. } => "h", Op::X { .. } => "x", Op::Y { .. } => "y", Op::Z { .. } => "z",
            Op::S { .. } => "s", Op::T { .. } => "t", Op::Rx { .. } => "rx", Op::Ry { .. } => "ry",
            Op::Rz { .. } => "rz", Op::Cx { .. } => "cx", Op::Cz { .. } => "cz", Op::Swap { .. } => "swap",
            Op::Measure { .. } => "measure", Op::Reset { .. } => "reset", Op::Barrier { .. } => "barrier",
        }
    }
}

/// A circuit. `meta` is a BTreeMap so serialization order is canonical.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Circuit {
    pub schema_ver: u16,
    pub level: Level,
    pub n_qubits: u32,
    pub n_clbits: u32,
    pub ops: Vec<Op>,
    #[serde(default)]
    pub meta: BTreeMap<String, String>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum IrError {
    #[error("op {index} ({op}) uses qubit {qubit} but circuit has {n_qubits}")]
    QubitOutOfRange { index: usize, op: &'static str, qubit: u32, n_qubits: u32 },
    #[error("op {index} uses clbit {clbit} but circuit has {n_clbits}")]
    ClbitOutOfRange { index: usize, clbit: u32, n_clbits: u32 },
    #[error("op {index} ({op}) repeats a qubit")]
    DuplicateQubit { index: usize, op: &'static str },
    #[error("schema version {found} is newer than supported {supported}")]
    SchemaTooNew { found: u16, supported: u16 },
    #[error("non-finite parameter in op {index}")]
    NonFiniteParam { index: usize },
}

impl Circuit {
    pub fn new(level: Level, n_qubits: u32, n_clbits: u32) -> Self {
        Self { schema_ver: SCHEMA_VER, level, n_qubits, n_clbits, ops: Vec::new(), meta: BTreeMap::new() }
    }

    pub fn push(&mut self, op: Op) -> &mut Self { self.ops.push(op); self }

    /// Structural invariants every pass must preserve.
    pub fn validate(&self) -> Result<(), IrError> {
        if self.schema_ver > SCHEMA_VER { return Err(IrError::SchemaTooNew { found: self.schema_ver, supported: SCHEMA_VER }); }
        for (index, op) in self.ops.iter().enumerate() {
            let qs = op.qubits();
            for q in &qs {
                if q.0 >= self.n_qubits { return Err(IrError::QubitOutOfRange { index, op: op.name(), qubit: q.0, n_qubits: self.n_qubits }); }
            }
            let mut sorted = qs.clone(); sorted.sort(); sorted.dedup();
            if sorted.len() != qs.len() { return Err(IrError::DuplicateQubit { index, op: op.name() }); }
            if let Op::Measure { c, .. } = op {
                if c.0 >= self.n_clbits { return Err(IrError::ClbitOutOfRange { index, clbit: c.0, n_clbits: self.n_clbits }); }
            }
            if let Op::Rx { theta: Param::Lit(v), .. } | Op::Ry { theta: Param::Lit(v), .. } | Op::Rz { theta: Param::Lit(v), .. } = op {
                if !v.is_finite() { return Err(IrError::NonFiniteParam { index }); }
            }
        }
        Ok(())
    }

    /// Canonical bytes: compact JSON with deterministic key order (serde struct
    /// order plus BTreeMap). Equal circuits give equal bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> { serde_json::to_vec(self).expect("IR is always serializable") }

    /// SHA-256 of the canonical bytes; the key for caches and certificates.
    pub fn hash(&self) -> String {
        let mut h = Sha256::new(); h.update(self.canonical_bytes());
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    pub fn from_json(s: &str) -> Result<Self, IrError> {
        let c: Circuit = serde_json::from_str(s).map_err(|_| IrError::SchemaTooNew { found: u16::MAX, supported: SCHEMA_VER })?;
        c.validate()?; Ok(c)
    }

    pub fn two_qubit_count(&self) -> usize { self.ops.iter().filter(|o| o.qubits().len() == 2).count() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn bell() -> Circuit {
        let mut c = Circuit::new(Level::Logical, 2, 2);
        c.push(Op::H { q: Qubit(0) }).push(Op::Cx { c: Qubit(0), t: Qubit(1) })
         .push(Op::Measure { q: Qubit(0), c: Clbit(0) }).push(Op::Measure { q: Qubit(1), c: Clbit(1) });
        c
    }

    #[test] fn bell_valid_and_hash_stable() {
        let c = bell(); c.validate().unwrap();
        assert_eq!(c.hash(), bell().hash());
        let back = Circuit::from_json(&String::from_utf8(c.canonical_bytes()).unwrap()).unwrap();
        assert_eq!(back, c);
    }
    #[test] fn meta_order_is_canonical() {
        let mut a = bell(); a.meta.insert("z".into(), "1".into()); a.meta.insert("a".into(), "2".into());
        let mut b = bell(); b.meta.insert("a".into(), "2".into()); b.meta.insert("z".into(), "1".into());
        assert_eq!(a.hash(), b.hash());
    }
    #[test] fn out_of_range_rejected() {
        let mut c = Circuit::new(Level::Logical, 1, 0); c.push(Op::Cx { c: Qubit(0), t: Qubit(1) });
        assert!(matches!(c.validate(), Err(IrError::QubitOutOfRange { .. })));
        let mut d = Circuit::new(Level::Logical, 2, 0); d.push(Op::Cx { c: Qubit(1), t: Qubit(1) });
        assert_eq!(d.validate(), Err(IrError::DuplicateQubit { index: 0, op: "cx" }));
    }
    #[test] fn newer_schema_rejected() {
        let mut c = bell(); c.schema_ver = SCHEMA_VER + 1;
        assert!(matches!(c.validate(), Err(IrError::SchemaTooNew { .. })));
    }

    fn arb_op(n: u32) -> impl Strategy<Value = Op> {
        let q = 0..n;
        prop_oneof![
            q.clone().prop_map(|a| Op::H { q: Qubit(a) }),
            (q.clone(), -10.0f64..10.0).prop_map(|(a, t)| Op::Rz { q: Qubit(a), theta: Param::Lit(t) }),
            (q.clone(), q.clone()).prop_filter("distinct", |(a, b)| a != b).prop_map(|(a, b)| Op::Cx { c: Qubit(a), t: Qubit(b) }),
        ]
    }
    proptest! {
        #[test] fn json_roundtrip(n in 2u32..6, ops in prop::collection::vec(arb_op(5), 0..20)) {
            let mut c = Circuit::new(Level::Logical, 5, 0);
            for o in ops { c.push(o); }
            let _ = n;
            prop_assert!(c.validate().is_ok());
            let back = Circuit::from_json(&String::from_utf8(c.canonical_bytes()).unwrap()).unwrap();
            prop_assert_eq!(back.hash(), c.hash());
        }
    }
}
