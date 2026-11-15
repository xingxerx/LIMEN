//! Pass pipeline. Every pass declares its level transition; the pipeline
//! re-validates the circuit and the declared invariants after each pass.

use limen_ir::{Circuit, IrError, Level, Op};

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PassError {
    #[error("pass {pass}: {source}")]
    Ir { pass: &'static str, #[source] source: IrError },
    #[error("pass {pass} changed qubit count {before} -> {after}")]
    QubitCountChanged { pass: &'static str, before: u32, after: u32 },
    #[error("pass {pass} expected level {expected:?}, got {got:?}")]
    WrongLevel { pass: &'static str, expected: Level, got: Level },
    #[error("pass {pass} failed: {msg}")]
    Failed { pass: &'static str, msg: String },
}

pub trait Pass {
    fn name(&self) -> &'static str;
    /// Level the pass consumes.
    fn input_level(&self) -> Level;
    /// Level the pass produces (same as input for optimizations).
    fn output_level(&self) -> Level;
    fn run(&self, c: &Circuit) -> Result<Circuit, PassError>;
}

pub struct Pipeline { passes: Vec<Box<dyn Pass>> }

impl Pipeline {
    pub fn new() -> Self { Self { passes: Vec::new() } }
    pub fn then(mut self, p: impl Pass + 'static) -> Self { self.passes.push(Box::new(p)); self }
    pub fn run(&self, input: &Circuit) -> Result<Circuit, PassError> {
        let mut c = input.clone();
        for p in &self.passes {
            if c.level != p.input_level() { return Err(PassError::WrongLevel { pass: p.name(), expected: p.input_level(), got: c.level }); }
            let out = p.run(&c)?;
            out.validate().map_err(|e| PassError::Ir { pass: p.name(), source: e })?;
            if out.n_qubits != c.n_qubits { return Err(PassError::QubitCountChanged { pass: p.name(), before: c.n_qubits, after: out.n_qubits }); }
            if out.level != p.output_level() { return Err(PassError::WrongLevel { pass: p.name(), expected: p.output_level(), got: out.level }); }
            c = out;
        }
        Ok(c)
    }
}

impl Default for Pipeline { fn default() -> Self { Self::new() } }

/// Removes adjacent self-inverse pairs (H H, X X, Y Y, Z Z, CX CX, SWAP SWAP) on identical qubits.
pub struct CancelInverses;

fn self_inverse_pair(a: &Op, b: &Op) -> bool {
    matches!((a, b), (Op::H { .. }, Op::H { .. }) | (Op::X { .. }, Op::X { .. }) | (Op::Y { .. }, Op::Y { .. })
        | (Op::Z { .. }, Op::Z { .. }) | (Op::Cx { .. }, Op::Cx { .. }) | (Op::Swap { .. }, Op::Swap { .. })) && a.qubits() == b.qubits()
}

impl Pass for CancelInverses {
    fn name(&self) -> &'static str { "cancel-inverses" }
    fn input_level(&self) -> Level { Level::Logical }
    fn output_level(&self) -> Level { Level::Logical }
    fn run(&self, c: &Circuit) -> Result<Circuit, PassError> {
        let mut out = c.clone(); out.ops.clear();
        for op in &c.ops {
            if let Some(last) = out.ops.last() { if self_inverse_pair(last, op) { out.ops.pop(); continue; } }
            out.ops.push(op.clone());
        }
        Ok(out)
    }
}

/// Logical -> Physical with the identity layout. Placeholder for real mapping;
/// records the layout in meta so the certificate can trace it.
pub struct TrivialLayout;

impl Pass for TrivialLayout {
    fn name(&self) -> &'static str { "trivial-layout" }
    fn input_level(&self) -> Level { Level::Logical }
    fn output_level(&self) -> Level { Level::Physical }
    fn run(&self, c: &Circuit) -> Result<Circuit, PassError> {
        let mut out = c.clone(); out.level = Level::Physical;
        out.meta.insert("layout".into(), (0..c.n_qubits).map(|i| i.to_string()).collect::<Vec<_>>().join(","));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use limen_ir::Qubit;
    use proptest::prelude::*;

    #[test] fn cancels_hh_and_keeps_rest() {
        let mut c = Circuit::new(Level::Logical, 2, 0);
        c.push(Op::H { q: Qubit(0) }).push(Op::H { q: Qubit(0) }).push(Op::Cx { c: Qubit(0), t: Qubit(1) }).push(Op::H { q: Qubit(1) });
        let out = Pipeline::new().then(CancelInverses).run(&c).unwrap();
        assert_eq!(out.ops.len(), 2);
    }
    #[test] fn level_mismatch_is_error() {
        let c = Circuit::new(Level::Physical, 1, 0);
        assert!(matches!(Pipeline::new().then(CancelInverses).run(&c), Err(PassError::WrongLevel { .. })));
    }
    #[test] fn layout_lowers_level() {
        let out = Pipeline::new().then(CancelInverses).then(TrivialLayout).run(&Circuit::new(Level::Logical, 3, 0)).unwrap();
        assert_eq!(out.level, Level::Physical);
        assert_eq!(out.meta["layout"], "0,1,2");
    }
    fn arb_op() -> impl Strategy<Value = Op> {
        prop_oneof![(0u32..3).prop_map(|a| Op::H { q: Qubit(a) }), (0u32..3).prop_map(|a| Op::X { q: Qubit(a) }),
            (0u32..3, 0u32..3).prop_filter("d", |(a, b)| a != b).prop_map(|(a, b)| Op::Cx { c: Qubit(a), t: Qubit(b) })]
    }
    proptest! {
        #[test] fn cancel_is_idempotent_and_never_grows(ops in prop::collection::vec(arb_op(), 0..40)) {
            let mut c = Circuit::new(Level::Logical, 3, 0); for o in ops { c.push(o); }
            let p = Pipeline::new().then(CancelInverses);
            let once = p.run(&c).unwrap(); let twice = p.run(&once).unwrap();
            prop_assert!(once.ops.len() <= c.ops.len());
            prop_assert_eq!(once.hash(), twice.hash());
        }
    }
}
