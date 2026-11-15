//! OpenQASM 3 subset <-> limen-ir. Exchange format, not a new syntax.
//! Subset: `OPENQASM 3.0; include "stdgates.inc"; qubit[n] q; bit[m] c;`
//! then one statement per line from the IR gate set. Parser never panics on input.

use limen_ir::{Circuit, Clbit, IrError, Level, Op, Param, Qubit};
use std::fmt::Write;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum QasmError {
    #[error("line {line}: {msg}")]
    Syntax { line: usize, msg: String },
    #[error("ir: {0}")]
    Ir(#[from] IrError),
}

fn err<T>(line: usize, msg: impl Into<String>) -> Result<T, QasmError> { Err(QasmError::Syntax { line, msg: msg.into() }) }

fn param(line: usize, s: &str) -> Result<Param, QasmError> {
    let s = s.trim();
    if s == "pi" { return Ok(Param::Lit(std::f64::consts::PI)); }
    if let Ok(v) = s.parse::<f64>() { return if v.is_finite() { Ok(Param::Lit(v)) } else { err(line, "non-finite angle") }; }
    if s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && s.chars().next().map_or(false, |c| c.is_ascii_alphabetic()) {
        return Ok(Param::Sym(s.to_string()));
    }
    err(line, format!("bad parameter {s:?}"))
}

/// Parse `q[3]` against a register name; returns the index.
fn idx(line: usize, s: &str, reg: &str) -> Result<u32, QasmError> {
    let s = s.trim();
    let inner = s.strip_prefix(reg).and_then(|r| r.strip_prefix('[')).and_then(|r| r.strip_suffix(']'));
    match inner.and_then(|i| i.parse::<u32>().ok()) { Some(i) => Ok(i), None => err(line, format!("expected {reg}[n], got {s:?}")) }
}

pub fn parse(src: &str) -> Result<Circuit, QasmError> {
    let (mut nq, mut nc, mut saw_header) = (None, None, false);
    let mut ops = Vec::new();
    for (i, raw) in src.lines().enumerate() {
        let line = i + 1;
        let stmt = raw.split("//").next().unwrap_or("").trim();
        if stmt.is_empty() { continue; }
        let stmt = match stmt.strip_suffix(';') { Some(s) => s.trim(), None => return err(line, "missing ';'") };
        if stmt.starts_with("OPENQASM") { saw_header = true; continue; }
        if stmt.starts_with("include") { continue; }
        if let Some(rest) = stmt.strip_prefix("qubit[") {
            let (n, name) = rest.split_once(']').ok_or_else(|| QasmError::Syntax { line, msg: "bad qubit decl".into() })?;
            if name.trim() != "q" { return err(line, "only register name 'q' is supported"); }
            nq = Some(n.parse::<u32>().map_err(|_| QasmError::Syntax { line, msg: "bad qubit count".into() })?); continue;
        }
        if let Some(rest) = stmt.strip_prefix("bit[") {
            let (n, name) = rest.split_once(']').ok_or_else(|| QasmError::Syntax { line, msg: "bad bit decl".into() })?;
            if name.trim() != "c" { return err(line, "only register name 'c' is supported"); }
            nc = Some(n.parse::<u32>().map_err(|_| QasmError::Syntax { line, msg: "bad bit count".into() })?); continue;
        }
        // measurement: c[i] = measure q[j];
        if let Some((lhs, rhs)) = stmt.split_once('=') {
            let rhs = rhs.trim().strip_prefix("measure").ok_or_else(|| QasmError::Syntax { line, msg: "expected measure".into() })?;
            ops.push(Op::Measure { q: Qubit(idx(line, rhs, "q")?), c: Clbit(idx(line, lhs, "c")?) }); continue;
        }
        // gate [(params)] args
        let (head, args) = match stmt.find(|ch: char| ch == ' ' || ch == '(') { Some(p) => stmt.split_at(p), None => return err(line, "bad statement") };
        let (name, params) = match args.trim().strip_prefix('(') {
            Some(r) => { let (p, a) = r.split_once(')').ok_or_else(|| QasmError::Syntax { line, msg: "unclosed '('".into() })?; (head, Some((p, a))) }
            None => (head, None),
        };
        let argstr = match params { Some((_, a)) => a, None => args };
        let qs: Vec<u32> = argstr.split(',').filter(|a| !a.trim().is_empty()).map(|a| idx(line, a, "q")).collect::<Result<_, _>>()?;
        let one = |qs: &[u32]| -> Result<Qubit, QasmError> { if qs.len() == 1 { Ok(Qubit(qs[0])) } else { err(line, "expected 1 qubit") } };
        let two = |qs: &[u32]| -> Result<(Qubit, Qubit), QasmError> { if qs.len() == 2 { Ok((Qubit(qs[0]), Qubit(qs[1]))) } else { err(line, "expected 2 qubits") } };
        let theta = || -> Result<Param, QasmError> { match params { Some((p, _)) => param(line, p), None => err(line, "missing angle") } };
        let op = match name {
            "h" => Op::H { q: one(&qs)? }, "x" => Op::X { q: one(&qs)? }, "y" => Op::Y { q: one(&qs)? }, "z" => Op::Z { q: one(&qs)? },
            "s" => Op::S { q: one(&qs)? }, "t" => Op::T { q: one(&qs)? },
            "rx" => Op::Rx { q: one(&qs)?, theta: theta()? }, "ry" => Op::Ry { q: one(&qs)?, theta: theta()? }, "rz" => Op::Rz { q: one(&qs)?, theta: theta()? },
            "cx" => { let (c, t) = two(&qs)?; Op::Cx { c, t } }, "cz" => { let (c, t) = two(&qs)?; Op::Cz { c, t } },
            "swap" => { let (a, b) = two(&qs)?; Op::Swap { a, b } },
            "reset" => Op::Reset { q: one(&qs)? },
            "barrier" => Op::Barrier { qs: qs.into_iter().map(Qubit).collect() },
            other => return err(line, format!("unsupported gate {other:?}")),
        };
        ops.push(op);
    }
    if !saw_header { return err(0, "missing OPENQASM 3 header"); }
    let mut c = Circuit::new(Level::Logical, nq.unwrap_or(0), nc.unwrap_or(0));
    c.ops = ops; c.validate()?; Ok(c)
}

fn fmt_param(p: &Param) -> String { match p { Param::Lit(v) => format!("{v:?}"), Param::Sym(s) => s.clone() } }

pub fn emit(c: &Circuit) -> String {
    let mut s = String::new();
    s.push_str("OPENQASM 3.0;\ninclude \"stdgates.inc\";\n");
    let _ = writeln!(s, "qubit[{}] q;", c.n_qubits);
    if c.n_clbits > 0 { let _ = writeln!(s, "bit[{}] c;", c.n_clbits); }
    for op in &c.ops {
        let line = match op {
            Op::Rx { q, theta } | Op::Ry { q, theta } | Op::Rz { q, theta } => format!("{}({}) q[{}];", op.name(), fmt_param(theta), q.0),
            Op::Measure { q, c } => format!("c[{}] = measure q[{}];", c.0, q.0),
            _ => format!("{} {};", op.name(), op.qubits().iter().map(|q| format!("q[{}]", q.0)).collect::<Vec<_>>().join(", ")),
        };
        s.push_str(&line); s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const BELL: &str = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nbit[2] c;\nh q[0];\ncx q[0], q[1];\nrz(pi) q[1]; // comment\nc[0] = measure q[0];\nc[1] = measure q[1];\n";

    #[test] fn parses_bell_and_roundtrips() {
        let c = parse(BELL).unwrap();
        assert_eq!(c.ops.len(), 5);
        let again = parse(&emit(&c)).unwrap();
        assert_eq!(again.hash(), c.hash());
    }
    #[test] fn errors_not_panics() {
        assert!(parse("qubit[2] q;").is_err());
        assert!(parse("OPENQASM 3.0;\nqubit[1] q;\ncx q[0], q[1];").is_err());
        assert!(parse("OPENQASM 3.0;\nqubit[1] q;\nfoo q[0];").is_err());
        assert!(parse("OPENQASM 3.0;\nqubit[1] q;\nh q[0]").is_err());
    }
    fn arb_op() -> impl Strategy<Value = Op> {
        prop_oneof![
            (0u32..4).prop_map(|a| Op::H { q: Qubit(a) }),
            (0u32..4, -7.0f64..7.0).prop_map(|(a, t)| Op::Ry { q: Qubit(a), theta: Param::Lit(t) }),
            (0u32..4).prop_map(|a| Op::Rz { q: Qubit(a), theta: Param::Sym("theta".into()) }),
            (0u32..4, 0u32..4).prop_filter("distinct", |(a, b)| a != b).prop_map(|(a, b)| Op::Cx { c: Qubit(a), t: Qubit(b) }),
            (0u32..4, 0u32..2).prop_map(|(a, b)| Op::Measure { q: Qubit(a), c: Clbit(b) }),
        ]
    }
    proptest! {
        #[test] fn roundtrip_lossless(ops in prop::collection::vec(arb_op(), 0..30)) {
            let mut c = Circuit::new(Level::Logical, 4, 2);
            for o in ops { c.push(o); }
            let back = parse(&emit(&c)).unwrap();
            prop_assert_eq!(back.hash(), c.hash());
        }
        #[test] fn fuzz_never_panics(s in "[ -~\n]{0,200}") { let _ = parse(&s); }
    }
}
