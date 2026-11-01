//! `limen.limen_core.ir`: Python face of the Rust IR workspace (rust/).
//! Circuits cross the boundary as canonical JSON or OpenQASM 3 text.

use limen_ir::Circuit;
use limen_pass::{CancelInverses, Pipeline, TrivialLayout};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn load(json: &str) -> PyResult<Circuit> { Circuit::from_json(json).map_err(|e| PyValueError::new_err(e.to_string())) }
fn dump(c: &Circuit) -> PyResult<String> { String::from_utf8(c.canonical_bytes()).map_err(|e| PyValueError::new_err(e.to_string())) }

/// OpenQASM 3 text -> canonical IR JSON.
#[pyfunction]
pub fn from_qasm3(src: &str) -> PyResult<String> {
    dump(&limen_qasm::parse(src).map_err(|e| PyValueError::new_err(e.to_string()))?)
}

/// Canonical IR JSON -> OpenQASM 3 text.
#[pyfunction]
pub fn to_qasm3(json: &str) -> PyResult<String> { Ok(limen_qasm::emit(&load(json)?)) }

/// SHA-256 of the canonical IR bytes; cache and certificate key.
#[pyfunction]
pub fn hash(json: &str) -> PyResult<String> { Ok(load(json)?.hash()) }

/// Validate and return the circuit's canonical JSON (normalizes key order).
#[pyfunction]
pub fn canonical(json: &str) -> PyResult<String> { dump(&load(json)?) }

/// Run the logical optimization pipeline (currently: cancel-inverses).
#[pyfunction]
pub fn optimize(json: &str) -> PyResult<String> {
    let out = Pipeline::new().then(CancelInverses).run(&load(json)?).map_err(|e| PyValueError::new_err(e.to_string()))?;
    dump(&out)
}

/// Lower Logical -> Physical with the identity layout.
#[pyfunction]
pub fn lower(json: &str) -> PyResult<String> {
    let out = Pipeline::new().then(TrivialLayout).run(&load(json)?).map_err(|e| PyValueError::new_err(e.to_string()))?;
    dump(&out)
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let ir = PyModule::new_bound(parent.py(), "ir")?;
    ir.add_function(wrap_pyfunction!(from_qasm3, &ir)?)?;
    ir.add_function(wrap_pyfunction!(to_qasm3, &ir)?)?;
    ir.add_function(wrap_pyfunction!(hash, &ir)?)?;
    ir.add_function(wrap_pyfunction!(canonical, &ir)?)?;
    ir.add_function(wrap_pyfunction!(optimize, &ir)?)?;
    ir.add_function(wrap_pyfunction!(lower, &ir)?)?;
    ir.add("SCHEMA_VER", limen_ir::SCHEMA_VER)?;
    parent.add_submodule(&ir)
}
