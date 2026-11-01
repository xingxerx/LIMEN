"""Rust IR bridge: Qiskit <-> limen-ir through OpenQASM 3.

The IR crosses the boundary as canonical JSON text. Hashes are stable, so
``hash(c)`` can key caches and certificates.
"""
from __future__ import annotations

from typing import Any

from limen.limen_core import ir as _ir

SCHEMA_VER: int = _ir.SCHEMA_VER


def from_qasm3(src: str) -> str:
    return _ir.from_qasm3(src)


def to_qasm3(circuit_json: str) -> str:
    return _ir.to_qasm3(circuit_json)


def hash(circuit_json: str) -> str:  # noqa: A001 - mirrors the Rust name
    return _ir.hash(circuit_json)


def canonical(circuit_json: str) -> str:
    return _ir.canonical(circuit_json)


def optimize(circuit_json: str) -> str:
    return _ir.optimize(circuit_json)


def lower(circuit_json: str) -> str:
    return _ir.lower(circuit_json)


def from_qiskit(qc: Any) -> str:
    """QuantumCircuit -> IR JSON. Registers must be the defaults ``q`` and ``c``."""
    from qiskit import qasm3

    return from_qasm3(qasm3.dumps(qc))


def to_qiskit(circuit_json: str) -> Any:
    """IR JSON -> QuantumCircuit."""
    from qiskit import qasm3

    return qasm3.loads(to_qasm3(circuit_json))
