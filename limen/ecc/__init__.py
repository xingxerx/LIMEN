"""Quantum error correction: surface code encoding, syndrome extraction, and decoding.

A logical-qubit layer above the physical compilation passes. Distinct
from limen.analog.certificate.CompilationCertificate (Hamiltonian
coefficient error / Theorem 1) - this package's LogicalErrorCertificate
quantifies quantum logical error rate instead.
"""

from limen.ecc.certificate import LogicalErrorCertificate, certify_logical_qubit
from limen.ecc.decoder import LookupDecoder
from limen.ecc.surface_code import SurfaceCodePatch, build_surface_code
from limen.ecc.syndrome import build_syndrome_circuit

__all__ = [
    "SurfaceCodePatch",
    "build_surface_code",
    "build_syndrome_circuit",
    "LookupDecoder",
    "LogicalErrorCertificate",
    "certify_logical_qubit",
]
