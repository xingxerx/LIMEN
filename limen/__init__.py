"""LIMEN — A physics-aware compiler stack for translating classical
optimization problems into native quantum and analog substrates."""

from limen.core.compiler import (
    PhysicalEncoding,
    compile_lexicographic,
    default_hardware_graph,
)
from limen.core.ir import Interaction, LogicalGraph, Variable
from limen.frontends.pyqubo import from_pyqubo, from_qubo_dict
from limen.validator.validator import ValidationResult, validate

__version__ = "0.4.0"

__all__ = [
    "Variable",
    "Interaction",
    "LogicalGraph",
    "PhysicalEncoding",
    "compile_lexicographic",
    "default_hardware_graph",
    "ValidationResult",
    "validate",
    "from_qubo_dict",
    "from_pyqubo",
    "CoDesignResult",
    "run_codesign",
    "PortfolioResult",
    "compile_portfolio",
    "HamiltonianIR",
    "HamiltonianTerm",
    "SubstrateType",
    "from_physical_encoding",
    "CompilationCertificate",
    "certify_ising",
    "load_quera_calibration",
    "load_ibmq_calibration",
    "load_live_ibmq_calibration",
    "QuantumChannel",
    "TeleportationResult",
    "QKDResult",
]



def __getattr__(name: str):
    if name in ("CoDesignResult", "run_codesign", "PortfolioResult", "compile_portfolio"):
        from limen.codesign import (
            CoDesignResult,
            PortfolioResult,
            compile_portfolio,
            run_codesign,
        )
        return {
            "CoDesignResult": CoDesignResult,
            "run_codesign": run_codesign,
            "PortfolioResult": PortfolioResult,
            "compile_portfolio": compile_portfolio,
        }[name]
    if name in ("HamiltonianIR", "HamiltonianTerm", "SubstrateType", "from_physical_encoding"):
        from limen.analog.hamiltonian import (
            HamiltonianIR,
            HamiltonianTerm,
            SubstrateType,
            from_physical_encoding,
        )
        return {
            "HamiltonianIR": HamiltonianIR,
            "HamiltonianTerm": HamiltonianTerm,
            "SubstrateType": SubstrateType,
            "from_physical_encoding": from_physical_encoding,
        }[name]
    if name in ("CompilationCertificate", "certify_ising"):
        from limen.analog.certificate import CompilationCertificate, certify_ising
        return {
            "CompilationCertificate": CompilationCertificate,
            "certify_ising": certify_ising,
        }[name]
    if name in ("load_quera_calibration", "load_ibmq_calibration", "load_live_ibmq_calibration"):
        from limen.analog.calibration_loader import (
            load_quera_calibration,
            load_ibmq_calibration,
            load_live_ibmq_calibration,
        )
        return {
            "load_quera_calibration": load_quera_calibration,
            "load_ibmq_calibration": load_ibmq_calibration,
            "load_live_ibmq_calibration": load_live_ibmq_calibration,
        }[name]
    if name in ("QuantumChannel", "TeleportationResult", "QKDResult"):
        from limen.communication.channel import (
            QuantumChannel,
            TeleportationResult,
            QKDResult,
        )
        return {
            "QuantumChannel": QuantumChannel,
            "TeleportationResult": TeleportationResult,
            "QKDResult": QKDResult,
        }[name]
    raise AttributeError(f"module 'limen' has no attribute {name!r}")


