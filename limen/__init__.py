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

__version__ = "0.1.0"

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
]
