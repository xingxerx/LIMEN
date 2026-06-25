class SizeViolation(Exception):
    """Raised when the input LogicalGraphIR contains more than 20 nodes."""
    pass


class GateSynthesisError(Exception):
    """Raised when a unitary cannot be synthesized into the supported gate set.

    Multi-qubit unitary synthesis (KAK decomposition etc.) is not
    implemented; use qiskit.transpile or qiskit.circuit.library.UnitaryGate
    for those cases.
    """
    pass
