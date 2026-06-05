"""Hardware backend adapters for LIMEN."""


def __getattr__(name: str):
    if name in ("DWaveResult", "run_dwave"):
        from limen.backends.dwave import DWaveResult, run_dwave
        return {"DWaveResult": DWaveResult, "run_dwave": run_dwave}[name]
    if name in ("QiskitResult", "run_qiskit"):
        from limen.backends.qiskit_backend import QiskitResult, run_qiskit
        return {"QiskitResult": QiskitResult, "run_qiskit": run_qiskit}[name]
    raise AttributeError(f"module 'limen.backends' has no attribute {name!r}")


__all__ = ["DWaveResult", "run_dwave", "QiskitResult", "run_qiskit"]
