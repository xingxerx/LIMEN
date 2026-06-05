"""Analog hardware backend stubs for LIMEN."""

from limen.analog.backends.neutral_atom import NeutralAtomResult, run_neutral_atom
from limen.analog.backends.photonic import PhotonicResult, run_photonic

__all__ = [
    "NeutralAtomResult",
    "run_neutral_atom",
    "PhotonicResult",
    "run_photonic",
]
