# limen/quantum_channel/channel_delta.py
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class ChannelDeltaModel:
    """
    Models classical feedforward latency and its effect on qubit
    coherence during teleportation. Analogous to HardwareDeltaModel
    but for the inter-node classical channel rather than QPU calibration.
    """
    latency_ms: float    # Classical channel round-trip latency
    t2_us: float         # QPU T2 coherence time in microseconds
    gate_time_us: float = 0.1

    def within_coherence(self) -> bool:
        """True if feedforward completes before T2 decay."""
        return (self.latency_ms * 1000.0) < self.t2_us

    def fidelity_penalty(self) -> float:
        """Exponential decay estimate: exp(-latency / T2)."""
        t_us = self.latency_ms * 1000.0
        return math.exp(-t_us / self.t2_us)

    def to_dict(self) -> dict:
        return {
            "latency_ms": self.latency_ms,
            "t2_us": self.t2_us,
            "gate_time_us": self.gate_time_us,
            "within_coherence": self.within_coherence(),
            "fidelity_penalty": self.fidelity_penalty(),
        }
