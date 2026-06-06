from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass
import limen_core

@dataclass
class ContinuousFieldIR:
    data: list[float]

@dataclass
class HardwareDeltaModel:
    calibration_data: list[float]

class AnalogTargetAdapter(ABC):
    @abstractmethod
    def compile_to_field(self, logical_graph: "limen_core.sim.LogicalGraph", delta_model: HardwareDeltaModel) -> ContinuousFieldIR:
        """
        Compiles the logical graph to a continuous field IR using hardware calibration data.
        """
        pass

    @abstractmethod
    def compute_convex_scaling(self, field: ContinuousFieldIR) -> None:
        """
        Adjusts problem weights dynamically according to live calibration telemetry.
        """
        pass
