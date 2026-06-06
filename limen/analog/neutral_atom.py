from dataclasses import dataclass
from typing import List
from limen.analog.base import AnalogTargetAdapter, ContinuousFieldIR, HardwareDeltaModel
import limen_core

class NeutralAtomCompiler(AnalogTargetAdapter):
    def __init__(self):
        self._core = limen_core.analog.NeutralAtomCompiler()

    def compile_to_field(self, logical_graph: "limen_core.sim.LogicalGraph", delta_model: HardwareDeltaModel) -> ContinuousFieldIR:
        # Internal mapping logic would call Rust optimized spatial solvers
        return self._core.compile_to_field(logical_graph, delta_model)

    def compute_convex_scaling(self, field: ContinuousFieldIR) -> None:
        self._core.compute_convex_scaling(field)

    def generate_layout(self, logical_graph: "limen_core.sim.LogicalGraph") -> "limen_core.analog.RydbergControlParameters":
        """
        Maps logical graph to spatial coordinates and generates Rydberg pulse schedules.
        """
        coords = self._core.map_to_spatial_geometry(logical_graph)
        return self._core.generate_rydberg_controls(logical_graph, coords)
