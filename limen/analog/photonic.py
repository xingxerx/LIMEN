from limen.analog.base import AnalogTargetAdapter, ContinuousFieldIR, HardwareDeltaModel
import limen_core

class PhotonicCompiler(AnalogTargetAdapter):
    def __init__(self):
        self._core = limen_core.analog.PhotonicCompiler()

    def compile_to_field(self, logical_graph: "limen_core.sim.LogicalGraph", delta_model: HardwareDeltaModel) -> ContinuousFieldIR:
        return self._core.compile_to_field(logical_graph, delta_model)

    def compute_convex_scaling(self, field: ContinuousFieldIR) -> None:
        self._core.compute_convex_scaling(field)

    def build_gbs_encoding(self, logical_graph: "limen_core.sim.LogicalGraph") -> "limen_core.analog.PhotonicGBSParameters":
        """
        Reformulates optimization into an explicit Arrazola-Bromley adjacency encoding matrix.
        """
        return self._core.build_arrazola_bromley_encoding(logical_graph)
