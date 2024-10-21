from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from typing import Optional

class DensoRobotView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "DensoRobotView",
    ) -> None:
        """Initialize articulation view for Denso robot without fingers."""

        super().__init__(
            prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False
        )

        self._end_effector = RigidPrimView(
            prim_paths_expr="/World/envs/.*/denso_robot/link_ee",
            name="end_effector_view",
            reset_xform_properties=False,
        )

    def initialize(self, physics_sim_view):
        """Initialize physics simulation view for Denso robot."""

        super().initialize(physics_sim_view)
