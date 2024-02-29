from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})
import argparse
import sys

import carb
import numpy as np
import omni.isaac.core.utils.deformable_mesh_utils as deformableMeshUtils
import torch
from omni.isaac.core import World
from omni.isaac.core.materials.deformable_material import DeformableMaterial
from omni.isaac.core.prims.soft.deformable_prim import DeformablePrim
from omni.isaac.core.prims.soft.deformable_prim_view import DeformablePrimView
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import Gf, UsdGeom, UsdLux
import omni.physxdemos as demo
# from omni.physxdemos import DeformableBodyAttachmentsDemo

# The example shows how to create and manipulate environments with deformable deformable through the DeformablePrimView
parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    sys.exit()


class DeformableExample:
    def __init__(self):
        self._array_container = torch.Tensor
        self.my_world = World(stage_units_in_meters=1.0, backend="torch", device="cuda")
        self.stage = simulation_app.context.get_stage()
        self.num_envs = 10
        self.dimx = 5
        self.dimy = 5
        self.my_world.scene.add_default_ground_plane()
        self.initial_positions = None
        self.makeEnvs()

    def makeEnvs(self):
        env_path = "/World/Envs/Env"
        env = UsdGeom.Xform.Define(self.stage, env_path)
        mesh_path = env.GetPrim().GetPath().AppendChild("deformable")
        demo.DeformableBodyAttachmentsDemo(demo.Base).create_sphere_mesh(self.stage, mesh_path)
        demo.DeformableBodyAttachmentsDemo(demo.Base).create(self.stage)

    def play(self):
        while simulation_app.is_running():
            if self.my_world.is_playing():
                # deal with sim re-initialization after restarting sim
                if self.my_world.current_time_step_index == 1:
                    # initialize simulation views
                    self.my_world.reset(soft=False)

            self.my_world.step(render=True)

            # if self.my_world.current_time_step_index == 200:
            #     for i in range(self.num_envs):
            #         print(
            #             "deformable {} average height = {:.2f}".format(
            #                 i, self.deformableView.get_simulation_mesh_nodal_positions()[i, :, 2].mean()
            #             )
            #         )
            #         print(
            #             "deformable {} average vertical speed = {:.2f}".format(
            #                 i, self.deformableView.get_simulation_mesh_nodal_velocities()[i, :, 2].mean()
            #             )
            #         )

            # # reset some random environments
            # if self.my_world.current_time_step_index % 500 == 1:
            #     indices = torch.tensor(
            #         np.random.choice(range(self.num_envs), self.num_envs // 2, replace=False), dtype=torch.long
            #     )
            #     new_positions = self.initial_positions[indices] + torch.tensor([0, 0, 5])
            #     new_velocities = self.initial_velocities[indices] + torch.tensor([0, 0, 3])
            #     self.deformableView.set_simulation_mesh_nodal_positions(new_positions, indices)
            #     self.deformableView.set_simulation_mesh_nodal_velocities(new_velocities, indices)
            #     updated_positions = self.deformableView.get_simulation_mesh_nodal_positions()
            #     updated_velocities = self.deformableView.get_simulation_mesh_nodal_velocities()
            #     for i in indices:
            #         print("reset index {} average height = {:.2f}".format(i, updated_positions[i, :, 2].mean()))
            #         print(
            #             "reset index {} average vertical speed = {:.2f}".format(i, updated_velocities[i, :, 2].mean())
            #         )

        simulation_app.close()


DeformableExample().play()
