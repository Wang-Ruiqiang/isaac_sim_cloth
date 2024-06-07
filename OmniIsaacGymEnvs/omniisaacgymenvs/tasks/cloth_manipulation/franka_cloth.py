import hydra
import numpy as np
import omegaconf
import os
import torch
import argparse

import matplotlib.pyplot as plt

import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.sensor import Camera

from omni.isaac.core.utils.nucleus import get_assets_root_path

from omniisaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from omniisaacgymenvs.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv
from omniisaacgymenvs.tasks.factory.factory_base import FactoryBase


from omniisaacgymenvs.robots.articulations.franka import Franka
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView

import omniisaacgymenvs.tasks.factory.factory_control as fc
import omni.isaac.core.utils.torch as torch_utils
import omni.isaac.core.utils.deformable_mesh_utils as deformableMeshUtils

from omni.physx.scripts import physicsUtils, deformableUtils
from omni.isaac.core.materials.deformable_material import DeformableMaterial
from omni.isaac.core.prims.soft.deformable_prim import DeformablePrim
from omni.isaac.core.prims.soft.deformable_prim_view import DeformablePrimView
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.prims.soft.particle_system import ParticleSystem
from omni.isaac.core.prims.soft.cloth_prim import ClothPrim
from omni.isaac.core.prims.soft.cloth_prim_view import ClothPrimView
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
from omni.isaac.core.materials import ParticleMaterial
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.objects import DynamicCuboid, DynamicCone, FixedCone

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.views.factory_franka_view import FactoryFrankaView

from pxr import Usd, Gf, UsdGeom, Sdf, UsdShade
import omni.kit.commands
from omni.physx.scripts import utils, physicsUtils

class FrankaCloth(FactoryBase, FactoryABCEnv):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._get_env_yaml_params()
        super().__init__(name, sim_config, env)
    

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_env', node=FactorySchemaConfigEnv)

        config_path = 'task/GarmentEnv.yaml'  # relative to Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

        sim_config_path = 'task/FrankaClothManipulation.yaml'
        self.sim_cfg_env = hydra.compose(config_name=sim_config_path)
        self.sim_cfg_env = self.sim_cfg_env['task']
        self.rendering_dt = self.sim_cfg_env["sim"]["dt"]
        self.fps = 1.0 / self.rendering_dt

        asset_info_path = '../tasks/cloth_manipulation/yaml/asset_info_garment.yaml'
        self.asset_info_garment = hydra.compose(config_name=asset_info_path)
        self.asset_info_garment = self.asset_info_garment['']['']['']['tasks']['cloth_manipulation']['yaml']  # strip superfluous nesting
        
        return


    def set_up_scene(self, scene) -> None:
        self.import_franka_assets(add_to_stage=True)
        # self.create_cloth_material()
        RLTask.set_up_scene(self, scene, replicate_physics=False)
        self._import_env_assets(add_to_stage=True)
        # self.create_cone()
        # self.add_attachment()
        self.import_camera()
        
        self.frankas = FactoryFrankaView(prim_paths_expr="/World/envs/.*/franka", name="frankas_view")
        # self.cloth = RigidPrimView(prim_paths_expr = "/World/envs/.*/garment/garment/Plane_Plane_002", name="cloth_view")
        self.cloth = ClothPrimView(prim_paths_expr = "/World/envs/.*/garment/cloth", 
                                   name="cloth_view",
                                   )
        # self.deformableView = DeformablePrimView(prim_paths_expr="/World/envs/.*/deformable_object/deformable", name="deformableView")
        
        scene.add(self.frankas)
        scene.add(self.frankas._hands)
        scene.add(self.frankas._lfingers)
        scene.add(self.frankas._rfingers)
        scene.add(self.frankas._fingertip_centered)
        scene.add(self.cloth)
        # scene.add(self.deformableView)

        return
    
    def initialize_views(self, scene) -> None:
        """Initialize views for extension workflow."""

        super().initialize_views(scene)
        if scene.object_exists("frankas_view"):
            scene.remove_object("frankas_view", registry_only=True)
        if scene.object_exists("cloth_view"):
            scene.remove_object("cloth_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)
        if scene.object_exists("lfingers_view"):
            scene.remove_object("lfingers_view", registry_only=True)
        if scene.object_exists("rfingers_view"):
            scene.remove_object("rfingers_view", registry_only=True)
        if scene.object_exists("fingertips_view"):
            scene.remove_object("fingertips_view", registry_only=True)

        self.import_franka_assets(add_to_stage=True)
        self._import_env_assets(add_to_stage=True)

        self.frankas = FactoryFrankaView(prim_paths_expr="/World/envs/.*/franka", name="frankas_view")
        self.cloth = ClothPrimView(prim_paths_expr = "/World/envs/.*/garment/cloth", 
                                   name="cloth_view")
        
        scene.add(self.cloth)
        # scene.add(self.deformableView)
        scene.add(self.frankas)
        scene.add(self.frankas._hands)
        scene.add(self.frankas._lfingers)
        scene.add(self.frankas._rfingers)
        scene.add(self.frankas._fingertip_centered)


    def create_cloth_material(self):
        self.nutboltPhysicsMaterialPath = "/World/Physics_Materials/GarmentMaterial"

        utils.addRigidBodyMaterial(
            self._stage,
            self.nutboltPhysicsMaterialPath,
            density=self.cfg_env.env.garment_density,
            staticFriction=self.cfg_env.env.garment_friction,
            dynamicFriction=0.9,
            restitution=0.0,
        )
    

    def _import_env_assets(self, add_to_stage=True):
        self.garment_heights = []
        self.garment_widths_max = []
        self.initial_positions = None

        for i in range(self._num_envs):
            self.garment_translation = torch.tensor([0.0, 0.0, self.cfg_base.env.table_height], device=self._device)
            # self.garment_orientation = torch.tensor([0.5, 0.5, 0.5, 0.5])
            self.garment_orientation = torch.tensor([1, 0, 0, 0])
            # garment_scale = torch.tensor([0.01, 0.01, 0.01])
            self.garment_scale = torch.tensor([0.3, 0.3, 0.3])
            
            subassembly = self.cfg_env.env.desired_subassemblies[1]
            components = list(self.asset_info_garment[subassembly])
            garment_height = self.asset_info_garment[subassembly][components[0]]['height']
            garment_width_max = self.asset_info_garment[subassembly][components[0]]['width_max']
            self.garment_heights.append(garment_height)
            self.garment_widths_max.append(garment_width_max)
            # garment_file = self.asset_info_garment[subassembly][components[0]]['usd_path']
            # add_reference_to_stage(garment_file, f"/World/envs/env_{i}" + "/garment")
            if add_to_stage:
                self.import_cloth_view(i)
                # self.create_cone(i)
            # self.import_XFormPrim_View(i)
            
        
        self.garment_heights = torch.tensor(self.garment_heights, device=self._device).unsqueeze(-1)
        self.garment_widths_max = torch.tensor(self.garment_widths_max, device=self._device).unsqueeze(-1)


    def import_camera(self):
        self.camera = Camera(
            prim_path="/World/camera",
            frequency=self.fps,
            resolution=(1600, 900),
        )
       
        # self.camera.set_world_pose(
        #     position = np.array([-0.3, -0.1, 0.7]),
        #     orientation=rot_utils.euler_angles_to_quats(np.array([0, -50, -90]), degrees=True, extrinsic = False),
        #     camera_axes="usd",
        # )

        # self.camera.set_world_pose(
        #     position = np.array([0.0, -0.6, 0.7]),
        #     orientation=rot_utils.euler_angles_to_quats(np.array([60, 0, 0]), degrees=True, extrinsic = False),
        #     camera_axes="usd",
        # )

        self.camera.set_world_pose(
            position = np.array([-0.3, -0.3, 0.6]),
            orientation=rot_utils.euler_angles_to_quats(np.array([30, -50, -60]), degrees=True, extrinsic = False),
            camera_axes="usd",
        )
        self.camera.initialize()
        self.camera.set_clipping_range(near_distance = 0.01, far_distance = 1000000)
        self.camera.set_focal_length(1.2)
        self.camera.add_motion_vectors_to_frame()


    def import_XFormPrim_View(self, idx):
        XFormPrim(
                prim_path=f"/World/envs/env_{idx}" + "/garment",
                translation=self.garment_translation,
                orientation=self.garment_orientation,
                scale=self.garment_scale,
            )
        physicsUtils.add_physics_material_to_prim(
                self._stage, 
                self._stage.GetPrimAtPath(f"/World/envs/env_{idx}" + f"/garment/garment/Plane_Plane_002"), 
                self.nutboltPhysicsMaterialPath
            )
        
    def import_cloth_view(self, idx):
        # radius = 0.15 * (0.6 / 5.0)
        # restOffset = radius
        # contactOffset = restOffset * 1.01
               

        cloth_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        cloth_noise_xy = cloth_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.cloth_pos_xy_initial_noise, device=self.device))
        
        # cloth_x_pos = self.cfg_task.randomize.cloth_pos_xy_initial[0] + cloth_noise_xy[idx][0].item()
        # cloth_y_pos = self.cfg_task.randomize.cloth_pos_xy_initial[1] + cloth_noise_xy[idx][1].item()

        cloth_x_pos = self.cfg_task.randomize.cloth_pos_xy_initial[0]
        cloth_y_pos = self.cfg_task.randomize.cloth_pos_xy_initial[1]

        cloth_z_pos = self.cfg_base.env.table_height + 0.001
        # garment_position = torch.tensor([cloth_x_pos, cloth_y_pos, cloth_z_pos], device=self._device) 

        env_path = f"/World/envs/env_{idx}/garment"
        env = UsdGeom.Xform.Define(self._stage, env_path)
        # cloth_path = f"/World/envs/env_{idx}" + "/garment/garment/Plane_Plane_002"
        cloth_path = env.GetPrim().GetPath().AppendChild("cloth")

        self.plane_mesh = UsdGeom.Mesh.Define(self._stage, cloth_path)
        physicsUtils.set_or_add_translate_op(UsdGeom.Xformable(env), Gf.Vec3f(0, 0, 0))

        self.tri_points, self.tri_indices = deformableUtils.create_triangle_mesh_square(dimx=8, dimy=8, scale=0.2)
        self.plane_mesh.GetPointsAttr().Set(self.tri_points)
        self.plane_mesh.GetFaceVertexIndicesAttr().Set(self.tri_indices)
        self.plane_mesh.GetFaceVertexCountsAttr().Set([3] * (len(self.tri_indices) // 3))

        init_loc = Gf.Vec3f(cloth_x_pos, cloth_y_pos, cloth_z_pos)
        physicsUtils.setup_transform_as_scale_orient_translate(self.plane_mesh)
        physicsUtils.set_or_add_translate_op(self.plane_mesh, init_loc)
        physicsUtils.set_or_add_orient_op(self.plane_mesh, Gf.Rotation(Gf.Vec3d([1, 0, 0]), 20).GetQuat()) #修改cloth的oritation

        # Define the material
        material_path = env.GetPrim().GetPath().AppendChild("clothMaterial")
        material = UsdShade.Material.Define(self._stage, material_path)
        shader_path = material_path.AppendChild("PBRShader")
        self.shader = UsdShade.Shader.Define(self._stage, shader_path)
        self.shader.CreateIdAttr("UsdPreviewSurface")

        # Set shader attributes for base color
        self.shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0.5))    #设置布料颜色
        material.CreateSurfaceOutput().ConnectToSource(self.shader.ConnectableAPI(), "surface")

        # Bind the material to the mesh
        UsdShade.MaterialBindingAPI(self.plane_mesh.GetPrim()).Bind(material)

        particle_system_path = env.GetPrim().GetPath().AppendChild("ParticleSystem")
        particle_material_path = env.GetPrim().GetPath().AppendChild("particleMaterial")
        self.particle_material = ParticleMaterial(
            prim_path=particle_material_path, drag=0.3, lift=0.6, friction=0.9
        )

        particle_system = ParticleSystem(
            particle_system_enabled = True,
            prim_path=particle_system_path,
            simulation_owner=self._env._world.get_physics_context().prim_path,
            rest_offset=0.005 * 0.99,
            contact_offset=0.005,
            solid_rest_offset=0.005 * 0.99,
            fluid_rest_offset=0.005 * 0.6 * 0.99,
            particle_contact_offset=0.005,
            max_neighborhood = 96,
            solver_position_iteration_count = 16,
            global_self_collision_enabled = True,
            non_particle_collision_enabled = True,
        )
        self.cloth1 = ClothPrim(
            prim_path=str(cloth_path),
            name="clothPrim" + str(idx),
            particle_system=particle_system,
            particle_material=self.particle_material,
            # stretch_stiffness=10000.0,
            # bend_stiffness=100.0,
            # shear_stiffness=100.0,
            # spring_damping=0.2,
            stretch_stiffness=10000.0,
            bend_stiffness=100.0,
            shear_stiffness=100.0,
            spring_damping=0.2,
            particle_mass=0.02,
        )


    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return torch.norm(goal_a - goal_b, p=2, dim=-1)


    def refresh_env_tensors(self):
        """Refresh tensors."""

        # self.cloth_pos, self.cloth_quat = self.cloth.get_world_poses(clone=False)
        self.cloth_pos, self.cloth_quat = self.cloth.get_world_poses() #对cloth来说这个坐标不会动
        self.particle_cloth_positon = self.cloth.get_world_positions()

        self.cloth_pos -= self.env_pos

        self.desired_goal = torch.cat((self.particle_cloth_positon[0, 8], self.particle_cloth_positon[0, 0], 
                                       self.particle_cloth_positon[0, 36], self.particle_cloth_positon[0, 44], 
                                       self.particle_cloth_positon[0, 8], self.particle_cloth_positon[0, 0]), dim = -1)
        self.desired_goal = self.desired_goal.unsqueeze(dim=0)
        
        cloth_velocities = self.cloth.get_velocities(clone=False)
        self.keypoint_vel = torch.cat((cloth_velocities[0, 0], cloth_velocities[0, 4], 
                                    cloth_velocities[0, 8], cloth_velocities[0, 36],
                                    cloth_velocities[0, 44], cloth_velocities[0, 72], 
                                    cloth_velocities[0, 76], cloth_velocities[0, 80]), dim = -1)
        self.keypoint_vel = self.keypoint_vel.unsqueeze(dim=0)

        self.keypoint_pos = torch.cat((self.particle_cloth_positon[0, 0], self.particle_cloth_positon[0, 4], 
                                    self.particle_cloth_positon[0, 8], self.particle_cloth_positon[0, 36],
                                    self.particle_cloth_positon[0, 44], self.particle_cloth_positon[0, 72], 
                                    self.particle_cloth_positon[0, 76], self.particle_cloth_positon[0, 80]), dim = -1)
        self.keypoint_pos = self.keypoint_pos.unsqueeze(dim=0)
        
        self.cloth_particle_vel = cloth_velocities[:, :]

