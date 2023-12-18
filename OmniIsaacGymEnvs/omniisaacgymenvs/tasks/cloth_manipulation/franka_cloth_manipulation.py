# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Factory: Class for nut-bolt pick task.

Inherits nut-bolt environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskNutBoltPick
"""

import asyncio

import hydra
import omni.kit
import omegaconf
import os
import torch
import PyKDL
import numpy as np

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from omniisaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from omniisaacgymenvs.tasks.cloth_manipulation.franka_cloth import FrankaCloth
from omniisaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask
import omniisaacgymenvs.tasks.factory.factory_control as fc
from omni.isaac.robot_assembler import RobotAssembler,AssembledRobot

from omni.isaac.core.simulation_context import SimulationContext
from omni.physx.scripts import physicsUtils, deformableUtils
from omni.isaac.core.utils.torch.transformations import *
import omni.isaac.core.utils.torch as torch_utils
from pxr import Gf, PhysxSchema, UsdGeom


from omni.isaac.core.prims.soft.cloth_prim_view import ClothPrimView



class FrankaClothManipulation(FrankaCloth, FactoryABCTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        super().__init__(name, sim_config, env)
        self._get_task_yaml_params()
    
    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self._task_cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        ppo_path = 'train/FrankaClothManipulationPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting
    

    def post_reset(self):
        """
        This method is called only one time right before sim begins. 
        The 'reset' here is referring to the reset of the world, which occurs before the sim starts.
        """
        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        super().post_reset()
        self.acquire_base_tensors()
        self._acquire_task_tensors()

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        # # randomize all envs
        # indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        # self.reset_idx(indices)
        # Reset all envs

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        asyncio.ensure_future(
            self.reset_idx_async(indices, randomize_gripper_pose=False)
        )

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        # Grasp pose tensors
        cloth_grasp_heights = self.garment_heights * 0.5 
        self.cloth_grasp_pos_local = cloth_grasp_heights * torch.tensor([0.0, 0.0, 1.0], device=self._device).repeat(
            (self._num_envs, 1))
        self.cloth_grasp_quat_local = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self._device).unsqueeze(0).repeat(
            self._num_envs, 1)

        # Keypoint tensors
        self.keypoint_offsets = self._get_keypoint_offsets(
            self.cfg_task.rl.num_keypoints) * self.cfg_task.rl.keypoint_scale
        self.keypoints_gripper = torch.zeros(
            (self._num_envs, self.cfg_task.rl.num_keypoints, 3),
            dtype=torch.float32,
            device=self._device
        )
        self.keypoints_cloth = torch.zeros_like(self.keypoints_gripper, device=self._device)
        self.identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self._device).unsqueeze(0).repeat(self._num_envs, 1)


    def pre_physics_step(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        if not self._env._world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(
            actions=self.actions,
            ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
            do_scale=True
        )


    async def pre_physics_step_async(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        if not self._env._world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            await self.reset_idx_async(env_ids, randomize_gripper_pose=True)

        self.actions = actions.clone().to(
            self.device
        )  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(
            actions=self.actions,
            ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
            do_scale=True,
        )

    def reset_idx(self, env_ids):
        """Reset specified environments."""
        self._reset_object(env_ids)
        self._reset_franka(env_ids)

        # self.add_attachment()
    
        self._randomize_gripper_pose(env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps)

        self._reset_buffers(env_ids)

    async def reset_idx_async(self, env_ids, randomize_gripper_pose) -> None:
        """Reset specified environments."""
        self._reset_object(env_ids)
        self._reset_franka(env_ids)

        if randomize_gripper_pose:
            await self._randomize_gripper_pose_async(
                env_ids, sim_steps=self.cfg_task.env.num_gripper_move_sim_steps
            )

        self._reset_buffers(env_ids)


    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""
        indices = env_ids.to(dtype=torch.int32)

        actions = torch.zeros((32, 12), device=self.device)
        actions[env_ids, 0:3] = torch.tensor([-0.0102, -0.1460, 0.5], device=self.device)

        self.target_postition, self.target_quat = self.cloth.get_world_poses()
        self.target_postition -= self.env_pos
        chain = self.create_panda_chain()
        joint_angles = self.compute_inverse_kinematics(chain)
        joint_angles = np.array(joint_angles)
        joint_goal = torch.rand([len(joint_angles), joint_angles[0].rows() + 2], device=self._device)
        for i in range(len(joint_angles)) :
            joint_goal[i, 7] = 0.08
            joint_goal[i, 8] = 0.08
            for j in range(joint_angles[0].rows()):
                joint_goal[i, j] = joint_angles[i][j]
                
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = joint_goal
        self.dof_pos[env_ids] = joint_goal

        self.frankas.set_joint_positions(joint_goal, indices=indices)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

    
    def _reset_object(self, env_ids):
        # self._env._world.scene.remove_object("cloth_view", False)
        # self._import_env_assets(True)
        # self.cloth = ClothPrimView(prim_paths_expr = "/World/envs/.*/garment/cloth", 
        #                            name="cloth_view",
        #                            )

        # self._env._world.scene.add(self.cloth)
        # self.initialize_views_cloth(self._env._world.scene)

        cloth_noise_xy = 2 * (torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        cloth_noise_xy = cloth_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.cloth_pos_xy_initial_noise, device=self.device))
        
        # self.cloth_pos[env_ids, 0] = self.cfg_task.randomize.cloth_pos_xy_initial[0] + cloth_noise_xy[env_ids, 0]
        # self.cloth_pos[env_ids, 1] = self.cfg_task.randomize.cloth_pos_xy_initial[1] + cloth_noise_xy[env_ids, 1]

        self.cloth_pos[env_ids, 0] = self.cfg_task.randomize.cloth_pos_xy_initial[0]
        self.cloth_pos[env_ids, 1] = self.cfg_task.randomize.cloth_pos_xy_initial[1]

        # self.cloth_pos[env_ids, 2] = self.cfg_base.env.table_height + self.garment_heights.squeeze(-1)
        self.cloth_pos[env_ids, 2] = self.cfg_base.env.table_height

        # self.cloth_quat[env_ids, :] = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device).repeat(len(env_ids), 1)

        self.cloth_particle_vel[env_ids, :] = 0.0

        # # 新加
        indices = env_ids.to(dtype=torch.int32)
        # self.target_postition, self.target_quat = self.cloth.get_world_poses()
        # for i in range(self._num_envs):
        #     # init_loc = Gf.Vec3f(self.cloth_pos[env_ids, 0][i].item(), self.cloth_pos[env_ids, 1][i].item() - 0.1, 0.4)

        #     init_loc = Gf.Vec3f(0, -0.15, 0.4)
        # #     # print("self.plan_mesh[i] = ", self.plane_meshs[i])
        # #     # physicsUtils.setup_transform_as_scale_orient_translate(self.plane_meshs[i])
        # #     # physicsUtils.set_or_add_translate_op(self.plane_meshs[i], init_loc)
        # #     # physicsUtils.set_or_add_orient_op(self.plane_meshs[i], Gf.Rotation(Gf.Vec3d([1, 0, 0]), 0).GetQuat())

        #     if self.initial_positions is None:
        #         self.initial_positions = torch.zeros((self.num_envs, len(self.tri_points), 3))
        #     # indices = torch.tensor(
        #     #         np.random.choice(range(self.num_envs), self.num_envs // 2, replace=False), dtype=torch.long
        #     #     )
        #     # print("torch.tensor(self.plane_meshs[i].GetPointsAttr().Get() = ", self.plane_meshs[i].GetPointsAttr().Get())
        #     self.initial_positions[i] = torch.tensor(init_loc) + torch.tensor(self.plane_meshs[i].GetPointsAttr().Get())
        #     # print("self.initial_positions[i] = ", self.initial_positions[i])
        #     self.cloth.set_world_positions(self.initial_positions[i], indices = [i])
        #     # print("world_positions = ", self.cloth.get_world_positions(indices = [i]))
        
        self.cloth.set_world_poses(self.cloth_pos[env_ids] + self.env_pos[env_ids], self.cloth_quat[env_ids], indices)
        self.cloth.set_velocities(self.cloth_particle_vel[env_ids], indices)


    def add_attachment(self):
        for i in range(32): 
            attachment_plane_mesh_path = f"/World/envs/env_{i}/garment/attachment"
            attachment_plane_mesh = PhysxSchema.PhysxPhysicsAttachment.Define(self._stage, attachment_plane_mesh_path)
            attachment_plane_mesh.GetActor0Rel().SetTargets([f"/World/envs_{i}/franka/panda_rightfinger/visuals/mesh_0"])
            attachment_plane_mesh.GetActor1Rel().SetTargets([f"/World/envs_{i}/garment/cloth"])
            PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment_plane_mesh.GetPrim())



    def create_panda_chain(self):
        robot = URDF.from_xml_file("/home/ruiqiang/workspace/omniverse_gym/OmniIsaacGymEnvs/omniisaacgymenvs/tasks/cloth_manipulation/urdf/panda.urdf")
        tree = kdl_tree_from_urdf_model(robot)
        chain = tree.getChain("panda_link0", "panda_link8")
        return chain
    

    def compute_inverse_kinematics(self, chain):
        fk = PyKDL.ChainFkSolverPos_recursive(chain)
        minjp = PyKDL.JntArray(7)
        maxjp = PyKDL.JntArray(7)
        minjp[0] = -2.9671
        minjp[1] = -1.8326
        minjp[2] = -2.9671
        minjp[3] = -3.1416
        minjp[4] = -2.9671
        minjp[5] = -0.0873
        minjp[6] = -2.9671

        maxjp[0] = 2.9671
        maxjp[1] = 1.8326
        maxjp[2] = 2.9671
        maxjp[3] = 0.0873
        maxjp[4] = 2.9671
        maxjp[5] = 3.8223
        maxjp[6] = 2.9671
        
        ikv = PyKDL.ChainIkSolverVel_pinv(chain)
        ik = PyKDL.ChainIkSolverPos_NR_JL(chain, minjp, maxjp, fk, ikv)
        result = []
        for i in range(self.target_postition.size(0)):
            # 创建目标位姿
            # target_x = self.target_postition[i, 0, 0].item()
            # target_y = self.target_postition[i, 0, 1].item()
            # target_z = self.target_postition[i, 0, 2].item()
            target_x = self.target_postition[i, 0].item() + 0.5
            target_y = -self.target_postition[i, 1].item()
            target_z = self.target_postition[i, 2].item() + 0.15

            target_frame = PyKDL.Frame(PyKDL.Rotation.RPY(3.1415926, 0, 0),
                                        PyKDL.Vector(target_x, target_y, target_z))
            # target_frame = PyKDL.Frame(PyKDL.Rotation.RPY(3.1415926, 0, 0), PyKDL.Vector(0.51, -0.1460, 0.5))
            # print ("target_frame = ", target_frame)
            # 创建起始关节角度
            initial_joint_angles = PyKDL.JntArray(7)
            initial_joint_angles_array = [0.012, -0.5697, 0, -2.8105, 0, 3.0312, 0.7853]
            for i in range(7):
                initial_joint_angles[i] = initial_joint_angles_array[i]
            single_result = PyKDL.JntArray(chain.getNrOfJoints())
            # 调用逆运动学求解器
            retval = ik.CartToJnt(initial_joint_angles, target_frame, single_result)
            if (retval >= 0):
                # print('single_result: ',single_result)
                result.append(single_result)
            else :
                print("Error: could not calculate ik kinematics :(")
        return result

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(
                angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                rot_actions_quat,
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)
            )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.generate_ctrl_signals()


    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        if self._env._world.is_playing():

            # In this policy, episode length is constant
            is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

            if self.cfg_task.env.close_and_lift:
                # At this point, robot has executed RL policy. Now close gripper and lift (open-loop)
                if is_last_step:
                    self._close_gripper(sim_steps=self.cfg_task.env.num_gripper_close_sim_steps)
                    self._lift_gripper(sim_steps=self.cfg_task.env.num_gripper_lift_sim_steps)

            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    

    async def post_physics_step_async(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        if self._env._world.is_playing():
            # In this policy, episode length is constant
            is_last_step = self.progress_buf[0] == self.max_episode_length - 1

            if self.cfg_task.env.close_and_lift:
                # At this point, robot has executed RL policy. Now close gripper and lift (open-loop)
                if is_last_step:
                    await self._close_gripper_async(
                        sim_steps=self.cfg_task.env.num_gripper_close_sim_steps
                    )
                    await self._lift_gripper_async(
                        sim_steps=self.cfg_task.env.num_gripper_lift_sim_steps
                    )

            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            self.get_observations()
            self.get_states()
            self.calculate_metrics()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        # Compute pose of cloth grasping frame
        self.cloth_grasp_quat, self.cloth_grasp_pos = tf_combine(
            self.cloth_quat,
            self.cloth_pos,
            self.cloth_grasp_quat_local,
            self.cloth_grasp_pos_local,
        )

        # Compute pos of keypoints on gripper and cloth in world frame
        for idx, keypoint_offset in enumerate(self.keypoint_offsets):
            self.keypoints_gripper[:, idx] = tf_combine(
                self.fingertip_midpoint_quat,
                self.fingertip_midpoint_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1)
            )[1]
            self.keypoints_cloth[:, idx] = tf_combine(
                self.cloth_grasp_quat,
                self.cloth_grasp_pos,
                self.identity_quat,
                keypoint_offset.repeat(self.num_envs, 1)
            )[1]

    def get_observations(self):
        """Compute observations."""

        # Shallow copies of tensors
        obs_tensors = [self.fingertip_midpoint_pos,
                       self.fingertip_midpoint_quat,
                       self.fingertip_midpoint_linvel,
                       self.fingertip_midpoint_angvel,
                       self.cloth_grasp_pos,
                       self.cloth_grasp_quat]

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)

        observations = {
            self.frankas.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations
    
    def calculate_metrics(self) -> None:
        """Update reward and reset buffers."""

        self._update_reset_buf()
        self._update_rew_buf()


    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf
        )

        
    def _update_rew_buf(self):
        """Compute reward at current timestep."""

        keypoint_reward = -self._get_keypoint_dist()
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale

        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale

        # In this policy, episode length is constant across all envs
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        if is_last_step:
            # Check if nut is picked up and above table
            lift_success = self._check_lift_success(height_multiple=3.0)
            self.rew_buf[:] += lift_success * self.cfg_task.rl.success_bonus
            self.extras['successes'] = torch.mean(lift_success.float())


    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self._device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self._device) - 0.5

        return keypoint_offsets
    

    def _get_keypoint_dist(self):
        """Get keypoint distance."""

        keypoint_dist = torch.sum(torch.norm(self.keypoints_cloth - self.keypoints_gripper, p=2, dim=-1), dim=-1)
        return keypoint_dist
    
    def _close_gripper(self, sim_steps=20):
        """Fully close gripper using controller. Called outside RL loop (i.e., after last step of episode)."""
        self._move_gripper_to_dof_pos(gripper_dof_pos=0.0, sim_steps=sim_steps)

        
    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)  # No hand motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos, do_scale=False)
        
        # Step sim
        for _ in range(sim_steps):
            SimulationContext.step(self._env._world, render=True)

    
    def _lift_gripper(self, franka_gripper_width=0.0, lift_distance=0.3, sim_steps=20):
        """Lift gripper by specified distance. Called outside RL loop (i.e., after last step of episode)."""

        delta_hand_pose = torch.zeros([self.num_envs, 6], device=self.device)
        delta_hand_pose[:, 2] = lift_distance

        # Step sim
        for _ in range(sim_steps):
            self._apply_actions_as_ctrl_targets(delta_hand_pose, franka_gripper_width, do_scale=False)
            SimulationContext.step(self._env._world, render=True)


    def _check_lift_success(self, height_multiple):
        """Check if nut is above table by more than specified multiple times height of nut."""

        lift_success = torch.where(
            self.cloth_pos[:, 2] > self.cfg_base.env.table_height + self.garment_heights.squeeze(-1) * height_multiple,
            torch.ones((self.num_envs,), device=self.device),
            torch.zeros((self.num_envs,), device=self.device))

        return lift_success
    

    def _randomize_gripper_pose(self, env_ids, sim_steps):
        """Move gripper to random pose."""
        # step once to update physx with the newly set joint positions from reset_franka()
        SimulationContext.step(self._env._world, render=True)

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor([0.0, 0.0, self.cfg_base.env.table_height], device=self.device) \
            + torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(self.num_envs, 1)

        fingertip_midpoint_pos_noise = 2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_pos_noise,
            device=self.device)
        )
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                                                            device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        fingertip_midpoint_rot_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device) - 0.5)  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device))
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2]
        )

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle'
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions), device=self.device)
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(
                actions=actions,
                ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                do_scale=False,
            )

            SimulationContext.step(self._env._world, render=True)

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])
        
        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

        # step once to update physx with the newly set joint velocities
        SimulationContext.step(self._env._world, render=True)


    async def _randomize_gripper_pose_async(self, env_ids, sim_steps) -> None:
        """Move gripper to random pose."""

        # step once to update physx with the newly set joint positions from reset_franka()
        await omni.kit.app.get_app().next_update_async()

        # Set target pos above table
        self.ctrl_target_fingertip_midpoint_pos = torch.tensor(
            [0.0, 0.0, self.cfg_base.env.table_height], device=self.device
        ) + torch.tensor(
            self.cfg_task.randomize.fingertip_midpoint_pos_initial, device=self.device
        )
        self.ctrl_target_fingertip_midpoint_pos = (
            self.ctrl_target_fingertip_midpoint_pos.unsqueeze(0).repeat(
                self.num_envs, 1
            )
        )

        fingertip_midpoint_pos_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_pos_noise = fingertip_midpoint_pos_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_pos_noise, device=self.device
            )
        )
        self.ctrl_target_fingertip_midpoint_pos += fingertip_midpoint_pos_noise

        # Set target rot
        ctrl_target_fingertip_midpoint_euler = (
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_initial,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        fingertip_midpoint_rot_noise = 2 * (
            torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        fingertip_midpoint_rot_noise = fingertip_midpoint_rot_noise @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.fingertip_midpoint_rot_noise, device=self.device
            )
        )
        ctrl_target_fingertip_midpoint_euler += fingertip_midpoint_rot_noise
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_fingertip_midpoint_euler[:, 0],
            ctrl_target_fingertip_midpoint_euler[:, 1],
            ctrl_target_fingertip_midpoint_euler[:, 2],
        )

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos,
                fingertip_midpoint_quat=self.fingertip_midpoint_quat,
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
                jacobian_type=self.cfg_ctrl["jacobian_type"],
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions), device=self.device
            )
            actions[:, :6] = delta_hand_pose

            self._apply_actions_as_ctrl_targets(
                actions=actions,
                ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_max,
                do_scale=False,
            )

            self._env._world.physics_sim_view.flush()
            await omni.kit.app.get_app().next_update_async()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        indices = env_ids.to(dtype=torch.int32)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

        # step once to update physx with the newly set joint velocities
        self._env._world.physics_sim_view.flush()
        await omni.kit.app.get_app().next_update_async()