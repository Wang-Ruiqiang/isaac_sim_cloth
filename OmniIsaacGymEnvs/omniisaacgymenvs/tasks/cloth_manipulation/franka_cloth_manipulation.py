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
import random

import hydra
import omni.kit
import omegaconf
import cv2
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

from omni.isaac.core.simulation_context import SimulationContext
from omni.physx.scripts import physicsUtils, deformableUtils
from omni.isaac.core.utils.torch.transformations import *
import omni.isaac.core.utils.torch as torch_utils
from pxr import Gf, Sdf


from omni.isaac.core.prims.soft.cloth_prim_view import ClothPrimView



class FrankaClothManipulation(FrankaCloth, FactoryABCTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        super().__init__(name, sim_config, env)
        self.frame_list = []
        self.frame_list2 = []
        self.counter = 0
        self.video_count = 0
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
        # self._update_camera_view()

        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(
            actions=self.actions,
            ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_min,   #初始状态夹爪位置
            do_scale=True
        )


    async def pre_physics_step_async(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        if not self._env._world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # self._update_camera_view()

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

        self._reset_buffers(env_ids)

    async def reset_idx_async(self, env_ids, randomize_gripper_pose) -> None:
        """Reset specified environments."""
        self._reset_object(env_ids)
        self._reset_franka(env_ids)

        self._reset_buffers(env_ids)


    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""
        indices = env_ids.to(dtype=torch.int32)

        # actions = torch.zeros((32, 12), device=self.device)
        # actions[env_ids, 0:3] = torch.tensor([-0.0102, -0.1460, 0.5], device=self.device)

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
        
        #通过关节角设置初始位置
        # initial_joint_angles_array = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0.08, 0.08], device = self._device)
        # joint_goal = torch.zeros((1, 9),
        #                            dtype=torch.float32,
        #                            device=self._device)
        # joint_goal = initial_joint_angles_array
                
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = joint_goal
        self.dof_pos[env_ids] = joint_goal

        self.frankas.set_joint_positions(joint_goal, indices=indices)
        self.frankas.set_joint_velocities(self.dof_vel[env_ids], indices=indices)

    
    def _reset_object(self, env_ids):
        self.cloth_positon = self.cloth.get_world_positions()

        cloth_x_pos = self.cfg_task.randomize.cloth_pos_xy_initial[0]
        cloth_y_pos = self.cfg_task.randomize.cloth_pos_xy_initial[1]

        cloth_z_pos = self.cfg_base.env.table_height + 0.001
        init_loc = Gf.Vec3f(cloth_x_pos, cloth_y_pos, cloth_z_pos)
        physicsUtils.setup_transform_as_scale_orient_translate(self.plane_mesh)
        physicsUtils.set_or_add_translate_op(self.plane_mesh, init_loc)
        physicsUtils.set_or_add_orient_op(self.plane_mesh, Gf.Rotation(Gf.Vec3d([1, 0, 0]), 5).GetQuat()) #修改cloth的oritation
        red_color = round(random.uniform(0, 2), 2)
        green_color = round(random.uniform(0, 2), 2)
        blue_color = round(random.uniform(0, 2), 2)
        self.shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(red_color, green_color, blue_color)) 

    def create_panda_chain(self):
        robot = URDF.from_xml_file("/home/ruiqiang/workspace/isaac_sim_cloth/OmniIsaacGymEnvs/omniisaacgymenvs/tasks/cloth_manipulation/urdf/panda.urdf")
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
            
            target_x = self.target_postition[i, 0].item() + 0.50 - 0.12
            target_y = -self.target_postition[i, 1].item() - 0.102
            target_z = self.target_postition[i, 2].item() + 0.097

            # 效果比较好的结果
            # target_x = self.target_postition[i, 0].item() + 0.50 - 0.12
            # target_y = -self.target_postition[i, 1].item() - 0.105
            # target_z = self.target_postition[i, 2].item() + 0.097

            # 最初的结果
            # target_x = self.target_postition[i, 0].item() + 0.50 - 0.09
            # target_y = -self.target_postition[i, 1].item() - 0.095
            # target_z = self.target_postition[i, 2].item() + 0.095

            # target_frame = PyKDL.Frame(PyKDL.Rotation.RPY(3.1415926, 0, -0.7854),
            #                             PyKDL.Vector(target_x, target_y, target_z))
            
            target_frame = PyKDL.Frame(PyKDL.Rotation.RPY(3.1415926, 0, 0.7854),
                                        PyKDL.Vector(target_x, target_y, target_z))
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
        pos_actions = actions[:, 0:3]   #增量
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))


        # 增加每一步或者最终位置的限制
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        if self.ctrl_target_fingertip_midpoint_pos[0][0] > 0.15:
            self.ctrl_target_fingertip_midpoint_pos[0][0] = 0.15

        if self.ctrl_target_fingertip_midpoint_pos[0][1] > 0.05:
            self.ctrl_target_fingertip_midpoint_pos[0][1] = 0.05
        elif self.ctrl_target_fingertip_midpoint_pos[0][1] < -0.40:
            self.ctrl_target_fingertip_midpoint_pos[0][1] = -0.40

        if self.ctrl_target_fingertip_midpoint_pos[0][2] > 0.55:
            self.ctrl_target_fingertip_midpoint_pos[0][2] = 0.55
            
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


    def _update_camera_view(self):
        if self.camera and self.progress_buf[:] >= 1:
            rgba_image = self.camera.get_rgba()
            if rgba_image is not None:
                # 转换图像格式
                rgb_image = rgba_image[:, :, :3]
                bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                self.frame_list.append(bgr_image)

            if self.camera2 and self.progress_buf[:] >= 1:
                rgba_image = self.camera2.get_rgba()
                if rgba_image is not None:
                    # 转换图像格式
                    rgb_image = rgba_image[:, :, :3]
                    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    self.frame_list2.append(bgr_image)


    def _save_video(self, video_path, frames, fps=30):
        if len(frames) == 0:
            print("No frames to save!")
            return
    
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        angle = random.uniform(-180, 180)  # 随机旋转角度
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        for frame in frames:
            rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
            video_writer.write(rotated_frame)

        video_writer.release()
        print(f"Video saved at {video_path}")

    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        if self._env._world.is_playing():
            if self.progress_buf[:] >= self.max_episode_length - 1 :
                self.counter += 1
                if self.counter == 10:
                    video_path = f"../tasks/cloth_manipulation/video/camera-1-{self.video_count}.avi"
                    video_path2 = f"../tasks/cloth_manipulation/video2/camera-2-{self.video_count}.avi"
                    # self._save_video(video_path, self.frame_list)
                    # self._save_video(video_path2, self.frame_list2)
                    self.counter = 0
                    self.video_count += 1
                    self.frame_list.clear()
                    self.frame_list2.clear()

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
            #TODO 这个应该现在没什么用
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
        self.achieved_goal = torch.cat((self.particle_cloth_positon[0, 80], self.particle_cloth_positon[0, 72], self.particle_cloth_positon[0, 36],
                            self.particle_cloth_positon[0, 44], self.particle_cloth_positon[0, 8], self.particle_cloth_positon[0, 0]), 0)
        # print("self.particle_cloth_positon[0, 0] = ", self.particle_cloth_positon[0, 0])
        # print("self.particle_cloth_positon[0, 8] = ", self.particle_cloth_positon[0, 8])
        # print("self.particle_cloth_positon[0, 72] = ", self.particle_cloth_positon[0, 72])
        # print("self.particle_cloth_positon[0, 80] = ", self.particle_cloth_positon[0, 80])
        self.achieved_goal = self.achieved_goal.unsqueeze(dim=0)
        obs_tensors = [self.fingertip_midpoint_pos,
                       self.fingertip_midpoint_quat,
                       self.fingertip_midpoint_linvel,
                       self.fingertip_midpoint_angvel,
                       self.achieved_goal,
                       self.desired_goal,
                       self.keypoint_vel,
                       self.keypoint_pos]
        
        # print("self.constraint_dis = ", self.constraint_dis)
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

        # keypoint_reward = -self._get_keypoint_dist()  #相减的值取负，所以随着实验进行会越来越大
        keypoint_reward = self._get_keypoint_dist()  #相减的值取负，所以随着实验进行会越来越大
        
        action_penalty = torch.norm(self.actions, p=2, dim=-1) * self.cfg_task.rl.action_penalty_scale

        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale
        
        if self.successes :
            # self.progress_buf[0] = self.max_episode_length - 2
            self.progress_buf[0] = self.max_episode_length - 1

        # # In this policy, episode length is constant across all envs
        # is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        # if is_last_step:
        #     # Check if nut is picked up and above table
        #     lift_success = self._check_lift_success(height_multiple=3.0)
        #     self.rew_buf[:] += lift_success * self.cfg_task.rl.success_bonus
        #     self.extras['successes'] = torch.mean(lift_success.float())


    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self._device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self._device) - 0.5

        return keypoint_offsets
    

    def _get_keypoint_dist(self):
        """Get keypoint distance."""
        achieved_oks = torch.zeros((1, 6),
                                   dtype=torch.float32,
                                   device=self._device)
        achieved_distances = torch.zeros((1, 6),
                                   dtype=torch.float32,
                                   device=self._device)
        success_reward = 0
        fail_reward = -1
        action_penalty = 0

        # constraint_distances = torch.tensor([0.04, 0.02, 0.02, 0.02, 0.02, 0.02], device=self._device)
        constraint_distances = torch.tensor([0.015, 0.01, 0.01, 0.01, 0.01, 0.01], device=self._device)

        for i, constraint_distance in enumerate(constraint_distances):
            achieved_distances_per_constraint = self.goal_distance(self.achieved_goal[0][i * 3 : (i + 1)* 3], 
                                                                   self.desired_goal[0][i * 3:(i + 1) * 3])
            constraint_ok = achieved_distances_per_constraint < constraint_distance
            # print("achieved_distances_per_constraint = ", achieved_distances_per_constraint)
            achieved_distances[:, i] = achieved_distances_per_constraint.item()
            achieved_oks[:, i] = constraint_ok.item()
            
        
        self.successes = torch.all(achieved_oks, axis=1)
        if self.successes:
            print("success")

        fails = torch.logical_not(self.successes)
        task_rewards = self.successes.float().flatten() * success_reward
        dist_rewards = torch.sum((1 - achieved_distances/constraint_distances), axis=1) / len(constraint_distances)
        point_one_dis = self.goal_distance(self.particle_cloth_positon[0, 80], self.particle_cloth_positon[0, 8])
        point_two_dis = self.goal_distance(self.particle_cloth_positon[0, 72], self.particle_cloth_positon[0, 0])

        if self.particle_cloth_positon[0, 80][0] - self.particle_cloth_positon[0, 8][0] > 0.03 :
            action_penalty += point_one_dis

        if self.particle_cloth_positon[0, 72][0] - self.particle_cloth_positon[0, 0][0] > 0.03 :
            action_penalty += point_two_dis

        
        task_rewards += dist_rewards # Extra for being closer to the goal
        task_rewards[fails] = fail_reward - action_penalty
        return task_rewards