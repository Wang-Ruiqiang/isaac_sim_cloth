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
import matplotlib.pyplot as plt

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

        self.y_displacements = []
        self.z_displacements = []

        self.is_first_run = True

        self.step_count = 0
        self.init_tensor_list()
        self.init_tensor_list1()
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
        # print("self.actions = ", self.actions)

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

        cloth_noise_xy = 2 * (
            torch.rand((self.num_envs, 2), dtype=torch.float32, device=self.device)
            - 0.5
        )  # [-1, 1]
        cloth_noise_xy = cloth_noise_xy @ torch.diag(
            torch.tensor(self.cfg_task.randomize.cloth_pos_xy_noise, device=self.device)
        )


        cloth_x_pos = self.cfg_task.randomize.cloth_pos_xy_initial[0] + cloth_noise_xy[env_ids, 0].item()
        cloth_y_pos = self.cfg_task.randomize.cloth_pos_xy_initial[1] 
        # cloth_x_pos = self.cfg_task.randomize.cloth_pos_xy_initial[0]
        # cloth_y_pos = self.cfg_task.randomize.cloth_pos_xy_initial[1]+ torch.rand(1).item() * 0.1
        self.cloth_pos_offset = cloth_noise_xy[env_ids, 0].item()

        cloth_z_pos = self.cfg_base.env.table_height + 0.001
        init_loc = Gf.Vec3f(cloth_x_pos, cloth_y_pos, cloth_z_pos)
        physicsUtils.setup_transform_as_scale_orient_translate(self.plane_mesh)
        physicsUtils.set_or_add_translate_op(self.plane_mesh, init_loc)
        physicsUtils.set_or_add_orient_op(self.plane_mesh, Gf.Rotation(Gf.Vec3d([1, 0, 0]), 5).GetQuat()) #修改cloth的oritation
        # physicsUtils.set_or_add_orient_op(self.plane_mesh, Gf.Rotation(Gf.Vec3d([1, 0, 0]), 0).GetQuat()) #修改cloth的oritation
        # red_color = round(random.uniform(0, 2), 2)
        # green_color = round(random.uniform(0, 2), 2)
        # blue_color = round(random.uniform(0, 2), 2)
        # self.shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(red_color, green_color, blue_color)) 
        self.shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1)) 

    def create_panda_chain(self):
        robot = URDF.from_xml_file("/home/ruiqiang/workspaces/isaac_ws/isaac_sim_cloth/OmniIsaacGymEnvs/omniisaacgymenvs/tasks/cloth_manipulation/urdf/panda.urdf")
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
        particle_cloth_positon = self.cloth.get_world_positions()

        for i in range(self.target_postition.size(0)):
            # 创建目标位姿
            # target_x = self.target_postition[i, 0, 0].item()
            # target_y = self.target_postition[i, 0, 1].item()
            # target_z = self.target_postition[i, 0, 2].item()
            
            target_x = -self.target_postition[i, 0].item() + 0.50 - 0.115
            target_y = -self.target_postition[i, 1].item() - 0.102
            target_z = self.target_postition[i, 2].item() + 0.097

            # target_x = -self.target_postition[i, 0].item() + 0.50 - 0.1 - 0.02
            # target_y = -self.target_postition[i, 1].item() - 0.10 
            # target_z = self.target_postition[i, 2].item() + 0.097


            # print("self.target_postition = ", self.target_postition)
            # print("target_x = ", target_x)
            # print("target_y = ", target_y)
            # print("target_z = ", target_z)

            # print("particle_cloth_positon[0, 80] = ", particle_cloth_positon[0, 80])

            # target_x = -particle_cloth_positon[0, 80][0] + 0.5 - 0.115 
            # target_y = -particle_cloth_positon[0, 80][1] - 0.002
            # target_z = particle_cloth_positon[0, 80][2] +0.097

            # 效果比较好的结果
            # target_x = self.target_postition[i, 0].item() + 0.50 - 0.12
            # target_y = -self.target_postition[i, 1].item() - 0.105
            # target_z = self.target_postition[i, 2].item() + 0.097

            # 最初的结果
            # target_x = self.target_postition[i, 0].item() + 0.50 - 0.09
            # target_y = -self.target_postition[i, 1].item() - 0.095
            # target_z = self.target_postition[i, 2].item() + 0.09
            # target_frame = PyKDL.Frame(PyKDL.Rotation.RPY(3.1415926, 0, -0.7854),
            #                             PyKDL.Vector(target_x, target_y, target_z))
            
            target_frame = PyKDL.Frame(PyKDL.Rotation.RPY(3.1415926, 0, 0.7854),
                                        PyKDL.Vector(target_x, target_y, target_z))
            # target_frame = PyKDL.Frame(PyKDL.Rotation.RPY(3.1415926, 0, -2.3546),
            #                             PyKDL.Vector(target_x, target_y, target_z))
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

        print("pos_actions =", pos_actions)

        self.y_displacements.append(pos_actions[0][1].item())
        self.z_displacements.append(pos_actions[0][2].item())


        # 增加每一步或者最终位置的限制
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        if self.ctrl_target_fingertip_midpoint_pos[0][0] > 0.15:
            self.ctrl_target_fingertip_midpoint_pos[0][0] = 0.15

        if self.ctrl_target_fingertip_midpoint_pos[0][1] > 0.1:
            self.ctrl_target_fingertip_midpoint_pos[0][1] = 0.1
        elif self.ctrl_target_fingertip_midpoint_pos[0][1] < -0.40:
            self.ctrl_target_fingertip_midpoint_pos[0][1] = -0.40

        # if self.ctrl_target_fingertip_midpoint_pos[0][2] > 0.55:
        #     self.ctrl_target_fingertip_midpoint_pos[0][2] = 0.55

        # if self.ctrl_target_fingertip_midpoint_pos[0][2] <= 0.40:
        #     self.ctrl_target_fingertip_midpoint_pos[0][2] = 0.40
            
        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        
        # print("rot_actions = ", rot_actions)

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        # rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.zeros(1, 4, device=self.device)
        # print("rot_actions_quat = ", rot_actions_quat)

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


    def plot_displacements(self):
        # 计算 x 和 z 的总和
        x_total = [sum(self.y_displacements[:i+1]) for i in range(len(self.y_displacements))]
        z_total = [sum(self.z_displacements[:i+1]) for i in range(len(self.z_displacements))]

        # 绘制图表
        plt.figure()
        plt.plot(x_total, z_total, marker='o', label='Z vs X Displacement')
        plt.xlabel('X Displacement Total')
        plt.ylabel('Z Displacement Total')
        plt.title('Z vs X Displacement')
        plt.legend()
        # 保存图像
        plt.savefig("image")

        # 关闭图表以释放内存
        plt.close()

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
        # print("--------------------------------------------------------------------")

        self.achieved_goal = self.achieved_goal.unsqueeze(dim=0)
        # origin_point = torch.tensor([0.1, -0.12, 0.4])
        # offset =  torch.rand(1, 3, device='cuda:0') * 0.1
        # self.fingertip_midpoint_pos = self.fingertip_midpoint_pos + offset

        # self.achieved_goal = self.achieved_goal.view(-1, 3)
        # self.achieved_goal = self.achieved_goal + offset
        # self.achieved_goal = self.achieved_goal.view(1, -1)

        # self.desired_goal = self.desired_goal.view(-1, 3)
        # self.desired_goal = self.desired_goal + offset
        # self.desired_goal = self.desired_goal.view(1, -1)

        # self.keypoint_pos = self.keypoint_pos.view(-1, 3)
        # self.keypoint_pos = self.keypoint_pos + offset
        # self.keypoint_pos = self.keypoint_pos.view(1, -1)

        print("self.fingertip_midpoint_pos = ", self.fingertip_midpoint_pos)
        # print("self.fingertip_midpoint_quat = ", self.fingertip_midpoint_quat)
        # print("self.fingertip_midpoint_linvel = ", self.fingertip_midpoint_linvel)
        # print("self.fingertip_midpoint_angvel = ", self.fingertip_midpoint_angvel)
        print("self.achieved_goal = ",self.achieved_goal)
        print("self.desired_goal = ", self.desired_goal)
        # print("self.keypoint_vel = ", self.keypoint_vel)
        # print("self.keypoint_pos = ", self.keypoint_pos)
        # print("--------------------------------------------------------------------")
        obs_tensors = [self.fingertip_midpoint_pos,
                    #    self.fingertip_midpoint_quat,
                       torch.tensor([[0.0, -0.0, 0.0, -0.0]], device='cuda:0'),
                    #    self.fingertip_midpoint_linvel,
                       torch.tensor([[0, 0,  0]], device='cuda:0'),
                    #    self.fingertip_midpoint_angvel,
                       torch.tensor([[0, 0,  0]], device='cuda:0'),
                       self.achieved_goal,
                       self.desired_goal,
                    #    self.keypoint_vel,
                       torch.zeros(1, 24, device = 'cuda:0'),
                       self.keypoint_pos]

        # obs_tensors = self.obs_tensors_list1[self.count]
        self.step_count += 1
        # if self.count >= len(self.obs_tensors_list):
        #     self.count = 0
        #     self.step_count = 0
        #     self.progress_buf[:] = self.max_episode_length - 1


        # print("step_count = ", self.step_count)
        # print("obs_tensors = ", obs_tensors)
        
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

        if self.progress_buf[:] >= self.max_episode_length - 1:
            self.plot_displacements()
            self.y_displacements.clear()
            self.z_displacements.clear()

        
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

        constraint_distances = torch.tensor([0.04, 0.02, 0.02, 0.02, 0.02, 0.02], device=self._device)
        # constraint_distances = torch.tensor([0.015, 0.01, 0.01, 0.01, 0.01, 0.01], device=self._device)

        for i, constraint_distance in enumerate(constraint_distances):
            achieved_distances_per_constraint = self.goal_distance(self.achieved_goal[0][i * 3 : (i + 1)* 3], 
                                                                   self.desired_goal[0][i * 3:(i + 1) * 3])
            constraint_ok = achieved_distances_per_constraint < constraint_distance
            # print("achieved_distances_per_constraint = ", achieved_distances_per_constraint)
            achieved_distances[:, i] = achieved_distances_per_constraint.item()
            achieved_oks[:, i] = constraint_ok.item()
            # print("achieved_oks = ", achieved_oks)
            
        
        # self.successes = torch.all(achieved_oks, axis=1)
        # print("achieved_oks[0, 0] = ", achieved_oks[0, 0])
        self.successes = achieved_oks[0, 0]
        if self.successes:
            print("success")

        fails = torch.logical_not(self.successes)
        task_rewards = self.successes.float().flatten() * success_reward

        #两对点匹配的距离reward
        # dist_rewards = torch.sum((1 - achieved_distances/constraint_distances), axis=1) / len(constraint_distances)

        # print("achieved_distances[0, 0] = ", achieved_distances[0, 0])
        # print("constraint_distances[0] = ", constraint_distances[0])
        dist_rewards = torch.sum((1 - achieved_distances[0, 0]/constraint_distances[0]))
        # print("dist_rewards = ", dist_rewards)
        point_one_dis = self.goal_distance(self.particle_cloth_positon[0, 80], self.particle_cloth_positon[0, 8])
        point_two_dis = self.goal_distance(self.particle_cloth_positon[0, 72], self.particle_cloth_positon[0, 0])

        if self.particle_cloth_positon[0, 8][1] - self.particle_cloth_positon[0, 80][1] > 0.03 :
            action_penalty +=  point_one_dis

        # print("self.particle_cloth_positon[0, 80] = ", self.particle_cloth_positon[0, 80])

        # if self.particle_cloth_positon[0, 0][1] - self.particle_cloth_positon[0, 72][1] > 0.03 :
        #     action_penalty += point_two_dis

        if self.is_first_run:
            self.left_top_point = self.particle_cloth_positon[0, 0]
            self.left_buttom_point = self.particle_cloth_positon[0, 8]
            self.is_first_run = False

        # penalty_point_move = abs(self.goal_distance(self.left_top_point, self.particle_cloth_positon[0, 0]))\
        #                     + abs(self.goal_distance(self.left_buttom_point, self.particle_cloth_positon[0, 8]))

        # if self.fingertip_midpoint_pos[0][2] <= 0.40:
        #     action_penalty += 0.5
        
        
        # action_penalty -= penalty_point_move

        # print("action_penalty = ", action_penalty)

        task_rewards += dist_rewards # Extra for being closer to the goal
        # if fails:
        #     task_rewards -= fail_reward - action_penalty*10
        # else:
        #     task_rewards = task_rewards
        task_rewards[fails] = fail_reward - action_penalty
        # print("task_rewards = ", task_rewards)
        return task_rewards
    


    def init_tensor_list(self):
        self.count = 0
        self.obs_tensors_list =[
                [torch.tensor([[0.1277, 0.0013, 0.4004]], device='cuda:0'), torch.tensor([[-0.0031, -0.7100,  0.7042, -0.0029]], device='cuda:0'), 
                    torch.tensor([[ 0.2706, -0.0422,  0.0177]], device='cuda:0'), torch.tensor([[-0.6896,  0.1037,  0.5244]], device='cuda:0'), 
                    torch.tensor([[ 0.0905, -0.0043,  0.4090, -0.0912, -0.0008,  0.4079, -0.1004, -0.1000,
                    0.4049,  0.1010, -0.1021,  0.4056,  0.1000, -0.2010,  0.4064, -0.1000,
                    -0.2001,  0.4063]], device='cuda:0'), 
                    torch.tensor([[ 0.1000, -0.2010,  0.4064, -0.1000, -0.2001,  0.4063, -0.1004, -0.1000,
                    0.4049,  0.1010, -0.1021,  0.4056,  0.1000, -0.2010,  0.4064, -0.1000,
                    -0.2001,  0.4063]], device='cuda:0'), 
                    torch.tensor([[-0.0080, -0.0434, -0.1288, -0.0068, -0.2415, -0.1050,  0.0041, -0.1899,
                    -0.1084,  0.0479, -0.0202, -0.0039, -0.0510, -0.1052, -0.0663,  1.4198,
                    -0.1006, -0.1059, -0.3131, -0.6110,  1.1232, -0.5701, -0.1294, -0.1172]],
                    device='cuda:0'), 
                    torch.tensor([[-1.0004e-01, -2.0009e-01,  4.0635e-01, -4.1613e-05, -2.0142e-01,
                    4.0643e-01,  1.0003e-01, -2.0098e-01,  4.0641e-01, -1.0037e-01,
                    -9.9998e-02,  4.0495e-01,  1.0095e-01, -1.0207e-01,  4.0563e-01,
                    -9.1184e-02, -7.5210e-04,  4.0787e-01,  3.6110e-03, -1.6349e-02,
                    4.3222e-01,  9.0478e-02, -4.3412e-03,  4.0903e-01]], device='cuda:0')],

                [torch.tensor([[0.1310, 0.0008, 0.4014]], device='cuda:0'), torch.tensor([[-0.0067, -0.7127,  0.7015, -0.0052]], device='cuda:0'), 
                    torch.tensor([[ 0.1778, -0.0314,  0.0736]], device='cuda:0'), torch.tensor([[-0.4455,  0.1104,  0.4500]], device='cuda:0'), 
                    torch.tensor([[ 0.1007, -0.0050,  0.4102, -0.0800, -0.0031,  0.4049, -0.0992, -0.0994,
                    0.4049,  0.1002, -0.1030,  0.4049,  0.1000, -0.2018,  0.4049, -0.1000,
                    -0.2001,  0.4049]], device='cuda:0'), 
                    torch.tensor([[ 0.1000, -0.2018,  0.4049, -0.1000, -0.2001,  0.4049, -0.0992, -0.0994,
                    0.4049,  0.1002, -0.1030,  0.4049,  0.1000, -0.2018,  0.4049, -0.1000,
                    -0.2001,  0.4049]], device='cuda:0'), 
                    torch.tensor([[-8.3447e-03, -6.2949e-03,  5.5502e-04,  4.1972e-03,  5.5118e-02,
                    9.9214e-04, -8.2492e-03, -9.9440e-03, -2.2859e-04,  8.4916e-02,
                    3.9478e-02,  5.1221e-03, -2.9583e-02, -5.5203e-02,  2.7019e-03,
                    1.0433e-01,  2.1241e-01, -1.4023e-02,  1.5990e-01, -1.4003e-01,
                    2.8662e-01,  3.8229e-01, -3.9841e-02,  8.5221e-02]], device='cuda:0'), 
                    torch.tensor([[-1.0004e-01, -2.0008e-01,  4.0495e-01, -9.9606e-05, -2.0274e-01,
                    4.0495e-01,  9.9973e-02, -2.0179e-01,  4.0495e-01, -9.9239e-02,
                    -9.9357e-02,  4.0495e-01,  1.0024e-01, -1.0302e-01,  4.0495e-01,
                    -8.0014e-02, -3.1199e-03,  4.0495e-01,  7.5736e-03, -2.1496e-02,
                    4.4012e-01,  1.0074e-01, -5.0208e-03,  4.1017e-01]], device='cuda:0')],

                [torch.tensor([[ 1.3293e-01, -2.3041e-05,  4.0319e-01]], device='cuda:0'), torch.tensor([[-0.0094, -0.7152,  0.6989, -0.0064]], device='cuda:0'), 
                    torch.tensor([[ 0.0956, -0.0538,  0.1222]], device='cuda:0'), torch.tensor([[-0.2956,  0.1425,  0.4258]], device='cuda:0'), 
                    torch.tensor([[ 0.1061, -0.0059,  0.4121, -0.0794, -0.0021,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4050,  0.1000, -0.2020,  0.4049, -0.1000,
                    -0.2001,  0.4049]], device='cuda:0'), 
                    torch.tensor([[ 0.1000, -0.2020,  0.4049, -0.1000, -0.2001,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4050,  0.1000, -0.2020,  0.4049, -0.1000,
                    -0.2001,  0.4049]], device='cuda:0'), 
                    torch.tensor([[-1.5686e-03,  5.3914e-03, -1.1344e-04, -6.8047e-05,  7.5679e-05,
                    -1.2156e-04, -5.5369e-04, -3.0808e-02, -1.1377e-04, -9.6164e-03,
                    3.2789e-02, -1.2107e-04, -2.8394e-05,  6.6516e-02,  5.8613e-03,
                    -8.5175e-02,  1.1857e-02,  3.4634e-04, -2.2265e-01,  1.6445e-01,
                    -1.4236e-02,  2.9371e-01, -5.1541e-02,  1.1264e-01]], device='cuda:0'), 
                    torch.tensor([[-1.0005e-01, -2.0007e-01,  4.0495e-01, -9.9719e-05, -2.0272e-01,
                    4.0495e-01,  9.9965e-02, -2.0201e-01,  4.0495e-01, -9.9481e-02,
                    -9.9378e-02,  4.0495e-01,  1.0028e-01, -1.0228e-01,  4.0495e-01,
                    -7.9447e-02, -2.0525e-03,  4.0495e-01,  4.6636e-03, -2.1577e-02,
                    4.4151e-01,  1.0606e-01, -5.9200e-03,  4.1212e-01]], device='cuda:0')],

                [torch.tensor([[ 0.1336, -0.0014,  0.4057]], device='cuda:0'), torch.tensor([[-0.0116, -0.7175,  0.6964, -0.0066]], device='cuda:0'), torch.tensor([[ 0.0247, -0.0933,  0.1599]], device='cuda:0'), torch.tensor([[-0.1743,  0.1866,  0.4062]], device='cuda:0'), torch.tensor([[ 0.1102, -0.0074,  0.4147, -0.0825, -0.0008,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4050,  0.1000, -0.2021,  0.4049, -0.1000,
                    -0.2001,  0.4049]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4049, -0.1000, -0.2001,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4050,  0.1000, -0.2021,  0.4049, -0.1000,
                    -0.2001,  0.4049]], device='cuda:0'), torch.tensor([[-5.6406e-04,  1.9050e-03, -1.1536e-04, -1.4695e-05,  5.5112e-05,
                    -1.3413e-04, -4.0570e-05, -9.8554e-05, -1.1694e-04, -3.5547e-05,
                    7.8652e-05, -1.2573e-04, -1.0584e-05,  2.5380e-04, -1.4032e-04,
                    -2.7902e-01,  1.0156e-01, -1.3475e-04,  6.9860e-02,  3.5431e-01,
                    -2.4389e-01,  2.2601e-01, -8.0863e-02,  1.4107e-01]], device='cuda:0'), torch.tensor([[-1.0005e-01, -2.0006e-01,  4.0495e-01, -9.9904e-05, -2.0271e-01,
                    4.0495e-01,  9.9959e-02, -2.0211e-01,  4.0495e-01, -9.9482e-02,
                    -9.9375e-02,  4.0495e-01,  1.0029e-01, -1.0227e-01,  4.0495e-01,
                    -8.2550e-02, -8.1222e-04,  4.0495e-01,  3.8438e-03, -1.6785e-02,
                    4.3826e-01,  1.1017e-01, -7.3707e-03,  4.1474e-01]], device='cuda:0')],

                [torch.tensor([[ 0.1333, -0.0034,  0.4087]], device='cuda:0'), torch.tensor([[-0.0134, -0.7198,  0.6940, -0.0059]], device='cuda:0'), torch.tensor([[-0.0352, -0.1317,  0.1897]], device='cuda:0'), torch.tensor([[-0.0766,  0.2164,  0.3902]], device='cuda:0'), torch.tensor([[ 1.1324e-01, -9.4303e-03,  4.1785e-01, -8.6607e-02,  2.3384e-04,
                    4.0495e-01, -9.9483e-02, -9.9373e-02,  4.0495e-01,  1.0029e-01,
                    -1.0227e-01,  4.0495e-01,  9.9958e-02, -2.0211e-01,  4.0495e-01,
                    -1.0005e-01, -2.0005e-01,  4.0495e-01]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4049, -0.1001, -0.2001,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4049, -0.1001,
                    -0.2001,  0.4049]], device='cuda:0'), torch.tensor([[-4.3042e-05,  1.3338e-04, -1.3198e-04, -1.2326e-05,  6.0690e-05,
                    -1.4168e-04, -1.5182e-05, -6.1210e-05, -1.3068e-04, -2.8220e-05,
                    5.6758e-05, -1.3341e-04, -2.0815e-06,  6.4515e-05, -1.2651e-04,
                    -2.3467e-01,  4.5881e-02, -1.9707e-04,  1.2811e-01,  3.5646e-01,
                    -5.1592e-01,  1.7009e-01, -1.1111e-01,  1.7001e-01]], device='cuda:0'), torch.tensor([[-1.0005e-01, -2.0005e-01,  4.0495e-01, -1.0017e-04, -2.0271e-01,
                    4.0495e-01,  9.9958e-02, -2.0211e-01,  4.0495e-01, -9.9483e-02,
                    -9.9373e-02,  4.0495e-01,  1.0029e-01, -1.0227e-01,  4.0495e-01,
                    -8.6607e-02,  2.3384e-04,  4.0495e-01,  6.6427e-03, -1.0176e-02,
                    4.3036e-01,  1.1324e-01, -9.4303e-03,  4.1785e-01]], device='cuda:0')],
                
                [torch.tensor([[ 0.1321, -0.0059,  0.4122]], device='cuda:0'), torch.tensor([[-0.0148, -0.7220,  0.6917, -0.0047]], device='cuda:0'), torch.tensor([[-0.0854, -0.1549,  0.2163]], device='cuda:0'), torch.tensor([[0.0025, 0.2295, 0.3769]], device='cuda:0'), torch.tensor([[ 0.1154, -0.0119,  0.4214, -0.0894,  0.0006,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4049, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-4.0223e-05,  1.3287e-04, -1.3278e-04, -1.1254e-05,  5.9925e-05,
                    -1.3832e-04, -1.4626e-05, -6.5868e-05, -1.3087e-04, -1.9142e-05,
                    -4.4702e-04, -1.2786e-04,  1.0902e-05,  3.8661e-05, -1.3088e-04,
                    -1.7207e-01,  1.2995e-02, -1.3426e-04,  3.1721e-02,  1.4576e-01,
                    -6.2750e-01,  1.2147e-01, -1.2743e-01,  2.0038e-01]], device='cuda:0'), torch.tensor([[-1.0005e-01, -2.0005e-01,  4.0495e-01, -1.0046e-04, -2.0271e-01,
                    4.0495e-01,  9.9958e-02, -2.0211e-01,  4.0495e-01, -9.9484e-02,
                    -9.9372e-02,  4.0495e-01,  1.0029e-01, -1.0227e-01,  4.0495e-01,
                    -8.9409e-02,  5.6545e-04,  4.0495e-01,  7.6157e-03, -6.3530e-03,
                    4.1929e-01,  1.1543e-01, -1.1903e-02,  4.2139e-01]], device='cuda:0')],


                [torch.tensor([[ 0.1301, -0.0087,  0.4162]], device='cuda:0'), torch.tensor([[-0.0159, -0.7242,  0.6894, -0.0031]], device='cuda:0'), torch.tensor([[-0.1279, -0.1683,  0.2424]], device='cuda:0'), torch.tensor([[0.0671, 0.2385, 0.3657]], device='cuda:0'), torch.tensor([[ 0.1169, -0.0146,  0.4253, -0.0903,  0.0006,  0.4050, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4049, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.8614e-05,  1.3216e-04, -1.3815e-04, -1.1857e-05,  5.6686e-05,
                    -1.3664e-04, -1.4059e-05, -6.7159e-05, -1.3107e-04, -2.8256e-05,
                    -2.1016e-04, -1.2546e-04,  1.5520e-04, -5.8556e-04, -1.2098e-04,
                    1.8394e-02, -7.4806e-03, -1.8716e-03,  2.3191e-01,  1.3097e-01,
                    -4.6501e-01,  7.7476e-02, -1.2781e-01,  2.2838e-01]], device='cuda:0'), torch.tensor([[-1.0005e-01, -2.0004e-01,  4.0495e-01, -1.0077e-04, -2.0271e-01,
                    4.0495e-01,  9.9957e-02, -2.0211e-01,  4.0495e-01, -9.9485e-02,
                    -9.9374e-02,  4.0495e-01,  1.0029e-01, -1.0227e-01,  4.0495e-01,
                    -9.0298e-02,  5.9607e-04,  4.0495e-01,  9.3931e-03, -4.0878e-03,
                    4.0868e-01,  1.1687e-01, -1.4604e-02,  4.2534e-01]], device='cuda:0')],
                
                [torch.tensor([[ 0.1275, -0.0115,  0.4205]], device='cuda:0'), torch.tensor([[-0.0166, -0.7262,  0.6873, -0.0010]], device='cuda:0'), torch.tensor([[-0.1638, -0.1767,  0.2646]], device='cuda:0'), torch.tensor([[0.1202, 0.2431, 0.3562]], device='cuda:0'), torch.tensor([[ 1.1768e-01, -1.7432e-02,  4.2964e-01, -8.8886e-02,  3.5322e-04,
                    4.0495e-01, -9.9486e-02, -9.9376e-02,  4.0495e-01,  1.0029e-01,
                    -1.0228e-01,  4.0495e-01,  9.9957e-02, -2.0211e-01,  4.0495e-01,
                    -1.0005e-01, -2.0004e-01,  4.0495e-01]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4049, -0.1001, -0.2000,  0.4050, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4050]], device='cuda:0'), torch.tensor([[-3.9453e-05,  1.3146e-04, -1.3489e-04, -1.2229e-05,  5.1461e-05,
                    -1.4194e-04, -1.3497e-05, -6.8930e-05, -1.3222e-04, -3.2729e-05,
                    -6.5369e-05, -1.2671e-04,  4.7046e-05, -1.8082e-04, -1.1959e-04,
                    1.1060e-01, -2.7729e-02,  3.6819e-04,  2.4324e-01, -8.9351e-02,
                    1.5040e-02,  3.9923e-02, -1.2669e-01,  2.5272e-01]], device='cuda:0'), torch.tensor([[-1.0005e-01, -2.0004e-01,  4.0495e-01, -1.0110e-04, -2.0271e-01,
                    4.0495e-01,  9.9957e-02, -2.0211e-01,  4.0495e-01, -9.9486e-02,
                    -9.9376e-02,  4.0495e-01,  1.0029e-01, -1.0228e-01,  4.0495e-01,
                    -8.8886e-02,  3.5322e-04,  4.0495e-01,  1.3483e-02, -3.7356e-03,
                    4.0498e-01,  1.1768e-01, -1.7432e-02,  4.2964e-01]], device='cuda:0')],

                [torch.tensor([[ 0.1244, -0.0147,  0.4251]], device='cuda:0'), torch.tensor([[-0.0172, -0.7282,  0.6851,  0.0014]], device='cuda:0'), torch.tensor([[-0.1953, -0.1941,  0.2841]], device='cuda:0'), torch.tensor([[0.1636, 0.2682, 0.3485]], device='cuda:0'), torch.tensor([[ 1.1796e-01, -2.0475e-02,  4.3425e-01, -8.7218e-02, -9.8164e-07,
                    4.0495e-01, -9.9487e-02, -9.9378e-02,  4.0495e-01,  1.0029e-01,
                    -1.0228e-01,  4.0495e-01,  9.9956e-02, -2.0212e-01,  4.0495e-01,
                    -1.0006e-01, -2.0004e-01,  4.0495e-01]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.8938e-05,  1.3048e-04, -1.3293e-04, -1.2737e-05,  5.4040e-05,
                    -1.3834e-04, -1.3066e-05, -7.0749e-05, -1.3520e-04, -2.2304e-05,
                    -5.8154e-05, -1.2675e-04,  4.0615e-05, -1.4317e-04, -1.2932e-04,
                    1.5192e-01, -4.4146e-02, -1.1069e-04,  3.9614e-02, -3.6173e-02,
                    3.3337e-01,  6.1289e-03, -1.3944e-01,  2.7213e-01]], device='cuda:0'), torch.tensor([[-1.0006e-01, -2.0004e-01,  4.0495e-01, -1.0144e-04, -2.0270e-01,
                    4.0495e-01,  9.9956e-02, -2.0212e-01,  4.0495e-01, -9.9487e-02,
                    -9.9378e-02,  4.0495e-01,  1.0029e-01, -1.0228e-01,  4.0495e-01,
                    -8.7218e-02, -9.8164e-07,  4.0495e-01,  1.4345e-02, -4.8185e-03,
                    4.0836e-01,  1.1796e-01, -2.0475e-02,  4.3425e-01]], device='cuda:0')],

                [torch.tensor([[ 0.1207, -0.0183,  0.4301]], device='cuda:0'), torch.tensor([[-0.0176, -0.7302,  0.6830,  0.0042]], device='cuda:0'), torch.tensor([[-0.2221, -0.2276,  0.3009]], device='cuda:0'), torch.tensor([[0.1983, 0.2955, 0.3425]], device='cuda:0'), torch.tensor([[ 1.1776e-01, -2.3979e-02,  4.3911e-01, -8.6003e-02, -3.6463e-04,
                    4.0495e-01, -9.9487e-02, -9.9380e-02,  4.0495e-01,  1.0029e-01,
                    -1.0229e-01,  4.0495e-01,  9.9956e-02, -2.0212e-01,  4.0495e-01,
                    -1.0006e-01, -2.0003e-01,  4.0495e-01]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.9150e-05,  1.2955e-04, -1.3856e-04, -1.2980e-05,  5.0861e-05,
                    -1.3670e-04, -1.2723e-05, -7.0691e-05, -1.3545e-04, -1.8623e-05,
                    -4.9860e-05, -1.2668e-04,  9.0110e-04, -3.2022e-03, -1.1851e-04,
                    9.1934e-02, -2.0528e-02, -1.1968e-04,  3.0129e-05, -1.2311e-01,
                    3.2516e-01, -2.1753e-02, -1.6359e-01,  2.9065e-01]], device='cuda:0'), torch.tensor([[-1.0006e-01, -2.0003e-01,  4.0495e-01, -1.0179e-04, -2.0270e-01,
                    4.0495e-01,  9.9956e-02, -2.0212e-01,  4.0495e-01, -9.9487e-02,
                    -9.9380e-02,  4.0495e-01,  1.0029e-01, -1.0229e-01,  4.0495e-01,
                    -8.6003e-02, -3.6463e-04,  4.0495e-01,  1.4430e-02, -6.3429e-03,
                    4.1419e-01,  1.1776e-01, -2.3979e-02,  4.3911e-01]], device='cuda:0')],

                [torch.tensor([[ 0.1170, -0.0224,  0.4353]], device='cuda:0'), torch.tensor([[-0.0181, -0.7321,  0.6809,  0.0073]], device='cuda:0'), torch.tensor([[-0.2463, -0.2566,  0.3155]], device='cuda:0'), torch.tensor([[0.2239, 0.3259, 0.3373]], device='cuda:0'), torch.tensor([[ 0.1171, -0.0280,  0.4442, -0.0852, -0.0005,  0.4050, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4050]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4050, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4050]], device='cuda:0'), torch.tensor([[-3.7356e-05,  1.2887e-04, -1.3520e-04, -1.3234e-05,  5.0201e-05,
                    -1.4205e-04, -1.2415e-05, -7.2830e-05, -1.3587e-04, -1.2554e-05,
                    -3.1551e-05, -1.2583e-04,  4.6308e-05,  1.8783e-04, -1.2049e-04,
                    5.0124e-02, -2.2797e-03, -1.1989e-04,  2.4076e-02, -1.0799e-01,
                    2.0923e-01, -4.7419e-02, -1.8292e-01,  3.0594e-01]], device='cuda:0'), torch.tensor([[-1.0006e-01, -2.0003e-01,  4.0495e-01, -1.0216e-04, -2.0270e-01,
                    4.0495e-01,  9.9955e-02, -2.0212e-01,  4.0495e-01, -9.9488e-02,
                    -9.9382e-02,  4.0495e-01,  1.0029e-01, -1.0229e-01,  4.0495e-01,
                    -8.5247e-02, -4.7759e-04,  4.0495e-01,  1.4476e-02, -8.3810e-03,
                    4.1799e-01,  1.1714e-01, -2.7954e-02,  4.4418e-01]], device='cuda:0')],

                [torch.tensor([[ 0.1131, -0.0269,  0.4407]], device='cuda:0'), torch.tensor([[-0.0186, -0.7340,  0.6788,  0.0107]], device='cuda:0'), torch.tensor([[-0.2652, -0.2811,  0.3274]], device='cuda:0'), torch.tensor([[0.2456, 0.3451, 0.3328]], device='cuda:0'), torch.tensor([[ 1.1338e-01, -3.2551e-02,  4.4955e-01, -8.4819e-02, -4.2887e-04,
                    4.0495e-01, -9.9489e-02, -9.9383e-02,  4.0495e-01,  1.0030e-01,
                    -1.0229e-01,  4.0495e-01,  9.9955e-02, -2.0212e-01,  4.0495e-01,
                    -1.0006e-01, -2.0002e-01,  4.0495e-01]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.8057e-05,  1.2814e-04, -1.3322e-04, -1.3460e-05,  4.9467e-05,
                    -1.3835e-04, -1.2127e-05, -7.1340e-05, -1.3703e-04, -6.1489e-06,
                    3.4372e-05, -1.2439e-04,  8.0624e-05, -5.6775e-05, -1.1995e-04,
                    7.5686e-02,  1.3917e-03, -1.2097e-04, -6.6139e-02, -1.3982e-01,
                    2.6105e-01, -2.3581e-01, -2.1564e-01,  3.2264e-01]], device='cuda:0'), torch.tensor([[-1.0006e-01, -2.0002e-01,  4.0495e-01, -1.0253e-04, -2.0270e-01,
                    4.0495e-01,  9.9955e-02, -2.0212e-01,  4.0495e-01, -9.9489e-02,
                    -9.9383e-02,  4.0495e-01,  1.0030e-01, -1.0229e-01,  4.0495e-01,
                    -8.4819e-02, -4.2887e-04,  4.0495e-01,  1.4095e-02, -1.0680e-02,
                    4.2160e-01,  1.1338e-01, -3.2551e-02,  4.4955e-01]], device='cuda:0')],

                [torch.tensor([[ 0.1087, -0.0318,  0.4461]], device='cuda:0'), torch.tensor([[-0.0190, -0.7359,  0.6767,  0.0143]], device='cuda:0'), torch.tensor([[-0.2798, -0.3016,  0.3262]], device='cuda:0'), torch.tensor([[0.2650, 0.3530, 0.3289]], device='cuda:0'), torch.tensor([[ 1.0915e-01, -3.7521e-02,  4.5496e-01, -8.4383e-02, -3.9082e-04,
                    4.0495e-01, -9.9489e-02, -9.9383e-02,  4.0495e-01,  1.0030e-01,
                    -1.0229e-01,  4.0495e-01,  9.9955e-02, -2.0212e-01,  4.0495e-01,
                    -1.0006e-01, -2.0002e-01,  4.0495e-01]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.7739e-05,  1.2741e-04, -1.3885e-04, -1.3726e-05,  4.8833e-05,
                    -1.3672e-04, -1.1722e-05, -7.1927e-05, -1.3660e-04, -1.2925e-07,
                    6.5573e-05, -1.2357e-04,  7.2240e-05, -4.9508e-05, -1.4192e-04,
                    2.0606e-02,  7.1554e-03, -1.2529e-04, -6.9823e-02, -1.5595e-01,
                    2.7585e-01, -2.6748e-01, -2.4300e-01,  3.2289e-01]], device='cuda:0'), torch.tensor([[-1.0006e-01, -2.0002e-01,  4.0495e-01, -1.0291e-04, -2.0270e-01,
                    4.0495e-01,  9.9955e-02, -2.0212e-01,  4.0495e-01, -9.9489e-02,
                    -9.9383e-02,  4.0495e-01,  1.0030e-01, -1.0229e-01,  4.0495e-01,
                    -8.4383e-02, -3.9082e-04,  4.0495e-01,  1.2680e-02, -1.3710e-02,
                    4.2593e-01,  1.0915e-01, -3.7521e-02,  4.5496e-01]], device='cuda:0')],

                [torch.tensor([[ 0.1039, -0.0369,  0.4515]], device='cuda:0'), torch.tensor([[-0.0194, -0.7377,  0.6746,  0.0180]], device='cuda:0'), torch.tensor([[-0.2931, -0.3191,  0.3182]], device='cuda:0'), torch.tensor([[0.2795, 0.3627, 0.3254]], device='cuda:0'), torch.tensor([[ 1.0452e-01, -4.2809e-02,  4.6026e-01, -8.4254e-02, -3.7163e-04,
                    4.0495e-01, -9.9489e-02, -9.9383e-02,  4.0495e-01,  1.0030e-01,
                    -1.0229e-01,  4.0495e-01,  9.9954e-02, -2.0213e-01,  4.0495e-01,
                    -1.0006e-01, -2.0001e-01,  4.0495e-01]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4050, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4050]], device='cuda:0'), torch.tensor([[-3.7472e-05,  1.2450e-04, -1.3548e-04, -1.3869e-05,  4.8432e-05,
                    -1.4205e-04, -1.1547e-05, -7.2330e-05, -1.3700e-04, -2.0742e-06,
                    4.9707e-05, -1.2475e-04,  6.1599e-05, -3.5900e-05, -1.3939e-04,
                    2.3253e-02,  7.3770e-04, -1.1439e-04, -6.4731e-02, -3.8588e-02,
                    2.2740e-01, -2.8602e-01, -2.6557e-01,  3.1788e-01]], device='cuda:0'), torch.tensor([[-1.0006e-01, -2.0001e-01,  4.0495e-01, -1.0329e-04, -2.0270e-01,
                    4.0495e-01,  9.9954e-02, -2.0213e-01,  4.0495e-01, -9.9489e-02,
                    -9.9383e-02,  4.0495e-01,  1.0030e-01, -1.0229e-01,  4.0495e-01,
                    -8.4254e-02, -3.7163e-04,  4.0495e-01,  1.1417e-02, -1.5371e-02,
                    4.2977e-01,  1.0452e-01, -4.2809e-02,  4.6026e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0988, -0.0424,  0.4566]], device='cuda:0'), torch.tensor([[-0.0197, -0.7395,  0.6726,  0.0219]], device='cuda:0'), torch.tensor([[-0.3038, -0.3340,  0.3044]], device='cuda:0'), torch.tensor([[0.2896, 0.3666, 0.3224]], device='cuda:0'), torch.tensor([[ 9.9513e-02, -4.8374e-02,  4.6537e-01, -8.4232e-02, -3.7184e-04,
                    4.0495e-01, -9.9490e-02, -9.9382e-02,  4.0495e-01,  1.0031e-01,
                    -1.0227e-01,  4.0495e-01,  9.9954e-02, -2.0213e-01,  4.0495e-01,
                    -1.0006e-01, -2.0001e-01,  4.0495e-01]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1023,  0.4049,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.6978e-05,  1.2244e-04, -1.3348e-04, -1.3835e-05,  4.7882e-05,
                    -1.3838e-04, -1.1581e-05, -7.3166e-05, -1.3728e-04, -9.9531e-06,
                    2.3913e-05, -1.2552e-04,  2.5101e-03,  1.3189e-02, -1.4123e-04,
                    3.5143e-04, -9.1653e-07, -4.8404e-04, -7.3869e-02, -6.4257e-02,
                    1.7994e-01, -3.1498e-01, -2.7955e-01,  3.0929e-01]], device='cuda:0'), torch.tensor([[-1.0006e-01, -2.0001e-01,  4.0495e-01, -1.0368e-04, -2.0270e-01,
                    4.0495e-01,  9.9954e-02, -2.0213e-01,  4.0495e-01, -9.9490e-02,
                    -9.9382e-02,  4.0495e-01,  1.0031e-01, -1.0227e-01,  4.0495e-01,
                    -8.4232e-02, -3.7184e-04,  4.0495e-01,  1.0242e-02, -1.6321e-02,
                    4.3268e-01,  9.9513e-02, -4.8374e-02,  4.6537e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0935, -0.0481,  0.4615]], device='cuda:0'), torch.tensor([[-0.0200, -0.7412,  0.6705,  0.0258]], device='cuda:0'), torch.tensor([[-0.3123, -0.3470,  0.2873]], device='cuda:0'), torch.tensor([[0.2955, 0.3757, 0.3194]], device='cuda:0'), torch.tensor([[ 9.4313e-02, -5.4167e-02,  4.7022e-01, -8.4231e-02, -3.7184e-04,
                    4.0495e-01, -9.9490e-02, -9.9381e-02,  4.0495e-01,  1.0034e-01,
                    -1.0214e-01,  4.0495e-01,  9.9953e-02, -2.0213e-01,  4.0495e-01,
                    -1.0006e-01, -2.0001e-01,  4.0495e-01]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1021,  0.4049,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.6669e-05,  1.2480e-04, -1.3907e-04, -1.4089e-05,  4.5877e-05,
                    -1.3672e-04, -1.1417e-05, -7.1877e-05, -1.3789e-04, -8.4917e-06,
                    1.9567e-05, -1.2813e-04,  2.3808e-03,  1.5987e-02, -3.1172e-03,
                    1.8222e-04,  5.3133e-05, -7.3996e-04, -1.0959e-01, -1.7696e-01,
                    1.8227e-01, -3.2156e-01, -2.8595e-01,  2.9633e-01]], device='cuda:0'), torch.tensor([[-1.0006e-01, -2.0001e-01,  4.0495e-01, -1.0408e-04, -2.0269e-01,
                    4.0495e-01,  9.9953e-02, -2.0213e-01,  4.0495e-01, -9.9490e-02,
                    -9.9381e-02,  4.0495e-01,  1.0034e-01, -1.0214e-01,  4.0495e-01,
                    -8.4231e-02, -3.7184e-04,  4.0495e-01,  8.5612e-03, -1.8789e-02,
                    4.3529e-01,  9.4313e-02, -5.4167e-02,  4.7022e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0881, -0.0539,  0.4661]], device='cuda:0'), torch.tensor([[-0.0204, -0.7429,  0.6684,  0.0297]], device='cuda:0'), torch.tensor([[-0.3127, -0.3580,  0.2694]], device='cuda:0'), torch.tensor([[0.3003, 0.3808, 0.3174]], device='cuda:0'), torch.tensor([[ 8.9062e-02, -6.0154e-02,  4.7477e-01, -8.4225e-02, -3.7204e-04,
                    4.0495e-01, -9.9490e-02, -9.9381e-02,  4.0495e-01,  1.0034e-01,
                    -1.0214e-01,  4.0495e-01,  9.9953e-02, -2.0213e-01,  4.0495e-01,
                    -1.0006e-01, -2.0000e-01,  4.0495e-01]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4050, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1021,  0.4049,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4050]], device='cuda:0'), torch.tensor([[-3.7516e-05,  1.2347e-04, -1.3565e-04, -1.4131e-05,  4.7261e-05,
                    -1.4205e-04, -1.2158e-05, -6.9946e-05, -1.3992e-04, -3.7151e-06,
                    2.2244e-05, -1.3377e-04, -1.0004e-03,  4.1486e-04, -6.9787e-03,
                    1.8587e-03, -5.0304e-05, -1.4569e-03, -1.3288e-01, -2.0978e-01,
                    1.9579e-01, -3.2322e-01, -2.9492e-01,  2.8041e-01]], device='cuda:0'), torch.tensor([[-1.0006e-01, -2.0000e-01,  4.0495e-01, -1.0448e-04, -2.0269e-01,
                    4.0495e-01,  9.9953e-02, -2.0213e-01,  4.0495e-01, -9.9490e-02,
                    -9.9381e-02,  4.0495e-01,  1.0034e-01, -1.0214e-01,  4.0495e-01,
                    -8.4225e-02, -3.7204e-04,  4.0495e-01,  6.3059e-03, -2.2631e-02,
                    4.3797e-01,  8.9062e-02, -6.0154e-02,  4.7477e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0828, -0.0600,  0.4704]], device='cuda:0'), torch.tensor([[-0.0207, -0.7446,  0.6663,  0.0337]], device='cuda:0'), torch.tensor([[-0.3086, -0.3674,  0.2500]], device='cuda:0'), torch.tensor([[0.3026, 0.3854, 0.3158]], device='cuda:0'), torch.tensor([[ 8.3869e-02, -6.6305e-02,  4.7901e-01, -8.3993e-02, -4.0930e-04,
                    4.0495e-01, -9.9490e-02, -9.9380e-02,  4.0495e-01,  1.0034e-01,
                    -1.0214e-01,  4.0495e-01,  9.9953e-02, -2.0213e-01,  4.0495e-01,
                    -1.0007e-01, -2.0000e-01,  4.0495e-01]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4050,  0.1003, -0.1021,  0.4050,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.6179e-05,  1.2290e-04, -1.3361e-04, -1.4261e-05,  4.6864e-05,
                    -1.3834e-04, -1.1272e-05, -7.2864e-05, -1.3855e-04, -3.2516e-06,
                    2.7909e-05, -1.3538e-04, -2.9425e-03,  2.3089e-03, -9.8314e-03,
                    3.8328e-02, -8.3065e-03, -3.5853e-03, -1.2673e-01, -1.4692e-01,
                    2.4695e-01, -3.1605e-01, -3.0753e-01,  2.6191e-01]], device='cuda:0'), torch.tensor([[-1.0007e-01, -2.0000e-01,  4.0495e-01, -1.0488e-04, -2.0269e-01,
                    4.0495e-01,  9.9953e-02, -2.0213e-01,  4.0495e-01, -9.9490e-02,
                    -9.9380e-02,  4.0495e-01,  1.0034e-01, -1.0214e-01,  4.0495e-01,
                    -8.3993e-02, -4.0930e-04,  4.0495e-01,  3.9165e-03, -2.5912e-02,
                    4.4125e-01,  8.3869e-02, -6.6305e-02,  4.7901e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0777, -0.0662,  0.4743]], device='cuda:0'), torch.tensor([[-0.0210, -0.7463,  0.6642,  0.0378]], device='cuda:0'), torch.tensor([[-0.3046, -0.3756,  0.2302]], device='cuda:0'), torch.tensor([[0.3009, 0.3899, 0.3151]], device='cuda:0'), torch.tensor([[ 0.0788, -0.0726,  0.4829, -0.0835, -0.0005,  0.4050, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1019,  0.4050,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.1003, -0.1019,  0.4050,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.5150e-05,  1.2255e-04, -1.3913e-04, -1.4190e-05,  4.6416e-05,
                    -1.3670e-04, -1.1304e-05, -6.4723e-05, -1.3969e-04, -2.1913e-06,
                    3.2842e-05, -1.3085e-04, -1.5461e-02,  4.0021e-02, -1.0976e-02,
                    5.8014e-02, -1.3663e-02, -4.7185e-03, -1.0210e-01, -1.5947e-01,
                    2.6994e-01, -3.0764e-01, -3.1801e-01,  2.4491e-01]], device='cuda:0'), torch.tensor([[-1.0007e-01, -2.0000e-01,  4.0495e-01, -1.0528e-04, -2.0269e-01,
                    4.0495e-01,  9.9952e-02, -2.0214e-01,  4.0495e-01, -9.9491e-02,
                    -9.9379e-02,  4.0495e-01,  1.0029e-01, -1.0191e-01,  4.0495e-01,
                    -8.3452e-02, -5.1008e-04,  4.0495e-01,  1.8388e-03, -2.8778e-02,
                    4.4520e-01,  7.8784e-02, -7.2592e-02,  4.8291e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0726, -0.0724,  0.4780]], device='cuda:0'), torch.tensor([[-0.0214, -0.7479,  0.6621,  0.0418]], device='cuda:0'), torch.tensor([[-0.2994, -0.3826,  0.2107]], device='cuda:0'), torch.tensor([[0.2992, 0.3929, 0.3146]], device='cuda:0'), torch.tensor([[ 0.0738, -0.0790,  0.4865, -0.0826, -0.0007,  0.4050, -0.0995, -0.0994,
                    0.4049,  0.1000, -0.1017,  0.4052,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4050]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4050, -0.0995, -0.0994,
                    0.4049,  0.1000, -0.1017,  0.4052,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4050]], device='cuda:0'), torch.tensor([[-3.4480e-05,  1.2123e-04, -1.3577e-04, -1.4048e-05,  4.5994e-05,
                    -1.4207e-04, -1.0911e-05, -1.0731e-05, -1.3025e-04, -8.7564e-07,
                    3.5971e-05, -1.3357e-04, -3.2843e-02,  3.0089e-02,  4.7053e-02,
                    7.7500e-02, -1.9679e-02, -5.3958e-03, -1.0769e-01, -2.2233e-01,
                    2.5408e-01, -3.0207e-01, -3.2513e-01,  2.2682e-01]], device='cuda:0'), torch.tensor([[-1.0007e-01, -1.9999e-01,  4.0495e-01, -1.0568e-04, -2.0269e-01,
                    4.0495e-01,  9.9952e-02, -2.0214e-01,  4.0495e-01, -9.9491e-02,
                    -9.9379e-02,  4.0495e-01,  1.0001e-01, -1.0165e-01,  4.0519e-01,
                    -8.2596e-02, -7.1416e-04,  4.0495e-01, -1.3992e-04, -3.2551e-02,
                    4.4897e-01,  7.3799e-02, -7.8998e-02,  4.8649e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0676, -0.0789,  0.4812]], device='cuda:0'), torch.tensor([[-0.0218, -0.7495,  0.6600,  0.0459]], device='cuda:0'), torch.tensor([[-0.2881, -0.3886,  0.1890]], device='cuda:0'), torch.tensor([[0.3002, 0.3932, 0.3146]], device='cuda:0'), torch.tensor([[ 0.0689, -0.0855,  0.4897, -0.0815, -0.0010,  0.4050, -0.0995, -0.0994,
                    0.4049,  0.0998, -0.1016,  0.4068,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4049,  0.0998, -0.1016,  0.4068,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.5473e-05,  1.2057e-04, -1.3372e-04, -1.3845e-05,  4.5750e-05,
                    -1.3841e-04, -1.4896e-05,  4.7462e-04, -1.1812e-04, -3.2194e-06,
                    5.4612e-05, -1.3490e-04, -9.4374e-03, -2.1961e-03,  1.6676e-01,
                    8.7978e-02, -2.0215e-02, -5.5765e-03, -1.4604e-01, -2.4419e-01,
                    2.5739e-01, -2.9737e-01, -3.3749e-01,  2.0706e-01]], device='cuda:0'), torch.tensor([[-1.0007e-01, -1.9999e-01,  4.0495e-01, -1.0609e-04, -2.0269e-01,
                    4.0495e-01,  9.9952e-02, -2.0214e-01,  4.0495e-01, -9.9491e-02,
                    -9.9378e-02,  4.0495e-01,  9.9840e-02, -1.0159e-01,  4.0681e-01,
                    -8.1516e-02, -9.6820e-04,  4.0495e-01, -2.6293e-03, -3.6997e-02,
                    4.5269e-01,  6.8916e-02, -8.5516e-02,  4.8972e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0629, -0.0854,  0.4841]], device='cuda:0'), torch.tensor([[-0.0222, -0.7512,  0.6579,  0.0499]], device='cuda:0'), torch.tensor([[-0.2728, -0.3940,  0.1630]], device='cuda:0'), torch.tensor([[0.2997, 0.3966, 0.3149]], device='cuda:0'), torch.tensor([[ 0.0643, -0.0921,  0.4925, -0.0805, -0.0013,  0.4050, -0.0995, -0.0994,
                    0.4050,  0.0996, -0.1021,  0.4105,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4050,  0.0996, -0.1021,  0.4105,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.5132e-05,  1.1959e-04, -1.3928e-04, -1.4165e-05,  4.3220e-05,
                    -1.3677e-04, -1.2376e-05,  2.8853e-05, -1.1713e-04,  2.8126e-06,
                    1.3541e-04, -1.2706e-04, -2.1244e-02, -3.7052e-02,  2.7260e-01,
                    8.1553e-02, -2.2641e-02, -5.2236e-03, -1.6428e-01, -2.5781e-01,
                    2.4788e-01, -2.7869e-01, -3.4775e-01,  1.8081e-01]], device='cuda:0'), torch.tensor([[-1.0007e-01, -1.9998e-01,  4.0495e-01, -1.0649e-04, -2.0269e-01,
                    4.0495e-01,  9.9951e-02, -2.0214e-01,  4.0495e-01, -9.9491e-02,
                    -9.9376e-02,  4.0495e-01,  9.9645e-02, -1.0212e-01,  4.1046e-01,
                    -8.0465e-02, -1.2632e-03,  4.0495e-01, -5.5593e-03, -4.1654e-02,
                    4.5633e-01,  6.4302e-02, -9.2123e-02,  4.9254e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0585, -0.0919,  0.4864]], device='cuda:0'), torch.tensor([[-0.0226, -0.7528,  0.6557,  0.0540]], device='cuda:0'), torch.tensor([[-0.2524, -0.3988,  0.1315]], device='cuda:0'), torch.tensor([[0.2992, 0.3972, 0.3154]], device='cuda:0'), torch.tensor([[ 0.0600, -0.0988,  0.4949, -0.0792, -0.0016,  0.4050, -0.0995, -0.0994,
                    0.4050,  0.0990, -0.1035,  0.4151,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4050]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4050, -0.1001, -0.2000,  0.4050, -0.0995, -0.0994,
                    0.4050,  0.0990, -0.1035,  0.4151,  0.1000, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4050]], device='cuda:0'), torch.tensor([[-3.4926e-05,  1.1887e-04, -1.3586e-04, -1.3999e-05,  4.4619e-05,
                    -1.4216e-04, -1.1471e-05,  8.2975e-05, -1.2654e-04,  1.3255e-05,
                    2.1769e-04, -1.2516e-04, -5.5768e-02, -9.0847e-02,  2.9245e-01,
                    1.0462e-01, -3.0953e-02,  8.7153e-03, -1.7644e-01, -2.7636e-01,
                    2.2439e-01, -2.5946e-01, -3.5811e-01,  1.4918e-01]], device='cuda:0'), torch.tensor([[-1.0007e-01, -1.9998e-01,  4.0495e-01, -1.0689e-04, -2.0268e-01,
                    4.0495e-01,  9.9951e-02, -2.0214e-01,  4.0495e-01, -9.9491e-02,
                    -9.9374e-02,  4.0495e-01,  9.8976e-02, -1.0353e-01,  4.1514e-01,
                    -7.9240e-02, -1.6463e-03,  4.0498e-01, -8.7107e-03, -4.6550e-02,
                    4.5961e-01,  6.0009e-02, -9.8815e-02,  4.9485e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0545, -0.0986,  0.4882]], device='cuda:0'), torch.tensor([[-0.0231, -0.7543,  0.6535,  0.0581]], device='cuda:0'), torch.tensor([[-0.2280, -0.4034,  0.0972]], device='cuda:0'), torch.tensor([[0.2972, 0.3979, 0.3161]], device='cuda:0'), torch.tensor([[ 0.0561, -0.1056,  0.4966, -0.0773, -0.0023,  0.4053, -0.0995, -0.0994,
                    0.4050,  0.0977, -0.1057,  0.4193,  0.1000, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4049, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4050,  0.0977, -0.1057,  0.4193,  0.1000, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.4741e-05,  1.1956e-04, -1.3381e-04, -1.4111e-05,  4.3831e-05,
                    -1.3846e-04, -1.1164e-05,  6.6340e-05, -1.2334e-04,  1.6639e-05,
                    2.3452e-04, -1.2813e-04, -8.6711e-02, -1.1751e-01,  2.3758e-01,
                    1.3118e-01, -4.1630e-02,  5.4882e-02, -1.8038e-01, -3.0656e-01,
                    2.0383e-01, -2.3405e-01, -3.6841e-01,  1.1543e-01]], device='cuda:0'), torch.tensor([[-1.0007e-01, -1.9998e-01,  4.0495e-01, -1.0729e-04, -2.0268e-01,
                    4.0495e-01,  9.9951e-02, -2.0213e-01,  4.0495e-01, -9.9491e-02,
                    -9.9371e-02,  4.0495e-01,  9.7736e-02, -1.0573e-01,  4.1925e-01,
                    -7.7306e-02, -2.2911e-03,  4.0529e-01, -1.1968e-02, -5.1906e-02,
                    4.6247e-01,  5.6100e-02, -1.0558e-01,  4.9660e-01]], device='cuda:0')],


                [torch.tensor([[ 0.0509, -0.1054,  0.4894]], device='cuda:0'), torch.tensor([[-0.0235, -0.7559,  0.6513,  0.0621]], device='cuda:0'), torch.tensor([[-0.2011, -0.4077,  0.0608]], device='cuda:0'), torch.tensor([[0.2924, 0.3987, 0.3172]], device='cuda:0'), torch.tensor([[ 0.0526, -0.1124,  0.4978, -0.0753, -0.0031,  0.4066, -0.0995, -0.0994,
                    0.4050,  0.0963, -0.1076,  0.4223,  0.1000, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[ 0.1000, -0.2021,  0.4049, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4050,  0.0963, -0.1076,  0.4223,  0.1000, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.3992e-05,  1.1709e-04, -1.3937e-04, -1.4219e-05,  4.3091e-05,
                    -1.3686e-04, -1.1539e-05,  7.2886e-05, -1.3148e-04,  1.7992e-05,
                    1.1368e-04, -1.2900e-04, -9.1540e-02, -7.1265e-02,  1.7521e-01,
                    1.1062e-01, -4.5083e-02,  1.2859e-01, -1.6913e-01, -3.2355e-01,
                    1.8846e-01, -2.0549e-01, -3.7734e-01,  8.1008e-02]], device='cuda:0'), torch.tensor([[-1.0007e-01, -1.9997e-01,  4.0495e-01, -1.0769e-04, -2.0268e-01,
                    4.0495e-01,  9.9950e-02, -2.0213e-01,  4.0495e-01, -9.9490e-02,
                    -9.9368e-02,  4.0495e-01,  9.6251e-02, -1.0763e-01,  4.2226e-01,
                    -7.5347e-02, -3.1239e-03,  4.0658e-01, -1.5135e-02, -5.7645e-02,
                    4.6497e-01,  5.2638e-02, -1.1243e-01,  4.9775e-01]], device='cuda:0')],


                [torch.tensor([[ 0.0478, -0.1122,  0.4900]], device='cuda:0'), torch.tensor([[-0.0241, -0.7574,  0.6491,  0.0661]], device='cuda:0'), torch.tensor([[-0.1740, -0.4120,  0.0206]], device='cuda:0'), torch.tensor([[0.2803, 0.3995, 0.3108]], device='cuda:0'), torch.tensor([[ 0.0496, -0.1193,  0.4982, -0.0741, -0.0039,  0.4093, -0.0995, -0.0994,
                    0.4050,  0.0948, -0.1085,  0.4246,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4050]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4049, -0.1001, -0.2000,  0.4050, -0.0995, -0.0994,
                    0.4050,  0.0948, -0.1085,  0.4246,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4050]], device='cuda:0'), torch.tensor([[-3.4248e-05,  1.1641e-04, -1.3597e-04, -1.4231e-05,  4.1901e-05,
                    -1.4224e-04, -1.1181e-05,  7.5451e-05, -1.3132e-04,  2.3461e-05,
                    1.6183e-04, -1.2594e-04, -8.6156e-02, -1.9014e-02,  1.6029e-01,
                    5.9388e-02, -3.7240e-02,  2.1431e-01, -1.6007e-01, -3.2953e-01,
                    1.8068e-01, -1.8009e-01, -3.7863e-01,  5.0835e-02]], device='cuda:0'), torch.tensor([[-1.0007e-01, -1.9997e-01,  4.0495e-01, -1.0810e-04, -2.0268e-01,
                    4.0495e-01,  9.9950e-02, -2.0213e-01,  4.0495e-01, -9.9490e-02,
                    -9.9364e-02,  4.0495e-01,  9.4771e-02, -1.0851e-01,  4.2462e-01,
                    -7.4118e-02, -3.9031e-03,  4.0928e-01, -1.8141e-02, -6.3504e-02,
                    4.6732e-01,  4.9646e-02, -1.1933e-01,  4.9825e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0452, -0.1191,  0.4897]], device='cuda:0'), torch.tensor([[-0.0247, -0.7589,  0.6470,  0.0701]], device='cuda:0'), torch.tensor([[-0.1458, -0.4163, -0.0262]], device='cuda:0'), torch.tensor([[0.2621, 0.3993, 0.2970]], device='cuda:0'), torch.tensor([[ 0.0471, -0.1263,  0.4980, -0.0739, -0.0047,  0.4133, -0.0995, -0.0994,
                    0.4050,  0.0940, -0.1092,  0.4264,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4049, -0.1001, -0.2000,  0.4049, -0.0995, -0.0994,
                    0.4050,  0.0940, -0.1092,  0.4264,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.4431e-05,  1.1569e-04, -1.3389e-04, -1.4135e-05,  4.0314e-05,
                    -1.3861e-04, -1.1118e-05,  7.7785e-05, -1.3191e-04,  3.4934e-04,
                    2.4157e-03, -1.1713e-04, -1.6885e-02, -1.9141e-02,  9.6723e-02,
                    -3.8917e-04, -5.0084e-02,  2.8435e-01, -1.4663e-01, -3.4600e-01,
                    1.3543e-01, -1.5839e-01, -3.7805e-01,  1.4786e-02]], device='cuda:0'), torch.tensor([[-1.0007e-01, -1.9997e-01,  4.0495e-01, -1.0850e-04, -2.0268e-01,
                    4.0495e-01,  9.9950e-02, -2.0213e-01,  4.0495e-01, -9.9489e-02,
                    -9.9357e-02,  4.0495e-01,  9.4035e-02, -1.0916e-01,  4.2639e-01,
                    -7.3901e-02, -4.7335e-03,  4.1331e-01, -2.0915e-02, -6.9513e-02,
                    4.6913e-01,  4.7125e-02, -1.2631e-01,  4.9798e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0431, -0.1261,  0.4887]], device='cuda:0'), torch.tensor([[-0.0255, -0.7602,  0.6450,  0.0738]], device='cuda:0'), torch.tensor([[-0.1167, -0.4208, -0.0744]], device='cuda:0'), torch.tensor([[0.2316, 0.3987, 0.2788]], device='cuda:0'), torch.tensor([[ 0.0451, -0.1333,  0.4969, -0.0747, -0.0063,  0.4181, -0.0995, -0.0993,
                    0.4049,  0.0940, -0.1094,  0.4273,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4049, -0.1001, -0.2000,  0.4049, -0.0995, -0.0993,
                    0.4049,  0.0940, -0.1094,  0.4273,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.3984e-05,  1.1665e-04, -1.3931e-04, -1.4375e-05,  3.9405e-05,
                    -1.3698e-04, -1.1151e-05,  7.0794e-05, -1.3382e-04,  1.7030e-03,
                    9.7831e-03, -1.1719e-04,  1.4406e-03, -1.1904e-02,  7.2870e-02,
                    -5.2633e-02, -1.1381e-01,  3.1182e-01, -1.3616e-01, -3.6440e-01,
                    8.8664e-02, -1.2981e-01, -3.8405e-01, -3.1271e-02]], device='cuda:0'), torch.tensor([[-1.0008e-01, -1.9996e-01,  4.0495e-01, -1.0891e-04, -2.0268e-01,
                    4.0495e-01,  9.9949e-02, -2.0213e-01,  4.0495e-01, -9.9483e-02,
                    -9.9325e-02,  4.0495e-01,  9.4044e-02, -1.0938e-01,  4.2735e-01,
                    -7.4689e-02, -6.3376e-03,  4.1810e-01, -2.3498e-02, -7.5828e-02,
                    4.7009e-01,  4.5087e-02, -1.3335e-01,  4.9692e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0415, -0.1331,  0.4869]], device='cuda:0'), torch.tensor([[-0.0264, -0.7614,  0.6431,  0.0774]], device='cuda:0'), torch.tensor([[-0.0841, -0.4251, -0.1192]], device='cuda:0'), torch.tensor([[0.1984, 0.3880, 0.2590]], device='cuda:0'), torch.tensor([[ 0.0436, -0.1404,  0.4951, -0.0761, -0.0091,  0.4230, -0.0995, -0.0993,
                    0.4050,  0.0939, -0.1102,  0.4280,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4050]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4049, -0.1001, -0.2000,  0.4050, -0.0995, -0.0993,
                    0.4050,  0.0939, -0.1102,  0.4280,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4050]], device='cuda:0'), torch.tensor([[-3.3812e-05,  1.1430e-04, -1.3597e-04, -1.4396e-05,  3.7350e-05,
                    -1.4239e-04, -1.0841e-05,  7.0875e-05, -1.3917e-04,  7.7315e-04,
                    3.7989e-03, -1.0623e-04, -1.8612e-02, -6.2120e-02,  6.1625e-02,
                    -6.7701e-02, -1.6135e-01,  3.1183e-01, -8.9760e-02, -3.6429e-01,
                    6.3056e-02, -9.9748e-02, -3.9020e-01, -7.1543e-02]], device='cuda:0'), torch.tensor([[-1.0008e-01, -1.9996e-01,  4.0495e-01, -1.0932e-04, -2.0268e-01,
                    4.0495e-01,  9.9949e-02, -2.0212e-01,  4.0495e-01, -9.9479e-02,
                    -9.9306e-02,  4.0495e-01,  9.3928e-02, -1.1024e-01,  4.2799e-01,
                    -7.6081e-02, -9.0897e-03,  4.2295e-01, -2.5514e-02, -8.2245e-02,
                    4.7046e-01,  4.3567e-02, -1.4045e-01,  4.9510e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0404, -0.1403,  0.4845]], device='cuda:0'), torch.tensor([[-0.0274, -0.7625,  0.6414,  0.0806]], device='cuda:0'), torch.tensor([[-0.0486, -0.4290, -0.1580]], device='cuda:0'), torch.tensor([[0.1650, 0.3601, 0.2402]], device='cuda:0'), torch.tensor([[ 0.0426, -0.1476,  0.4926, -0.0775, -0.0123,  0.4278, -0.0995, -0.0993,
                    0.4050,  0.0935, -0.1115,  0.4285,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4050, -0.1001, -0.2000,  0.4049, -0.0995, -0.0993,
                    0.4050,  0.0935, -0.1115,  0.4285,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.4848e-05,  1.1351e-04, -1.3401e-04, -1.4327e-05,  3.6343e-05,
                    -1.3874e-04, -1.1748e-05,  6.6528e-05, -1.3571e-04,  4.5767e-05,
                    2.6796e-05, -1.1746e-04, -3.4088e-02, -6.3036e-02,  4.4755e-02,
                    -5.7789e-02, -1.6920e-01,  3.2315e-01, -5.9981e-02, -3.6415e-01,
                    5.2817e-02, -6.3087e-02, -3.9676e-01, -1.0746e-01]], device='cuda:0'), torch.tensor([[-1.0008e-01, -1.9996e-01,  4.0495e-01, -1.0973e-04, -2.0268e-01,
                    4.0495e-01,  9.9949e-02, -2.0212e-01,  4.0495e-01, -9.9478e-02,
                    -9.9302e-02,  4.0495e-01,  9.3482e-02, -1.1146e-01,  4.2847e-01,
                    -7.7455e-02, -1.2263e-02,  4.2784e-01, -2.6852e-02, -8.8576e-02,
                    4.7060e-01,  4.2616e-02, -1.4761e-01,  4.9261e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0400, -0.1475,  0.4814]], device='cuda:0'), torch.tensor([[-0.0284, -0.7635,  0.6398,  0.0834]], device='cuda:0'), torch.tensor([[-0.0120, -0.4324, -0.1917]], device='cuda:0'), torch.tensor([[0.1325, 0.3182, 0.2220]], device='cuda:0'), torch.tensor([[ 0.0423, -0.1548,  0.4895, -0.0786, -0.0156,  0.4330, -0.0995, -0.0993,
                    0.4049,  0.0929, -0.1124,  0.4286,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4049, -0.1001, -0.2000,  0.4049, -0.0995, -0.0993,
                    0.4049,  0.0929, -0.1124,  0.4286,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.2000,  0.4049]], device='cuda:0'), torch.tensor([[-3.3580e-05,  1.1396e-04, -1.3940e-04, -1.4573e-05,  3.5052e-05,
                    -1.3710e-04, -1.0674e-05,  6.3710e-05, -1.3420e-04,  5.8878e-05,
                    8.7651e-05, -1.2744e-04, -3.5057e-02, -4.1222e-02,  3.0782e-02,
                    -4.4913e-02, -1.8359e-01,  3.4918e-01, -5.1764e-02, -3.8387e-01,
                    2.4901e-02, -2.1607e-02, -4.0321e-01, -1.3756e-01]], device='cuda:0'), torch.tensor([[-1.0008e-01, -1.9995e-01,  4.0495e-01, -1.1014e-04, -2.0267e-01,
                    4.0495e-01,  9.9948e-02, -2.0212e-01,  4.0495e-01, -9.9477e-02,
                    -9.9299e-02,  4.0495e-01,  9.2919e-02, -1.1240e-01,  4.2862e-01,
                    -7.8629e-02, -1.5600e-02,  4.3305e-01, -2.7909e-02, -9.5073e-02,
                    4.7038e-01,  4.2272e-02, -1.5483e-01,  4.8954e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0403, -0.1547,  0.4779]], device='cuda:0'), torch.tensor([[-0.0293, -0.7644,  0.6383,  0.0857]], device='cuda:0'), torch.tensor([[ 0.0268, -0.4353, -0.2210]], device='cuda:0'), torch.tensor([[0.1002, 0.2629, 0.2037]], device='cuda:0'), torch.tensor([[ 0.0426, -0.1621,  0.4860, -0.0797, -0.0196,  0.4385, -0.0995, -0.0993,
                    0.4050,  0.0925, -0.1131,  0.4289,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1999,  0.4050]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4049, -0.1001, -0.1999,  0.4050, -0.0995, -0.0993,
                    0.4050,  0.0925, -0.1131,  0.4289,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1999,  0.4050]], device='cuda:0'), torch.tensor([[-3.3448e-05,  1.1244e-04, -1.3599e-04, -1.4797e-05,  3.0417e-05,
                    -1.4245e-04, -1.0553e-05,  6.1084e-05, -1.3988e-04,  1.0704e-04,
                    1.2264e-04, -1.1817e-04, -2.1697e-02, -3.2977e-02,  4.9342e-02,
                    -4.2107e-02, -2.4363e-01,  3.5878e-01, -4.2072e-02, -4.0697e-01,
                    -4.0894e-03,  2.2246e-02, -4.0944e-01, -1.6188e-01]], device='cuda:0'), torch.tensor([[-1.0008e-01, -1.9995e-01,  4.0495e-01, -1.1056e-04, -2.0267e-01,
                    4.0495e-01,  9.9948e-02, -2.0212e-01,  4.0495e-01, -9.9475e-02,
                    -9.9295e-02,  4.0495e-01,  9.2477e-02, -1.1306e-01,  4.2887e-01,
                    -7.9691e-02, -1.9622e-02,  4.3853e-01, -2.8865e-02, -1.0194e-01,
                    4.6962e-01,  4.2555e-02, -1.6209e-01,  4.8596e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0411, -0.1620,  0.4739]], device='cuda:0'), torch.tensor([[-0.0301, -0.7653,  0.6370,  0.0874]], device='cuda:0'), torch.tensor([[ 0.0671, -0.4378, -0.2465]], device='cuda:0'), torch.tensor([[0.0674, 0.1999, 0.1869]], device='cuda:0'), torch.tensor([[ 0.0435, -0.1694,  0.4819, -0.0809, -0.0250,  0.4439, -0.0995, -0.0993,
                    0.4050,  0.0923, -0.1140,  0.4296,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.1999,  0.4049]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4050, -0.1001, -0.1999,  0.4049, -0.0995, -0.0993,
                    0.4050,  0.0923, -0.1140,  0.4296,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.1999,  0.4049]], device='cuda:0'), torch.tensor([[-3.3192e-05,  1.1173e-04, -1.3393e-04, -1.4676e-05,  3.0099e-05,
                    -1.3875e-04, -9.3343e-06,  5.8317e-05, -1.3651e-04,  8.3762e-05,
                    1.8726e-05, -1.1547e-04, -7.1878e-03, -6.2908e-02,  8.9674e-02,
                    -6.2274e-02, -3.3610e-01,  3.4221e-01, -1.5203e-02, -4.2972e-01,
                    -1.9449e-02,  7.1643e-02, -4.1401e-01, -1.8219e-01]], device='cuda:0'), torch.tensor([[-1.0008e-01, -1.9995e-01,  4.0495e-01, -1.1098e-04, -2.0267e-01,
                    4.0495e-01,  9.9948e-02, -2.0212e-01,  4.0495e-01, -9.9473e-02,
                    -9.9292e-02,  4.0495e-01,  9.2300e-02, -1.1401e-01,  4.2959e-01,
                    -8.0910e-02, -2.4996e-02,  4.4386e-01, -2.9464e-02, -1.0917e-01,
                    4.6847e-01,  4.3490e-02, -1.6939e-01,  4.8195e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0427, -0.1693,  0.4695]], device='cuda:0'), torch.tensor([[-0.0307, -0.7661,  0.6358,  0.0886]], device='cuda:0'), torch.tensor([[ 0.1066, -0.4399, -0.2685]], device='cuda:0'), torch.tensor([[0.0354, 0.1345, 0.1741]], device='cuda:0'), torch.tensor([[ 0.0451, -0.1767,  0.4776, -0.0827, -0.0320,  0.4486, -0.0995, -0.0993,
                    0.4050,  0.0922, -0.1159,  0.4311,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1999,  0.4049]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4049, -0.1001, -0.1999,  0.4049, -0.0995, -0.0993,
                    0.4050,  0.0922, -0.1159,  0.4311,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1999,  0.4049]], device='cuda:0'), torch.tensor([[-3.3278e-05,  1.1135e-04, -1.3944e-04, -1.4872e-05,  2.9075e-05,
                    -1.3710e-04, -1.0356e-05,  5.7074e-05, -1.3495e-04,  1.0030e-04,
                    1.0922e-04, -1.2783e-04, -5.7184e-03, -1.2692e-01,  1.4191e-01,
                    -1.0768e-01, -4.3601e-01,  2.9955e-01,  3.0724e-03, -4.5861e-01,
                    -4.2390e-02,  1.2193e-01, -4.1452e-01, -1.9436e-01]], device='cuda:0'), torch.tensor([[-1.0008e-01, -1.9994e-01,  4.0495e-01, -1.1140e-04, -2.0267e-01,
                    4.0495e-01,  9.9947e-02, -2.0211e-01,  4.0495e-01, -9.9471e-02,
                    -9.9289e-02,  4.0495e-01,  9.2230e-02, -1.1586e-01,  4.3113e-01,
                    -8.2711e-02, -3.2016e-02,  4.4862e-01, -2.9662e-02, -1.1683e-01,
                    4.6700e-01,  4.5088e-02, -1.7672e-01,  4.7756e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0449, -0.1766,  0.4648]], device='cuda:0'), torch.tensor([[-0.0312, -0.7670,  0.6347,  0.0892]], device='cuda:0'), torch.tensor([[ 0.1458, -0.4353, -0.2872]], device='cuda:0'), torch.tensor([[0.0076, 0.0694, 0.1652]], device='cuda:0'), torch.tensor([[ 0.0473, -0.1840,  0.4729, -0.0853, -0.0407,  0.4524, -0.0995, -0.0993,
                    0.4050,  0.0921, -0.1188,  0.4332,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1999,  0.4050]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4049, -0.1001, -0.1999,  0.4050, -0.0995, -0.0993,
                    0.4050,  0.0921, -0.1188,  0.4332,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1999,  0.4050]], device='cuda:0'), torch.tensor([[-3.3080e-05,  1.1112e-04, -1.3595e-04, -1.4958e-05,  2.6970e-05,
                    -1.4244e-04, -1.0209e-05,  5.3763e-05, -1.4040e-04,  5.6861e-05,
                    -1.1185e-04, -1.2571e-04, -4.8865e-03, -1.8124e-01,  1.6327e-01,
                    -1.5901e-01, -5.2789e-01,  2.5328e-01,  7.5704e-03, -4.9388e-01,
                    -7.8701e-02,  1.6624e-01, -4.0728e-01, -2.1156e-01]], device='cuda:0'), torch.tensor([[-1.0008e-01, -1.9994e-01,  4.0495e-01, -1.1182e-04, -2.0267e-01,
                    4.0495e-01,  9.9947e-02, -2.0211e-01,  4.0495e-01, -9.9459e-02,
                    -9.9274e-02,  4.0495e-01,  9.2109e-02, -1.1881e-01,  4.3325e-01,
                    -8.5348e-02, -4.0668e-02,  4.5243e-01, -2.9714e-02, -1.2505e-01,
                    4.6494e-01,  4.7330e-02, -1.8401e-01,  4.7286e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0478, -0.1837,  0.4599]], device='cuda:0'), torch.tensor([[-0.0315, -0.7678,  0.6337,  0.0892]], device='cuda:0'), torch.tensor([[ 0.1845, -0.4188, -0.3031]], device='cuda:0'), torch.tensor([[-0.0151,  0.0004,  0.1586]], device='cuda:0'), torch.tensor([[ 0.0502, -0.1911,  0.4679, -0.0886, -0.0508,  0.4553, -0.0995, -0.0993,
                    0.4050,  0.0917, -0.1221,  0.4352,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.1999,  0.4049]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4050, -0.1001, -0.1999,  0.4049, -0.0995, -0.0993,
                    0.4050,  0.0917, -0.1221,  0.4352,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.1999,  0.4049]], device='cuda:0'), torch.tensor([[-3.2896e-05,  1.1107e-04, -1.3389e-04, -1.4992e-05,  2.3560e-05,
                    -1.3873e-04, -1.0142e-05,  5.0754e-05, -1.3740e-04,  1.2198e-04,
                    1.9186e-05, -1.3251e-04, -3.9856e-02, -1.7643e-01,  1.3223e-01,
                    -1.8182e-01, -6.1765e-01,  1.9309e-01,  2.9637e-02, -5.1377e-01,
                    -1.1038e-01,  2.0917e-01, -3.8916e-01, -2.3111e-01]], device='cuda:0'), torch.tensor([[-1.0008e-01, -1.9993e-01,  4.0495e-01, -1.1225e-04, -2.0267e-01,
                    4.0495e-01,  9.9947e-02, -2.0211e-01,  4.0495e-01, -9.9456e-02,
                    -9.9273e-02,  4.0495e-01,  9.1700e-02, -1.2212e-01,  4.3516e-01,
                    -8.8559e-02, -5.0823e-02,  4.5532e-01, -2.9540e-02, -1.3371e-01,
                    4.6224e-01,  5.0206e-02, -1.9106e-01,  4.6789e-01]], device='cuda:0')],

                
                [torch.tensor([[ 0.0512, -0.1903,  0.4546]], device='cuda:0'), torch.tensor([[-0.0315, -0.7686,  0.6327,  0.0887]], device='cuda:0'), torch.tensor([[ 0.2183, -0.3863, -0.3165]], device='cuda:0'), torch.tensor([[-0.0290, -0.0666,  0.1531]], device='cuda:0'), torch.tensor([[ 0.0537, -0.1976,  0.4627, -0.0916, -0.0622,  0.4568, -0.0994, -0.0992,
                    0.4049,  0.0914, -0.1251,  0.4366,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1999,  0.4049]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4049, -0.1001, -0.1999,  0.4049, -0.0994, -0.0992,
                    0.4049,  0.0914, -0.1251,  0.4366,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1999,  0.4049]], device='cuda:0'), torch.tensor([[-3.2513e-05,  1.1461e-04, -1.3890e-04, -1.5000e-05,  2.1665e-05,
                    -1.3707e-04, -9.7990e-06,  4.7246e-05, -1.3618e-04,  1.6137e-02,
                    1.5661e-02, -1.5297e-04,  6.8639e-03, -1.5183e-01,  1.1140e-01,
                    -1.6506e-01, -6.8632e-01,  1.0057e-01,  5.3779e-02, -5.1861e-01,
                    -1.2077e-01,  2.4721e-01, -3.5582e-01, -2.5283e-01]], device='cuda:0'), torch.tensor([[-1.0008e-01, -1.9993e-01,  4.0495e-01, -1.1268e-04, -2.0267e-01,
                    4.0495e-01,  9.9946e-02, -2.0211e-01,  4.0495e-01, -9.9391e-02,
                    -9.9204e-02,  4.0495e-01,  9.1402e-02, -1.2506e-01,  4.3658e-01,
                    -9.1643e-02, -6.2189e-02,  4.5680e-01, -2.8926e-02, -1.4253e-01,
                    4.5921e-01,  5.3666e-02, -1.9763e-01,  4.6268e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0552, -0.1961,  0.4492]], device='cuda:0'), torch.tensor([[-0.0312, -0.7695,  0.6317,  0.0878]], device='cuda:0'), torch.tensor([[ 0.2423, -0.3332, -0.3274]], device='cuda:0'), torch.tensor([[-0.0293, -0.1231,  0.1593]], device='cuda:0'), torch.tensor([[ 0.0576, -0.2034,  0.4573, -0.0944, -0.0743,  0.4569, -0.0992, -0.0991,
                    0.4049,  0.0921, -0.1278,  0.4375,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1999,  0.4050]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4049, -0.1001, -0.1999,  0.4050, -0.0992, -0.0991,
                    0.4049,  0.0921, -0.1278,  0.4375,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1999,  0.4050]], device='cuda:0'), torch.tensor([[-3.1719e-05,  2.4720e-04, -1.2564e-04, -1.4970e-05,  1.8907e-05,
                    -1.4225e-04, -9.7689e-06,  4.4060e-05, -1.4147e-04,  2.9346e-02,
                    1.2410e-02, -1.6308e-04,  6.7304e-02, -1.5641e-01,  7.5132e-02,
                    -1.5555e-01, -7.3526e-01,  2.8173e-02,  7.1014e-02, -5.1915e-01,
                    -1.3117e-01,  2.7104e-01, -3.0383e-01, -2.7251e-01]], device='cuda:0'), torch.tensor([[-1.0009e-01, -1.9993e-01,  4.0495e-01, -1.1311e-04, -2.0267e-01,
                    4.0495e-01,  9.9946e-02, -2.0211e-01,  4.0495e-01, -9.9233e-02,
                    -9.9087e-02,  4.0495e-01,  9.2097e-02, -1.2781e-01,  4.3754e-01,
                    -9.4358e-02, -7.4333e-02,  4.5691e-01, -2.7954e-02, -1.5135e-01,
                    4.5601e-01,  5.7594e-02, -2.0339e-01,  4.5728e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0594, -0.2009,  0.4437]], device='cuda:0'), torch.tensor([[-0.0307, -0.7706,  0.6307,  0.0866]], device='cuda:0'), torch.tensor([[ 0.2555, -0.2683, -0.3332]], device='cuda:0'), torch.tensor([[-0.0223, -0.1710,  0.1736]], device='cuda:0'), torch.tensor([[ 0.0618, -0.2081,  0.4518, -0.0972, -0.0872,  0.4561, -0.0984, -0.0991,
                    0.4049,  0.0937, -0.1309,  0.4374,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.1999,  0.4050]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4050, -0.1001, -0.1999,  0.4050, -0.0984, -0.0991,
                    0.4049,  0.0937, -0.1309,  0.4374,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.1999,  0.4050]], device='cuda:0'), torch.tensor([[-2.1105e-03,  1.3470e-02, -1.1847e-04, -1.2868e-05,  7.0575e-06,
                    -1.3718e-04, -8.6021e-06,  4.0198e-05, -1.3806e-04,  1.0922e-01,
                    -8.6605e-03, -5.8197e-03,  1.1576e-01, -1.8594e-01, -1.5889e-02,
                    -1.7800e-01, -7.8748e-01, -2.7852e-02,  7.2964e-02, -5.0876e-01,
                    -1.4867e-01,  2.8270e-01, -2.4162e-01, -2.8350e-01]], device='cuda:0'), torch.tensor([[-1.0009e-01, -1.9989e-01,  4.0495e-01, -1.1355e-04, -2.0267e-01,
                    4.0495e-01,  9.9946e-02, -2.0211e-01,  4.0495e-01, -9.8368e-02,
                    -9.9081e-02,  4.0495e-01,  9.3690e-02, -1.3092e-01,  4.3738e-01,
                    -9.7199e-02, -8.7210e-02,  4.5605e-01, -2.6846e-02, -1.6006e-01,
                    4.5252e-01,  6.1833e-02, -2.0810e-01,  4.5175e-01]], device='cuda:0')],
                
                [torch.tensor([[ 0.0637, -0.2043,  0.4384]], device='cuda:0'), torch.tensor([[-0.0298, -0.7717,  0.6295,  0.0853]], device='cuda:0'), torch.tensor([[ 0.2486, -0.1900, -0.3131]], device='cuda:0'), torch.tensor([[ 0.0056, -0.2091,  0.2035]], device='cuda:0'), torch.tensor([[ 0.0661, -0.2115,  0.4464, -0.1005, -0.1009,  0.4541, -0.0967, -0.0992,
                    0.4049,  0.0958, -0.1340,  0.4352,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1998,  0.4050]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4049, -0.1001, -0.1998,  0.4050, -0.0967, -0.0992,
                    0.4049,  0.0958, -0.1340,  0.4352,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1998,  0.4050]], device='cuda:0'), torch.tensor([[-1.6911e-03,  1.7972e-02, -1.1884e-04, -1.4892e-05,  8.7148e-06,
                    -1.3657e-04, -9.9087e-06,  3.5823e-05, -1.3669e-04,  1.2566e-01,
                    -1.4091e-02, -1.2093e-02,  1.4180e-01, -1.4733e-01, -1.4516e-01,
                    -2.0798e-01, -8.3682e-01, -1.0052e-01,  8.0033e-02, -4.7569e-01,
                    -1.7321e-01,  2.7772e-01, -1.6678e-01, -2.7185e-01]], device='cuda:0'), torch.tensor([[-1.0010e-01, -1.9981e-01,  4.0495e-01, -1.1397e-04, -2.0267e-01,
                    4.0495e-01,  9.9945e-02, -2.0211e-01,  4.0495e-01, -9.6728e-02,
                    -9.9152e-02,  4.0495e-01,  9.5772e-02, -1.3400e-01,  4.3522e-01,
                    -1.0054e-01, -1.0093e-01,  4.5411e-01, -2.5681e-02, -1.6841e-01,
                    4.4864e-01,  6.6149e-02, -2.1153e-01,  4.4643e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0679, -0.2065,  0.4335]], device='cuda:0'), torch.tensor([[-0.0286, -0.7731,  0.6281,  0.0839]], device='cuda:0'), torch.tensor([[ 0.2381, -0.1095, -0.2826]], device='cuda:0'), torch.tensor([[ 0.0352, -0.2452,  0.2304]], device='cuda:0'), torch.tensor([[ 0.0703, -0.2136,  0.4416, -0.1042, -0.1152,  0.4507, -0.0951, -0.0994,
                    0.4049,  0.0982, -0.1360,  0.4311,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1997,  0.4050]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4049, -0.1001, -0.1997,  0.4050, -0.0951, -0.0994,
                    0.4049,  0.0982, -0.1360,  0.4311,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1997,  0.4050]], device='cuda:0'), torch.tensor([[-9.0391e-04,  2.0959e-02, -1.1887e-04, -1.5088e-05,  5.4634e-06,
                    -1.4169e-04, -9.1841e-06,  2.9233e-05, -1.4210e-04,  1.3072e-01,
                    -2.3118e-02, -1.4640e-02,  1.4849e-01, -7.7158e-02, -2.3472e-01,
                    -2.1436e-01, -8.5413e-01, -1.8982e-01,  1.0131e-01, -4.0583e-01,
                    -2.0613e-01,  2.5776e-01, -9.1119e-02, -2.4908e-01]], device='cuda:0'), torch.tensor([[-1.0010e-01, -1.9970e-01,  4.0495e-01, -1.1441e-04, -2.0267e-01,
                    4.0495e-01,  9.9945e-02, -2.0210e-01,  4.0495e-01, -9.5096e-02,
                    -9.9397e-02,  4.0495e-01,  9.8206e-02, -1.3598e-01,  4.3108e-01,
                    -1.0418e-01, -1.1519e-01,  4.5069e-01, -2.4258e-02, -1.7581e-01,
                    4.4425e-01,  7.0289e-02, -2.1363e-01,  4.4157e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0718, -0.2073,  0.4293]], device='cuda:0'), torch.tensor([[-0.0270, -0.7746,  0.6265,  0.0824]], device='cuda:0'), torch.tensor([[ 0.2254, -0.0323, -0.2447]], device='cuda:0'), torch.tensor([[ 0.0641, -0.2751,  0.2495]], device='cuda:0'), torch.tensor([[ 0.0742, -0.2144,  0.4373, -0.1077, -0.1293,  0.4457, -0.0936, -0.0997,
                    0.4049,  0.1003, -0.1365,  0.4277,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.1996,  0.4050]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4050, -0.1001, -0.1996,  0.4050, -0.0936, -0.0997,
                    0.4049,  0.1003, -0.1365,  0.4277,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.1996,  0.4050]], device='cuda:0'), torch.tensor([[-4.0145e-04,  1.7125e-02, -1.1875e-04, -1.5984e-05,  4.9276e-07,
                    -1.3702e-04, -5.5359e-05,  4.5340e-05, -2.4220e-05,  1.0413e-01,
                    -1.3787e-02, -1.4294e-02,  1.1912e-01,  5.4339e-03,  4.0582e-02,
                    -1.9172e-01, -8.3061e-01, -2.8170e-01,  9.8762e-02, -3.3899e-01,
                    -2.3631e-01,  2.3953e-01, -1.3643e-02, -2.1657e-01]], device='cuda:0'), torch.tensor([[-1.0011e-01, -1.9964e-01,  4.0495e-01, -1.1485e-04, -2.0267e-01,
                    4.0495e-01,  9.9945e-02, -2.0210e-01,  4.0495e-01, -9.3579e-02,
                    -9.9657e-02,  4.0495e-01,  1.0027e-01, -1.3652e-01,  4.2769e-01,
                    -1.0767e-01, -1.2931e-01,  4.4568e-01, -2.2626e-02, -1.8200e-01,
                    4.3932e-01,  7.4188e-02, -2.1445e-01,  4.3732e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0755, -0.2071,  0.4255]], device='cuda:0'), torch.tensor([[-0.0250, -0.7761,  0.6249,  0.0809]], device='cuda:0'), torch.tensor([[ 0.2212,  0.0321, -0.2174]], device='cuda:0'), torch.tensor([[ 0.0991, -0.3025,  0.2594]], device='cuda:0'), torch.tensor([[ 0.0779, -0.2142,  0.4336, -0.1107, -0.1428,  0.4392, -0.0928, -0.0998,
                    0.4050,  0.1014, -0.1357,  0.4284,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.1996,  0.4050]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4050, -0.1001, -0.1996,  0.4050, -0.0928, -0.0998,
                    0.4050,  0.1014, -0.1357,  0.4284,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.1996,  0.4050]], device='cuda:0'), torch.tensor([[-1.3135e-04,  4.7584e-03, -1.1800e-04, -1.4798e-05, -1.3740e-06,
                    -1.3591e-04, -2.2603e-04,  2.1003e-05, -4.3668e-04,  6.0699e-02,
                    -1.1079e-03, -1.1568e-02,  3.0512e-02,  3.8439e-02,  1.0791e-04,
                    -1.6933e-01, -7.8250e-01, -3.6071e-01,  8.7386e-02, -3.0409e-01,
                    -2.6864e-01,  2.1862e-01,  2.8929e-02, -2.0468e-01]], device='cuda:0'), torch.tensor([[-1.0011e-01, -1.9962e-01,  4.0495e-01, -1.1528e-04, -2.0267e-01,
                    4.0495e-01,  9.9945e-02, -2.0210e-01,  4.0495e-01, -9.2802e-02,
                    -9.9801e-02,  4.0495e-01,  1.0137e-01, -1.3573e-01,  4.2839e-01,
                    -1.1075e-01, -1.4281e-01,  4.3921e-01, -2.1200e-02, -1.8728e-01,
                    4.3392e-01,  7.7901e-02, -2.1418e-01,  4.3361e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0792, -0.2059,  0.4222]], device='cuda:0'), torch.tensor([[-0.0227, -0.7777,  0.6232,  0.0794]], device='cuda:0'), torch.tensor([[ 0.2271,  0.0870, -0.1937]], device='cuda:0'), torch.tensor([[ 0.1321, -0.3252,  0.2582]], device='cuda:0'), torch.tensor([[ 0.0816, -0.2130,  0.4304, -0.1132, -0.1556,  0.4315, -0.0931, -0.0999,
                    0.4050,  0.1011, -0.1362,  0.4283,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.1996,  0.4050]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4050, -0.1001, -0.1996,  0.4050, -0.0931, -0.0999,
                    0.4050,  0.1011, -0.1362,  0.4283,  0.0999, -0.2021,  0.4050, -0.1001,
                    -0.1996,  0.4050]], device='cuda:0'), torch.tensor([[-1.5193e-04,  3.5552e-03, -1.1744e-04, -1.6875e-05, -4.0111e-06,
                    -1.4108e-04, -2.5541e-04, -3.0506e-05, -3.2029e-05, -3.8593e-02,
                    1.9786e-02,  2.2929e-02, -5.4873e-02, -3.3769e-02,  3.6911e-03,
                    -1.1039e-01, -7.3950e-01, -4.1747e-01,  1.1193e-01, -2.6279e-01,
                    -3.0143e-01,  2.1765e-01,  7.6480e-02, -1.9114e-01]], device='cuda:0'), torch.tensor([[-1.0011e-01, -1.9961e-01,  4.0495e-01, -1.1573e-04, -2.0267e-01,
                    4.0495e-01,  9.9946e-02, -2.0210e-01,  4.0495e-01, -9.3121e-02,
                    -9.9922e-02,  4.0502e-01,  1.0113e-01, -1.3620e-01,  4.2832e-01,
                    -1.1317e-01, -1.5557e-01,  4.3153e-01, -1.9710e-02, -1.9188e-01,
                    4.2797e-01,  8.1645e-02, -2.1302e-01,  4.3038e-01]], device='cuda:0')],


                [torch.tensor([[ 0.0830, -0.2039,  0.4191]], device='cuda:0'), torch.tensor([[-0.0202, -0.7792,  0.6216,  0.0779]], device='cuda:0'), torch.tensor([[ 0.2344,  0.1306, -0.1801]], device='cuda:0'), torch.tensor([[ 0.1437, -0.3431,  0.2423]], device='cuda:0'), torch.tensor([[ 0.0854, -0.2110,  0.4273, -0.1143, -0.1673,  0.4232, -0.0949, -0.0996,
                    0.4059,  0.1001, -0.1358,  0.4285,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1996,  0.4050]], device='cuda:0'), torch.tensor([[ 0.0999, -0.2021,  0.4049, -0.1001, -0.1996,  0.4050, -0.0949, -0.0996,
                    0.4059,  0.1001, -0.1358,  0.4285,  0.0999, -0.2021,  0.4049, -0.1001,
                    -0.1996,  0.4050]], device='cuda:0'), torch.tensor([[-1.3289e-05,  1.9775e-04, -1.3072e-04, -1.5618e-05, -3.9304e-06,
                    -1.3769e-04, -7.6497e-05,  2.7635e-05, -2.1920e-04, -1.2213e-01,
                    2.2813e-02,  1.1435e-01, -3.7664e-02,  9.2401e-02,  5.6041e-02,
                    -2.1245e-02, -7.0490e-01, -4.5855e-01,  1.4613e-01, -2.6421e-01,
                    -2.9454e-01,  2.2514e-01,  1.1451e-01, -1.6472e-01]], device='cuda:0'), torch.tensor([[-1.0011e-01, -1.9961e-01,  4.0495e-01, -1.1617e-04, -2.0267e-01,
                    4.0495e-01,  9.9946e-02, -2.0210e-01,  4.0495e-01, -9.4890e-02,
                    -9.9552e-02,  4.0592e-01,  1.0007e-01, -1.3577e-01,  4.2851e-01,
                    -1.1427e-01, -1.6734e-01,  4.2318e-01, -1.7691e-02, -1.9629e-01,
                    4.2177e-01,  8.5387e-02, -2.1104e-01,  4.2734e-01]], device='cuda:0')],


                [torch.tensor([[ 0.0870, -0.2012,  0.4162]], device='cuda:0'), torch.tensor([[-0.0177, -0.7806,  0.6201,  0.0762]], device='cuda:0'), torch.tensor([[ 0.2379,  0.1705, -0.1725]], device='cuda:0'), torch.tensor([[ 0.1328, -0.3584,  0.2120]], device='cuda:0'), torch.tensor([[ 0.0893, -0.2083,  0.4244, -0.1134, -0.1789,  0.4142, -0.0974, -0.0995,
                    0.4084,  0.0991, -0.1338,  0.4292,  0.1011, -0.2020,  0.4049, -0.1001,
                    -0.1996,  0.4050]], device='cuda:0'), torch.tensor([[ 0.1011, -0.2020,  0.4049, -0.1001, -0.1996,  0.4050, -0.0974, -0.0995,
                    0.4084,  0.0991, -0.1338,  0.4292,  0.1011, -0.2020,  0.4049, -0.1001,
                    -0.1996,  0.4050]], device='cuda:0'), torch.tensor([[-1.0383e-05,  1.4148e-04, -1.3285e-04,  4.9043e-02,  2.5143e-04,
                    -1.1710e-04, -2.5267e-02,  7.9302e-04, -3.8137e-04, -1.6991e-01,
                    5.8264e-03,  2.1598e-01, -1.0894e-01,  9.1717e-02,  4.2073e-02,
                    1.2596e-01, -6.5405e-01, -4.7188e-01,  2.0537e-01, -2.0210e-01,
                    -2.8561e-01,  2.3213e-01,  1.5302e-01, -1.5960e-01]], device='cuda:0'), torch.tensor([[-1.0011e-01, -1.9960e-01,  4.0495e-01,  3.1017e-05, -2.0267e-01,
                    4.0495e-01,  1.0111e-01, -2.0205e-01,  4.0495e-01, -9.7357e-02,
                    -9.9518e-02,  4.0841e-01,  9.9078e-02, -1.3382e-01,  4.2918e-01,
                    -1.1340e-01, -1.7893e-01,  4.1416e-01, -1.4863e-02, -2.0013e-01,
                    4.1580e-01,  8.9317e-02, -2.0835e-01,  4.2442e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0910, -0.1979,  0.4135]], device='cuda:0'), torch.tensor([[-0.0151, -0.7818,  0.6189,  0.0744]], device='cuda:0'), torch.tensor([[ 0.2386,  0.2048, -0.1651]], device='cuda:0'), torch.tensor([[ 0.1192, -0.3717,  0.1753]], device='cuda:0'), torch.tensor([[ 0.0933, -0.2051,  0.4216, -0.1104, -0.1888,  0.4054, -0.1003, -0.1006,
                    0.4130,  0.0973, -0.1312,  0.4296,  0.1008, -0.2021,  0.4050, -0.1001,
                    -0.1996,  0.4049]], device='cuda:0'), torch.tensor([[ 0.1008, -0.2021,  0.4050, -0.1001, -0.1996,  0.4049, -0.1003, -0.1006,
                    0.4130,  0.0973, -0.1312,  0.4296,  0.1008, -0.2021,  0.4050, -0.1001,
                    -0.1996,  0.4049]], device='cuda:0'), torch.tensor([[-9.9714e-07,  1.3718e-04, -1.3189e-04,  7.5425e-05, -2.8155e-06,
                    -1.3060e-04, -3.6984e-02, -1.4198e-03,  2.9220e-04, -1.7422e-01,
                    -9.7237e-02,  3.5369e-01, -9.3755e-02,  1.7585e-01,  4.7967e-02,
                    2.0828e-01, -5.4224e-01, -4.2990e-01,  2.4589e-01, -1.3914e-01,
                    -2.7568e-01,  2.2841e-01,  1.7698e-01, -1.4865e-01]], device='cuda:0'), torch.tensor([[-1.0011e-01, -1.9960e-01,  4.0495e-01,  1.7780e-04, -2.0266e-01,
                    4.0495e-01,  1.0084e-01, -2.0206e-01,  4.0495e-01, -1.0034e-01,
                    -1.0058e-01,  4.1296e-01,  9.7321e-02, -1.3122e-01,  4.2964e-01,
                    -1.1040e-01, -1.8882e-01,  4.0544e-01, -1.1065e-02, -2.0294e-01,
                    4.0999e-01,  9.3277e-02, -2.0506e-01,  4.2164e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0950, -0.1943,  0.4106]], device='cuda:0'), torch.tensor([[-0.0126, -0.7828,  0.6179,  0.0724]], device='cuda:0'), torch.tensor([[ 0.2371,  0.2183, -0.1715]], device='cuda:0'), torch.tensor([[ 0.0812, -0.3833,  0.1310]], device='cuda:0'), torch.tensor([[ 0.0973, -0.2015,  0.4188, -0.1092, -0.1962,  0.4050, -0.1028, -0.1037,
                    0.4190,  0.0963, -0.1280,  0.4298,  0.1008, -0.2018,  0.4045, -0.1001,
                    -0.1996,  0.4050]], device='cuda:0'), torch.tensor([[ 0.1008, -0.2018,  0.4045, -0.1001, -0.1996,  0.4050, -0.1028, -0.1037,
                    0.4190,  0.0963, -0.1280,  0.4298,  0.1008, -0.2018,  0.4045, -0.1001,
                    -0.1996,  0.4050]], device='cuda:0'), torch.tensor([[ 2.1186e-02,  7.1219e-03, -1.4477e-04,  1.7550e-02,  2.4045e-03,
                    1.6670e-03,  3.5990e-02,  3.9178e-02, -6.1206e-02, -1.0837e-01,
                    -2.1384e-01,  3.4347e-01, -3.8251e-02,  1.9581e-01,  2.9559e-02,
                    -5.4922e-02, -5.1543e-01,  3.2063e-02,  7.1205e-02, -1.1043e-01,
                    1.4841e-01,  2.3218e-01,  1.9980e-01, -1.5530e-01]], device='cuda:0'), torch.tensor([[-1.0007e-01, -1.9958e-01,  4.0495e-01,  7.5895e-05, -2.0258e-01,
                    4.0495e-01,  1.0076e-01, -2.0185e-01,  4.0453e-01, -1.0282e-01,
                    -1.0366e-01,  4.1903e-01,  9.6284e-02, -1.2805e-01,  4.2983e-01,
                    -1.0919e-01, -1.9624e-01,  4.0495e-01, -9.1149e-03, -2.0464e-01,
                    4.0801e-01,  9.7257e-02, -2.0145e-01,  4.1881e-01]], device='cuda:0')],

                [torch.tensor([[ 0.0989, -0.1907,  0.4075]], device='cuda:0'), torch.tensor([[-0.0103, -0.7835,  0.6174,  0.0701]], device='cuda:0'), torch.tensor([[ 0.2338,  0.2129, -0.1933]], device='cuda:0'), torch.tensor([[ 0.0475, -0.3938,  0.0785]], device='cuda:0'), torch.tensor([[ 0.1012, -0.1979,  0.4157, -0.1090, -0.2001,  0.4049, -0.1039, -0.1094,
                0.4248,  0.0962, -0.1248,  0.4297,  0.1027, -0.2000,  0.4028, -0.0997,
                -0.1997,  0.4050]], device='cuda:0'), torch.tensor([[ 0.1027, -0.2000,  0.4028, -0.0997, -0.1997,  0.4050, -0.1039, -0.1094,
                0.4248,  0.0962, -0.1248,  0.4297,  0.1027, -0.2000,  0.4028, -0.0997,
                -0.1997,  0.4050]], device='cuda:0'), torch.tensor([[ 0.0336, -0.0443,  0.0010,  0.1409, -0.0255, -0.0043,  0.1567,  0.1414,
                -0.1018,  0.0066, -0.3204,  0.3704,  0.0250,  0.1731,  0.0275,  0.1082,
                -0.1044,  0.0057,  0.1333, -0.0326,  0.0018,  0.2280,  0.2041, -0.1888]],
                device='cuda:0'), torch.tensor([[-0.0997, -0.1997,  0.4050,  0.0014, -0.2027,  0.4049,  0.1027, -0.2000,
                    0.4028, -0.1039, -0.1094,  0.4248,  0.0962, -0.1248,  0.4297, -0.1090,
                    -0.2001,  0.4049, -0.0075, -0.2053,  0.4083,  0.1012, -0.1979,  0.4157]],
                device='cuda:0')],
            ]




    def init_tensor_list1(self):
        self.count = 0
        self.obs_tensors_list1 =[
            [torch.tensor([[0.1400, 0.0500, 0.4300]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.1407,  0.0501,  0.4292, -0.0812,  0.0185,  0.4429, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1461, -0.1469,  0.4000, -0.0646,
         -0.1700,  0.4312]], device='cuda:0'), torch.tensor([[ 0.1461, -0.1469,  0.4000, -0.0646, -0.1700,  0.4312, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1461, -0.1469,  0.4000, -0.0646,
         -0.1700,  0.4312]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
            device='cuda:0'), torch.tensor([[-0.0646, -0.1700,  0.4312,  0.0408, -0.1585,  0.4130,  0.1461, -0.1469,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0812,
          0.0185,  0.4429,  0.0298,  0.0343,  0.4361,  0.1407,  0.0501,  0.4292]],
            device='cuda:0')],

            [torch.tensor([[0.1400, 0.0500, 0.4300]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.1401,  0.0499,  0.4301, -0.0812,  0.0185,  0.4428, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1458, -0.1493,  0.4000, -0.0640,
         -0.1702,  0.4303]], device='cuda:0'), torch.tensor([[ 0.1458, -0.1493,  0.4000, -0.0640, -0.1702,  0.4303, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1458, -0.1493,  0.4000, -0.0640,
         -0.1702,  0.4303]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0640, -0.1702,  0.4303,  0.0409, -0.1597,  0.4135,  0.1458, -0.1493,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0812,
          0.0185,  0.4428,  0.0294,  0.0342,  0.4365,  0.1401,  0.0499,  0.4301]],
       device='cuda:0')],
            [torch.tensor([[0.1399, 0.0460, 0.4326]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.1395,  0.0499,  0.4310, -0.0804,  0.0188,  0.4423, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1462, -0.1461,  0.4000, -0.0640,
         -0.1703,  0.4302]], device='cuda:0'), torch.tensor([[ 0.1462, -0.1461,  0.4000, -0.0640, -0.1703,  0.4302, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1462, -0.1461,  0.4000, -0.0640,
         -0.1703,  0.4302]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0640, -0.1703,  0.4302,  0.0411, -0.1582,  0.4123,  0.1462, -0.1461,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0804,
          0.0188,  0.4423,  0.0295,  0.0343,  0.4367,  0.1395,  0.0499,  0.4310]],
       device='cuda:0')],
            [torch.tensor([[0.1398, 0.0410, 0.4361]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.1437,  0.0469,  0.4319, -0.0799,  0.0194,  0.4412, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1460, -0.1462,  0.4000, -0.0641,
         -0.1701,  0.4300]], device='cuda:0'), torch.tensor([[ 0.1460, -0.1462,  0.4000, -0.0641, -0.1701,  0.4300, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1460, -0.1462,  0.4000, -0.0641,
         -0.1701,  0.4300]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0641, -0.1701,  0.4300,  0.0409, -0.1582,  0.4120,  0.1460, -0.1462,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0799,
          0.0194,  0.4412,  0.0319,  0.0331,  0.4365,  0.1437,  0.0469,  0.4319]],
       device='cuda:0')],

            [torch.tensor([[0.1396, 0.0360, 0.4395]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.1478,  0.0361,  0.4339, -0.0793,  0.0195,  0.4404, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1456, -0.1471,  0.4000, -0.0645,
         -0.1702,  0.4313]], device='cuda:0'), torch.tensor([[ 0.1456, -0.1471,  0.4000, -0.0645, -0.1702,  0.4313, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1456, -0.1471,  0.4000, -0.0645,
         -0.1702,  0.4313]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0645, -0.1702,  0.4313,  0.0406, -0.1586,  0.4129,  0.1456, -0.1471,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0793,
          0.0195,  0.4404,  0.0343,  0.0278,  0.4371,  0.1478,  0.0361,  0.4339]],
       device='cuda:0')],
       [torch.tensor([[0.1393, 0.0309, 0.4431]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.1488,  0.0318,  0.4313, -0.0833,  0.0177,  0.4451, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1459, -0.1464,  0.4000, -0.0638,
         -0.1702,  0.4306]], device='cuda:0'), torch.tensor([[ 0.1459, -0.1464,  0.4000, -0.0638, -0.1702,  0.4306, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1459, -0.1464,  0.4000, -0.0638,
         -0.1702,  0.4306]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0638, -0.1702,  0.4306,  0.0411, -0.1583,  0.4122,  0.1459, -0.1464,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0833,
          0.0177,  0.4451,  0.0327,  0.0248,  0.4382,  0.1488,  0.0318,  0.4313]],
       device='cuda:0')],
       [torch.tensor([[0.1389, 0.0259, 0.4467]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.1515,  0.0278,  0.4297, -0.0797,  0.0192,  0.4396, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1455, -0.1488,  0.4000, -0.0639,
         -0.1702,  0.4304]], device='cuda:0'), torch.tensor([[ 0.1455, -0.1488,  0.4000, -0.0639, -0.1702,  0.4304, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1455, -0.1488,  0.4000, -0.0639,
         -0.1702,  0.4304]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0639, -0.1702,  0.4304,  0.0408, -0.1595,  0.4130,  0.1455, -0.1488,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0797,
          0.0192,  0.4396,  0.0359,  0.0235,  0.4346,  0.1515,  0.0278,  0.4297]],
       device='cuda:0')],
       [torch.tensor([[0.1387, 0.0209, 0.4498]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.1518,  0.0230,  0.4280, -0.0816,  0.0184,  0.4423, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1450, -0.1512,  0.4000, -0.0644,
         -0.1704,  0.4314]], device='cuda:0'), torch.tensor([[ 0.1450, -0.1512,  0.4000, -0.0644, -0.1704,  0.4314, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1450, -0.1512,  0.4000, -0.0644,
         -0.1704,  0.4314]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0644, -0.1704,  0.4314,  0.0403, -0.1608,  0.4144,  0.1450, -0.1512,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0816,
          0.0184,  0.4423,  0.0351,  0.0207,  0.4351,  0.1518,  0.0230,  0.4280]],
       device='cuda:0')],
       [torch.tensor([[0.1382, 0.0159, 0.4535]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.1492,  0.0177,  0.4309, -0.0815,  0.0177,  0.4423, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1452, -0.1515,  0.4000, -0.0643,
         -0.1706,  0.4316]], device='cuda:0'), torch.tensor([[ 0.1452, -0.1515,  0.4000, -0.0643, -0.1706,  0.4316, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1452, -0.1515,  0.4000, -0.0643,
         -0.1706,  0.4316]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0643, -0.1706,  0.4316,  0.0405, -0.1610,  0.4143,  0.1452, -0.1515,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0815,
          0.0177,  0.4423,  0.0338,  0.0177,  0.4366,  0.1492,  0.0177,  0.4309]],
       device='cuda:0')],
       [torch.tensor([[0.1378, 0.0109, 0.4569]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.1882,  0.0190,  0.4272, -0.0824,  0.0162,  0.4445, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1452, -0.1530,  0.4000, -0.0629,
         -0.1708,  0.4300]], device='cuda:0'), torch.tensor([[ 0.1452, -0.1530,  0.4000, -0.0629, -0.1708,  0.4300, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1452, -0.1530,  0.4000, -0.0629,
         -0.1708,  0.4300]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0629, -0.1708,  0.4300,  0.0411, -0.1619,  0.4142,  0.1452, -0.1530,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0824,
          0.0162,  0.4445,  0.0529,  0.0176,  0.4359,  0.1882,  0.0190,  0.4272]],
       device='cuda:0')],
       [torch.tensor([[0.1375, 0.0059, 0.4602]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.1971,  0.0115,  0.4296, -0.0794,  0.0169,  0.4438, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1457, -0.1514,  0.4000, -0.0637,
         -0.1702,  0.4307]], device='cuda:0'), torch.tensor([[ 0.1457, -0.1514,  0.4000, -0.0637, -0.1702,  0.4307, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1457, -0.1514,  0.4000, -0.0637,
         -0.1702,  0.4307]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0637, -0.1702,  0.4307,  0.0410, -0.1608,  0.4139,  0.1457, -0.1514,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0794,
          0.0169,  0.4438,  0.0589,  0.0142,  0.4367,  0.1971,  0.0115,  0.4296]],
       device='cuda:0')],
       [torch.tensor([[0.1366, 0.0009, 0.4637]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2013,  0.0085,  0.4247, -0.0778,  0.0163,  0.4443, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1457, -0.1516,  0.4000, -0.0643,
         -0.1704,  0.4315]], device='cuda:0'), torch.tensor([[ 0.1457, -0.1516,  0.4000, -0.0643, -0.1704,  0.4315, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1457, -0.1516,  0.4000, -0.0643,
         -0.1704,  0.4315]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0643, -0.1704,  0.4315,  0.0407, -0.1610,  0.4142,  0.1457, -0.1516,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0778,
          0.0163,  0.4443,  0.0617,  0.0124,  0.4345,  0.2013,  0.0085,  0.4247]],
       device='cuda:0')],

       [torch.tensor([[ 0.1356, -0.0042,  0.4673]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2005,  0.0021,  0.4230, -0.0750,  0.0158,  0.4432, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1461, -0.1498,  0.4000, -0.0637,
         -0.1702,  0.4307]], device='cuda:0'), torch.tensor([[ 0.1461, -0.1498,  0.4000, -0.0637, -0.1702,  0.4307, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1461, -0.1498,  0.4000, -0.0637,
         -0.1702,  0.4307]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0637, -0.1702,  0.4307,  0.0412, -0.1600,  0.4133,  0.1461, -0.1498,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0750,
          0.0158,  0.4432,  0.0628,  0.0090,  0.4331,  0.2005,  0.0021,  0.4230]],
       device='cuda:0')],

       [torch.tensor([[ 0.1344, -0.0092,  0.4708]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.1994, -0.0059,  0.4244, -0.0707,  0.0166,  0.4405, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1461, -0.1501,  0.4000, -0.0621,
         -0.1713,  0.4293]], device='cuda:0'), torch.tensor([[ 0.1461, -0.1501,  0.4000, -0.0621, -0.1713,  0.4293, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1461, -0.1501,  0.4000, -0.0621,
         -0.1713,  0.4293]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0621, -0.1713,  0.4293,  0.0420, -0.1607,  0.4123,  0.1461, -0.1501,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0707,
          0.0166,  0.4405,  0.0643,  0.0053,  0.4325,  0.1994, -0.0059,  0.4244]],
       device='cuda:0')],
       [torch.tensor([[ 0.1331, -0.0142,  0.4747]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.1977, -0.0101,  0.4252, -0.0708,  0.0156,  0.4423, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1459, -0.1519,  0.4000, -0.0621,
         -0.1715,  0.4293]], device='cuda:0'), torch.tensor([[ 0.1459, -0.1519,  0.4000, -0.0621, -0.1715,  0.4293, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1459, -0.1519,  0.4000, -0.0621,
         -0.1715,  0.4293]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0621, -0.1715,  0.4293,  0.0419, -0.1617,  0.4128,  0.1459, -0.1519,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0708,
          0.0156,  0.4423,  0.0635,  0.0027,  0.4337,  0.1977, -0.0101,  0.4252]],
       device='cuda:0')],
       [torch.tensor([[ 0.1315, -0.0192,  0.4789]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2004, -0.0159,  0.4234, -0.0672,  0.0155,  0.4404, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1464, -0.1504,  0.4000, -0.0637,
         -0.1713,  0.4306]], device='cuda:0'), torch.tensor([[ 0.1464, -0.1504,  0.4000, -0.0637, -0.1713,  0.4306, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1464, -0.1504,  0.4000, -0.0637,
         -0.1713,  0.4306]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-6.3688e-02, -1.7133e-01,  4.3061e-01,  4.1357e-02, -1.6085e-01,
          4.1279e-01,  1.4640e-01, -1.5037e-01,  4.0000e-01, -7.1869e-02,
         -7.5567e-02,  4.3583e-01,  1.3653e-01, -4.1649e-02,  4.2444e-01,
         -6.7208e-02,  1.5461e-02,  4.4043e-01,  6.6592e-02, -2.0035e-04,
          4.3193e-01,  2.0039e-01, -1.5861e-02,  4.2342e-01]], device='cuda:0')],
          [torch.tensor([[ 0.1300, -0.0242,  0.4830]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2078, -0.0233,  0.4215, -0.0677,  0.0137,  0.4433, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1463, -0.1519,  0.4000, -0.0638,
         -0.1712,  0.4304]], device='cuda:0'), torch.tensor([[ 0.1463, -0.1519,  0.4000, -0.0638, -0.1712,  0.4304, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1463, -0.1519,  0.4000, -0.0638,
         -0.1712,  0.4304]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0638, -0.1712,  0.4304,  0.0412, -0.1615,  0.4134,  0.1463, -0.1519,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0677,
          0.0137,  0.4433,  0.0701, -0.0048,  0.4324,  0.2078, -0.0233,  0.4215]],
       device='cuda:0')],

       [torch.tensor([[ 0.1285, -0.0293,  0.4872]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2115, -0.0295,  0.4212, -0.0616,  0.0129,  0.4417, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1466, -0.1504,  0.4000, -0.0644,
         -0.1710,  0.4313]], device='cuda:0'), torch.tensor([[ 0.1466, -0.1504,  0.4000, -0.0644, -0.1710,  0.4313, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1466, -0.1504,  0.4000, -0.0644,
         -0.1710,  0.4313]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0644, -0.1710,  0.4313,  0.0411, -0.1607,  0.4131,  0.1466, -0.1504,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0616,
          0.0129,  0.4417,  0.0749, -0.0083,  0.4314,  0.2115, -0.0295,  0.4212]],
       device='cuda:0')],

       [torch.tensor([[ 0.1268, -0.0343,  0.4911]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2173, -0.0350,  0.4188, -0.0610,  0.0109,  0.4424, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1470, -0.1497,  0.4000, -0.0630,
         -0.1715,  0.4298]], device='cuda:0'), torch.tensor([[ 0.1470, -0.1497,  0.4000, -0.0630, -0.1715,  0.4298, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1470, -0.1497,  0.4000, -0.0630,
         -0.1715,  0.4298]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0630, -0.1715,  0.4298,  0.0420, -0.1606,  0.4120,  0.1470, -0.1497,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0610,
          0.0109,  0.4424,  0.0781, -0.0120,  0.4306,  0.2173, -0.0350,  0.4188]],
       device='cuda:0')],

       [torch.tensor([[ 0.1251, -0.0393,  0.4951]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2142, -0.0413,  0.4191, -0.0607,  0.0079,  0.4427, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1467, -0.1505,  0.4000, -0.0639,
         -0.1711,  0.4303]], device='cuda:0'), torch.tensor([[ 0.1467, -0.1505,  0.4000, -0.0639, -0.1711,  0.4303, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1467, -0.1505,  0.4000, -0.0639,
         -0.1711,  0.4303]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0639, -0.1711,  0.4303,  0.0414, -0.1608,  0.4126,  0.1467, -0.1505,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0607,
          0.0079,  0.4427,  0.0768, -0.0167,  0.4309,  0.2142, -0.0413,  0.4191]],
       device='cuda:0')],

       [torch.tensor([[ 0.1231, -0.0443,  0.4994]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2187, -0.0457,  0.4185, -0.0577,  0.0056,  0.4398, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1467, -0.1523,  0.4000, -0.0641,
         -0.1711,  0.4300]], device='cuda:0'), torch.tensor([[ 0.1467, -0.1523,  0.4000, -0.0641, -0.1711,  0.4300, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1467, -0.1523,  0.4000, -0.0641,
         -0.1711,  0.4300]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0641, -0.1711,  0.4300,  0.0413, -0.1617,  0.4128,  0.1467, -0.1523,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0577,
          0.0056,  0.4398,  0.0805, -0.0200,  0.4291,  0.2187, -0.0457,  0.4185]],
       device='cuda:0')],

       [torch.tensor([[ 0.1211, -0.0494,  0.5035]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2181, -0.0552,  0.4186, -0.0575,  0.0036,  0.4436, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1492, -0.1563,  0.4000, -0.0641,
         -0.1711,  0.4301]], device='cuda:0'), torch.tensor([[ 0.1492, -0.1563,  0.4000, -0.0641, -0.1711,  0.4301, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1492, -0.1563,  0.4000, -0.0641,
         -0.1711,  0.4301]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0641, -0.1711,  0.4301,  0.0426, -0.1637,  0.4134,  0.1492, -0.1563,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0575,
          0.0036,  0.4436,  0.0803, -0.0258,  0.4311,  0.2181, -0.0552,  0.4186]],
       device='cuda:0')],

       [torch.tensor([[ 0.1190, -0.0544,  0.5079]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2360, -0.0575,  0.4188, -0.0503,  0.0028,  0.4381, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1494, -0.1547,  0.4000, -0.0642,
         -0.1712,  0.4298]], device='cuda:0'), torch.tensor([[ 0.1494, -0.1547,  0.4000, -0.0642, -0.1712,  0.4298, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1494, -0.1547,  0.4000, -0.0642,
         -0.1712,  0.4298]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0642, -0.1712,  0.4298,  0.0426, -0.1630,  0.4127,  0.1494, -0.1547,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0503,
          0.0028,  0.4381,  0.0929, -0.0273,  0.4285,  0.2360, -0.0575,  0.4188]],
       device='cuda:0')],

       [torch.tensor([[ 0.1170, -0.0593,  0.5119]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2397, -0.0623,  0.4167, -0.0479,  0.0024,  0.4400, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1490, -0.1583,  0.4000, -0.0642,
         -0.1708,  0.4298]], device='cuda:0'), torch.tensor([[ 0.1490, -0.1583,  0.4000, -0.0642, -0.1708,  0.4298, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1490, -0.1583,  0.4000, -0.0642,
         -0.1708,  0.4298]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0642, -0.1708,  0.4298,  0.0424, -0.1646,  0.4135,  0.1490, -0.1583,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0479,
          0.0024,  0.4400,  0.0959, -0.0299,  0.4283,  0.2397, -0.0623,  0.4167]],
       device='cuda:0')],

       [torch.tensor([[ 0.1146, -0.0644,  0.5165]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2452, -0.0684,  0.4138, -0.0444, -0.0015,  0.4416, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1492, -0.1596,  0.4000, -0.0641,
         -0.1707,  0.4301]], device='cuda:0'), torch.tensor([[ 0.1492, -0.1596,  0.4000, -0.0641, -0.1707,  0.4301, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1492, -0.1596,  0.4000, -0.0641,
         -0.1707,  0.4301]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0641, -0.1707,  0.4301,  0.0426, -0.1652,  0.4136,  0.1492, -0.1596,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0444,
         -0.0015,  0.4416,  0.1004, -0.0350,  0.4277,  0.2452, -0.0684,  0.4138]],
       device='cuda:0')],

       [torch.tensor([[ 0.1121, -0.0693,  0.5211]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2611, -0.0767,  0.4158, -0.0375, -0.0029,  0.4392, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1495, -0.1618,  0.4000, -0.0641,
         -0.1711,  0.4300]], device='cuda:0'), torch.tensor([[ 0.1495, -0.1618,  0.4000, -0.0641, -0.1711,  0.4300, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1495, -0.1618,  0.4000, -0.0641,
         -0.1711,  0.4300]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0641, -0.1711,  0.4300,  0.0427, -0.1665,  0.4136,  0.1495, -0.1618,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0375,
         -0.0029,  0.4392,  0.1118, -0.0398,  0.4275,  0.2611, -0.0767,  0.4158]],
       device='cuda:0')],

       [torch.tensor([[ 0.1095, -0.0743,  0.5256]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2582, -0.0829,  0.4145, -0.0363, -0.0047,  0.4392, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1497, -0.1629,  0.4000, -0.0651,
         -0.1715,  0.4303]], device='cuda:0'), torch.tensor([[ 0.1497, -0.1629,  0.4000, -0.0651, -0.1715,  0.4303, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1497, -0.1629,  0.4000, -0.0651,
         -0.1715,  0.4303]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0651, -0.1715,  0.4303,  0.0423, -0.1672,  0.4138,  0.1497, -0.1629,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0363,
         -0.0047,  0.4392,  0.1109, -0.0438,  0.4269,  0.2582, -0.0829,  0.4145]],
       device='cuda:0')],

       [torch.tensor([[ 0.1066, -0.0794,  0.5303]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2645, -0.0961,  0.4161, -0.1348,  0.0711,  0.4000, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1500, -0.1631,  0.4000, -0.0636,
         -0.1718,  0.4289]], device='cuda:0'), torch.tensor([[ 0.1500, -0.1631,  0.4000, -0.0636, -0.1718,  0.4289, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1500, -0.1631,  0.4000, -0.0636,
         -0.1718,  0.4289]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0636, -0.1718,  0.4289,  0.0432, -0.1675,  0.4130,  0.1500, -0.1631,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.1348,
          0.0711,  0.4000,  0.0649, -0.0125,  0.4014,  0.2645, -0.0961,  0.4161]],
       device='cuda:0')],

       [torch.tensor([[ 0.1037, -0.0844,  0.5350]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2662, -0.1013,  0.4166, -0.1362,  0.0709,  0.4000, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1502, -0.1623,  0.4000, -0.0643,
         -0.1718,  0.4296]], device='cuda:0'), torch.tensor([[ 0.1502, -0.1623,  0.4000, -0.0643, -0.1718,  0.4296, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1502, -0.1623,  0.4000, -0.0643,
         -0.1718,  0.4296]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0643, -0.1718,  0.4296,  0.0430, -0.1670,  0.4130,  0.1502, -0.1623,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.1362,
          0.0709,  0.4000,  0.0650, -0.0152,  0.4024,  0.2662, -0.1013,  0.4166]],
       device='cuda:0')],

       [torch.tensor([[ 0.1010, -0.0894,  0.5399]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2739, -0.1074,  0.4133, -0.1391,  0.0769,  0.4000, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1504, -0.1632,  0.4000, -0.0644,
         -0.1716,  0.4294]], device='cuda:0'), torch.tensor([[ 0.1504, -0.1632,  0.4000, -0.0644, -0.1716,  0.4294, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1504, -0.1632,  0.4000, -0.0644,
         -0.1716,  0.4294]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0644, -0.1716,  0.4294,  0.0430, -0.1674,  0.4132,  0.1504, -0.1632,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.1391,
          0.0769,  0.4000,  0.0674, -0.0153,  0.4000,  0.2739, -0.1074,  0.4133]],
       device='cuda:0')],

       [torch.tensor([[ 0.0984, -0.0945,  0.5450]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2789, -0.1141,  0.4113, -0.1415,  0.0767,  0.4000, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1505, -0.1633,  0.4000, -0.0647,
         -0.1706,  0.4310]], device='cuda:0'), torch.tensor([[ 0.1505, -0.1633,  0.4000, -0.0647, -0.1706,  0.4310, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1505, -0.1633,  0.4000, -0.0647,
         -0.1706,  0.4310]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0647, -0.1706,  0.4310,  0.0429, -0.1669,  0.4139,  0.1505, -0.1633,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.1415,
          0.0767,  0.4000,  0.0687, -0.0187,  0.4000,  0.2789, -0.1141,  0.4113]],
       device='cuda:0')],

       [torch.tensor([[ 0.0958, -0.0994,  0.5499]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2859, -0.1205,  0.4088, -0.1500,  0.0833,  0.4000, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1508, -0.1631,  0.4000, -0.0645,
         -0.1711,  0.4313]], device='cuda:0'), torch.tensor([[ 0.1508, -0.1631,  0.4000, -0.0645, -0.1711,  0.4313, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1508, -0.1631,  0.4000, -0.0645,
         -0.1711,  0.4313]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0645, -0.1711,  0.4313,  0.0432, -0.1671,  0.4142,  0.1508, -0.1631,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.1500,
          0.0833,  0.4000,  0.0679, -0.0186,  0.4000,  0.2859, -0.1205,  0.4088]],
       device='cuda:0')],

       [torch.tensor([[ 0.0932, -0.1044,  0.5549]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2833, -0.1285,  0.4097, -0.1548,  0.0876,  0.4000, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1509, -0.1608,  0.4000, -0.0626,
         -0.1719,  0.4304]], device='cuda:0'), torch.tensor([[ 0.1509, -0.1608,  0.4000, -0.0626, -0.1719,  0.4304, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1509, -0.1608,  0.4000, -0.0626,
         -0.1719,  0.4304]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0626, -0.1719,  0.4304,  0.0441, -0.1664,  0.4137,  0.1509, -0.1608,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.1548,
          0.0876,  0.4000,  0.0642, -0.0204,  0.4000,  0.2833, -0.1285,  0.4097]],
       device='cuda:0')],
       [torch.tensor([[ 0.0907, -0.1094,  0.5599]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.3095, -0.1348,  0.4059, -0.1599,  0.0926,  0.4000, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1502, -0.1610,  0.4000, -0.0604,
         -0.1718,  0.4302]], device='cuda:0'), torch.tensor([[ 0.1502, -0.1610,  0.4000, -0.0604, -0.1718,  0.4302, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1502, -0.1610,  0.4000, -0.0604,
         -0.1718,  0.4302]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0604, -0.1718,  0.4302,  0.0449, -0.1664,  0.4144,  0.1502, -0.1610,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.1599,
          0.0926,  0.4000,  0.0748, -0.0211,  0.4000,  0.3095, -0.1348,  0.4059]],
       device='cuda:0')],

       [torch.tensor([[ 0.0882, -0.1145,  0.5650]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.3113, -0.1425,  0.4063, -0.1646,  0.0937,  0.4000, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1505, -0.1557,  0.4000, -0.0604,
         -0.1724,  0.4300]], device='cuda:0'), torch.tensor([[ 0.1505, -0.1557,  0.4000, -0.0604, -0.1724,  0.4300, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1505, -0.1557,  0.4000, -0.0604,
         -0.1724,  0.4300]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0604, -0.1724,  0.4300,  0.0450, -0.1640,  0.4140,  0.1505, -0.1557,
          0.4000, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.1646,
          0.0937,  0.4000,  0.0734, -0.0244,  0.4000,  0.3113, -0.1425,  0.4063]],
       device='cuda:0')],

       [torch.tensor([[ 0.0855, -0.1195,  0.5700]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.3151, -0.1500,  0.4080, -0.0474, -0.0516,  0.4351, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1498, -0.1529,  0.4005, -0.0603,
         -0.1721,  0.4303]], device='cuda:0'), torch.tensor([[ 0.1498, -0.1529,  0.4005, -0.0603, -0.1721,  0.4303, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1498, -0.1529,  0.4005, -0.0603,
         -0.1721,  0.4303]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0603, -0.1721,  0.4303,  0.0448, -0.1625,  0.4154,  0.1498, -0.1529,
          0.4005, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0474,
         -0.0516,  0.4351,  0.1339, -0.1008,  0.4216,  0.3151, -0.1500,  0.4080]],
       device='cuda:0')],

       [torch.tensor([[ 0.0828, -0.1245,  0.5750]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.3193, -0.1614,  0.4063, -0.1713,  0.0942,  0.4000, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1466, -0.1532,  0.4039, -0.0578,
         -0.1717,  0.4325]], device='cuda:0'), torch.tensor([[ 0.1466, -0.1532,  0.4039, -0.0578, -0.1717,  0.4325, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1466, -0.1532,  0.4039, -0.0578,
         -0.1717,  0.4325]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0578, -0.1717,  0.4325,  0.0444, -0.1625,  0.4182,  0.1466, -0.1532,
          0.4039, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.1713,
          0.0942,  0.4000,  0.0740, -0.0336,  0.4000,  0.3193, -0.1614,  0.4063]],
       device='cuda:0')],

       [torch.tensor([[ 0.0789, -0.1295,  0.5800]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[-0.4889, -0.1428,  0.9440, -0.0393, -0.0599,  0.4324, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1448, -0.1579,  0.4097, -0.0582,
         -0.1729,  0.4335]], device='cuda:0'), torch.tensor([[ 0.1448, -0.1579,  0.4097, -0.0582, -0.1729,  0.4335, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1448, -0.1579,  0.4097, -0.0582,
         -0.1729,  0.4335]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0582, -0.1729,  0.4335,  0.0433, -0.1654,  0.4216,  0.1448, -0.1579,
          0.4097, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0393,
         -0.0599,  0.4324, -0.2641, -0.1013,  0.6882, -0.4889, -0.1428,  0.9440]],
       device='cuda:0')],

       [torch.tensor([[ 0.0757, -0.1344,  0.5850]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[-0.4889, -0.1428,  0.9440, -0.0423, -0.0635,  0.4329, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1430, -0.1598,  0.4122, -0.0585,
         -0.1741,  0.4330]], device='cuda:0'), torch.tensor([[ 0.1430, -0.1598,  0.4122, -0.0585, -0.1741,  0.4330, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1430, -0.1598,  0.4122, -0.0585,
         -0.1741,  0.4330]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0585, -0.1741,  0.4330,  0.0422, -0.1670,  0.4226,  0.1430, -0.1598,
          0.4122, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0423,
         -0.0635,  0.4329, -0.2656, -0.1032,  0.6885, -0.4889, -0.1428,  0.9440]],
       device='cuda:0')],

       [torch.tensor([[ 0.0843, -0.1444,  0.5768]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.3605, -0.1923,  0.4018, -0.0344, -0.0903,  0.4303, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1426, -0.1617,  0.4091, -0.4889,
         -0.1428,  0.9440]], device='cuda:0'), torch.tensor([[ 0.1426, -0.1617,  0.4091, -0.4889, -0.1428,  0.9440, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1426, -0.1617,  0.4091, -0.4889,
         -0.1428,  0.9440]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.4889, -0.1428,  0.9440, -0.1731, -0.1522,  0.6766,  0.1426, -0.1617,
          0.4091, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0344,
         -0.0903,  0.4303,  0.1630, -0.1413,  0.4161,  0.3605, -0.1923,  0.4018]],
       device='cuda:0')],

       [torch.tensor([[ 0.0811, -0.1494,  0.5800]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[-0.4889, -0.1428,  0.9440, -0.0309, -0.1057,  0.4292, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1448, -0.1613,  0.4078, -0.1269,
         -0.1699,  0.5143]], device='cuda:0'), torch.tensor([[ 0.1448, -0.1613,  0.4078, -0.1269, -0.1699,  0.5143, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1448, -0.1613,  0.4078, -0.1269,
         -0.1699,  0.5143]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.1269, -0.1699,  0.5143,  0.0089, -0.1656,  0.4610,  0.1448, -0.1613,
          0.4078, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0309,
         -0.1057,  0.4292, -0.2599, -0.1242,  0.6866, -0.4889, -0.1428,  0.9440]],
       device='cuda:0')],

       [torch.tensor([[ 0.0807, -0.1544,  0.5768]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.3579, -0.2115,  0.4019, -0.0317, -0.1179,  0.4308, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1421, -0.1614,  0.4076, -0.1224,
         -0.1704,  0.5083]], device='cuda:0'), torch.tensor([[ 0.1421, -0.1614,  0.4076, -0.1224, -0.1704,  0.5083, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1421, -0.1614,  0.4076, -0.1224,
         -0.1704,  0.5083]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.1224, -0.1704,  0.5083,  0.0098, -0.1659,  0.4579,  0.1421, -0.1614,
          0.4076, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0317,
         -0.1179,  0.4308,  0.1631, -0.1647,  0.4164,  0.3579, -0.2115,  0.4019]],
       device='cuda:0')],

       [torch.tensor([[ 0.0849, -0.1594,  0.5718]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[-0.4889, -0.1428,  0.9440, -0.0376, -0.1281,  0.4282, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1431, -0.1620,  0.4066, -0.1196,
         -0.1710,  0.5052]], device='cuda:0'), torch.tensor([[ 0.1431, -0.1620,  0.4066, -0.1196, -0.1710,  0.5052, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1431, -0.1620,  0.4066, -0.1196,
         -0.1710,  0.5052]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.1196, -0.1710,  0.5052,  0.0118, -0.1665,  0.4559,  0.1431, -0.1620,
          0.4066, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0376,
         -0.1281,  0.4282, -0.2632, -0.1355,  0.6861, -0.4889, -0.1428,  0.9440]],
       device='cuda:0')],

       [torch.tensor([[ 0.0840, -0.1645,  0.5712]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.3206, -0.2201,  0.4022, -0.0430, -0.1426,  0.4279, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1443, -0.1609,  0.4065, -0.1137,
         -0.1711,  0.5015]], device='cuda:0'), torch.tensor([[ 0.1443, -0.1609,  0.4065, -0.1137, -0.1711,  0.5015, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1443, -0.1609,  0.4065, -0.1137,
         -0.1711,  0.5015]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.1137, -0.1711,  0.5015,  0.0153, -0.1660,  0.4540,  0.1443, -0.1609,
          0.4065, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0430,
         -0.1426,  0.4279,  0.1388, -0.1813,  0.4151,  0.3206, -0.2201,  0.4022]],
       device='cuda:0')],

       [torch.tensor([[ 0.0876, -0.1694,  0.5672]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.3201, -0.2247,  0.4029, -0.0491, -0.1533,  0.4306, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1435, -0.1618,  0.4069, -0.1108,
         -0.1713,  0.4966]], device='cuda:0'), torch.tensor([[ 0.1435, -0.1618,  0.4069, -0.1108, -0.1713,  0.4966, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1435, -0.1618,  0.4069, -0.1108,
         -0.1713,  0.4966]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.1108, -0.1713,  0.4966,  0.0164, -0.1666,  0.4518,  0.1435, -0.1618,
          0.4069, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0491,
         -0.1533,  0.4306,  0.1355, -0.1890,  0.4168,  0.3201, -0.2247,  0.4029]],
       device='cuda:0')],

       [torch.tensor([[ 0.0866, -0.1745,  0.5671]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.3320, -0.2350,  0.4004, -0.0956, -0.1656,  0.4761, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1437, -0.1610,  0.4063, -0.1010,
         -0.1722,  0.4802]], device='cuda:0'), torch.tensor([[ 0.1437, -0.1610,  0.4063, -0.1010, -0.1722,  0.4802, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1437, -0.1610,  0.4063, -0.1010,
         -0.1722,  0.4802]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.1010, -0.1722,  0.4802,  0.0214, -0.1666,  0.4433,  0.1437, -0.1610,
          0.4063, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0956,
         -0.1656,  0.4761,  0.1182, -0.2003,  0.4382,  0.3320, -0.2350,  0.4004]],
       device='cuda:0')],

       [torch.tensor([[ 0.0843, -0.1794,  0.5680]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.3272, -0.2441,  0.4000, -0.4889, -0.1428,  0.9440, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1426, -0.1629,  0.4071, -0.0992,
         -0.1720,  0.4774]], device='cuda:0'), torch.tensor([[ 0.1426, -0.1629,  0.4071, -0.0992, -0.1720,  0.4774, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1426, -0.1629,  0.4071, -0.0992,
         -0.1720,  0.4774]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0992, -0.1720,  0.4774,  0.0217, -0.1674,  0.4423,  0.1426, -0.1629,
          0.4071, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.4889,
         -0.1428,  0.9440, -0.0808, -0.1935,  0.6719,  0.3272, -0.2441,  0.4000]],
       device='cuda:0')],

       [torch.tensor([[ 0.0838, -0.1845,  0.5646]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2966, -0.2480,  0.4000, -0.4889, -0.1428,  0.9440, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1420, -0.1638,  0.4075, -0.0936,
         -0.1738,  0.4712]], device='cuda:0'), torch.tensor([[ 0.1420, -0.1638,  0.4075, -0.0936, -0.1738,  0.4712, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1420, -0.1638,  0.4075, -0.0936,
         -0.1738,  0.4712]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0936, -0.1738,  0.4712,  0.0242, -0.1688,  0.4394,  0.1420, -0.1638,
          0.4075, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.4889,
         -0.1428,  0.9440, -0.0961, -0.1954,  0.6718,  0.2966, -0.2480,  0.4000]],
       device='cuda:0')],

       [torch.tensor([[ 0.0879, -0.1895,  0.5596]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2849, -0.2511,  0.4016, -0.0929, -0.1866,  0.4732, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1413, -0.1629,  0.4070, -0.0996,
         -0.1920,  0.4791]], device='cuda:0'), torch.tensor([[ 0.1413, -0.1629,  0.4070, -0.0996, -0.1920,  0.4791, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1413, -0.1629,  0.4070, -0.0996,
         -0.1920,  0.4791]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0996, -0.1920,  0.4791,  0.0208, -0.1775,  0.4430,  0.1413, -0.1629,
          0.4070, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0929,
         -0.1866,  0.4732,  0.0960, -0.2188,  0.4374,  0.2849, -0.2511,  0.4016]],
       device='cuda:0')],

       [torch.tensor([[ 0.0929, -0.1945,  0.5546]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2908, -0.2577,  0.4000, -0.0684, -0.1084,  0.4314, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1411, -0.1643,  0.4066, -0.0959,
         -0.2086,  0.4763]], device='cuda:0'), torch.tensor([[ 0.1411, -0.1643,  0.4066, -0.0959, -0.2086,  0.4763, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1411, -0.1643,  0.4066, -0.0959,
         -0.2086,  0.4763]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0959, -0.2086,  0.4763,  0.0226, -0.1864,  0.4414,  0.1411, -0.1643,
          0.4066, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0684,
         -0.1084,  0.4314,  0.1112, -0.1830,  0.4153,  0.2908, -0.2577,  0.4000]],
       device='cuda:0')],

       [torch.tensor([[ 0.0940, -0.1995,  0.5498]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2942, -0.2648,  0.4000, -0.0675, -0.1122,  0.4289, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1376, -0.1793,  0.4185, -0.0937,
         -0.2216,  0.4712]], device='cuda:0'), torch.tensor([[ 0.1376, -0.1793,  0.4185, -0.0937, -0.2216,  0.4712, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1376, -0.1793,  0.4185, -0.0937,
         -0.2216,  0.4712]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0937, -0.2216,  0.4712,  0.0219, -0.2005,  0.4449,  0.1376, -0.1793,
          0.4185, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0675,
         -0.1122,  0.4289,  0.1134, -0.1885,  0.4136,  0.2942, -0.2648,  0.4000]],
       device='cuda:0')],

       [torch.tensor([[ 0.0924, -0.2045,  0.5494]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2864, -0.2711,  0.4000, -0.0680, -0.1144,  0.4298, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1303, -0.2037,  0.4360, -0.0903,
         -0.2282,  0.4631]], device='cuda:0'), torch.tensor([[ 0.1303, -0.2037,  0.4360, -0.0903, -0.2282,  0.4631, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1303, -0.2037,  0.4360, -0.0903,
         -0.2282,  0.4631]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0903, -0.2282,  0.4631,  0.0200, -0.2160,  0.4496,  0.1303, -0.2037,
          0.4360, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0680,
         -0.1144,  0.4298,  0.1092, -0.1928,  0.4143,  0.2864, -0.2711,  0.4000]],
       device='cuda:0')],

       [torch.tensor([[ 0.0902, -0.2095,  0.5504]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2834, -0.2809,  0.4000, -0.0679, -0.1156,  0.4299, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1273, -0.2221,  0.4500, -0.0968,
         -0.2321,  0.4656]], device='cuda:0'), torch.tensor([[ 0.1273, -0.2221,  0.4500, -0.0968, -0.2321,  0.4656, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1273, -0.2221,  0.4500, -0.0968,
         -0.2321,  0.4656]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0968, -0.2321,  0.4656,  0.0152, -0.2271,  0.4578,  0.1273, -0.2221,
          0.4500, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0679,
         -0.1156,  0.4299,  0.1078, -0.1982,  0.4134,  0.2834, -0.2809,  0.4000]],
       device='cuda:0')],

       [torch.tensor([[ 0.0875, -0.2146,  0.5517]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2883, -0.2854,  0.4000, -0.0667, -0.1197,  0.4295, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1233, -0.2388,  0.4609, -0.0942,
         -0.2369,  0.4638]], device='cuda:0'), torch.tensor([[ 0.1233, -0.2388,  0.4609, -0.0942, -0.2369,  0.4638, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1233, -0.2388,  0.4609, -0.0942,
         -0.2369,  0.4638]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0942, -0.2369,  0.4638,  0.0146, -0.2378,  0.4624,  0.1233, -0.2388,
          0.4609, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0667,
         -0.1197,  0.4295,  0.1108, -0.2026,  0.4132,  0.2883, -0.2854,  0.4000]],
       device='cuda:0')],

       [torch.tensor([[ 0.0847, -0.2196,  0.5528]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.2616, -0.2894,  0.4000, -0.0641, -0.1248,  0.4278, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1221, -0.2483,  0.4665, -0.0948,
         -0.2417,  0.4625]], device='cuda:0'), torch.tensor([[ 0.1221, -0.2483,  0.4665, -0.0948, -0.2417,  0.4625, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1221, -0.2483,  0.4665, -0.0948,
         -0.2417,  0.4625]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0948, -0.2417,  0.4625,  0.0137, -0.2450,  0.4645,  0.1221, -0.2483,
          0.4665, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0641,
         -0.1248,  0.4278,  0.0988, -0.2071,  0.4127,  0.2616, -0.2894,  0.4000]],
       device='cuda:0')],

       [torch.tensor([[ 0.0817, -0.2247,  0.5541]], device='cuda:0'), torch.tensor([[0, 0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[0, 0, 0]], device='cuda:0'), torch.tensor([[ 0.3003, -0.3047,  0.4000, -0.0644, -0.1274,  0.4289, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1151, -0.2528,  0.4666, -0.0983,
         -0.2477,  0.4622]], device='cuda:0'), torch.tensor([[ 0.1151, -0.2528,  0.4666, -0.0983, -0.2477,  0.4622, -0.0719, -0.0756,
          0.4358,  0.1365, -0.0416,  0.4244,  0.1151, -0.2528,  0.4666, -0.0983,
         -0.2477,  0.4622]], device='cuda:0'), torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
       device='cuda:0'), torch.tensor([[-0.0983, -0.2477,  0.4622,  0.0084, -0.2502,  0.4644,  0.1151, -0.2528,
          0.4666, -0.0719, -0.0756,  0.4358,  0.1365, -0.0416,  0.4244, -0.0644,
         -0.1274,  0.4289,  0.1180, -0.2161,  0.4105,  0.3003, -0.3047,  0.4000]],
       device='cuda:0')]
        ]