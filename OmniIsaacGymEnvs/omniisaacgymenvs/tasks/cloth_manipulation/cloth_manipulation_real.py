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
import time

import hydra
import omegaconf
import cv2
import os
import torch
import numpy as np
from gym import spaces

from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from omniisaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray



class ClothManipulationReal():
    def __init__(self, name, config, env, offset=None) -> None:
        self._task_cfg = config.get("task", dict())
        self.device = 'cuda:0'
        self._get_task_yaml_params()

        self.cleanup()

        self.frame_list = []
        self.frame_list2 = []
        self.counter = 0
        self.video_count = 0

        self.y_displacements = []
        self.z_displacements = []

        self.is_first_run = True

        self.keypoint_param_init()
        self.ros_param_init(name)
        
        self.step_count = 0

    
    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self._task_cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        ppo_path = 'train/FrankaClothManipulationPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

        self.num_observations = self._task_cfg["env"]["numObservations"]
        self.num_actions = self._task_cfg["env"]["numActions"]
        self.num_envs = self._task_cfg["env"]["numEnvs"]
        self.clip_actions = self._task_cfg["env"].get("clipActions", 1)
        

        self.observation_space = spaces.Box(
            np.ones(self.num_observations, dtype=np.float32) * -np.Inf,  # 下限为 -inf
            np.ones(self.num_observations, dtype=np.float32) * np.Inf,   # 上限为 inf
            dtype=np.float32
        )

        self.action_space = spaces.Box(
                np.ones(self.num_actions, dtype=np.float32) * -1.0,
                np.ones(self.num_actions, dtype=np.float32) * 1.0
            )

    
    def ros_param_init(self, name):
        rclpy.init(args=None)
        self.node = rclpy.create_node(name)

        self.keypoint_pose = torch.zeros(1, 24, device=self.device)

        self.is_first_call_keypoint_callback = True

        # ROS 2 订阅者，用于接收关键点数据
        self.keypoint_subscriber = self.node.create_subscription(
            Float32MultiArray,
            'front_camera/keypoints',  # 订阅关键点话题
            self.front_camera_callback,  # 指定回调函数
            10
        )

    def keypoint_param_init(self):
        self.keypoint_offsets = None
        self.desired_initial_pose = torch.tensor([0.0887,  0.0788,  0.4049], device=self.device)

    def spin_ros(self):
        rclpy.spin_once(self.node, timeout_sec=0.1)  # 每次只 spin 一次，这样可以在任务中间运行 ROS
        time.sleep(0.5)


    def destroy(self):
        self.node.destroy_node()
        rclpy.shutdown()

        
    def pre_physics_step(self, actions) -> None:
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        self.actions = actions.clone().to(self.device) 

        # self._apply_actions_as_ctrl_targets(
        #     actions=self.actions,
        #     ctrl_target_gripper_dof_pos=self.asset_info_franka_table.franka_gripper_width_min,   #初始状态夹爪位置
        #     do_scale=True
        # )


    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        self.spin_ros()

        self.refresh_env_tensors()
        self.get_observations()
        self.calculate_metrics()


        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    
    def refresh_env_tensors(self):
        """Refresh tensors."""

        self.desired_goal = torch.cat((self.keypoint_pose[1], self.keypoint_pose[3], 
                                       self.keypoint_pose[5], self.keypoint_pose[4], 
                                       self.keypoint_pose[1], self.keypoint_pose[3]), dim = -1)
        self.desired_goal = self.desired_goal.unsqueeze(dim=0)


        self.achieved_goal = torch.cat((self.keypoint_pose[0], self.keypoint_pose[2], self.keypoint_pose[5],
                            self.keypoint_pose[4], self.keypoint_pose[1], self.keypoint_pose[3]), 0)
        self.achieved_goal = self.achieved_goal.unsqueeze(dim=0)


        self.keypoint_pos = torch.cat((self.keypoint_pose[3], self.keypoint_pose[7], 
                                    self.keypoint_pose[1], self.keypoint_pose[5],
                                    self.keypoint_pose[4], self.keypoint_pose[2], 
                                    self.keypoint_pose[6], self.keypoint_pose[0]), dim = -1)
        self.keypoint_pos = self.keypoint_pos.unsqueeze(dim=0)
        
    

    def get_observations(self):
        """Compute observations."""
        
        # print("--------------------------------------------------------------------")
        # print("self.fingertip_midpoint_pos = ", self.fingertip_midpoint_pos)
        # print("self.fingertip_midpoint_quat = ", self.fingertip_midpoint_quat)
        # print("self.fingertip_midpoint_linvel = ", self.fingertip_midpoint_linvel)
        # print("self.fingertip_midpoint_angvel = ", self.fingertip_midpoint_angvel)
        # print("self.achieved_goal = ",self.achieved_goal)
        # print("self.desired_goal = ", self.desired_goal)
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

        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets
    

    def _get_keypoint_dist(self):
        """Get keypoint distance."""
        achieved_oks = torch.zeros((1, 6),
                                   dtype=torch.float32,
                                   device=self.device)
        achieved_distances = torch.zeros((1, 6),
                                   dtype=torch.float32,
                                   device=self.device)
        success_reward = 0
        fail_reward = -1
        action_penalty = 0

        constraint_distances = torch.tensor([0.04, 0.02, 0.02, 0.02, 0.02, 0.02], device=self.device)
        # constraint_distances = torch.tensor([0.015, 0.01, 0.01, 0.01, 0.01, 0.01], device=self.device)

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

        point_one_dis = self.goal_distance(self.keypoint_pose[0], self.keypoint_pose[1])
        point_two_dis = self.goal_distance(self.keypoint_pose[2], self.keypoint_pose[3])

        if self.particle_cloth_positon[0, 8][1] - self.particle_cloth_positon[0, 80][1] > 0.03 :
            action_penalty +=  point_one_dis

        # print("self.particle_cloth_positon[0, 80] = ", self.particle_cloth_positon[0, 80])

        # if self.particle_cloth_positon[0, 0][1] - self.particle_cloth_positon[0, 72][1] > 0.03 :
        #     action_penalty += point_two_dis

        if self.is_first_run:
            self.left_top_point = self.keypoint_pose[3]
            self.left_buttom_point = self.keypoint_pose[1]
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
    

    def cleanup(self) -> None:
        """Prepares torch buffers for RL data collection."""

        # prepare tensors
        self.obs_buf = torch.zeros((self.num_envs, self.num_observations), device=self.device, dtype=torch.float)
        # self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}


    def reset(self):
        """Flags all environments for reset."""
        self.reset_buf = torch.ones_like(self.reset_buf)


    def front_camera_callback(self, msg):
        # 将接收到的关键点数据更新到 keypoint_pose
        data = msg.data
        if len(data) == 12:  # 确认数据长度是12，表示4个关键点的x、y、z坐标
        # 将数据转换为形状为 (4, 3) 的 PyTorch 张量
            keypoint_array = torch.tensor(data, device=self.device).view(4, 3)
            keypoint_array[:, 0] = -keypoint_array[:, 0]  # x坐标取反
            keypoint_array[:, 1] = -keypoint_array[:, 1]  # y坐标取反

        # 如果offset还未计算，计算offset并存储
            if self.keypoint_offsets is None:
                # 计算keypoint_pose[0]与期望初始坐标之间的偏移量
                self.keypoint_offsets = self.desired_initial_pose - keypoint_array[0]
                print("Offset calculated: ", self.keypoint_offsets)

            keypoint_array[0] += self.keypoint_offsets
            keypoint_array[1] += self.keypoint_offsets
            keypoint_array[2] += self.keypoint_offsets
            keypoint_array[3] += self.keypoint_offsets

        self.keypoint_pose = keypoint_array


        if self.is_first_call_keypoint_callback:
            self.middle_point_of_zero_one = (self.keypoint_pose[0] + self.keypoint_pose[1]) / 2
            self.middle_point_of_two_three = (self.keypoint_pose[2] + self.keypoint_pose[3]) / 2
            self.is_first_call_keypoint_callback = False
            
        
        self.middle_point_of_zero_two= (self.keypoint_pose[0] + self.keypoint_pose[2]) / 2
        self.middle_point_of_one_three = (self.keypoint_pose[1] + self.keypoint_pose[3]) / 2


        self.keypoint_pose = torch.cat(
            (self.keypoint_pose, self.middle_point_of_zero_one.unsqueeze(0), self.middle_point_of_two_three.unsqueeze(0),
             self.middle_point_of_zero_two.unsqueeze(0), self.middle_point_of_one_three.unsqueeze(0)),
            dim=0
        )

        print("self.keypoint_pose = ", self.keypoint_pose)