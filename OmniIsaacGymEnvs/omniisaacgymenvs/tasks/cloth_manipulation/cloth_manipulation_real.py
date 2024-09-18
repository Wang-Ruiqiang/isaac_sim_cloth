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
from geometry_msgs.msg import PoseStamped



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
        self.robot_param_init()
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
        self.clip_obs = self._task_cfg["env"].get("clipObservations", np.Inf)
        

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

        self.is_first_call_keypoint_callback = True

        # ROS 2 订阅者，用于接收关键点数据
        self.front_camera_subscriber = self.node.create_subscription(
            Float32MultiArray,
            'front_camera/keypoints',  # 订阅关键点话题
            self.front_camera_callback,  # 指定回调函数
            10
        )


        self.side_camera_subscriber = self.node.create_subscription(
            Float32MultiArray,
            'side_camera/keypoints',  # 订阅关键点话题
            self.side_camera_callback,  # 指定回调函数
            10
        )

        self.robot_subscription = self.node.create_subscription(
            PoseStamped,
            '/cartesian_compliance_controller/current_pose',
            self.pose_callback,
            10)
        
        
        self.robot_control_publisher = self.node.create_publisher(Float32MultiArray, '/cloth_folding/robot_control', 10)



    def keypoint_param_init(self):
        self.keypoint_offsets = None
        self.keypoint_pose = torch.zeros(8, 3, device=self.device)
        self.desired_initial_pose = torch.tensor([0.0908,  0.0993,  0.4049], device=self.device)

        self.previous_keypoint_pose = torch.zeros(8, 3, device=self.device)  # 上次使用的关键点坐标
        self.front_keypoint_pose = torch.zeros(4, 3, device=self.device)     # 存储前置相机的关键点数据
        self.side_keypoint_pose = torch.zeros(4, 3, device=self.device)      # 存储侧面相机的关键点数据

    
    def robot_param_init(self):
        self.current_pose = torch.zeros(1, 3, device=self.device)
        self.robot_end_offsets = None
        self.desired_initial_pose_robot = torch.tensor([0.1205, 0.1022, 0.4007], device=self.device)
        self.grip_offset = self.desired_initial_pose - self.desired_initial_pose_robot


    def spin_ros(self):
        rclpy.spin_once(self.node, timeout_sec=0.1)  # 每次只 spin 一次，这样可以在任务中间运行 ROS
        time.sleep(1)


    def destroy(self):
        self.node.destroy_node()
        rclpy.shutdown()

        
    def pre_physics_step(self, actions) -> None:
        self.actions = actions.clone().to(self.device) 
        if torch.all(torch.eq(self.current_pose, 0)) or torch.all(torch.eq(self.keypoint_pose, 0)):
            print("Current pose or keypoint pose is all zeros, skipping step.")
            return

        pos_actions = actions[:, 0:3]   #增量
        pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        print("pos_actions = ", pos_actions)

        rot_actions = actions[:, 3:6]
        rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch.zeros(1, 4, device=self.device)

        rot_actions_quat[0, 0] = torch.cos(angle / 2.0)                      # 四元数的 w 分量
        rot_actions_quat[0, 1] = axis[0, 0] * torch.sin(angle / 2.0)         # 四元数的 x 分量
        rot_actions_quat[0, 2] = axis[0, 1] * torch.sin(angle / 2.0)         # 四元数的 y 分量
        rot_actions_quat[0, 3] = axis[0, 2] * torch.sin(angle / 2.0)         # 四元数的 z 分量


        control_msg = Float32MultiArray()

        control_data = torch.cat((pos_actions.flatten(), rot_actions_quat.flatten()))

        control_msg.data = control_data.cpu().numpy().tolist()

        self.robot_control_publisher.publish(control_msg)
        print("publish success")


    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        self.spin_ros()
        # 阻塞检查，直到 current_pose 和 keypoint_pose 被赋值（不全为 0）
        while torch.all(torch.eq(self.current_pose, 0)) or torch.all(torch.eq(self.keypoint_pose, 0)):
            # print("Waiting for valid current_pose and keypoint_pose data...")
            self.spin_ros()

        self.refresh_env_tensors()
        self.get_observations()
        self.calculate_metrics()


        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    
    def refresh_env_tensors(self):
        """Refresh tensors."""
        grip_point = self.current_pose + self.grip_offset

        # self.desired_goal = torch.cat((self.keypoint_pose[1], self.keypoint_pose[3], 
        #                                self.keypoint_pose[5], self.keypoint_pose[4], 
        #                                self.keypoint_pose[1], self.keypoint_pose[3]), dim = -1)



        self.desired_goal = torch.cat((self.key_point_one, self.key_point_three, 
                                       self.keypoint_pose[5], self.keypoint_pose[4], 
                                       self.key_point_one, self.key_point_three), dim = -1)
        self.desired_goal = self.desired_goal.unsqueeze(dim=0)


        # self.achieved_goal = torch.cat((self.keypoint_pose[0], self.keypoint_pose[2], self.keypoint_pose[5],
        #                     self.keypoint_pose[4], self.keypoint_pose[1], self.keypoint_pose[3]), 0)
        self.achieved_goal = torch.cat((grip_point.squeeze(0), self.keypoint_pose[2], self.keypoint_pose[5],
                            self.keypoint_pose[4], self.key_point_one, self.key_point_three), 0)
        self.achieved_goal = self.achieved_goal.unsqueeze(dim=0)


        # self.keypoint_pos = torch.cat((self.keypoint_pose[3], self.keypoint_pose[7], 
        #                             self.keypoint_pose[1], self.keypoint_pose[5],
        #                             self.keypoint_pose[4], self.keypoint_pose[2], 
        #                             self.keypoint_pose[6], self.keypoint_pose[0]), dim = -1)

        self.keypoint_pos = torch.cat((self.key_point_three, self.keypoint_pose[7], 
                                    self.key_point_one, self.keypoint_pose[5],
                                    self.keypoint_pose[4], self.keypoint_pose[2], 
                                    self.keypoint_pose[6], grip_point.squeeze(0)), dim = -1)
        self.keypoint_pos = self.keypoint_pos.unsqueeze(dim=0)
        
    

    def get_observations(self):
        """Compute observations."""
        
        # print("--------------------------------------------------------------------")
        print("self.current_pose = ", self.current_pose)
        # print("self.fingertip_midpoint_quat = ", self.fingertip_midpoint_quat)
        # print("self.fingertip_midpoint_linvel = ", self.fingertip_midpoint_linvel)
        # print("self.fingertip_midpoint_angvel = ", self.fingertip_midpoint_angvel)
        print("self.achieved_goal = ",self.achieved_goal)
        print("self.desired_goal = ", self.desired_goal)
        # print("self.keypoint_vel = ", self.keypoint_vel)
        # print("self.keypoint_pos = ", self.keypoint_pos)
        # print("--------------------------------------------------------------------")
        obs_tensors = [self.current_pose,
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
            'denso': {
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
        # self.reset_buf[:] = torch.where(
        #     self.progress_buf[:] >= self.max_episode_length - 1,
        #     torch.ones_like(self.reset_buf),
        #     self.reset_buf
        # )

        # if self.progress_buf[:] >= self.max_episode_length - 1:
        #     self.plot_displacements()
        #     self.y_displacements.clear()
        #     self.z_displacements.clear()

        
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

        if self.keypoint_pose[1][1] - self.keypoint_pose[0][1] > 0.03 :
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
    

    def goal_distance(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return torch.norm(goal_a - goal_b, p=2, dim=-1)
    

    def cleanup(self) -> None:
        """Prepares torch buffers for RL data collection."""

        # prepare tensors
        self.obs_buf = torch.zeros((self.num_envs, self.num_observations), device=self.device, dtype=torch.float)
        # self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}


    def reset(self):
        """Flags all environments for reset."""
        # self.reset_buf = torch.ones_like(self.reset_buf)


    def pose_callback(self, msg):
        position_data = msg.pose.position
        position_array = torch.tensor([-position_data.x, -position_data.y, position_data.z], device=self.device).view(1, 3)

        if self.robot_end_offsets is None:
            self.robot_end_offsets = self.desired_initial_pose_robot - position_array[0]
        
        position_array[0] += self.robot_end_offsets
        self.current_pose = position_array
        # print("self.current_pose = ", self.current_pose)
        


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
                if not torch.all(keypoint_array[0] == 0):
                    # 计算keypoint_pose[0]与期望初始坐标之间的偏移量
                    self.keypoint_offsets = self.desired_initial_pose - keypoint_array[0]
                    print("Offset calculated: ", self.keypoint_offsets)
                else:
                    # 如果第一个点为0，不计算 offset
                    print("First keypoint is 0, unable to calculate offset.")
                    return


            for i in range(4):
                if not torch.all(keypoint_array[i] == 0):  # 只有非 0 点才应用偏移量
                    keypoint_array[i] += self.keypoint_offsets

        self.front_keypoint_pose[:4] = keypoint_array


        # 检查是否有任何坐标为 0, self.keypoint_pose[0] is the grasp point
        for i in range(4):
            if torch.all(self.front_keypoint_pose[i] == 0):
                if torch.all(self.side_keypoint_pose[i] == 0):  # 侧面相机也没有数据
                    self.front_keypoint_pose[i] = self.previous_keypoint_pose[i]  # 使用上一次的坐标
                elif not torch.all(torch.eq(self.previous_keypoint_pose, 0)):
                    delta_side_x = torch.abs(self.side_keypoint_pose[i][0] - self.previous_keypoint_pose[i][0])  # 侧面相机 x 方向增量
                    delta_side_y = torch.abs(self.side_keypoint_pose[i][1] - self.previous_keypoint_pose[i][1])  # 侧面相机 y 方向增量

                    if delta_side_x > 0.04 and delta_side_y > 0.04:
                        print(f"Side camera keypoint {i} invalid, using previous data.")
                        self.front_keypoint_pose[i] = self.previous_keypoint_pose[i]
                    else:
                        print(f"Using side camera keypoint {i} data.")
                        self.front_keypoint_pose[i] = self.side_keypoint_pose[i] + self.keypoint_offsets 

            elif not torch.all(torch.eq(self.previous_keypoint_pose, 0)): 
                # delta = torch.abs(self.front_keypoint_pose[i] - self.previous_keypoint_pose[i])
                delta_x = torch.abs(self.front_keypoint_pose[i][0] - self.previous_keypoint_pose[i][0])  # x 方向增量
                delta_y = torch.abs(self.front_keypoint_pose[i][1] - self.previous_keypoint_pose[i][1])  # y 方向增量


                if delta_x > 0.04 and delta_y > 0.04:
                    print(f"Warning: Keypoint {i} has moved too much in both x and y, delta_x: {delta_x}, delta_y: {delta_y}")

                    if not torch.all(self.side_keypoint_pose[i] == 0):
                        delta_side_x = torch.abs(self.side_keypoint_pose[i][0] - self.previous_keypoint_pose[i][0])
                        delta_side_y = torch.abs(self.side_keypoint_pose[i][1] - self.previous_keypoint_pose[i][1])

                        if delta_side_x > 0.04 and delta_side_y > 0.04:
                            print(f"Side camera keypoint {i} also invalid, using previous data.")
                            self.front_keypoint_pose[i] = self.previous_keypoint_pose[i]

                        else:
                            print(f"Using side camera keypoint {i} data.")
                            self.front_keypoint_pose[i] = self.side_keypoint_pose[i] + self.keypoint_offsets  # 使用侧面相机的数据

            
        self.keypoint_pose = self.front_keypoint_pose


        if self.is_first_call_keypoint_callback:
            self.key_point_one = self.keypoint_pose[1].clone()
            self.key_point_three = self.keypoint_pose[3].clone()
            self.middle_point_of_zero_one = (self.keypoint_pose[0] + self.keypoint_pose[1]) / 2
            self.middle_point_of_two_three = (self.keypoint_pose[2] + self.keypoint_pose[3]) / 2
            self.middle_point_of_one_three = self.key_point_one + self.key_point_three / 2
            self.is_first_call_keypoint_callback = False
            
        self.middle_point_of_zero_two= (self.keypoint_pose[0] + self.keypoint_pose[2]) / 2
        # self.middle_point_of_one_three = (self.keypoint_pose[1] + self.keypoint_pose[3]) / 2


        self.keypoint_pose = torch.cat(
            (self.keypoint_pose, self.middle_point_of_zero_one.unsqueeze(0), self.middle_point_of_two_three.unsqueeze(0),
             self.middle_point_of_zero_two.unsqueeze(0), self.middle_point_of_one_three.unsqueeze(0)),
            dim=0
        )
        self.previous_keypoint_pose = self.keypoint_pose.clone()

        # print("self.keypoint_pose = ", self.keypoint_pose)

    def side_camera_callback(self, msg):
        data = msg.data
        if len(data) == 12:  # 确认数据长度是12，表示4个关键点的x、y、z坐标
            keypoint_array = torch.tensor(data, device=self.device).view(4, 3)
            keypoint_array[:, 0] = -keypoint_array[:, 0]  # x坐标取反
            keypoint_array[:, 1] = -keypoint_array[:, 1]  # y坐标取反

            # 存储侧面相机的关键点数据，不应用 offset
            self.side_keypoint_pose[:4] = keypoint_array