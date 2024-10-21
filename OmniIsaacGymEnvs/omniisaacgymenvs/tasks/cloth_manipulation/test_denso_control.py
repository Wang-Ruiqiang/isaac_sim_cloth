#!/usr/bin/env python3

import os
import torch
import numpy as np
from omni.isaac.kit import SimulationApp
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.urdf import _urdf
import torch
import numpy as np

# 1. 初始化 Isaac Sim 应用
from omni.isaac.lab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser(description="This script demonstrates different dexterous hands.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 定义 Denso 机械臂类，基于URDF文件
class DensoRobot(Robot):
    def __init__(self, prim_path, urdf_path, translation=None, orientation=None):
        self._urdf_path = urdf_path
        self._translation = translation if translation is not None else [0, 0, 0]
        self._orientation = orientation if orientation is not None else [0, 0, 0, 1]

        # 加载 URDF 文件
        urdf_loader = _urdf.Urdf()
        urdf_loader.parse(self._urdf_path)
        
        # 初始化机械臂
        super().__init__(prim_path=prim_path, name="denso_robot",
                         translation=self._translation, orientation=self._orientation)
    
    def get_end_effector_position(self):
        """获取末端执行器的位置（示例：假设末端执行器在 link6 位置）"""
        return self.get_joint_positions()[-1]
    
    def get_jacobian(self):
        """获取机械臂的雅可比矩阵"""
        # 这里只是示例，实际上 Isaac Sim 中的 API 可能有所不同
        return self.get_jacobians()

# 3. 逆运动学控制逻辑
def ik_control(robot, target_position):
    # 获取当前关节位置和雅可比矩阵
    joint_positions = robot.get_joint_positions()
    jacobian = robot.get_jacobian()

    # 设置目标末端位置
    target_pos_tensor = torch.tensor(target_position, dtype=torch.float32)
    current_pos = torch.tensor(robot.get_end_effector_position(), dtype=torch.float32)
    
    # 计算位置误差
    error_pos = target_pos_tensor - current_pos

    # 使用伪逆雅可比矩阵计算关节角度变化
    joint_velocity = torch.matmul(torch.pinverse(jacobian), error_pos)

    # 更新关节位置
    new_joint_positions = joint_positions + joint_velocity
    robot.set_joint_positions(new_joint_positions)

# 4. 启动仿真环境并加载 Denso 机械臂
def load_denso_robot():
    # 加载场景中的 URDF 文件
    stage = get_current_stage()
    urdf_path = "/home/ruiqiang/workspaces/isaac_ws/isaac_sim_cloth/OmniIsaacGymEnvs/omniisaacgymenvs/tasks/cloth_manipulation/urdf/denso_robot.urdf"  # 将路径替换为你的URDF文件路径
    denso_robot = DensoRobot(prim_path="/World/denso_robot", urdf_path=urdf_path)
    return denso_robot

# 5. 运行仿真，并通过逆运动学控制机械臂
def run_simulation():
    # 加载 Denso 机械臂
    robot = load_denso_robot()

    # 仿真循环，设定目标位置并通过 IK 控制机械臂
    target_position = [0.5, 0.2, 0.3]  # 定义目标末端执行器的位置

    while simulation_app.is_running():
        ik_control(robot, target_position)  # 控制机械臂到目标位置
        simulation_app.update()  # 更新仿真
        simulation_app.render()  # 渲染场景

# 启动仿真
if __name__ == "__main__":
    run_simulation()
    simulation_app.close()
