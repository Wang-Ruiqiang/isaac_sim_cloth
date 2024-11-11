# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
from typing import Optional

import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.usd import get_context
from pxr import Usd
from pxr import PhysxSchema


class Denso(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "denso_robot",
        translation: Optional[torch.tensor] = None,
        orientation: Optional[torch.tensor] = None,
    ) -> None:
        """[summary]"""

        self._name = name

        print("translation = ", translation)

        self._position = torch.tensor([1.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([0.0, 0.0, 0.0, 1.0]) if orientation is None else orientation

        self._usd_path = "/home/ruiqiang/workspaces/isaac_ws/isaac_sim_cloth/OmniIsaacGymEnvs/omniisaacgymenvs/tasks/cloth_manipulation/urdf/denso_robot/denso_robot_with_hand3.usda"

        add_reference_to_stage(self._usd_path, prim_path)


        # 获取当前的 stage
        stage = get_context().get_stage()

        # 获取所有 prim
        all_prims = stage.Traverse()

        # 输出所有 prim 的路径
        for prim in all_prims:
            print(prim.GetPath())


        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

        dof_paths = [
            "denso_robot/base_link/joint1",
            "denso_robot/link1/joint2",
            "denso_robot/link2/joint3",
            "denso_robot/link3/joint4",
            "denso_robot/link4/joint5",
            "denso_robot/link5/joint6",
            "denso_robot/palm_lower/mcp_joint_1",
            "denso_robot/mcp_link/pip_joint",
            "denso_robot/pip_link/dip_joint",
            "denso_robot/dip_link/fingertip_joint",
            "denso_robot/palm_lower/mcp_joint_2",
            "denso_robot/mcp_link_2/pip_joint_2",
            "denso_robot/pip_link_2/dip_joint_2",
            "denso_robot/dip_link_2/fingertip_joint_2",
            "denso_robot/palm_lower/mcp_joint_3",
            "denso_robot/mcp_link_3/pip_joint_3",
            "denso_robot/pip_link_3/dip_joint_3",
            "denso_robot/dip_link_3/fingertip_joint_3",
            "denso_robot/palm_lower/mcp_joint_4",
            "denso_robot/mcp_link_4/thumb_pip_joint",
            "denso_robot/thumb_pip_link/thumb_dip_joint",
            "denso_robot/thumb_dip_link/thumb_fingertip_joint",
        ]

        drive_type = ["angular"] * 6 + ["linear"] * 16
        default_dof_pos = [
            0.0,    # joint1
            0.0,    # joint2
            1.57,   # joint3 (在URDF文件中的初始值)
            0.0,    # joint4
            0.0,    # joint5
            0.0,    # joint6
        ] + [0] * 16
        stiffness = [70] * 6 + [8] * 16
        damping = [10] * 6 + [8] * 16
        max_force = [87, 87, 87, 87, 87, 50] + [0.95] * 16# 你可以根据 URDF 文件中的 limit.effort 设置
        max_velocity = [124.618, 124.618, 149.541, 149.541, 149.541, 200] + [8.48] * 16



        for i, dof in enumerate(dof_paths):
            full_prim_path = f"{self.prim_path}/{dof}"
            prim = get_prim_at_path(full_prim_path)
            print(f"Prim at {full_prim_path}: {prim.IsValid()}")

            set_drive(
                prim_path=f"{self.prim_path}/{dof}",
                drive_type=drive_type[i],
                target_type="position",
                target_value=default_dof_pos[i],
                stiffness=stiffness[i],
                damping=damping[i],
                max_force=max_force[i],
            )

            PhysxSchema.PhysxJointAPI(get_prim_at_path(f"{self.prim_path}/{dof}")).CreateMaxJointVelocityAttr().Set(
                max_velocity[i]
            )
        
        