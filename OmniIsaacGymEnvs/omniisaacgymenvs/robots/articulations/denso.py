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

        self._usd_path = "/home/ruiqiang/workspaces/isaac_ws/isaac_sim_cloth/OmniIsaacGymEnvs/omniisaacgymenvs/tasks/cloth_manipulation/urdf/denso_robot_3.usda"

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
        ]

        drive_type = ["angular"] * 6
        default_dof_pos = [
            0.0,    # joint1
            0.0,    # joint2
            1.57,   # joint3 (在URDF文件中的初始值)
            0.0,    # joint4
            0.0,    # joint5
            0.0,    # joint6
        ]
        stiffness = [100] * 6
        damping = [1.4] * 6
        max_force = [87, 87, 87, 87, 87, 50]  # 你可以根据 URDF 文件中的 limit.effort 设置
        max_velocity = [124.618, 124.618, 149.541, 149.541, 149.541, 200]

        print("stiffness = ", stiffness)
        print("damping = ", damping)
        print("max_force = ", max_force)
        print("max_velocity = ", max_velocity)



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
        
        