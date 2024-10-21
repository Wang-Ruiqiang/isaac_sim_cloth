from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.kit import SimulationApp

# 初始化仿真应用
simulation_app = SimulationApp({"headless": False})

# 指定URDF文件路径和加载到的prim路径
urdf_path = "/home/ruiqiang/workspaces/isaac_ws/isaac_sim_cloth/OmniIsaacGymEnvs/omniisaacgymenvs/tasks/cloth_manipulation/urdf/denso_robot.urdf"
prim_path = "/World/denso_robot"

# 加载URDF
add_reference_to_stage(usd_path=urdf_path, prim_path=prim_path)

# 运行仿真
simulation_app.update()
simulation_app.render()