import PyKDL
import torch
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
import math
import numpy as np

def create_panda_chain():
    # # DH参数
    # a = [0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088]
    # alpha = [0.0, -math.pi / 2, math.pi / 2, math.pi / 2, -math.pi / 2, math.pi / 2, math.pi / 2]
    # d = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0]
    # theta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # chain = PyKDL.Chain()
    # for i in range(7):
    #     chain.addSegment(PyKDL.Segment(PyKDL.Joint(PyKDL.Joint.RotZ), PyKDL.Frame(PyKDL.Rotation.RotZ(theta[i]) * PyKDL.Rotation.RotX(alpha[i]), PyKDL.Vector(a[i], -d[i] * math.sin(alpha[i]), d[i] * math.cos(alpha[i])) )))
    # return chain
    robot = URDF.from_xml_file("/home/ruiqiang/workspace/OmniIsaacGymEnvs/omniisaacgymenvs/tasks/cloth_manipulation/urdf/panda.urdf")
    tree = kdl_tree_from_urdf_model(robot)
    chain = tree.getChain("panda_link0", "panda_link8")
    return chain

def quat_to_angle(quat):
  rot = PyKDL.Rotation.Quaternion(quat[3], quat[4], quat[5], quat[6])
  print("rot.GetRPY()[0] = ", rot.GetRPY()[0])
  print("rot.GetRPY()[1] = ", rot.GetRPY()[1])
  print("rot.GetRPY()[2] = ", rot.GetRPY()[2])
  return rot

def compute_inverse_kinematics(chain, target_pose):
    '''
    正运动学
    '''
 
    fk = PyKDL.ChainFkSolverPos_recursive(chain)
    pos = PyKDL.Frame()
    q = PyKDL.JntArray(7)
    qq = [-0.917812, -0.917812, 43.2983, 21.2432, 16.8387, -27.4167, 19.5677]
 
    for i in range(7):
        q[i] = qq[i]
    fk_flag = fk.JntToCart(q, pos)
    print("fk_flag", fk_flag)
    print("pos", pos)
 
    '''
    逆运动学
    '''
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
    result1 = []
    for i in range(10):
        # 创建目标位姿
        target_frame = PyKDL.Frame(PyKDL.Rotation.RPY(3.1415926, 0, 0), PyKDL.Vector(0.51, -0.1460, 0.5))
        print ("target_frame = ", target_frame)
        # 创建起始关节角度
        # initial_joint_angles = PyKDL.JntArray(chain.getNrOfJoints())
        initial_joint_angles = PyKDL.JntArray(7)
        initial_joint_angles_array = [0.012, -0.5697, 0, -2.8105, 0, 3.0312, 0.7853]
        for i in range(7):
            initial_joint_angles[i] = initial_joint_angles_array[i]
        print("initial_joint_angles = ", initial_joint_angles)
        # result = PyKDL.JntArray(chain.getNrOfJoints())
        result = PyKDL.JntArray(chain.getNrOfJoints())
        #print(target_frame)
        # 调用逆运动学求解器
        retval = ik.CartToJnt(initial_joint_angles, target_frame,result)
        if (retval >= 0):
            print('result: ',result)
            result1.append(result)
        else :
            print("Error: could not calculate ik kinematics :(")
    return result1
 
if __name__=="__main__":
    # test = torch.randn(3, 3 , 3)
    # test1 = test[0,0,0].item() / 10
    # test2 = test[1,1,0] / 10
    # test3 = test[2,0,0] / 10
    # print("torch = ", test)
    # print("test1 = ", test1)
    # print("test2 = ", test2)
    # print("test3 = ", test3)
    # vec = PyKDL.Vector(test1, test2, test3)
    # print("vec = ", vec)
    # 创建机器人链
    chain = create_panda_chain()
    # 设置目标位姿
    target_pose = [-0.0102, -0.1460, 0.5, 0, 1, 0, 0]
    euler_angle = quat_to_angle(target_pose)
    target_pose_euler = [target_pose[0], target_pose[1], target_pose[2], euler_angle.GetRPY()[0], euler_angle.GetRPY()[1], euler_angle.GetRPY()[2]]
    # 调用逆运动学求解函数
    joint_angles = compute_inverse_kinematics(chain, target_pose_euler)
    # print("关节角度: ", joint_angles)
    joint_angles = np.array(joint_angles)
    print("关节角度: ", joint_angles)
    print("joint_angles[0][0] = ", joint_angles[0][0])
    print("len(joint_angles) = ", len(joint_angles))
    print("len(joint_angles[0]) = ", joint_angles[0].rows())
    joint_goal = torch.rand(10, 7)
    for i in range(len(joint_angles)) :
        for j in range(7):
            joint_goal[i, j] = joint_angles[i][j]
    
    print("joint_goal = ", joint_goal)
    