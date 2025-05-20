import math
import numpy as np
import mujoco, mujoco.viewer
import mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch

max_action_value: 100.0
action_scale = 0.25
num_actions = 18
policy_path_ = "mani-centric-wbc/resources/robots/aliengo_description/policy/model_17500.pt"
xml_path_ = "/home/george/code/umi-on-legs/mani-centric-wbc/resources/robots/aliengo_description/urdf/aliengoPiper.xml"

kps = [50,50,50,50,50,50,50,50,50,50,50,50,150,150,100,50,50,20]
kds = [1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,3.0,3.0,3.0,2.0,1.0,0.5]

tau_limit = [35.278,35.278,44.400,35.278,35.278,44.400,35.278,35.278,44.400,35.278,35.278,44.400,20.0,20.0,15.0,7.0,5.0,5.0]

class EnvSetup:
    kp: torch.Tensor
    kd: torch.Tensor

    rigidbody_mass: torch.Tensor
    rigidbody_com_offset: torch.Tensor
    rigidbody_restitution_coef: torch.Tensor
    rigid_shape_friction: torch.Tensor

    dof_friction: torch.Tensor
    dof_damping: torch.Tensor
    dof_velocity: torch.Tensor

def local_root_gravity(data) -> torch.Tensor:
    norm_gravity = torch.Tensor([[0,0,-1]])
    xyzw_quat = data.sensor('orientation')
    return quat_rotate_inverse(
        xyzw_quat,
        norm_gravity,
    )

def quat_rotate_inverse(q_, v):
    q = torch.Tensor(q_.data)
    print(q)
    q = q.reshape(1,4)
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    print(q_vec.shape)
    print(v.shape)
    print("*****")
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, xml_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((num_actions), dtype=np.double)
    action = np.zeros((num_actions), dtype=np.double)

    while True:
        mujoco.mj_step(model, data)
        obs = []
        action_history = torch.zeros(1,18)
        # data.sensordata
        # root_ang_vel = data.sensor('angular-velocity').data
        # print("root_ang_vel",root_ang_vel)
        # qri = local_root_gravity(data)
        # print("dof_pos",data.qpos[7:])
        # print("dof_vel",data.qvel[6:])

        # state_obs
        obs[0:18] = data.qpos[7:]
        obs[18:36] = data.qvel[6:]
        obs[36:39] = local_root_gravity(data)
        obs[39:42] = data.sensor('angular-velocity').data
        # task_obs
        obs[42:78] = [-0.0511, -0.0401,  0.0033, -0.0393, -0.0430,  0.0062, -0.0314, -0.0443,
          0.0069,  0.9702, -0.3058,  0.3965,  1.4994, -0.0395,  0.0128,  0.0400,
          1.4977, -0.0724,  1.4991, -0.0459,  0.0233,  0.0471,  1.4973, -0.0767,
          1.4988, -0.0520,  0.0310,  0.0535,  1.4970, -0.0778,  1.3867,  0.2040,
         -0.5342, -0.2987,  1.4533, -0.2205]
        # control
        obs[78:97] = action_history[1,:]

        obs_ = torch.Tensor(obs)
        action[:] = policy(obs_).detach().numpy()
        action_history = action
        action = np.clip(action, -max_action_value, max_action_value)
        target_q = action * action_scale
        target_dq = np.zeros((num_actions), dtype=np.double)

        tau = pd_control(target_q, data.qpos[7:], kps,
                        target_dq, data.qvel[6:], kds)  # Calc torques
        tau = np.clip(tau, -tau_limit, tau_limit)
        data.ctrl = tau

        mujoco.mj_step(model,data)
        viewer.render()

    # obs {state_obs,setup_obs,tasks,action}
    # state_obs: 
    #       root_ang_vel: (data.sensor('angular-velocity').data), 
    #       local_root_gravity(finished), 
    #       dof_pos:(data.qpos[7:])， 
    #       dof_vel:(data.qvel[6:])
    # setup_obs: EnvSetup(finished)
    # task: reachingtask
    # action: last_action




if __name__ == '__main__':
    
    policy = torch.jit.load(policy_path_)
    run_mujoco(policy, xml_path_)