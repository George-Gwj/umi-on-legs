# 备份fixedArm loco

import os
import pickle
import re
from isaacgym import gymapi, gymutil  # must be improved before torch
from argparse import ArgumentParser

import hydra
import imageio.v2 as imageio
import numpy as np
import zarr
import torch
from omegaconf import OmegaConf
from rich.progress import track
from transforms3d import affines, quaternions
from legged_gym.rsl_rl.runners.on_policy_runner import OnPolicyRunner

import wandb
from legged_gym.env.isaacgym.env import IsaacGymEnv
from train import setup

import math
import numpy as np
import mujoco, mujoco.viewer
import mujoco_viewer
from tqdm import tqdm
from collections import deque
from scipy.spatial.transform import Rotation as R
import torch
import pytorch3d.transforms as pt3d
from scipy.spatial.transform import Rotation as R

def recursively_replace_device(obj, device: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "device":
                obj[k] = device
            else:
                obj[k] = recursively_replace_device(v, device)
        return obj
    elif isinstance(obj, list):
        return [recursively_replace_device(v, device) for v in obj]
    else:
        return obj
    return obj


count = 0

def local_root_gravity(data) -> torch.Tensor:
    norm_gravity = torch.Tensor([[0,0,-1]])
    xyzw_quat = np.array([data.qpos[4],data.qpos[5],data.qpos[6],data.qpos[3]])
    return quat_rotate_inverse(
        xyzw_quat,
        norm_gravity,
    )

def quat_rotate_inverse(q_, v):
    q = torch.Tensor(q_.data)
    q = q.reshape(1,4)
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]

    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    torques = (target_q - q) * kp - dq * kd
    torques[0] = -torques[0]
    torques[3] = -torques[3]
    torques[6] = -torques[6]
    torques[9] = -torques[9]
    return torques

link_pose_history = np.eye(4)

def get_target_pose(env, times: torch.Tensor, sim_dt: float):
    # returns the current target pose in the local frame of the robot
    target_pose = (
        torch.eye(4, device='cuda:0').unsqueeze(0)
    )

    pos, rot_mat = get_targets_at_times(env, times=times, sim_dt=sim_dt)
    target_pose[..., :3, 3] = pos
    target_pose[..., :3, :3] = rot_mat
    return target_pose

def get_targets_at_times(
    tasks,
    times: torch.Tensor,
    sim_dt: float,
):
    episode_step = torch.clamp(
        # (torch.zeros_like(times) / sim_dt).long(),
        (times / sim_dt).long(),
        min=0,
        max=tasks.target_pos_seq.shape[1] - 1,
    )
    episode_step = torch.clamp(
        episode_step, min=0, max=tasks.target_pos_seq.shape[1] - 1
    ).to(tasks.storage_device)
    env_idx = torch.arange(0, tasks.num_envs)
    return (
        tasks.target_pos_seq[env_idx, episode_step].to(tasks.device),
        tasks.target_rot_mat_seq[env_idx, episode_step].to(tasks.device),
    )

def observe(env, MjData) -> torch.Tensor:
    target_obs_times = torch.Tensor([0.02, 0.04, 0.06, 1.0])
    sim_time = torch.Tensor([MjData.time])
    global_target_pose = torch.stack(
        [
            get_target_pose(
                env,
                times= MjData.time + t_offset,
                sim_dt=0.005,
            )
            for t_offset in target_obs_times
        ],
        dim=1,
    )  # (num_envs, num_obs, 4, 4)
    

    global_target_pose = global_target_pose[0,:,:,:]

    # global_target_pose = torch.Tensor(
    #     [[[1.0, 0.0,  0.0, 0.2],
    #      [0.0, 1.0, 0.0,  0.0],
    #      [0.0, 0.0, 1.0,  0.4],
    #      [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    #     [[1.0, 0.0,  0.0, 0.2],
    #      [0.0, 1.0, 0.0,  0.0],
    #      [0.0, 0.0, 1.0,  0.4],
    #      [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    #     [[1.0, 0.0,  0.0, 0.2],
    #      [0.0, 1.0, 0.0,  0.0],
    #      [0.0, 0.0, 1.0,  0.4],
    #      [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],

    #     [[1.0, 0.0,  0.0, 0.5],
    #      [0.0, 1.0, 0.0,  0.0],
    #      [0.0, 0.0, 1.0,  0.5],
    #      [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]]
    # ).to('cuda:0')

    # print("global_target_pose",global_target_pose)
    # 延迟 1 step
    observation_link_pose = torch.from_numpy(link_pose_history).to('cuda:0')  # (num_envs, 4, 4), clone otherwise sim state will be modified


    ee_id = MjData.body("end_effector").id
    pos_ee = MjData.xpos[ee_id]
    quat_ee = MjData.xquat[ee_id]
    quat_ee = [quat_ee[1],quat_ee[2],quat_ee[3],quat_ee[0]]

    pos_body = MjData.qpos[:3]
    quat_body = MjData.qpos[3:7]
    quat_body = [quat_body[1],quat_body[2],quat_body[3],quat_body[0]]



    # print("pos:" ,pos)
    R_ee = R.from_quat(quat_ee).as_matrix()
    R_body = R.from_quat(quat_body).as_matrix()

    local_pos = R_body.T @ (pos_ee -pos_body)
    local_quat = R_body.T @ R_ee

    link_pose_history[:3,:3] = local_quat
    link_pose_history[:3,3] = local_pos

    

    local_target_pose = (
        torch.linalg.inv(observation_link_pose.float()) @ global_target_pose
    )

    pos_obs = (local_target_pose[..., :3, 3] * env.pos_obs_scale).view(
        env.num_envs, -1
    )

    if env.pos_obs_clip is not None:
        pos_obs = torch.clamp(pos_obs, -env.pos_obs_clip, env.pos_obs_clip)
    orn_obs = (
        pt3d.matrix_to_rotation_6d(local_target_pose[..., :3, :3])
        * env.orn_obs_scale
    ).view(env.num_envs, -1)

    relative_pose_obs = torch.cat((pos_obs, orn_obs), dim=1)
    # NOTE after episode resetting, the first pose will be outdated
    # (this is a quirk of isaacgym, where state resets don't apply until the
    # next physics step), we will have to wait for `pose_latency` seconds
    # to get the first pose so just return special values for such cases
    waiting_for_pose_mask = (
        (observation_link_pose == torch.eye(4, device=env.device))
        .all(dim=-1)
        .all(dim=-1)
    )
    relative_pose_obs[waiting_for_pose_mask] = -1.0

    return relative_pose_obs, global_target_pose


print_count = 0
# offsets = [
#     0.1,  0.8, -1.5,
#     -0.1,  0.8, -1.5,
#     0.1,  1.0, -1.5,
#     -0.1,  1.0, -1.5,
#     0.0,  0.5, -0.5,  0.0,  0.0,  0.0
# ]
offsets = [
    0.1,  0.8, -1.5,
    -0.1,  0.8, -1.5,
    0.1,  1.0, -1.5,
    -0.1,  1.0, -1.5,
    0.0,  0.0, 0.0,  0.0,  0.0,  0.0
]
# kps = [56.8180,  56.6650,  51.0831,  73.0741,  65.1575,  45.9261,  40.9404,
#         59.3790,  46.9374,  42.8220,  39.4459,  49.8876, 223.6642, 212.9659,
#         53.2537,  74.9537,  56.0480,  15.1038]
# kds = [1.3544, 1.3853, 1.7969, 1.5305, 1.6285, 1.3584, 1.2268, 1.6558, 1.2969,
#         1.2228, 1.3105, 1.2119, 3.5078, 2.9678, 3.4770, 1.2140, 1.4190, 0.4582]

# kps = [70,50,80,
#        70,50,80,
#        70,50,50,
#        70,50,50,
#        150,150,100,50,50,20]
# kds = [1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,1.25,3.0,3.0,3.0,2.0,1.0,0.5]

kps = [ 67.0836+200,  53.4219+40,  73.2470+40,  42.9056+200,  27.5247+40,  37.3351+40,  58.8479,
          36.0806,  41.5540,  41.4776,  53.5915,  72.7477,  79.4844, 110.4279,
          74.2736,  26.8182,  28.3079,   6.7272]
kds = [1.1288+10, 0.5270, 1.1730, 0.9112+10, 1.0497, 0.9425, 0.9452, 0.6217, 0.9993,
         1.4791, 1.3940, 0.5742, 0.5306, 1.1236, 1.3222, 0.5369, 0.6784, 0.5892]

target_lin_vel_x = 0.3
target_lin_vel_y = 0.0
target_ang_vel_z = 0.0
target_z_height = 0.35
target_local_gravity = [0.0, 0.0, -1.0]

def play():
    # mujoco init
    xml_path_ = "/home/george/code/umi-on-legs/mani-centric-wbc/resources/robots/aliengo_description/urdf/aliengoPiper.xml"
    # xml_path_ = "/home/george/code/umi-on-legs/mani-centric-wbc/resources/robots/aliengoLerobot/go1_torque.xml"
    model = mujoco.MjModel.from_xml_path(xml_path_)
    model.opt.timestep = 0.005

    data = mujoco.MjData(model)

    data.qpos[7:] = offsets
    data.qpos[0] = data.qpos[1] = 0.0
    data.qpos[2] = 0.38
    data.qvel[:6] = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

            max_action_value = 100.0
            action_scale = 0.25
            num_actions = 18


            tau_limit = [35.278,35.278,44.400,35.278,35.278,44.400,35.278,35.278,44.400,35.278,35.278,44.400,20.0,20.0,15.0,7.0,5.0,5.0]


            parser = ArgumentParser()
            parser.add_argument("--ckpt_path", type=str)
            parser.add_argument("--visualize", action="store_true")
            parser.add_argument("--record_video", action="store_true")
            parser.add_argument("--device", type=str, default="cuda:0")
            parser.add_argument("--trajectory_file_path", type=str, required=True)
            parser.add_argument("--num_envs", type=int, default=1)
            parser.add_argument("--num_steps", type=int, default=1000)
            args = parser.parse_args()
            if args.visualize:
                args.num_envs = 1

            config = OmegaConf.create(
                pickle.load(
                    open(os.path.join(os.path.dirname(args.ckpt_path), "config.pkl"), "rb")
                )
            )
            sim_params = gymapi.SimParams()
            gymutil.parse_sim_config(config.env.cfg.sim, sim_params)
            config = recursively_replace_device(
                OmegaConf.to_container(
                    config,
                    resolve=True,
                ),
                device=args.device,
            )
            config["_convert_"] = "all"
            config["wandb"]["mode"] = "offline"  # type: ignore
            config["env"]["headless"] = not args.visualize  # type: ignore
            config["env"]["graphics_device_id"] = int(args.device.split("cuda:")[-1]) if "cuda" in args.device else 0  # type: ignore
            config["env"]["attach_camera"] = args.visualize  # type: ignore
            config["env"]["sim_device"] = args.device
            config["env"]["dof_pos_reset_range_scale"] = 0
            config["env"]["controller"]["num_envs"] = args.num_envs  # type: ignore
            config["env"]["cfg"]["env"]["num_envs"] = args.num_envs  # type: ignore
            config["env"]["controller"]["num_envs"] = args.num_envs  # type: ignore
            config["env"]["cfg"]["domain_rand"]["push_robots"] = False  # type: ignore
            config["env"]["cfg"]["domain_rand"]["transport_robots"] = False  # type: ignore

            # reset episode before commands change
            config["env"]["cfg"]["terrain"]["mode"] = "plane"
            config["env"]["cfg"]["init_state"]["pos_noise"] = [0.0, 0.0, 0.0]
            config["env"]["cfg"]["init_state"]["euler_noise"] = [0.0, 0.0, 0.0]
            config["env"]["cfg"]["init_state"]["lin_vel_noise"] = [0.0, 0.0, 0.0]
            config["env"]["cfg"]["init_state"]["ang_vel_noise"] = [0.0, 0.0, 0.0]
            config["env"]["cfg"]["init_state"]["pos"] = [0.0, 0.0, 0.0]
            # config["env"]["tasks"]["reaching"]["sequence_sampler"][
            #     "file_path"
            # ] = args.trajectory_file_path

            config["env"]["tasks"]["locomotion"]["lin_vel_range"] = [[target_lin_vel_x, target_lin_vel_x],[0.0, target_lin_vel_y],[0.0,0.0]]
            config["env"]["tasks"]["locomotion"]["ang_vel_range"] = [[0.0, 0.0],[0.0, 0.0],[0.0, target_ang_vel_z]]
            config["env"]["tasks"]["locomotion"]["z_height_range"] = [target_z_height, target_z_height]

            config["env"]["constraints"] = {}

            setup(config, seed=config["seed"])  # type: ignore

            env: IsaacGymEnv = hydra.utils.instantiate(
                config["env"],
                sim_params=sim_params,
            )
            config["runner"]["ckpt_dir"] = wandb.run.dir
            runner: OnPolicyRunner = hydra.utils.instantiate(
                config["runner"], env=env, eval_fn=None
            )
            runner.load(args.ckpt_path)
            policy = runner.alg.get_inference_policy(device=env.device)
            actor_idx: int = config["env"]["cfg"]["env"]["num_envs"] // 2



            def update_cam_pos():
                cam_rotating_frequency: float = 0.025
                offset = np.array([0.8, 0.3, 0.3]) * 1.5
                target_position = env.state.root_pos[actor_idx, :]
                # rotate camera around target's z axis
                angle = np.sin(2 * np.pi * env.gym_dt * cam_rotating_frequency * count)
                target_transform = affines.compose(
                    T=target_position.cpu().numpy(),
                    R=np.array(
                        [
                            [np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle), 0],
                            [0, 0, 1],
                        ]
                    ),
                    Z=np.ones(3),
                )

                camera_transform = target_transform @ affines.compose(
                    T=offset,
                    R=np.identity(3),
                    Z=np.ones(3),
                )
                try:
                    camera_position = affines.decompose(camera_transform)[0]
                    env.set_camera(camera_position, target_position)
                except np.linalg.LinAlgError:
                    pass
                finally:
                    pass

            # obs = torch.Tensor([[1,61]]).to('cuda:0')
            actions_ = torch.Tensor([[0,0,0,0,0,0,0,0,0,0,0,0]]).to('cuda:0')
            obs = step(self=env, action=actions_, data=data, model=model, viewer=viewer)
            # print("obs.size",obs.size())

            # if args.visualize:
            #     env.render()  # render once to initialize viewer
            count = 0
            if args.num_steps == -1:
                with torch.inference_mode():
                    action_history = np.zeros(18)
                    while True:

                        actions_ = policy(obs)
                        # print("actions_",actions_.cpu().numpy() - offsets[:12])
                        obs = step(self=env, action=actions_, data=data, model=model, viewer=viewer)
                        print("target",obs[0,42:49])
                        # if count < 150:
                        #     print("actions_",actions_)
                        #     count += 1




def step(
    self,
    action: torch.Tensor,
    data: mujoco.MjData,
    model: mujoco.MjModel,
    viewer: mujoco_viewer.MujocoViewer,
):
    """
    Apply actions, simulate, call

    Args:
        actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
    """
    info = {}
    self.ctrl.push(
        torch.clip(action, -self.max_action_value, self.max_action_value).to(
            self.device
        )
    )
    decimation_count = self.controller.decimation_count
    for decimation_step in range(decimation_count):
        # handle delay by indexing into the buffer of past targets
        # since new actions are pushed to the front of the buffer,
        # the current target is further back in the buffer for larger
        # delays.
        curr_target_idx = torch.ceil(
            ((self.ctrl_delay_steps - decimation_step)) / decimation_count
        ).long()

        assert (curr_target_idx >= 0).all()


        actions = self.ctrl.buffer.permute(2, 1, 0)[
                torch.arange(self.num_actions, device=self.device),
                curr_target_idx,
                :,
            ].permute(1, 0)
        
        actions = actions.cpu().numpy()
    
        normalized_action = np.zeros(18)
        normalized_action[:12] = actions[0,:] * 0.25 + offsets[:12]  # action * scale + offset
        normalized_action[12:] = offsets[12:]
        
        torques = kps * (normalized_action - data.qpos[7:]) - kps * data.qvel[6:]
        # print("normalized_action",normalized_action)
        # print("data.qpos[7:]",data.qpos[7:])
        # print("data.time",data.time)
        torques = np.clip(torques, -44.4, 44.4)
        data.ctrl = torques
        mujoco.mj_step(model,data)
        
        viewer.sync()
        # print("diff",normalized_action - data.qpos[7:])



    obs = self.get_observations(
        state=self.state,
        setup=self.setup,
        state_obs=self.state_obs,
        setup_obs=self.setup_obs,
    )
    #     # tasks
    # all_task_obs = []
    # for k, task in self.tasks.items():
    #     task_obs, global_task = observe(task ,data)
    #     all_task_obs.append(task_obs)
    # task_obs_tensor = torch.cat(
    #     all_task_obs,
    #     dim=1,
    # )
    
    # task_obs_tensor = task_obs_tensor.cpu().clone().numpy()
    # global_task = global_task.cpu().clone().numpy()
    # positions = []
    # global_offset = np.zeros(3).T
    # global_offset[0] = 1
    # positions.append(global_task[0,:3,3]+global_offset)
    # positions.append(global_task[1,:3,3]+global_offset)
    # positions.append(global_task[2,:3,3]+global_offset)
    # positions.append(global_task[3,:3,3]+global_offset)

    # draw_trajectory(viewer, positions)

    # 每一项要乘各自的scale
    obs = obs.cpu().numpy()
    obs_ = np.zeros_like(obs)
    obs_[0,0:18] = data.qpos[7:] - offsets  + 0.01*np.random.randn(18)
    obs_[0,18:36] = data.qvel[6:] * 0.05 + 0.075*np.random.randn(18)
    # print("dof_vel",obs_[0,18:36])
    obs_[0,36:39] = local_root_gravity(data) + 0.05*np.random.randn(3)
    # print("local_root_gravity(data)",local_root_gravity(data))
    obs_[0,39:42] = data.qvel[3:6] * 0.25  + 0.05*np.random.randn(3)
    # obs_[0,42:78] = task_obs_tensor[0,:]
    # obs_[0,78:96] = action.cpu().numpy()
    obs_[0,42:49] = [target_lin_vel_x*2.0, target_lin_vel_y*2.0,  target_ang_vel_z*2.0,  target_z_height, target_local_gravity[0], target_local_gravity[1], target_local_gravity[2]] # 需要修改
    obs_[0,49:61] = action.cpu().numpy()

    obs_ = torch.from_numpy(obs_).to('cuda:0')

    return obs_

def draw_trajectory(viewer, positions, color=[0, 1, 0, 1], width=0.01):
    """
    绘制轨迹线段。

    参数：
    - viewer: Mujoco 查看器对象。
    - positions: 存储位置的列表。
    - color: 线段颜色，默认为绿色。
    - width: 线条宽度，默认为 0.002。
    """
    # 清除之前的可视化几何体
    viewer.user_scn.ngeom = 0

    # 使用 mjv_connector 绘制路径线段
    for i in range(len(positions) - 1):
        mujoco.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_LINE,
            width=width,
            from_=positions[i],
            to=positions[i + 1]
        )
        # 设置几何体的颜色
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = color
        viewer.user_scn.ngeom += 1


if __name__ == "__main__":
    play()