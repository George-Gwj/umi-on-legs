import os
import pickle
import re
from isaacgym import gymapi, gymutil  # must be improved before torch
from argparse import ArgumentParser
import dill
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

import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.pose_util import pose_to_mat, mat_to_pose

OmegaConf.register_new_resolver("eval", eval, replace=True)

def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal
                
                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))


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

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_swap', is_flag=True, default=False)


def umi(input, output, robot_config, 
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, 
    no_mirror, sim_fov, camera_intrinsics, mirror_swap):
    max_gripper_width = 0.09
    gripper_speed = 0.2
    
    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    
    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right
    
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, \
            KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(
                output_dir=output,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                # latency
                camera_obs_latency=0.17,
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_swap=mirror_swap,
                # action
                max_pos_speed=2.0,
                max_rot_speed=6.0,
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break

                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # creating model
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)


            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference")
            obs = env.get_obs()
            episode_start_pose = list()
            for robot_id in range(len(robots_config)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10 * len(robots_config)
                action = get_real_umi_action(action, obs, action_pose_repr)
                assert action.shape[-1] == 7 * len(robots_config)
                del result

            print('Ready!')
            while True:
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # get current pose
                    obs = env.get_obs()    # 改成从issac gym中获得obs
                    episode_start_pose = list()
                    for robot_id in range(len(robots_config)):
                        pose = np.concatenate([
                            obs[f'robot{robot_id}_eef_pos'],
                            obs[f'robot{robot_id}_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    perv_target_pose = None
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                tx_robot1_robot0=tx_robot1_robot0,
                                episode_start_pose=episode_start_pose)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            print('Inference latency:', time.time() - s)
                        
                        # convert policy action to env actions
                        this_target_poses = action
                        assert this_target_poses.shape[1] == len(robots_config) * 7
                        for target_pose in this_target_poses:
                            for robot_idx in range(len(robots_config)):
                                solve_table_collision(
                                    ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                                    gripper_width=target_pose[robot_idx * 7 + 6],
                                    height_threshold=robots_config[robot_idx]['height_threshold']
                                )
                            
                            # solve collison between two robots
                            solve_sphere_collision(
                                ee_poses=target_pose.reshape([len(robots_config), -1]),
                                robots_config=robots_config
                            )

                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1]
                        print(dt)
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)
                        if np.sum(is_new) == 0:
                            # exceeded time budget, still do something
                            this_target_poses = this_target_poses[[-1]]
                            # schedule on next available step
                            next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                            action_timestamp = eval_t_start + (next_step_idx) * dt
                            print('Over budget', action_timestamp - curr_time)
                            action_timestamps = np.array([action_timestamp])
                        else:
                            this_target_poses = this_target_poses[is_new]
                            action_timestamps = action_timestamps[is_new]

                        # execute actions
                        env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                        )
                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        obs_left_img = obs['camera0_rgb'][-1]
                        obs_right_img = obs['camera0_rgb'][-1]
                        vis_img = np.concatenate([obs_left_img, obs_right_img], axis=1)
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])

                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                print('Stopped.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")

def play():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--trajectory_file_path", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--dp_path", type=str)

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
    config["env"]["tasks"]["reaching"]["sequence_sampler"][
        "file_path"
    ] = args.trajectory_file_path

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

    obs, privileged_obs = env.reset()

    if args.visualize:
        env.render()  # render once to initialize viewer

    if args.num_steps == -1:
        with torch.inference_mode():
            while True:
                actions = policy(obs)
                obs = env.step(actions)[0]
                update_cam_pos()
                env.render()

    state_logs = {
        "root_state": [],
        "root_pos": [],
        "root_xyzw_quat": [],
        "root_lin_vel": [],
        "root_ang_vel": [],
        "rigid_body_pos": [],
        "rigid_body_xyzw_quat": [],
        "dof_pos": [],
        "dof_vel": [],
        "contact_forces": [],
        "episode_time": [],
    }
    action_logs = {
        "torque": [],
        "action": [],
        "obs": [],
    }
    task_logs = {
        "target_positions": [],
        "target_quats_wxyz": [],
    }
    episode_logs = {}
    task = list(env.tasks.values())[0]
    with imageio.get_writer(
        uri=os.path.join(wandb.run.dir, "video.mp4"), mode="I", fps=24
    ) as writer, torch.inference_mode():

        def render_cb(env, writer=writer):
            global count
            if env.state.time * 24 < count:
                return
            if args.visualize:
                update_cam_pos()
                env.visualize(vis_env_ids=[0])
                env.render()
                count += 1
                if args.record_video:
                    env.gym.write_viewer_image_to_file(
                        env.viewer, f"/{wandb.run.dir}/out.png"
                    )
                    img = imageio.imread(f"/{wandb.run.dir}/out.png")
                    writer.append_data(img)
            for k, v in state_logs.items():
                v.append(getattr(env.state, k)[:].cpu().numpy())
            action_logs["torque"].append(env.ctrl.torque[:].cpu().numpy())
            action_logs["action"].append(actions.view(args.num_envs, -1).cpu().numpy())
            action_logs["obs"].append(obs.view(args.num_envs, -1).cpu().numpy())
            global_target_pose = torch.stack(
                [
                    task.get_target_pose(
                        times=env.state.episode_time + t_offset,
                        sim_dt=env.state.sim_dt,
                    )
                    # NOTE: only for visualization purposes,
                    # you can change the target pose times to any
                    # time interval which visualizes the gripper
                    # movements best
                    for t_offset in np.linspace(0.05, 1.0, 8)
                ],
                dim=1,
            )
            task_logs["target_positions"].append(
                global_target_pose[..., :3, 3].cpu().squeeze().numpy()
            )
            task_logs["target_quats_wxyz"].append(
                np.vstack(
                    [
                        quaternions.mat2quat(rot_mat)
                        for rot_mat in global_target_pose[..., :3, :3]
                        .cpu()
                        .squeeze()
                        .numpy()
                    ]
                )
            )

        for step_idx in track(range(args.num_steps), description="Playing"):
            actions = policy(obs)
            obs, privileged_obs, rews, dones, infos = env.step(
                actions, callback=render_cb
            )
            for k, v in infos.items():
                episode_logs.setdefault(k, []).append(v[:].cpu().numpy())

            if args.visualize and dones[actor_idx]:
                break
    root = zarr.group(
        store=zarr.DirectoryStore(wandb.run.dir + "/logs.zarr"), overwrite=True
    )
    for k, v in {
        **state_logs,
        **action_logs,
        **episode_logs,
        **task_logs,
    }.items():
        v = np.array(v)
        k = k.replace("/", "_")
        if len(v.shape) == 2:
            root.create_dataset(
                k, data=v, chunks=(1, *list(v.shape)[1:]), dtype=v.dtype
            )
        else:
            root.create_dataset(k, data=v, dtype=v.dtype)
    pickle.dump(
        {
            "config": config,
            "state_logs": state_logs,
            "action_logs": action_logs,
            "episode_logs": episode_logs,
            "task_logs": task_logs,
        },
        open(wandb.run.dir + "/logs.pkl", "wb"),
    )


if __name__ == "__main__":
    play()