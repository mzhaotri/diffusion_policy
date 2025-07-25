import os
from unittest import result
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robocasa.robocasa_image_wrapper import RobocasaImageWrapper
import robomimic.utils.file_utils as FileUtils
import robocasa.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robocasa.scripts.playback_dataset import get_env_metadata_from_dataset, get_env_from_dataset
from diffusion_policy.common.robocasa_util import create_environment, create_eval_environment
from diffusion_policy.common.robocasa_util import create_environment_interactive
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper, DAggerDataCollectionWrapper
# from robocasa.scripts.collect_demos import gather_dagger_demonstrations_as_hdf5
from robocasa.scripts.collect_dagger_demos import gather_human_only_dagger_demonstrations_as_hdf5, gather_demonstrations_as_hdf5, gather_single_human_only_dagger_demonstrations_as_hdf5, gather_single_combined_demonstrations_as_hdf5
from robocasa.utils.robomimic.robomimic_dataset_utils import convert_to_robomimic_format
import pdb
import robosuite
from robosuite.devices import SpaceMouse
import copy
import open_clip
import json
import matplotlib.pyplot as plt
import time
import pickle
import asyncio
import shutil
import threading
from pynput import keyboard
from termcolor import colored
# import cv2
# import robocasa.utils.eval_utils as EvalUtils


def create_interactive_eval_env_modified(
    env_name,
    # robosuite-related configs
    robots="PandaMobile",
    controllers="OSC_POSE",
    camera_names=[
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    ],
    camera_widths=256,
    camera_heights=256,
    seed=None,
    # robocasa-related configs
    obj_instance_split="B",
    generative_textures=None,
    randomize_cameras=False,
    layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
    controller_configs=None,
    id_selection=None,
):
    # controller_configs = load_controller_config(default_controller=controllers)   # somehow this line doesn't work for me

    # layout_and_style_ids = (layout_and_style_ids[id_selection],)

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=True,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        # seed=seed,
        renderer = 'mjviewer',
        render_camera="robot0_agentview_left",
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        # layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )
    

    env = robosuite.make(**env_kwargs)
    # pdb.set_trace()
    return env, env_kwargs

def create_env(dataset_path, env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env, env_meta = create_environment_interactive(dataset_path=dataset_path)

    return env, env_meta

def display_controller_cv2(controller_text):
    print(f"Current controller: {controller_text}")

def is_empty_input_spacemouse(action_dict):
    if not np.all(action_dict["right_delta"] == 0):
        return False
    if "base_mode" in action_dict and action_dict["base_mode"] != -1:
        return False
    if "base" in action_dict and not np.all(action_dict["base"] == 0):
        return False

    return True


## Get metrics
def adjust_xshape(x, in_dim):
    total_dim = x.shape[1]
    # Calculate the padding needed to make total_dim a multiple of in_dim
    remain_dim = total_dim % in_dim
    if remain_dim > 0:
        pad = in_dim - remain_dim
        total_dim += pad
        x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device)], dim=1)
    # Calculate the padding needed to make (total_dim // in_dim) a multiple of 4
    reshaped_dim = total_dim // in_dim
    if reshaped_dim % 4 != 0:
        extra_pad = (4 - (reshaped_dim % 4)) * in_dim
        x = torch.cat([x, torch.zeros(x.shape[0], extra_pad, device=x.device)], dim=1)
    return x.reshape(x.shape[0], -1, in_dim)

def logpZO_UQ(baseline_model, observation, action_pred = None, task_name = 'square'):
    observation = observation
    in_dim = 7
    observation = adjust_xshape(observation, in_dim)
    if action_pred is not None:
        action_pred = action_pred
        observation = torch.cat([observation, action_pred], dim=1)
    with torch.no_grad():
        timesteps = torch.zeros(observation.shape[0], device=observation.device)
        pred_v = baseline_model(observation, timesteps)
        observation = observation + pred_v
        logpZO = observation.reshape(len(observation), -1).pow(2).sum(dim=-1)
    return logpZO




class DAggerRobocasaImageRunner(BaseImageRunner):
    """
    Robocasa envs also enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            task_name,
            controller_configs,
            shape_meta:dict,
            max_steps=2000,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            device='cuda',
            n_dagger_rollouts=1,
            teleop_device='spacemouse',
            pos_sensitivity=1.0,
            rot_sensitivity=1.0,
            vendor_id=None,
            product_id=None,
        ):
        super().__init__(output_dir)
        max_steps=2000
        self.device = device
        self.task_name = task_name
        self.controller_configs = controller_configs

        # Setup spacemouse for teleop
        self.n_dagger_rollouts = n_dagger_rollouts
        # self.pos_sensitivity = pos_sensitivity
        # self.rot_sensitivity = rot_sensitivity
        self.pos_sensitivity = 4
        self.rot_sensitivity = 4
        self.vendor_id = vendor_id
        self.product_id = product_id
        print(f"Using vendor_id: {self.vendor_id}, product_id: {self.product_id}")

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        # steps_per_render = max(robosuite_fps // fps, 1)
        steps_per_render = 1

        # read from dataset
        # pdb.set_trace()
        # dataset_path = 'datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams_im256.hdf5'
        env_meta = get_env_metadata_from_dataset(dataset_path)
        self.dataset_path = dataset_path
        # disable object state observation
        # env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        # pdb.set_trace()
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        # self.robomimic_env
        self.env_meta = env_meta
        self.shape_meta = shape_meta
        # robomimic_env, robomimic_env_meta = create_env(dataset_path=self.dataset_path, env_meta=self.env_meta, shape_meta=self.shape_meta)
        robomimic_env, robomimic_env_meta =  create_interactive_eval_env_modified(env_name=self.task_name, controller_configs=self.controller_configs)
        
        
        robomimic_env = VisualizationWrapper(robomimic_env)
        self.env_info = json.dumps(robomimic_env_meta)
        # pdb.set_trace()

        # create folder for dagger data if it does not exist
        dagger_dir = pathlib.Path(output_dir) / 'dagger_data'
        if not os.path.exists(dagger_dir):
            os.makedirs(dagger_dir)
        self.dagger_dir = dagger_dir

        # create folder for dagger data if it does not exist
        processed_dagger_dir = pathlib.Path(output_dir) / 'processed_dagger_data'
        if not os.path.exists(processed_dagger_dir):
            os.makedirs(processed_dagger_dir)
        self.processed_dagger_dir = processed_dagger_dir

        # set env indicators to transparent
        # Make indicator sites fully transparent
        indicators = ['gripper0_right_grip_site', 'gripper0_right_grip_site_cylinder']
        for indicator_name in indicators:
            site_id = robomimic_env.env.sim.model.site_name2id(indicator_name)
            robomimic_env.env.sim.model.site_rgba[site_id] = [1, 0, 0, 0.0]

        robomimic_env = DAggerDataCollectionWrapper(robomimic_env, dagger_dir)

        # Make indicator sites fully transparent
        indicators = ['gripper0_right_grip_site', 'gripper0_right_grip_site_cylinder']
        for indicator_name in indicators:
            site_id = robomimic_env.env.sim.model.site_name2id(indicator_name)
            robomimic_env.env.sim.model.site_rgba[site_id] = [1, 0, 0, 0.0]

        

        self.env = MultiStepWrapper(
                RobocasaImageWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=render_obs_key
                ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )

        self.teleop_device = SpaceMouse(
                env=robomimic_env,
                pos_sensitivity=self.pos_sensitivity,
                rot_sensitivity=self.rot_sensitivity,
                vendor_id=int(self.vendor_id),
                product_id=int(self.product_id),
            )
        
        self.render_obs_key = render_obs_key
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.steps_per_render = steps_per_render
        self.output_dir = output_dir
        self.adjusted_cp_band = None
        self.fig = None

        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            threading.Thread(target=self.loop.run_forever, daemon=True).start()

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

        self.stop_flag = False

    def on_press(self, key):
        try:
            if key.char == 'q':
                self.stop_flag = True
        except AttributeError:
            pass  # handle special keys if needed

    def convert_observations(self, dataset, obs, clip_embedding):

        left_image = obs['robot0_agentview_left_image']
        right_image = obs['robot0_agentview_right_image']
        gripper_image = obs['robot0_eye_in_hand_image']
        # pdb.set_trace()

        # flip right and left image over, currently they are upside down
        left_image[0] = np.flipud(left_image[0])
        right_image[0] = np.flipud(right_image[0])
        gripper_image[0] = np.flipud(gripper_image[0])

        

        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # axs[0].imshow(left_image[-1])
        # axs[0].set_title('Left Image')
        # axs[1].imshow(right_image[-1])
        # axs[1].set_title('Right Image')
        # axs[2].imshow(gripper_image[-1])
        # axs[2].set_title('Gripper Image')
        # plt.show()

        left_image = dataset.convert_frame(frame=left_image[0], swap_rgb=dataset.swap_rgb)
        right_image = dataset.convert_frame(frame=right_image[0], swap_rgb=dataset.swap_rgb)
        gripper_image = dataset.convert_frame(frame=gripper_image[0], swap_rgb=dataset.swap_rgb)
        # pdb.set_trace()

        left_image = torch.tensor(left_image, dtype=torch.float32)
        right_image = torch.tensor(right_image, dtype=torch.float32)
        gripper_image = torch.tensor(gripper_image, dtype=torch.float32)

        left_image = (left_image + 1) / 2
        right_image = (right_image + 1) / 2
        gripper_image = (gripper_image + 1) / 2

        left_image = dataset.augmentation_transform(left_image, dataset.transform_rgb)
        right_image = dataset.augmentation_transform(right_image, dataset.transform_rgb)
        gripper_image = dataset.augmentation_transform(gripper_image, dataset.transform_rgb)
        
        left_image = left_image.unsqueeze(0)
        right_image = right_image.unsqueeze(0)
        gripper_image = gripper_image.unsqueeze(0)

        # unsqueeze again
        left_image = left_image.unsqueeze(0)
        right_image = right_image.unsqueeze(0)
        gripper_image = gripper_image.unsqueeze(0)

        return {
            "task_description": clip_embedding,
            "left_image": left_image,
            "right_image": right_image,
            "gripper_image": gripper_image,
            }


    def recreate_env(self, idx_seed):
        self.env.close()
        robomimic_env, robomimic_env_meta =  create_interactive_eval_env_modified(env_name=self.task_name, seed=idx_seed, controller_configs=self.controller_configs)
        
        
        robomimic_env = VisualizationWrapper(robomimic_env)
        self.env_info = json.dumps(robomimic_env_meta)
        # pdb.set_trace()
        
        # Make indicator sites fully transparent
        indicators = ['gripper0_right_grip_site', 'gripper0_right_grip_site_cylinder']
        for indicator_name in indicators:
            site_id = robomimic_env.env.sim.model.site_name2id(indicator_name)
            robomimic_env.env.sim.model.site_rgba[site_id] = [1, 0, 0, 0.0]

        robomimic_env = DAggerDataCollectionWrapper(robomimic_env, self.dagger_dir)

        # Make indicator sites fully transparent
        indicators = ['gripper0_right_grip_site', 'gripper0_right_grip_site_cylinder']
        for indicator_name in indicators:
            site_id = robomimic_env.env.sim.model.site_name2id(indicator_name)
            robomimic_env.env.sim.model.site_rgba[site_id] = [1, 0, 0, 0.0]

        

        self.env = MultiStepWrapper(
                RobocasaImageWrapper(
                    env=robomimic_env,
                    shape_meta=self.shape_meta,
                    init_state=None,
                    render_obs_key=self.render_obs_key
                ),
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            max_episode_steps=self.max_steps
        )

        self.teleop_device.set_env(robomimic_env)
        


    def run_interactive_dagger_rollout_constant_band(self, video_dataset, policy: BaseImagePolicy, score_network, CP_band, dagger_ep_idx, end_ep_idx):

        self.video_dataset = video_dataset
        self.score_network = score_network
        self.policy = policy
        self.recreate_env(dagger_ep_idx)
        env = self.env

        # start CP band plot
        # fig, ax = plt.subplots()
        
        # xaxis = np.arange(len(CP_band))
        # upper = CP_band
        # lower = np.zeros_like(CP_band)
        # ax.fill_between(xaxis, upper, lower, color='blue', alpha=0.25)
        # auton_logpzo_scores = [0]
        # line, = ax.plot(range(len(auton_logpzo_scores)), auton_logpzo_scores, color='green', label='Log PZO')
        # ax.set_xlabel('Timestep')
        # ax.set_ylabel('Log PZO')
        # ax.set_title('Log PZO vs Timestep')
        # ax.set_ylim(0, 3500)
        # ax.set_xlim(0, 60)

        # Initialize figure with 2 subplots: left (plot), right (text)
        # if dagger_ep_idx == 0:
        #     fig, (self.ax_plot, self.ax_text) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={'width_ratios': [3, 1]})
        #     plt.subplots_adjust(wspace=0.4)

        # Plot band
        CP_band = list(CP_band)
        while len(CP_band) < 60:
            CP_band.append(CP_band[-1])
        xaxis = np.arange(len(CP_band))
        upper = CP_band
        if self.adjusted_cp_band is None:
            self.adjusted_cp_band = CP_band[0]
        gamma = 100
        adjusted_alpha = 0.2
        lower = np.zeros_like(CP_band)
        # self.ax_plot.fill_between(xaxis, upper, lower, color='skyblue', alpha=0.3, label='CP Band')

        # Score line
        auton_logpzo_scores = [0]
        # line, = self.ax_plot.plot(range(len(auton_logpzo_scores)), auton_logpzo_scores, color='green', linewidth=2, label='Log PZO')

        # Axes styling
        # self.ax_plot.set_xlabel('Timestep', fontsize=12)
        # self.ax_plot.set_ylabel('Log PZO', fontsize=12)
        # self.ax_plot.set_title('Conformal Prediction and Log PZO', fontsize=14)
        # self.ax_plot.set_ylim(0, 3500)
        # self.ax_plot.set_xlim(0, 60)
        # self.ax_plot.grid(True, linestyle='--', alpha=0.6)
        # self.ax_plot.legend(loc='upper left', fontsize=10)

        # # Initialize text display on the right
        # self.ax_text.axis('off')
        # text_display = self.ax_text.text(0.5, 0.5, '', fontsize=14, ha='center', va='center')
        # plt.draw()
        # plt.pause(0.1)

        
        # start rollout
        obs = env.reset()
        past_action = None
        policy.reset()
        self.timestep = 0
        first_switch_to_human_occurred = False

        # Initialize plot
        if dagger_ep_idx == 0:
            self.fig, self.ax = plt.subplots()
            gripper_img = obs['robot0_eye_in_hand_image'][0]
            gripper_img = np.flipud(gripper_img)  # Flip the image upside down
            self.img = self.ax.imshow(gripper_img, cmap='gray', vmin=0, vmax=1)
            plt.ion()  # Turn on interactive mode
            plt.show()
        else:
            gripper_img = obs['robot0_eye_in_hand_image'][0]
            gripper_img = np.flipud(gripper_img)  # Flip the image upside down
            self.img.set_data(gripper_img)                
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()



        

        # setup teleop
        self.teleop_device.env = env.env.env
        self.teleop_device.start_control()
        nonzero_ac_seen = False
        # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
        # pdb.set_trace()
        all_prev_gripper_actions = [
            {
                f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                for robot_arm in robot.arms
                if robot.gripper[robot_arm].dof > 0
            }
            for robot in env.env.env.robots
        ]
        zero_action = np.zeros(env.env.env.action_dim)
        discard_dagger_traj = False
        acting_agent = 'robot'
        self.teleop_device.is_acting_agent = False
        env.env.env.set_dagger_acting_agent(acting_agent)
        new_controller = "Teleop Policy (Human)" if acting_agent=='human' else "Auto Policy (Robot)"
        display_controller_cv2(new_controller)
        mirror_actions = True


        env_name = self.env_meta['env_name']
        pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Timestep {self.timestep}/{self.max_steps}", 
            leave=False, mininterval=self.tqdm_interval_sec)
        
        
        done = False
        timestep = 0
        num_human_segments = 0
        human_segment_idx = 0
        human_segment_to_length = {}
        self.timestep = 0

        # pdb.set_trace() 
        task_description = env.env.env.env.get_ep_meta()["lang"]
        task_description = open_clip.tokenize([task_description]) # returns torch.Size([1, 77])
        with torch.no_grad():
            clip_embedding = self.video_dataset.lang_model(task_description.to(self.device)).cpu().unsqueeze(0) # returns torch.Size([1, 1, 1024])

        dagger_episode_meta = {}
        dagger_episode_meta['obs_list'] = []
        dagger_episode_meta['action_list'] = []
        dagger_episode_meta['n_human_timesteps'] = 0
        dagger_episode_meta['n_robot_timesteps'] = 0
        dagger_episode_meta['cp_band'] = CP_band
        dagger_episode_meta['adjusted_cp_band'] = [self.adjusted_cp_band]
        prev_score = 0

        while not done:
            # Handle any switches in acting agent
            if acting_agent == 'human' and self.teleop_device.is_acting_agent is False:
                # pdb.set_trace()
                acting_agent = 'robot'
                env.env.env.set_dagger_acting_agent(acting_agent)
            
                
                new_controller = "Teleop Policy (Human)" if acting_agent=='human' else "Auto Policy (Robot)"
                display_controller_cv2(new_controller)
            elif acting_agent == 'robot' and self.teleop_device.is_acting_agent is True:
                # pdb.set_trace()
                num_human_segments += 1
                human_segment_idx += 1
                human_segment_to_length[human_segment_idx] = 0
                acting_agent = 'human'
                env.env.env.set_dagger_acting_agent(acting_agent)

                # adjust cp band
                y_t = prev_score
                err_t = int(self.adjusted_cp_band > y_t)
                self.adjusted_cp_band = self.adjusted_cp_band - gamma *  (err_t - (adjusted_alpha))
                dagger_episode_meta['adjusted_cp_band'].append(self.adjusted_cp_band)
                print(f"Adjusted CP band at timestep {self.timestep}: {self.adjusted_cp_band}")
                adj_upper = [self.adjusted_cp_band] * len(lower)
                # self.ax_plot.fill_between(xaxis, adj_upper, lower, color='red', alpha=0.3, label='CP Band')
                # # Update plot
                # plt.draw()
                # plt.pause(0.1)

                new_controller = "Teleop Policy (Human)" if acting_agent=='human' else "Auto Policy (Robot)"
                display_controller_cv2(new_controller)
                human_segment_timestep = 0
                control_text = "Human acting"
                control_color = "red"

                # pdb.set_trace()
                current_gripper_qpos = self.env.env.env.env.env._get_observations()['robot0_gripper_qpos']
                closed_gripper_qpos = np.array([ 0.00068734, -0.00048299])
                distance_to_closed_gripper = np.linalg.norm(current_gripper_qpos - closed_gripper_qpos)
                # pdb.set_trace()
                # if distance_to_closed_gripper < 0.001:
                #     self.teleop_device.single_click_and_hold = True
                # else:
                #     self.teleop_device.single_click_and_hold = False

                gripper_input = input("gripper closed? (y/n): ")
                if gripper_input.lower() == 'y':
                    self.teleop_device.single_click_and_hold = True
                elif gripper_input.lower() == 'n':
                    self.teleop_device.single_click_and_hold = False
                # Update text display
                # text_display.set_text(control_text)
                # text_display.set_color(control_color)
                # plt.draw()
                # plt.pause(0.1)
            gripper_img = obs['robot0_eye_in_hand_image'][0]
            gripper_img = np.flipud(gripper_img)  # Flip the image upside down
            self.img.set_data(gripper_img)                
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

            if acting_agent == 'human':
                

                # Set active robot
                active_robot = env.env.env.robots[self.teleop_device.active_robot]
                active_arm = self.teleop_device.active_arm

                # Get the newest action
                input_ac_dict = self.teleop_device.input2action(mirror_actions=mirror_actions)

                # If action is none, then this a reset so we should break
                if input_ac_dict is None:
                    discard_traj = True
                    break

                action_dict = copy.deepcopy(input_ac_dict)

                # set arm actions
                for arm in active_robot.arms:
                    controller_input_type = active_robot.part_controllers[arm].input_type
                    if controller_input_type == "delta":
                        action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                    elif controller_input_type == "absolute":
                        action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                    else:
                        raise ValueError
                    
                if is_empty_input_spacemouse(action_dict):
                    if not nonzero_ac_seen:
                        env.render()
                        continue
                else:
                    nonzero_ac_seen = True
                    human_segment_timestep += 1
                    dagger_episode_meta['n_human_timesteps'] += 1

                human_segment_to_length[human_segment_idx] += 1

                # Maintain gripper state for each robot but only update the active robot with action
                env_action = [
                    robot.create_action_vector(all_prev_gripper_actions[i])
                    for i, robot in enumerate(env.env.env.robots)
                ]
                env_action[self.teleop_device.active_robot] = active_robot.create_action_vector(action_dict)
                env_action = np.concatenate(env_action)
                env_action =  np.expand_dims(env_action, axis=0) # add time dimension
                # pdb.set_trace()
                obs, _, _, _ = env.step(env_action)
                dagger_episode_meta['obs_list'].append((obs, acting_agent, self.timestep))
                dagger_episode_meta['action_list'].append((env_action, acting_agent, self.timestep))
                env.render()

                
                if human_segment_timestep == 16:
                    human_segment_timestep = 0
                    self.timestep += 1

                if env.env.env._check_success():
                    print("finished task")
                    break

            else:
                # current_gripper_qpos = np.mean(abs(self.env.env.env.env.env._get_observations()['robot0_gripper_qpos']))
                # print("current_gripper_qpos", current_gripper_qpos)
                batch = self.convert_observations(self.video_dataset, obs, clip_embedding)
                batch = {key: value.to(self.device, dtype=torch.float32) for key, value in batch.items()}

                action_pred, action_pred_infos_result = self.policy.predict_action_with_infos(batch)
                action_pred = ((action_pred.detach().cpu().numpy() + 1) / 2) * (self.video_dataset.max - self.video_dataset.min) + self.video_dataset.min
                action_pred = np.squeeze(action_pred)
                action_pred = np.hstack((action_pred, [[0, 0, 0, 0, -1]] * action_pred.shape[0]))
                action_horizon = 16
                action_pred = action_pred[0:action_horizon]

                # baseline_metric = logpZO_UQ(self.score_network, action_pred_infos_result['global_cond'])
                # print("LOG PZO", baseline_metric)
                # if self.timestep < len(CP_band):
                #     CP_threshold_at_t = CP_band[self.timestep]
                # else:
                #     CP_threshold_at_t = CP_band[-1]
                # print("CP threshold at t", CP_threshold_at_t)

                # # pdb.set_trace()
                # auton_logpzo_scores.append(baseline_metric.item())
                # line.set_data(range(len(auton_logpzo_scores)), auton_logpzo_scores)
                # # ax.relim()
                # # ax.autoscale_view()
                # plt.draw()
                # plt.pause(0.001)
                baseline_metric = logpZO_UQ(self.score_network, action_pred_infos_result['global_cond'])
                print("LOG PZO", baseline_metric)
                prev_score = baseline_metric.item()

                # Get CP threshold at current timestep
                CP_threshold_at_t = CP_band[self.timestep] if self.timestep < len(CP_band) else CP_band[-1]
                print("CP threshold at t", CP_threshold_at_t)
                print(f"Adjusted CP band: {self.adjusted_cp_band}")

                dagger_episode_meta['obs_list'].append((obs, acting_agent, baseline_metric, CP_threshold_at_t, self.timestep))
                dagger_episode_meta['action_list'].append((action_pred, acting_agent, self.timestep))

                # Update score history
                auton_logpzo_scores.append(baseline_metric.item())
                # line.set_data(range(len(auton_logpzo_scores)), auton_logpzo_scores)

                # Decide control status
                if acting_agent == 'human':
                    control_text = "Human requested"
                    control_color = "red"
                else:
                    control_text = "Robot acting"
                    control_color = "green"

                # Update text display
                # text_display.set_text(control_text)
                # text_display.set_color(control_color)

                # Refresh plot
                # self.ax_plot.set_xlim(0, max(60, len(auton_logpzo_scores)))
                # if adjusted_cp_band != CP_band[0]:
                #     adj_upper = [adjusted_cp_band] * len(lower)
                #     self.ax_plot.fill_between(xaxis, adj_upper, lower, color='red', alpha=0.3, label='CP Band')
                # plt.draw()
                # plt.pause(0.1)
    
                if baseline_metric > CP_threshold_at_t or baseline_metric > self.adjusted_cp_band:
                    print("FAILURE DETECTED: SWITCHING TO HUMAN")
                    self.teleop_device.is_acting_agent = True
                    first_switch_to_human_occurred = True
                    control_text = "Human requested"
                    control_color = "red"
                    env.env.env.set_dagger_acting_agent("human")
                    # Update text display
                    # text_display.set_text(control_text)
                    # text_display.set_color(control_color)
                    # plt.draw()
                    # plt.pause(0.1)
                    continue



                obs, reward, done, info = env.step(action_pred, render=True)
                dagger_episode_meta['n_robot_timesteps'] += 1
                self.timestep += 1
                if env.env.env._check_success():
                    print("finished task")
                    break

            print("done rendering")
            done = np.all(done)
            # self.timestep += 1
            # update pbar
            pbar.update(self.n_action_steps)
        pbar.close()
        print("human_segment_to_length", human_segment_to_length)
        # Close the figure after loop
        # plt.ioff()
        # plt.close(fig)
        
        want_to_save = input("Done with this episode, do you want to save the demo? (y/n): ").lower()
        if want_to_save == 'y':
            discard_dagger_traj = False
        else:
            discard_dagger_traj = True

        # clear matplotlib figure
        # plt.clf()

        # clear out video buffer
        _ = env.reset()
        # env.close()
        # Cleanup
        if nonzero_ac_seen and hasattr(env, "ep_directory"):
            ep_directory = env.env.env.ep_directory
        else:
            ep_directory = None

        hdf5_path = None
        if end_ep_idx == dagger_ep_idx:
            excluded_eps = []
            if discard_dagger_traj and ep_directory is not None:
                excluded_eps.append(ep_directory.split("/")[-1])
            hdf5_path = gather_human_only_dagger_demonstrations_as_hdf5(
                    self.dagger_dir, self.processed_dagger_dir, self.env_info, excluded_episodes=excluded_eps
                )
            print(f"Gathered Human-only DAgger demonstrations to {hdf5_path}")
            convert_to_robomimic_format(hdf5_path)
            print(f"Converted DAgger demonstrations to robomimic format at {hdf5_path}")
        
        
        # save dagger_episode_meta as pkl
        with open(os.path.join(self.dagger_dir, f"dagger_episode_meta_{dagger_ep_idx}.pkl"), 'wb') as f:
            pickle.dump(dagger_episode_meta, f)
            print(f"Saved dagger_episode_meta to {os.path.join(self.dagger_dir, f'dagger_episode_meta_{dagger_ep_idx}.pkl')}")


        hdf5_path = None
        if end_ep_idx == dagger_ep_idx:
            excluded_eps = []
            if discard_dagger_traj and ep_directory is not None:
                excluded_eps.append(ep_directory.split("/")[-1])
            hdf5_path = gather_demonstrations_as_hdf5(
                    self.dagger_dir, self.processed_dagger_dir, self.env_info, excluded_episodes=excluded_eps
                )
            print(f"Gathered Combined DAgger demonstrations to {hdf5_path}")
            convert_to_robomimic_format(hdf5_path)
            print(f"Converted DAgger demonstrations to robomimic format at {hdf5_path}")

        return hdf5_path
    

    def initialize_helper_plots(self, obs, CP_band, auton_logpzo_scores):
        self.fig, (self.ax_cp, self.ax_cam) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 1]})
        xaxis = np.arange(len(CP_band))
        lower = np.zeros_like(CP_band)
        self.ax_cp.fill_between(xaxis, CP_band, lower, color='skyblue', alpha=0.3, label='CP Band')

        self.ax_cp.set_xlabel('Timestep', fontsize=12)
        self.ax_cp.set_ylabel('Log PZO', fontsize=12)
        self.ax_cp.set_title('Conformal Prediction and Log PZO', fontsize=14)
        self.ax_cp.set_ylim(0, max(CP_band) * 1.1)
        self.line, = self.ax_cp.plot(range(len(auton_logpzo_scores)), auton_logpzo_scores, color='green', linewidth=2, label='Log PZO')

        gripper_img = obs['robot0_eye_in_hand_image'][0]
        gripper_img = np.flipud(gripper_img)  # Flip the image upside down
        self.img = self.ax_cam.imshow(gripper_img, cmap='gray', vmin=0, vmax=1)
        plt.ion()  # Turn on interactive mode
        plt.show()
    
    def update_helper_plots(self, obs, auton_logpzo_scores):
        gripper_img = obs['robot0_eye_in_hand_image'][0]
        gripper_img = np.flipud(gripper_img)  # Flip the image upside down
        self.img.set_data(gripper_img)                
        

        # update self.line
        self.line.set_data(range(len(auton_logpzo_scores)), auton_logpzo_scores)
        self.ax_cp.set_xlim(0, max(60, len(auton_logpzo_scores)))
        self.ax_cp.relim()
        self.ax_cp.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    async def predict_robot_action(self, obs, clip_embedding):
        # print("entered predict_robot_action")
        batch = self.convert_observations(self.video_dataset, obs, clip_embedding)
        batch = {key: value.to(self.device, dtype=torch.float32) for key, value in batch.items()}
        batch['robot0_eef_pos'] = torch.tensor(obs['robot0_eef_pos']).unsqueeze(0).to(self.device, dtype=torch.float32)
        batch['robot0_eef_quat'] = torch.tensor(obs['robot0_eef_quat']).unsqueeze(0).to(self.device, dtype=torch.float32)
        batch['robot0_gripper_qpos'] = torch.tensor(obs['robot0_gripper_qpos']).unsqueeze(0).to(self.device, dtype=torch.float32)

        action_pred, action_pred_infos_result = self.policy.predict_action_with_infos(batch)
        action_pred = ((action_pred.detach().cpu().numpy() + 1) / 2) * (self.video_dataset.max - self.video_dataset.min) + self.video_dataset.min
        action_pred = np.squeeze(action_pred)
        action_pred = np.hstack((action_pred, [[0, 0, 0, 0, -1]] * action_pred.shape[0]))
        # action_horizon = 1
        # action_pred = action_pred[0:action_horizon]
        # print("during teleop action_pred", action_pred)

        baseline_metric = logpZO_UQ(self.score_network, action_pred_infos_result['global_cond'])
        # print("during teleop LOG PZO", baseline_metric)
        bool_CP = 0
        thresh = self.CP_band[self.timestep] if self.timestep < len(self.CP_band) else self.CP_band[-1]
        if baseline_metric < thresh:
            bool_CP = 1

        return action_pred, bool_CP, baseline_metric, action_pred_infos_result

    

    def run_interactive_dagger_rollout(self, video_dataset, policy: BaseImagePolicy, score_network, CP_band, dagger_ep_idx, end_ep_idx):
        self.CP_band = CP_band
        dagger_episode_complete = False
        while dagger_episode_complete is False:
            self.video_dataset = video_dataset
            self.score_network = score_network
            self.policy = policy
            self.recreate_env(dagger_ep_idx)
            env = self.env


            # Set CP band
            CP_band = list(CP_band)
            
            
            # Score line
            auton_logpzo_scores = [0]
            

            
            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()
            self.timestep = 0
            first_switch_to_human_occurred = False

            # Initialize plot
            if self.fig is None:
                self.initialize_helper_plots(obs, CP_band, auton_logpzo_scores)
            else:
                self.update_helper_plots(obs, auton_logpzo_scores)


            # setup teleop
            self.teleop_device.env = env.env.env
            self.teleop_device.start_control()
            nonzero_ac_seen = False
            # Keep track of prev gripper actions when using since they are position-based and must be maintained when arms switched
            # pdb.set_trace()
            all_prev_gripper_actions = [
                {
                    f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                    for robot_arm in robot.arms
                    if robot.gripper[robot_arm].dof > 0
                }
                for robot in env.env.env.robots
            ]
            zero_action = np.zeros(env.env.env.action_dim)
            discard_dagger_traj = False
            acting_agent = 'robot'
            self.teleop_device.is_acting_agent = False
            env.env.env.set_dagger_acting_agent(acting_agent)
            new_controller = "Teleop Policy (Human)" if acting_agent=='human' else "Auto Policy (Robot)"
            display_controller_cv2(new_controller)
            mirror_actions = True


            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Timestep {self.timestep}/{self.max_steps}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            
            done = False
            timestep = 0
            num_human_segments = 0
            human_segment_idx = 0
            human_segment_to_length = {}
            self.timestep = 0
            begin_mixing_on_human_handback = False

            # pdb.set_trace() 
            task_description = env.env.env.env.get_ep_meta()["lang"]

            # print language command in green
            print(colored(f"TASK DESCRIPTION: {task_description}", "green"))
            task_description = open_clip.tokenize([task_description]) # returns torch.Size([1, 77])
            with torch.no_grad():
                clip_embedding = self.video_dataset.lang_model(task_description.to(self.device)).cpu().unsqueeze(0) # returns torch.Size([1, 1, 1024])

            dagger_episode_meta = {}
            dagger_episode_meta['obs_list'] = []
            dagger_episode_meta['action_list'] = []
            dagger_episode_meta['n_human_timesteps'] = 0
            dagger_episode_meta['n_robot_timesteps'] = 0
            dagger_episode_meta['cp_band'] = CP_band
            dagger_episode_meta['switch_initiated_by'] = []
            dagger_episode_meta['arbitration_to_robot'] = []
            dagger_episode_meta['robot_pred_during_teleop'] = []
            dagger_episode_meta['mixed_teleop_action'] = []
            dagger_episode_meta['during_teleop_robot_predictions'] = []

            # Save robot predictions during human teleop
            self.during_teleop_robot_prediction_task = None
            self.during_teleop_robot_action_chunk_prediction = None
            self.current_chunk_timestep = 0
            self.human_teleop_timestep = 0
            self.num_mixing_timesteps = 0
            self.current_bool_CP = 0
            arbitration_to_robot = 0

            prev_score = 0

            while not done:
                if self.stop_flag:
                    print("Quitting loop...")
                    self.stop_flag = False
                    break
                    # Handle any switches in acting agent
                if acting_agent == 'human' and self.teleop_device.is_acting_agent is False:
                    # acting_agent = 'robot'
                    # env.env.env.set_dagger_acting_agent(acting_agent)
                    # new_controller = "Teleop Policy (Human)" if acting_agent=='human' else "Auto Policy (Robot)"
                    # display_controller_cv2(new_controller)
                    dagger_episode_meta['switch_initiated_by'].append('human')
                    begin_mixing_on_human_handback = True
                    self.num_mixing_timesteps = 0
                elif acting_agent == 'robot' and self.teleop_device.is_acting_agent is True:
                    # pdb.set_trace()
                    num_human_segments += 1
                    human_segment_idx += 1
                    human_segment_to_length[human_segment_idx] = 0
                    acting_agent = 'human'
                    env.env.env.set_dagger_acting_agent(acting_agent)
                    new_controller = "Teleop Policy (Human)" if acting_agent=='human' else "Auto Policy (Robot)"
                    display_controller_cv2(new_controller)
                    human_segment_timestep = 0
                    self.human_teleop_timestep = 0
                    robot_action_prediction = None
                    robot_action_prediction_obs_score = None
                    arbitration_to_robot = 0
                    human_action_chunk = []
                    dagger_episode_meta['switch_initiated_by'].append('human')

                    gripper_input = input("gripper closed? (y/n): ")
                    if gripper_input.lower() == 'y':
                        self.teleop_device.single_click_and_hold = True
                    elif gripper_input.lower() == 'n':
                        self.teleop_device.single_click_and_hold = False

                self.update_helper_plots(obs, auton_logpzo_scores)

                if acting_agent == 'human':
                    # Set active robot
                    active_robot = env.env.env.robots[self.teleop_device.active_robot]
                    active_arm = self.teleop_device.active_arm

                    # Get the newest action
                    input_ac_dict = self.teleop_device.input2action(mirror_actions=mirror_actions)

                    # If action is none, then this a reset so we should break
                    if input_ac_dict is None:
                        discard_traj = True
                        break

                    action_dict = copy.deepcopy(input_ac_dict)

                    # set arm actions
                    for arm in active_robot.arms:
                        controller_input_type = active_robot.part_controllers[arm].input_type
                        if controller_input_type == "delta":
                            action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                        elif controller_input_type == "absolute":
                            action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                        else:
                            raise ValueError
                        
                    if is_empty_input_spacemouse(action_dict):
                        if not nonzero_ac_seen:
                            env.render()
                            continue
                    else:
                        nonzero_ac_seen = True
                        human_segment_timestep += 1
                        self.human_teleop_timestep += 1
                        self.current_chunk_timestep += 1
                        dagger_episode_meta['n_human_timesteps'] += 1

                    human_segment_to_length[human_segment_idx] += 1

                    # Maintain gripper state for each robot but only update the active robot with action
                    env_action = [
                        robot.create_action_vector(all_prev_gripper_actions[i])
                        for i, robot in enumerate(env.env.env.robots)
                    ]
                    env_action[self.teleop_device.active_robot] = active_robot.create_action_vector(action_dict)
                    env_action = np.concatenate(env_action)
                    env_action =  np.expand_dims(env_action, axis=0) # add time dimension

                    
                    

                    # Step-wise loop
                    if self.during_teleop_robot_prediction_task is None or self.human_teleop_timestep % 16 == 0:
                        self.during_teleop_robot_prediction_task = asyncio.run_coroutine_threadsafe(
                            self.predict_robot_action(obs, clip_embedding),
                            self.loop
                        )


                    # Await and process prediction if ready
                    if self.during_teleop_robot_prediction_task and self.during_teleop_robot_prediction_task.done():
                        action_pred, bool_CP, baseline_metric, action_pred_infos_result = self.during_teleop_robot_prediction_task.result()
                        self.during_teleop_robot_prediction_task = None  # Reset for future prediction

                        # save current action chunk robot prediction
                        self.during_teleop_robot_action_chunk_prediction = action_pred
                        self.current_chunk_timestep = 0
                        self.current_bool_CP = bool_CP
                        dagger_episode_meta['during_teleop_robot_predictions'].append((action_pred, bool_CP, baseline_metric, self.timestep, self.human_teleop_timestep))

                    arbitration_to_robot = 0
                    if begin_mixing_on_human_handback is True and self.during_teleop_robot_action_chunk_prediction is not None and self.current_chunk_timestep < len(self.during_teleop_robot_action_chunk_prediction):
                        current_action_pred = self.during_teleop_robot_action_chunk_prediction[self.current_chunk_timestep]
                        l2_diff = np.linalg.norm(env_action - current_action_pred)
                        print("L2 diff between env_action and action_pred", l2_diff)

                        arbitration_to_robot = 1 / (1 + np.exp(8 * (l2_diff - 1.0))) * self.current_bool_CP
                        print("arbitration_to_robot", arbitration_to_robot)

                        during_teleop_robot_action_prediction = current_action_pred

                        dagger_episode_meta['arbitration_to_robot'].append(arbitration_to_robot)
                        dagger_episode_meta['robot_pred_during_teleop'].append((current_action_pred, env_action, l2_diff, self.timestep, self.human_teleop_timestep, self.current_chunk_timestep, self.current_bool_CP))

                        # Blend human and robot actions
                        env_action = (1 - arbitration_to_robot) * env_action + arbitration_to_robot * during_teleop_robot_action_prediction
                        self.num_mixing_timesteps += 1
                        if self.num_mixing_timesteps >= 64:
                            begin_mixing_on_human_handback = False
                            self.num_mixing_timesteps = 0
                            print("robot not ready for handover, handing back full control to human")
                    dagger_episode_meta['mixed_teleop_action'].append((env_action, acting_agent, self.timestep, arbitration_to_robot))

                    # Step the environment

                    # pdb.set_trace()
                    obs, _, _, _ = env.step(env_action)
                    dagger_episode_meta['obs_list'].append((obs, acting_agent, self.timestep))
                    dagger_episode_meta['action_list'].append((env_action, acting_agent, self.timestep))
                    env.render()

                    if abs(1-arbitration_to_robot) < 0.1:
                        print("ROBOT CONFIDENT: SWITCHING TO ROBOT")
                        acting_agent = 'robot'
                        self.teleop_device.is_acting_agent = False
                        env.env.env.set_dagger_acting_agent("robot")
                        dagger_episode_meta['switch_initiated_by'].append('robot')
                        begin_mixing_on_human_handback = False

                    
                    if human_segment_timestep >= 16 and human_segment_timestep%16 == 0:
                        self.timestep += 1

                    if env.env.env._check_success() or self.timestep >= self.max_steps:
                        print("finished task")
                        break

                else:
                    # current_gripper_qpos = np.mean(abs(self.env.env.env.env.env._get_observations()['robot0_gripper_qpos']))
                    # print("current_gripper_qpos", current_gripper_qpos)
                    batch = self.convert_observations(self.video_dataset, obs, clip_embedding)
                    batch = {key: value.to(self.device, dtype=torch.float32) for key, value in batch.items()}
                    batch['robot0_eef_pos'] = torch.tensor(obs['robot0_eef_pos']).unsqueeze(0).to(self.device, dtype=torch.float32)
                    batch['robot0_eef_quat'] = torch.tensor(obs['robot0_eef_quat']).unsqueeze(0).to(self.device, dtype=torch.float32)
                    batch['robot0_gripper_qpos'] = torch.tensor(obs['robot0_gripper_qpos']).unsqueeze(0).to(self.device, dtype=torch.float32)

                    action_pred, action_pred_infos_result = self.policy.predict_action_with_infos(batch)
                    action_pred = ((action_pred.detach().cpu().numpy() + 1) / 2) * (self.video_dataset.max - self.video_dataset.min) + self.video_dataset.min
                    action_pred = np.squeeze(action_pred)
                    action_pred = np.hstack((action_pred, [[0, 0, 0, 0, -1]] * action_pred.shape[0]))
                    action_horizon = 16
                    action_pred = action_pred[0:action_horizon]

                    
                    baseline_metric = logpZO_UQ(self.score_network, action_pred_infos_result['global_cond'])
                    print("LOG PZO", baseline_metric)
                    prev_score = baseline_metric.item()

                    # Get CP threshold at current timestep
                    CP_threshold_at_t = CP_band[self.timestep] if self.timestep < len(CP_band) else CP_band[-1]
                    print("CP threshold at t", CP_threshold_at_t)

                    dagger_episode_meta['obs_list'].append((obs, acting_agent, baseline_metric, CP_threshold_at_t, self.timestep))
                    dagger_episode_meta['action_list'].append((action_pred, acting_agent, self.timestep))

                    # Update score history
                    auton_logpzo_scores.append(baseline_metric.item())
        
                    if baseline_metric > CP_threshold_at_t:
                        print("FAILURE DETECTED: SWITCHING TO HUMAN")
                        self.teleop_device.is_acting_agent = True
                        env.env.env.set_dagger_acting_agent("human")
                        dagger_episode_meta['switch_initiated_by'].append('robot')
                        continue



                    obs, reward, done, info = env.step(action_pred, render=True)
                    dagger_episode_meta['n_robot_timesteps'] += 1
                    self.timestep += 1
                    if env.env.env._check_success():
                        print("finished task")
                        break

                print("done rendering")
                done = np.all(done)
                # self.timestep += 1
                # update pbar
                pbar.update(self.n_action_steps)
            pbar.close()
            print("human_segment_to_length", human_segment_to_length)
            # Close the figure after loop
            # plt.ioff()
            # plt.close(fig)
            
            want_to_save = input("Done with this episode, do you want to save the demo? (y/n): ").lower()
            if want_to_save == 'y':
                dagger_episode_complete = True
            else:
                print("trying trajectory again")
                # clear out video buffer
                _ = env.reset()
                
                # for ep_directory in os.listdir(directory):
                # for ep_directory in self.dagger_dir, delete the episode directory
                for ep_directory in os.listdir(self.dagger_dir):
                    ep_directory = os.path.join(self.dagger_dir, ep_directory)
                    if os.path.isdir(ep_directory):
                        print(f"Deleting episode directory {ep_directory}")
                        shutil.rmtree(ep_directory)
                continue

            _ = env.reset()


            
        # env.close()
        # Cleanup
        if nonzero_ac_seen and hasattr(env, "ep_directory"):
            ep_directory = env.env.env.ep_directory
        else:
            ep_directory = None

        hdf5_path = None
        savefilename = f'demo_{dagger_ep_idx}'
        excluded_eps = []
        hdf5_path = gather_single_human_only_dagger_demonstrations_as_hdf5(
                self.dagger_dir, self.processed_dagger_dir, self.env_info, savefilename, excluded_episodes=excluded_eps
            )
        print(f"Gathered Human-only DAgger demonstrations to {hdf5_path} at filename {savefilename}")
        if hdf5_path is not None:
            convert_to_robomimic_format(hdf5_path)
            print(f"Converted human-only DAgger demonstrations to robomimic format at {hdf5_path} at filename {savefilename}")
        else:
            print("no human-only demonstrations gathered, skipping conversion to robomimic format")

        
        hdf5_path = None
        savefilename = f'combined_demo_{dagger_ep_idx}'
        excluded_eps = []
        hdf5_path = gather_single_combined_demonstrations_as_hdf5(
                self.dagger_dir, self.processed_dagger_dir, self.env_info, savefilename, excluded_episodes=excluded_eps
            )
        print(f"Gathered Combined DAgger demonstrations to {hdf5_path} at filename {savefilename}")
        if hdf5_path is not None:
            convert_to_robomimic_format(hdf5_path)
            print(f"Converted combined DAgger demonstrations to robomimic format at {hdf5_path} at filename {savefilename}")
        else:
            print("no combined demonstrations gathered, skipping conversion to robomimic format")

        print("emptying dagger dir")
        # clear dagger dir
        for ep_directory in os.listdir(self.dagger_dir):
            ep_directory = os.path.join(self.dagger_dir, ep_directory)
            if os.path.isdir(ep_directory):
                print(f"Deleting episode directory {ep_directory}")
                shutil.rmtree(ep_directory)
        
        # save dagger_episode_meta as pkl
        with open(os.path.join(self.dagger_dir, f"dagger_episode_meta_{dagger_ep_idx}.pkl"), 'wb') as f:
            pickle.dump(dagger_episode_meta, f)
            print(f"Saved dagger_episode_meta to {os.path.join(self.dagger_dir, f'dagger_episode_meta_{dagger_ep_idx}.pkl')}")

        return hdf5_path
    

    def collect_dagger_rollout_data(self, aggregation_type, savefilename):


        hdf5_path = gather_human_only_dagger_demonstrations_as_hdf5(
                self.dagger_dir, self.processed_dagger_dir, self.env_info
            )
        print(f"Gathered DAgger demonstrations to {hdf5_path}")
        convert_to_robomimic_format(hdf5_path)
        print(f"Converted DAgger demonstrations to robomimic format at {hdf5_path}")
        
        
        return hdf5_path