if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import copy
import random
import pickle
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.clip_dataset import InMemoryVideoDataset
import dill
import json
from filelock import FileLock
import imageio
import open_clip
from collections import deque
from termcolor import colored
import robosuite
from scipy.spatial.transform import Rotation as R
from robocasa.utils.dataset_registry import get_ds_path
# from robocasa.utils.eval_utils import create_eval_env_modified
# import robocasa.utils.eval_utils
from hydra.core.hydra_config import HydraConfig
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper, DAggerDataCollectionWrapper
from robosuite.utils.binding_utils import MjRenderContext

OmegaConf.register_new_resolver("eval", eval, replace=True)
import mujoco
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pdb
from diffusion_policy.failure_detection.UQ_baselines.CFM.net_CFM import get_unet
import diffusion_policy.failure_detection.UQ_baselines.data_loader as data_loader
import robocasa.utils.robomimic.robomimic_env_utils as EnvUtils
from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS
from robocasa.utils.env_utils import create_env, run_random_rollouts
import numpy as np
from copy import deepcopy
from omegaconf import open_dict
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

def inject_camera_mounts(xml_str):
    # Insert our camera mount bodies just before </body> for mobilebase0_support
    insert_after = "<body name=\"mobilebase0_support\""
    insert_index = xml_str.find(insert_after)
    if insert_index == -1:
        raise ValueError("Could not find mobilebase0_support in MJCF!")

    # Find closing tag for the body (safe heuristic)
    body_close_index = xml_str.find("</body>", insert_index)
    if body_close_index == -1:
        raise ValueError("Could not find </body> for mobilebase0_support")

    # Camera mount body XML
    camera_mounts = """
    <body name="robot0_agentview_mount_left" pos="-0.52755 0.38896 1.08304">
        <geom type="sphere" size="0.001" rgba="0 0 0 0" contype="0" conaffinity="0"/>
    </body>
    <body name="robot0_agentview_mount_right" pos="-0.46641 -0.40810 1.00255">
        <geom type="sphere" size="0.001" rgba="0 0 0 0" contype="0" conaffinity="0"/>
    </body>
    """

    return xml_str[:body_close_index] + camera_mounts + xml_str[body_close_index:]





def render_camera_mujoco(env, camera_name, width=256, height=256, save_path=None):
    model = env.sim.model._model
    data = env.sim.data._data

    # Set up renderer
    option = mujoco.MjvOption()
    cam = mujoco.MjvCamera()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    # Set camera by name
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

    # Viewport and image buffer
    viewport = mujoco.MjrRect(0, 0, width, height)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    # Render scene
    mujoco.mjv_updateScene(model, data, option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    mujoco.mjr_render(viewport, scene, context)
    mujoco.mjr_readPixels(rgb, None, viewport, context)

    # Flip vertically
    rgb = np.flipud(rgb)

    if save_path:
        Image.fromarray(rgb).save(save_path)

    # show image for 0.001 seconds
    plt.imshow(rgb)
    plt.axis('off')
    plt.pause(0.001)
    plt.clf()

    return rgb


def create_eval_env_modified(
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
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        # renderer = 'mjviewer',
        # render_camera="robot0_agentview_left",
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )
    

    env = robosuite.make(**env_kwargs)

    
    return env

def create_eval_env_modified_old(
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

    layout_and_style_ids = (layout_and_style_ids[id_selection],)

    env_kwargs = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_configs,
        camera_names=camera_names,
        camera_widths=camera_widths,
        camera_heights=camera_heights,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        # renderer = 'mjviewer',
        # render_camera="robot0_agentview_left",
        obj_instance_split=obj_instance_split,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
        layout_and_style_ids=layout_and_style_ids,
        translucent_robot=False,
    )
    

    env = robosuite.make(**env_kwargs)

    
    return env

def get_camera_transform(model, data, cam_name):
    from scipy.spatial.transform import Rotation as R
    cam_id = model.camera_name2id(cam_name)
    cam_local_pos = model.cam_pos[cam_id]
    cam_local_quat = model.cam_quat[cam_id]
    quat = np.roll(cam_local_quat, -1)  # convert [w,x,y,z] → [x,y,z,w]
    cam_local_mat = R.from_quat(quat).as_matrix()
    cam_body_id = model.cam_bodyid[cam_id]

    if cam_body_id == -1:
        world_pos = cam_local_pos
        world_mat = cam_local_mat
    else:
        body_pos = data.xpos[cam_body_id]
        body_mat = data.xmat[cam_body_id].reshape(3, 3)
        world_pos = body_pos + body_mat @ cam_local_pos
        world_mat = body_mat @ cam_local_mat

    T = np.eye(4)
    T[:3, :3] = world_mat
    T[:3, 3] = world_pos
    return T




TASK_NAME_TO_HUMAN_PATH = {'PnPCabToCounter': "../robocasa/datasets_first/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams_im256.hdf5",
                           'PnPSinkToCounter': "../robocasa/datasets_first/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams_im256.hdf5",
                           'OpenSingleDoor': "../robocasa/datasets_first/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams_im256.hdf5",
                           'OpenDrawer': "../robocasa/datasets_first/v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03/demo_gentex_im128_randcams_im256.hdf5",
                           'CloseDrawer': "../robocasa/datasets_first/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams_im256.hdf5",
                           'TurnOnStove': "../robocasa/datasets_first/v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02/demo_gentex_im128_randcams_im256.hdf5",
                           'TurnOnSinkFaucet': "../robocasa/datasets_first/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25/demo_gentex_im128_randcams_im256.hdf5",
                           'CoffeePressButton': "../robocasa/datasets_first/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25/demo_gentex_im128_randcams_im256.hdf5",
                            'CoffeeServeMug': "../robocasa/datasets_first/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01/demo_gentex_im128_randcams_im256.hdf5",
                            'TurnOnMicrowave': "../robocasa/datasets_first/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25/demo_gentex_im128_randcams_im256.hdf5",
                            'CloseSingleDoor': "../robocasa/datasets_first/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24/demo_gentex_im128_randcams_im256.hdf5",
                           }


class EvalComputeFDScoresDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        payload = torch.load(open(cfg.ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.payload_cfg = payload['cfg']

        self.payload_cfg.task.dataset.mode = cfg.dataset_mode

        # Read task name and configure human_path and tasks
        task_name = cfg.task_name
        self.task_name = task_name
        with open_dict(self.payload_cfg.task.dataset):
            self.payload_cfg.task.dataset.tasks = {
                task_name: None,
            }
            self.payload_cfg.task.dataset.human_path = TASK_NAME_TO_HUMAN_PATH[task_name]

        for key in self.payload_cfg.task.dataset.tasks:
            self.payload_cfg.task.dataset.tasks[key] = {
                "environment_file": cfg.environment_file,
                "num_experiments": cfg.num_experiments
            }

        cls = hydra.utils.get_class(self.payload_cfg._target_)
        workspace = cls(self.payload_cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.ema_model
        policy.num_inference_steps = self.cfg.num_inference_steps

        self.device = torch.device('cuda')
        policy.eval().to(self.device)
        self.policy = policy
        
        dataset: InMemoryVideoDataset
        dataset = hydra.utils.instantiate(self.payload_cfg.task.dataset)
        self.dataset = dataset

        self.run_dir = HydraConfig.get().run.dir

        ## Get logpZO
        input_dim = 7
        net = get_unet(input_dim)
        # pdb.set_trace()
        ckpt = torch.load(cfg.fail_detect.save_score_network_path)
        net.load_state_dict(ckpt['model'])
        net.eval()

        # move net to device
        net.to(self.device)
        self.score_network = net

    def create_environment_data_from_yaml(self, config, output_json_file):

        # Initialize data structure
        data = {"tasks": {}}

        # Populate the data structure with experiments
        for env_name, env_details in config["tasks"].items():
            num_experiments = env_details.get("num_experiments", 0)
            experiments = {
                f"demo_{i}": {"status": "pending", "success": -1}
                for i in range(num_experiments)
            }
            data["tasks"][env_name] = {
                "environment_file": env_details["environment_file"],
                "experiments": experiments
            }

        # Write the JSON file
        with open(output_json_file, "w") as f:
            json.dump(data, f, indent=4)

        print(f"JSON file '{output_json_file}' created successfully.")

    def create_fd_scores_pickle_data_from_yaml(self, config, output_pickle_file):

        # Initialize data structure
        data = {"tasks": {}}

        # Populate the data structure with experiments
        for env_name, env_details in config["tasks"].items():
            num_experiments = env_details.get("num_experiments", 0)
            experiments = {
                f"demo_{i}": {"status": "pending", 
                              "success": -1,
                              'logpzo_scores':[],
                              'img_observations':[],
                              'obs_embeddings':[],
                              'action_predictions':[],
                            #   'rewards':[],
                              'success_at_times':[]}
                for i in range(num_experiments)
            }
            data["tasks"][env_name] = {
                "environment_file": env_details["environment_file"],
                "experiments": experiments
            }

        # Write the JSON file
        with open(output_pickle_file, "wb") as f:
            pickle.dump(data, f)

        print(f"FAILDETECT pickle file '{output_pickle_file}' created successfully.")

    def get_earliest_pending_experiments(self, json_file, max_experiments):

        # Load the JSON data
        with open(json_file, "r") as f:
            data = json.load(f)

        # Iterate through environments
        for env_name, env_details in data["tasks"].items():
            # Filter pending experiments
            pending_experiments = {
                key: experiment
                for key, experiment in env_details["experiments"].items()
                if experiment["status"] == "pending"
            }

            # If there are pending experiments, build and return the structure
            if pending_experiments:
                # Limit to max_experiments
                limited_pending_experiments = dict(
                    list(pending_experiments.items())[:max_experiments]
                )
                return {
                    "tasks": {
                        env_name: {
                            "environment_file": env_details["environment_file"],
                            "experiments": limited_pending_experiments
                        }
                    }
                }

        return None
    
    def get_fd_earliest_pending_experiments(self, fd_pickle_file, max_experiments):

        # Load the JSON data
        with open(fd_pickle_file, "rb") as f:
            data = pickle.load(f)

        # Iterate through environments
        for env_name, env_details in data["tasks"].items():
            # Filter pending experiments
            pending_experiments = {
                key: experiment
                for key, experiment in env_details["experiments"].items()
                if experiment["status"] == "pending"
            }

            # If there are pending experiments, build and return the structure
            if pending_experiments:
                # Limit to max_experiments
                limited_pending_experiments = dict(
                    list(pending_experiments.items())[:max_experiments]
                )
                return {
                    "tasks": {
                        env_name: {
                            "environment_file": env_details["environment_file"],
                            "experiments": limited_pending_experiments
                        }
                    }
                }

        return None

    def set_all_status_to_in_progress(self, data):

        # Iterate through environments
        for env in data["tasks"].values():
            # Iterate through experiments within the environment
            for experiment in env["experiments"].values():
                # Update the status
                experiment["status"] = "in_progress"
        
        return data

    def update_json_file(self, json_file, updated_data):

        with open(json_file, "r") as f:
            existing_data = json.load(f)

        def merge_dicts(source, target):

            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    merge_dicts(value, target[key])
                else:
                    target[key] = value

        merge_dicts(updated_data, existing_data)

        with open(json_file, "w") as f:
            json.dump(existing_data, f, indent=4)

        print(f"JSON file '{json_file}' updated successfully.")

    def update_fd_pickle_file(self, fd_pickle_file, updated_data):

        with open(fd_pickle_file, "rb") as f:
            existing_data = pickle.load(f)

        def merge_dicts(source, target):

            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    merge_dicts(value, target[key])
                else:
                    target[key] = value

        merge_dicts(updated_data, existing_data)

        with open(fd_pickle_file, "wb") as f:
            pickle.dump(existing_data, f)

        print(f"FAIL-Detect Pickle file '{fd_pickle_file}' updated successfully.")

    def reset_to(self, env, state):
        """
        Reset to a specific simulator state.

        Args:
            state (dict): current simulator state that contains one or more of:
                - states (np.ndarray): initial state of the mujoco environment
                - model (str): mujoco scene xml

        Returns:
            observation (dict): observation dictionary after setting the simulator state (only
                if "states" is in @state)
        """
        should_ret = False
        if "model" in state:
            if state.get("ep_meta", None) is not None:
                # set relevant episode information
                ep_meta = json.loads(state["ep_meta"])
            else:
                ep_meta = {}
            if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
                env.set_attrs_from_ep_meta(ep_meta)
            elif hasattr(env, "set_ep_meta"):  # newer versions
                env.set_ep_meta(ep_meta)
            # this reset is necessary.
            # while the call to env.reset_from_xml_string does call reset,
            # that is only a "soft" reset that doesn't actually reload the model.
            print("ep_meta", ep_meta)
            env.reset()
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml

                xml = postprocess_model_xml(state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = env.edit_model_xml(state["model"])

            env.reset_from_xml_string(xml)
            env.sim.reset()
            # hide teleop visualization after restoring from model
            # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
            # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
        if "states" in state:
            env.sim.set_state_from_flattened(state["states"])
            env.sim.forward()
            should_ret = True

        # update state as needed
        if hasattr(env, "update_sites"):
            # older versions of environment had update_sites function
            env.update_sites()
        if hasattr(env, "update_state"):
            # later versions renamed this to update_state
            env.update_state()

        # if should_ret:
        #     # only return obs if we've done a forward call - otherwise the observations will be garbage
        #     return get_observation()
        return None

    def convert_observations(self, dataset, left_image_queue, right_image_queue, gripper_image_queue, clip_embedding):
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # axs[0].imshow(left_image_queue[-1])
        # axs[0].set_title('Left Image')
        # axs[1].imshow(right_image_queue[-1])
        # axs[1].set_title('Right Image')
        # axs[2].imshow(gripper_image_queue[-1])
        # axs[2].set_title('Gripper Image')
        # plt.show()

        left_image = np.stack([dataset.convert_frame(frame=frame, swap_rgb=dataset.swap_rgb) for frame in left_image_queue])
        right_image = np.stack([dataset.convert_frame(frame=frame, swap_rgb=dataset.swap_rgb) for frame in right_image_queue])
        gripper_image = np.stack([dataset.convert_frame(frame=frame, swap_rgb=dataset.swap_rgb) for frame in gripper_image_queue])

        # print(left_image.shape)
        # dataset.undo_operations_and_save(left_image, 'left_image.png')
        # dataset.undo_operations_and_save(right_image, 'right_image.png')
        # dataset.undo_operations_and_save(gripper_image, 'gripper_image.png')
        # exit()
        # print(left_image.shape, right_image.shape, gripper_image.shape)
        
                

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

        return {
            "task_description": clip_embedding,
            "left_image": left_image,
            "right_image": right_image,
            "gripper_image": gripper_image,
            }


    def run(self):
        for i in range(0,50):
            print(colored(f"Running experiment {i+1}/50", "green"))
            self.run_single_idx(i)

    def run_single_idx(self, idx):
        
        max_traj_len = self.cfg['max_traj_len']

        action_horizon = self.cfg['execution_horizon']
        demo_number = idx
        
        task_name = self.task_name
        # env = create_env(
            
        #     render_onscreen=True,
        #     # camera_height=256,
        #     # camera_width=256,
        #     seed=None, # set seed=None to run unseeded
        # )
        # from robosuite import load_composite_controller_config

        # Load the desired controller config with default Basic controller
        # controller_config = {'type': 'HYBRID_MOBILE_BASE', 'body_parts': {'right': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping_ratio': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_ratio_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': False, 'interpolation': None, 'ramp_ratio': 0.2, 'gripper': {'type': 'GRIP'}}, 'torso': {'type': 'JOINT_POSITION', 'interpolation': 'null', 'kp': 2000, 'control_delta': False}, 'base': {'type': 'JOINT_VELOCITY', 'interpolation': 'null', 'control_delta': False}}, 'composite_controller_specific_configs': {'body_part_ordering': ['right', 'right_gripper', 'base', 'torso'], 'right': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping_ratio': 1, 'impedance_mode': 'fixed', 'kp_limits': [0, 300], 'damping_ratio_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': False, 'interpolation': None, 'ramp_ratio': 0.2, 'gripper': {'type': 'GRIP'}}, 'torso': {'type': 'JOINT_POSITION', 'interpolation': 'null', 'kp': 2000, 'control_delta': False}, 'base': {'type': 'JOINT_VELOCITY', 'interpolation': 'null', 'control_delta': False}}}

        # config = {
        #     "env_name": task_name,
        #     "robots": "PandaOmron",
        #     "camera_names": [
        #         "robot0_agentview_left",
        #         "robot0_agentview_right",
        #         "robot0_eye_in_hand",
        #     ],
        #     "camera_widths": 256,
        #     "camera_heights": 256,
        #     "controller_configs": controller_config,
        # }
        # env = robosuite.make(
        #     **config,
        #     has_renderer=False,
        #     has_offscreen_renderer=True,
        #     ignore_done=False,
        #     use_camera_obs=False,
        #     # control_freq=20,
        #     layout_and_style_ids=((1, 1), (2, 2), (4, 4), (6, 9), (7, 10)),
        #     renderer='mjviewer',
        # )

        with open("datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im256_randcams_100_train_envs.pkl", "rb") as pickle_file:
            environment_data = pickle.load(pickle_file)

        # env = robosuite.make(**environment_data['env_kwargs'])

        max_traj_len = self.cfg['max_traj_len']
        camera_names = environment_data['env_kwargs']['camera_names']   # ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']
        camera_height = round(self.payload_cfg.task.dataset['frame_height']/self.payload_cfg.task.dataset['aug']['crop'])
        camera_width = round(self.payload_cfg.task.dataset['frame_width']/self.payload_cfg.task.dataset['aug']['crop'])
        pred_horizon = self.payload_cfg.task['action_horizon']
        action_horizon = self.cfg['execution_horizon']

        environment_data['env_kwargs']['has_renderer'] = True
        environment_data['env_kwargs']["renderer"] = "mjviewer"
        env = create_eval_env_modified(env_name=task_name, controller_configs=environment_data['env_kwargs']['controller_configs'], id_selection=demo_number//10, seed=idx)
        # pdb.set_trace()

        # Wrap this with visualization wrapper
        env = VisualizationWrapper(env)

        

        # Grab reference to controller config and convert it to json-encoded string
        # pdb.set_trace()
        # env_info = json.dumps(config)

        # wrap the environment with data collection wrapper
        tmp_directory = self.run_dir
        env = DAggerDataCollectionWrapper(env, tmp_directory)

        # Make indicator sites fully transparent
        indicators = ['gripper0_right_grip_site', 'gripper0_right_grip_site_cylinder']
        for indicator_name in indicators:
            site_id = env.env.sim.model.site_name2id(indicator_name)
            env.env.sim.model.site_rgba[site_id] = [1, 0, 0, 0.0]

        # sphere_id = env.sim.model.site_name2id("gripper0_right_grip_site")
        # cylinder_id = env.sim.model.site_name2id("gripper0_right_grip_site_cylinder")

        # env.sim.model.site_rgba[sphere_id] = [0.0, 0.0, 0.0, 0.0]
        # env.sim.model.site_rgba[cylinder_id] = [0.0, 0.0, 0.0, 0.0]


        # env.render()

        task_description = env.get_ep_meta()["lang"]
        task_description = open_clip.tokenize([task_description]) # returns torch.Size([1, 77])
        with torch.no_grad():
            clip_embedding = self.dataset.lang_model(task_description.to(self.device)).cpu().unsqueeze(0) # returns torch.Size([1, 1, 1024])

        video_path = f'{self.run_dir}/{task_name}_{demo_number}.mp4'
        video_writer = imageio.get_writer(video_path, fps=30)

        left_image_queue = deque(maxlen=self.payload_cfg.task.img_obs_horizon)
        right_image_queue = deque(maxlen=self.payload_cfg.task.img_obs_horizon)
        gripper_image_queue = deque(maxlen=self.payload_cfg.task.img_obs_horizon)

        list_of_logpZO_scores = []
        list_of_img_observations = []
        list_of_obs_embeddings = []
        list_of_action_predictions = []
        # list_of_rewards = []
        list_of_success_at_times = []

        # reset the environment
        env.reset()
        zero_action = np.zeros(env.action_dim)
        env.step(zero_action)

        # get task language
        lang = env.get_ep_meta()["lang"]
        print("Instruction:", lang)
        
        # open matplotlib window for visualizing the camera images
        
        # plt.show()


        for i in range(max_traj_len // action_horizon):
            # action = np.random.randn(*env.action_spec[0].shape) * 0.1
            
            # pdb.set_trace()
            # env.render()  # render on display
            # get three camera images
            # cam1 = obs['robot0_agentview_left_image']
            # cam2 = obs['robot0_agentview_right_image']
            # cam3 = obs['robot0_eye_in_hand_image']
            cam1 = env.sim.render(height=256, width=256, camera_name='robot0_agentview_left')[::-1]
            cam2 = env.sim.render(height=256, width=256, camera_name='robot0_agentview_right')[::-1]
            cam3 = env.sim.render(height=256, width=256, camera_name='robot0_eye_in_hand')[::-1]

            # concatenate the three camera images horizontally
            video_img = [cam1, cam2, cam3]
            left_image_queue.append(video_img[0])
            right_image_queue.append(video_img[1])
            gripper_image_queue.append(video_img[2])

            while(len(left_image_queue)) < self.payload_cfg.task.img_obs_horizon:
                left_image_queue.append(video_img[0])
                right_image_queue.append(video_img[1])
                gripper_image_queue.append(video_img[2])

            batch = self.convert_observations(self.dataset, left_image_queue, right_image_queue, gripper_image_queue, clip_embedding)
            batch = {key: value.to(self.device, dtype=torch.float32) for key, value in batch.items()}

            
            # task_description torch.Size([1, 1, 1024])
            # left_image torch.Size([1, 1, 3, 224, 224])
            # right_image torch.Size([1, 1, 3, 224, 224])
            # gripper_image torch.Size([1, 1, 3, 224, 224])

            action_pred, action_pred_infos_result = self.policy.predict_action_with_infos(batch)
            action_pred = ((action_pred.detach().cpu().numpy() + 1) / 2) * (self.dataset.max - self.dataset.min) + self.dataset.min
            action_pred = np.squeeze(action_pred)
            action_pred = np.hstack((action_pred, [[0, 0, 0, 0, -1]] * action_pred.shape[0]))
            action_pred = action_pred[0:action_horizon]

            # compute scores
            baseline_metric = logpZO_UQ(self.score_network, action_pred_infos_result['global_cond'])

            print(i)
            # pdb.set_trace()

            for step in range(action_pred.shape[0]):
                # print("action_pred[step]", action_pred[step])
                obs, reward, done, info = env.step(action_pred[step])  # take action in the environment
                cam1 = env.sim.render(height=256, width=256, camera_name='robot0_agentview_left')[::-1]
                cam2 = env.sim.render(height=256, width=256, camera_name='robot0_agentview_right')[::-1]
                cam3 = env.sim.render(height=256, width=256, camera_name='robot0_eye_in_hand')[::-1]

                # concatenate the three camera images horizontally
                video_img = [cam1, cam2, cam3]
                video_img = np.concatenate(
                    video_img, axis=1
                )  # concatenate horizontally
                video_writer.append_data(video_img)
                # update the matplotlib window with the new images
                # if i%10==0:
                #     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                #     axs[0].imshow(cam1)
                #     axs[1].imshow(cam2)
                #     axs[2].imshow(cam3)
                #     plt.savefig(f"{self.run_dir}/rollout{idx}_video_frame_{i}_{step}.png")
                #     plt.close()
                if env._check_success():
                    break


            # collect data
            is_success = env._check_success()
            list_of_success_at_times.append(is_success)
            print("LOGPZO: baseline_metric:", baseline_metric)
            list_of_logpZO_scores.append(baseline_metric)
            list_of_img_observations.append([batch['left_image'], batch['right_image'], batch['gripper_image']])
            list_of_obs_embeddings.append(action_pred_infos_result['global_cond'])
            list_of_action_predictions.append(action_pred)
            # list_of_rewards.append(env._get_reward())

            if is_success:
                break

        # save data

        
        fd_score_experiments_data = {
            "tasks": {
                task_name: {
                    "experiments": {
                        idx: {
                            "demo_number": idx,
                            "status": "pending",
                            "success": -1,
                            "logpzo_scores": [],
                            "img_observations": [],
                            "obs_embeddings": [],
                            "action_predictions": [],
                            # "rewards": [],
                            "success_at_times": []  
                        }
                    }
                }
            }
        }

        fd_score_experiments_data['tasks'][task_name]['experiments'][demo_number]['status'] = 'done'
        if is_success:
            fd_score_experiments_data['tasks'][task_name]['experiments'][demo_number]['success'] = 1
        else:
            fd_score_experiments_data['tasks'][task_name]['experiments'][demo_number]['success'] = 0

        fd_score_experiments_data['tasks'][task_name]['experiments'][demo_number]['logpzo_scores'] = list_of_logpZO_scores
        fd_score_experiments_data['tasks'][task_name]['experiments'][demo_number]['img_observations'] = list_of_img_observations
        fd_score_experiments_data['tasks'][task_name]['experiments'][demo_number]['obs_embeddings'] = list_of_obs_embeddings
        fd_score_experiments_data['tasks'][task_name]['experiments'][demo_number]['action_predictions'] = list_of_action_predictions
        # fd_score_experiments_data['tasks'][task_name]['experiments'][demo]['rewards'] = list_of_rewards
        fd_score_experiments_data['tasks'][task_name]['experiments'][demo_number]['success_at_times'] = list_of_success_at_times

        # dump to pickle with demo_idx 
        with open(f"{self.run_dir}/{task_name}_{demo_number}_fd_scores.pkl", "wb") as f:
            pickle.dump(fd_score_experiments_data, f)
        fd_score_experiments_data = {}
            

        print(colored(f"Saved video to {video_path}", "green"))
        print("success?", is_success)
        video_writer.close()

        # close the matplotlib window
        # plt.close(fig)
        
        # close renderer
        # env._renderer.close()
        env.close()
        return 



    def run_all(self):

        experiment_record = f"{self.run_dir}/multi_environment_experiment_record.json"
        experiment_record_lock = f"{experiment_record}.lock"
        lock = FileLock(experiment_record_lock)

        # fd_score_experiment_record = f"{self.run_dir}/multi_environment_experiment_fd_score_record.pkl"
        # fd_score_experiment_record_lock = f"{fd_score_experiment_record}.lock"
        # fd_score_lock = FileLock(fd_score_experiment_record_lock)

        with lock:
            if os.path.exists(experiment_record):
                print("Experiment record already exists.")
            else:
                print("Experiment record does not exist.")
            self.create_environment_data_from_yaml(self.payload_cfg.task.dataset, experiment_record)

            experiments_data = self.get_earliest_pending_experiments(experiment_record, self.cfg.number_of_tasks)
            # if experiments_data is None:
            #     print('All experiments are in progress or completed.')
            #     exit()
            experiments_data = self.set_all_status_to_in_progress(experiments_data)
            self.update_json_file(experiment_record, experiments_data)

        # with fd_score_lock:
        #     if os.path.exists(fd_score_experiment_record):
        #         print("Experiment record already exists.")
        #     else:
        #         print("Experiment record does not exist.")
        #         self.create_fd_scores_pickle_data_from_yaml(self.payload_cfg.task.dataset, fd_score_experiment_record)

        #     fd_score_experiments_data = self.get_fd_earliest_pending_experiments(fd_score_experiment_record, self.cfg.number_of_tasks)
        #     if fd_score_experiments_data is None:
        #         print('All experiments are in progress or completed.')
        #         exit()
        #     fd_score_experiments_data = self.set_all_status_to_in_progress(fd_score_experiments_data)
        #     self.update_fd_pickle_file(fd_score_experiment_record, fd_score_experiments_data)

        task_name = list(experiments_data['tasks'].keys())[0]

        with open("datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im256_randcams_100_train_envs.pkl", "rb") as pickle_file:
            environment_data = pickle.load(pickle_file)

        # env = robosuite.make(**environment_data['env_kwargs'])

        demos = list(experiments_data['tasks'][task_name]['experiments'].keys())

        max_traj_len = self.cfg['max_traj_len']
        camera_names = environment_data['env_kwargs']['camera_names']   # ['robot0_agentview_left', 'robot0_agentview_right', 'robot0_eye_in_hand']
        camera_height = round(self.payload_cfg.task.dataset['frame_height']/self.payload_cfg.task.dataset['aug']['crop'])
        camera_width = round(self.payload_cfg.task.dataset['frame_width']/self.payload_cfg.task.dataset['aug']['crop'])
        pred_horizon = self.payload_cfg.task['action_horizon']
        action_horizon = self.cfg['execution_horizon']

        for demo in demos:

            demo_number = int(demo.replace("demo_", ""))

            env = create_eval_env_modified(env_name=task_name, controller_configs=environment_data['env_kwargs']['controller_configs'], id_selection=demo_number//10)
            
            # initial_state = environment_data['demos'][demo]['initial_state']
            # self.reset_to(env, initial_state)

            obs = env.reset()

            task_description = env.get_ep_meta()["lang"]
            task_description = open_clip.tokenize([task_description]) # returns torch.Size([1, 77])
            with torch.no_grad():
                clip_embedding = self.dataset.lang_model(task_description.to(self.device)).cpu().unsqueeze(0) # returns torch.Size([1, 1, 1024])

            video_path = f'{self.run_dir}/{task_name}_{demo}.mp4'
            video_writer = imageio.get_writer(video_path, fps=30)

            left_image_queue = deque(maxlen=self.payload_cfg.task.img_obs_horizon)
            right_image_queue = deque(maxlen=self.payload_cfg.task.img_obs_horizon)
            gripper_image_queue = deque(maxlen=self.payload_cfg.task.img_obs_horizon)

            list_of_logpZO_scores = []
            list_of_img_observations = []
            list_of_obs_embeddings = []
            list_of_action_predictions = []
            # list_of_rewards = []
            list_of_success_at_times = []

            for i in range(int(max_traj_len/action_horizon)):

                video_img = []
                # visual_obs = env._get_observations()
                for cam_name in camera_names:
                    # env.viewer.make_current()
                    env.sim.forward()
                    # im_og = env.sim.render(
                    #     height=camera_height, width=camera_width, camera_name=cam_name
                    # )[::-1]
                    # pdb.set_trace()
                    im = env.sim.render(
                        height=camera_height, width=camera_width, camera_name=cam_name
                    )[::-1]
                    # 
                    # im = visual_obs[cam_name+'_image']
                    # flip im, currently it is upside down
                    # im = np.flip(im, axis=0)
                    # pdb.set_trace()
                    video_img.append(im)

                left_image_queue.append(video_img[0])
                right_image_queue.append(video_img[1])
                gripper_image_queue.append(video_img[2])

                while(len(left_image_queue)) < self.payload_cfg.task.img_obs_horizon:
                    left_image_queue.append(video_img[0])
                    right_image_queue.append(video_img[1])
                    gripper_image_queue.append(video_img[2])

                batch = self.convert_observations(self.dataset, left_image_queue, right_image_queue, gripper_image_queue, clip_embedding)
                batch = {key: value.to(self.device, dtype=torch.float32) for key, value in batch.items()}
                batch['robot0_eef_pos'] = torch.tensor(obs['robot0_eef_pos']).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
                batch['robot0_eef_quat'] = torch.tensor(obs['robot0_eef_quat']).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
                batch['robot0_gripper_qpos'] = torch.tensor(obs['robot0_gripper_qpos']).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)

                action_pred, action_pred_infos_result = self.policy.predict_action_with_infos(batch)
                action_pred = ((action_pred.detach().cpu().numpy() + 1) / 2) * (self.dataset.max - self.dataset.min) + self.dataset.min
                action_pred = np.squeeze(action_pred)
                action_pred = np.hstack((action_pred, [[0, 0, 0, 0, -1]] * action_pred.shape[0]))
                action_pred = action_pred[0:action_horizon]

                # compute scores
                baseline_metric = logpZO_UQ(self.score_network, action_pred_infos_result['global_cond'])

                print(i)

                for step in range(action_pred.shape[0]):

                    env.step(action_pred[step])

                    # video render
                    video_img = []
                    # visual_obs = env._get_observations()
                    for cam_name in camera_names:
                        # im = visual_obs[cam_name+'_image']
                        # flip im, currently it is upside down
                        # im = np.flip(im, axis=0)
                        # env.viewer.make_current()
                        env.sim.forward()
                        im = env.sim.render(
                            height=camera_height, width=camera_width, camera_name=cam_name
                        )[::-1]
                        video_img.append(im)
                    video_img = np.concatenate(
                        video_img, axis=1
                    )  # concatenate horizontally
                    video_writer.append_data(video_img)

                    if env._check_success():
                        break
                
                # collect data
                is_success = env._check_success()
                list_of_success_at_times.append(is_success)
                print("LOGPZO: baseline_metric:", baseline_metric)
                list_of_logpZO_scores.append(baseline_metric)
                list_of_img_observations.append(batch)
                list_of_obs_embeddings.append(action_pred_infos_result['global_cond'])
                list_of_action_predictions.append(action_pred)
                # list_of_rewards.append(env._get_reward())

                if env._check_success():
                    break

            # save data
            fd_score_experiments_data = {
                "tasks": {
                    task_name: {
                        "environment_file": experiments_data['tasks'][task_name]['environment_file'],
                        "experiments": {
                            demo: {
                                "demo_number": demo_number,
                                "status": "pending",
                                "success": -1,
                                "logpzo_scores": [],
                                "img_observations": [],
                                "obs_embeddings": [],
                                "action_predictions": [],
                                # "rewards": [],
                                "success_at_times": []  
                            }
                        }
                    }
                }
            }

            
            experiments_data['tasks'][task_name]['experiments'][demo]['status'] = 'done'
            fd_score_experiments_data['tasks'][task_name]['experiments'][demo]['status'] = 'done'
            if env._check_success():
                experiments_data['tasks'][task_name]['experiments'][demo]['success'] = 1
                fd_score_experiments_data['tasks'][task_name]['experiments'][demo]['success'] = 1
            else:
                experiments_data['tasks'][task_name]['experiments'][demo]['success'] = 0
                fd_score_experiments_data['tasks'][task_name]['experiments'][demo]['success'] = 0

            print(colored(f"success? {env._check_success()}", 'green'))
            fd_score_experiments_data['tasks'][task_name]['experiments'][demo]['logpzo_scores'] = list_of_logpZO_scores
            fd_score_experiments_data['tasks'][task_name]['experiments'][demo]['img_observations'] = list_of_img_observations
            fd_score_experiments_data['tasks'][task_name]['experiments'][demo]['obs_embeddings'] = list_of_obs_embeddings
            fd_score_experiments_data['tasks'][task_name]['experiments'][demo]['action_predictions'] = list_of_action_predictions
            # fd_score_experiments_data['tasks'][task_name]['experiments'][demo]['rewards'] = list_of_rewards
            fd_score_experiments_data['tasks'][task_name]['experiments'][demo]['success_at_times'] = list_of_success_at_times

            # dump to pickle with demo_idx 
            with open(f"{self.run_dir}/{task_name}_{demo}_fd_scores.pkl", "wb") as f:
                pickle.dump(fd_score_experiments_data, f)
            fd_score_experiments_data = {}
                

            print(colored(f"Saved video to {video_path}", "green"))
            video_writer.close()

            with lock:
                self.update_json_file(experiment_record, experiments_data)

            # env.close()

        pass


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = EvalDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
