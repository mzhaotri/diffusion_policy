import os
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
from diffusion_policy.common.robocasa_util import create_environment
import pdb
import robosuite
import torchvision.transforms as T


def create_env(dataset_path, env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = create_environment(dataset_path=dataset_path)

    return env

def plot_and_save_images(obs, output_dir, timestep = 0, start=0):
    """
    Plots and saves images from a tensor of shape (A, 3, W, H).
    
    Parameters:
    tensor (torch.Tensor): Input tensor of shape (A, 3, W, H) with integer values between 0 and 255.
    save_dir (str): Directory where images will be saved.
    """
    # Ensure the save directory exists
    # Convert the tensor to a format suitable for plotting
    transform = T.ToPILImage()
    for key, img_tensor in obs.items():
        if len(img_tensor.shape) < 5:
            continue
        img_tensor = torch.from_numpy(img_tensor[:, 0, :, :, :])
        for i, img_tensor_ in enumerate(img_tensor):
            save_dir = os.path.join(output_dir, 'media', f'test_{start+i}')
            os.makedirs(save_dir, exist_ok=True)
            img = transform(img_tensor_)
            # Save the image
            img.save(os.path.join(save_dir, f'{key}_t={timestep}.png'))

class RobocasaFDImageRunner(BaseImageRunner):
    """
    Robocasa envs also enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = get_env_metadata_from_dataset(dataset_path)

        for val in shape_meta['obs'].values():
            if len(val['shape']) == 3:
                # To get higher resolution images!
                env_meta['env_kwargs']['camera_heights'] = val['shape'][-1]
                env_meta['env_kwargs']['camera_widths'] = val['shape'][-1]
                break
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        # load initial states from robocasa dataset
        self.load_initial_states(dataset_path)

        def env_fn():
            robomimic_env = create_env(dataset_path=dataset_path, env_meta=env_meta, shape_meta=shape_meta)
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            
            # robomimic_env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobocasaImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    dataset_path=dataset_path,
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobocasaImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        

        # train
        with h5py.File(dataset_path, 'r') as f:
            # pdb.set_trace()
            dataset_demos = list(f['data'].keys())


            for i in range(n_train):
                train_demo_key = dataset_demos[i]
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis

                init_state = self.demo_idx_to_initial_state[train_idx]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobocasaImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobocasaImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        env = SyncVectorEnv(env_fns)


        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def load_initial_states(self, dataset_path):
        self.demo_idx_to_initial_state = {}
        with h5py.File(dataset_path, 'r') as f:

            # list of all demonstration episodes (sorted in increasing number order)
            demos = list(f["data"].keys())

            inds = np.argsort([int(elem[5:]) for elem in demos])
            demos = [demos[i] for i in inds]

            for ind in range(len(demos)):
                ep = demos[ind]
                
                # prepare initial state to reload from
                states = f["data/{}/states".format(ep)][()]
                initial_state = dict(states=states[0])
                initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
                initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)
                self.demo_idx_to_initial_state[ind] = initial_state



    def run(self, policy: BaseImagePolicy, modify=False):
        '''
        The logic:
        horizon_now = 0
        while horizon_now < max_steps:
            for n_envs rollouts in rollouts:
                - get "env_action" and corresponding log_p of length n_envs, where env_action is of shape (n_envs, T_a, action_dim)
                - horizon_now += n_envs
        Then, for each rollout, we get max_steps // T_a log_p values, which we can plot to see how the log_p changes over time
        between success and failure rollouts
        '''
        device = policy.device
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills) 
        n_chunks = math.ceil(n_inits / n_envs) 

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_logpZO = [None] * n_inits # Stores logpZO for all rollout across all steps

        # Loop over batches of environments
        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs
            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])
            
            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()
            
            env_name = self.env_meta['env_name']
            # max_steps = max time horizon
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            logpZO_local_slices = []

            reward_local_slices = []
            modify_again = True if modify else False
            save_freq = 1 # Will be updated
            STAC_prev_actions = None
            while not done:
                actual_t = pbar.n
                render_obs_key = self.render_obs_key
                if (modify and modify_again) and (actual_t >= self.modify_t):
                    print("Modifying the environment...")
                    # Dynamically modify the existing environments in place
                    delta = 0.1 if 'tool' not in self.task_name else 0.02
                    env.call_each('modify_environment', args_list=[(delta, render_obs_key)] * len(env.env_fns))
                    modify_again = False  # Only modify once
                                    
                # Save some observation images when new action is predicted
                if actual_t % save_freq == 0:
                    plot_and_save_images(obs, self.output_dir, 
                                         timestep=actual_t, start=start)
                    
                # create obs dict
                np_obs_dict = dict(obs)
                # Resize image if the input has lenth 5 for policy network
                from scipy.ndimage import zoom
                def resize_image_array_nd(array, new_shape):
                    """
                    Resize a numpy array of shape (N1, N2, 3, a, b) to (N1, N2, 3, c, d) with interpolation.

                    Parameters:
                    array (numpy.ndarray): The input array of shape (N1, N2, 3, a, b).
                    new_shape (tuple): The desired shape (N1, N2, 3, c, d).

                    Returns:
                    numpy.ndarray: The resized array of shape (N1, N2, 3, c, d).
                    """
                    if array.shape == new_shape:
                        print("The input array already has the desired shape.")
                        return array
                    
                    if array.shape[2] != 3:
                        raise ValueError("The third dimension of the input array must be 3 (color channels)")

                    N1, N2, _, a, b = array.shape
                    _, _, _, c, d = new_shape

                    # Initialize an empty array with the new shape
                    resized_array = np.empty((N1, N2, 3, c, d), dtype=array.dtype)

                    # Calculate the zoom factors for the spatial dimensions
                    zoom_factors = (
                        1,  # Keep the first dimension (color channels) the same
                        c / a,  # Scale factor for the height dimension
                        d / b   # Scale factor for the width dimension
                    )

                    # Iterate over the first two dimensions and resize each (3, a, b) sub-array
                    for i in range(N1):
                        for j in range(N2):
                            resized_array[i, j] = zoom(array[i, j], zoom_factors, order=1)  # order=1 for bilinear interpolation

                    return resized_array
                for key, val in np_obs_dict.items():
                    if len(val.shape) == 5:
                        np_obs_dict[key] = resize_image_array_nd(val, (*val.shape[:3], self.curr_shape, self.curr_shape))
                ####
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)

                # import elb
                import sys
                sys.path.append('../../UQ_baselines/')
                import eval_load_baseline as elb
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    # policy.store_var_AO = True if self.num_rep == 0 else False
                    action_dict = policy.predict_action(obs_dict)
                
                baseline_metric = elb.logpZO_UQ(self.baseline_model_logpZO, 
                                                     action_dict['global_cond'], 
                                                     task_name = self.task_name)
                logpZO_local_slices.append(baseline_metric)
                
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)
                # print(f'Action shape for the current local slice: {env_action.shape}')
                obs, reward, done, info = env.step(env_action)
                # reward is a list of 0/1, length = n_envs (if 1 then done)
                # done is a list of booleans, length = n_envs (if 1 then means no more simulation, not necessarily success)
                reward_local_slices.append(torch.from_numpy(reward))
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
                save_freq = action.shape[1]
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            logpZO_local_slices = torch.stack(logpZO_local_slices, dim=1) # (n_envs, max_steps // T_p)
            all_logpZO[this_global_slice] = logpZO_local_slices
            
            if modify and modify_again == False:
                env.call_each('modify_environment', args_list=[(-delta, render_obs_key)] * len(env.env_fns))
        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            def helper(input, i):
                return [f'{float(x.item()):.4f}' for x in input[i]]
            # max reward, log_p_cond, log_p_marginal, reward
            log_data[prefix+f'sim_max_reward_{seed}'] = [max_reward, 
                                                         # Baseline

                                                         '/'.join(map(str, helper(all_logpZO, i))),
                                                         
                                                         ]
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction