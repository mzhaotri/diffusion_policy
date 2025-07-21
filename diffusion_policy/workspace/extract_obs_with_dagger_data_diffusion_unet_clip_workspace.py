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
from torch.utils.data import DataLoader
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_clip_policy import DiffusionUnetTimmPolicy
from diffusion_policy.dataset.clip_dataset import InMemoryVideoDataset
from diffusion_policy.model.vision.clip_obs_encoder import FrozenOpenCLIPImageEmbedder
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator
import dill
OmegaConf.register_new_resolver("eval", eval, replace=True)
import matplotlib.pyplot as plt
from openTSNE import TSNE as OpenTSNE
import open_clip
import pdb
import h5py
import json


TASK_NAME_TO_HUMAN_PATH = {'PnPCabToCounter': "../robocasa/datasets_first/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams_im256.hdf5",
                           'PnPSinkToCounter': "../robocasa/datasets_first/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams_im256.hdf5",
                           'OpenSingleDoor': "../robocasa/datasets_first/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams_im256.hdf5",
                           'OpenDrawer': "../robocasa/datasets_first/v0.1/single_stage/kitchen_drawer/OpenDrawer/2024-05-03/demo_gentex_im128_randcams_im256.hdf5",
                           'CloseDrawer': "../robocasa/datasets_first/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams_im256.hdf5",
                           'TurnOnStove': "../robocasa/datasets_first/v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02/demo_gentex_im128_randcams_im256.hdf5",
                           'TurnOnSinkFaucet': "../robocasa/datasets_first/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25/demo_gentex_im128_randcams_im256.hdf5",
                           'CoffeePressButton': "../robocasa/datasets_first/v0.1/single_stage/kitchen_coffee/CoffeePressButton/2024-04-25/demo_gentex_im128_randcams_im256.hdf5",
                            'CoffeeServeMug': "../robocasa/datasets_first/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01/demo_gentex_im128_randcams_im256.hdf5",
                            # 'CoffeeServeMug': "/home/michellezhao/Downloads/myoriginal50.hdf5",
                            'TurnOnMicrowave': "../robocasa/datasets_first/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25/demo_gentex_im128_randcams_im256.hdf5",
                            'CloseSingleDoor': "../robocasa/datasets_first/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24/demo_gentex_im128_randcams_im256.hdf5",
                            "CloseDoubleDoor": "../robocasa/datasets_first/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29/demo_gentex_im128_randcams_im256.hdf5"
                           }


TASK_NAME_TO_DAGGER_PATH = {
                            'CoffeeServeMug': "data/outputs/ST_OOD_DAgger_train_diffusion_unet_clip_CoffeeServeMug/dagger_episode_0/processed_dagger_data/merged_dagger_data.hdf5",
                            # 'CoffeeServeMug': "/home/michellezhao/Downloads/human_only_demo.hdf5",

                           }


class ExtractObsDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        lastest_ckpt_path = cfg.training.FD_base_policy_ckpt_path
        payload = torch.load(open(lastest_ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        self.payload_cfg = payload['cfg']
        # self.payload_cfg.task_description = 'closedrawer'

        # self.payload_cfg.task.dataset.mode = cfg.dataset_mode

        cls = hydra.utils.get_class(self.payload_cfg._target_)
        workspace = cls(self.payload_cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.ema_model
        policy.num_inference_steps = self.cfg.training.eval_num_inference_steps

        self.device = torch.device('cuda')
        policy.eval().to(self.device)
        self.policy = policy

        # Read task name and configure human_path and tasks
        task_name = cfg.task_name
        cfg.task.dataset.tasks = {task_name: None}
        cfg.task.dataset.human_path = TASK_NAME_TO_HUMAN_PATH[task_name]
        
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        self.dataset = dataset
        self.train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # Read task name and configure human_path and tasks
        cfg.task.dataset.human_path = TASK_NAME_TO_DAGGER_PATH[task_name]
        self.cfg = cfg
        
        # dagger_dataset = hydra.utils.instantiate(cfg.task.dataset)
        # self.dagger_dataset = dagger_dataset
        # self.dagger_train_dataloader = DataLoader(dagger_dataset, **cfg.dataloader)

    def get_dagger_success_fails(self):
        full_dagger_auton_success_x = []; full_dagger_auton_success_y = []
        full_dagger_combined_success_x = []; full_dagger_combined_success_y = []
        full_dagger_fail_x = []; full_dagger_fail_y = []
        full_dagger_human_success_x = []; full_dagger_human_success_y = []
        datapath = self.cfg.dagger.dagger_data_path
        processed_datapath = self.cfg.dagger.processed_dagger_data_path
        for demo_num in range(self.cfg.dagger.N_dagger_eps):
            print(f'Processing demo {demo_num}/{self.cfg.dagger.N_dagger_eps}')
            dataset_path_robocasa = f'{datapath}/dagger_episode_meta_{demo_num}.pkl'
            with open(dataset_path_robocasa, "rb") as pickle_file:
                data = pickle.load(pickle_file)
                actors = [x[-2] for x in data['action_list']]
                action_meta_list = data['action_list']
                obs_meta_list = data['obs_list']
                human_idx = 0

                clip_embedding = None
                with h5py.File(f'{processed_datapath}/demo_{demo_num}.hdf5', "r") as f:
                    
                    demo_keys = f['data'].keys()
                    if len(demo_keys) == 0:
                        task_description = 'pick the mug from under the coffee machine dispenser and place it on the counter'
                    # pdb.set_trace()
                    else:
                        task_description = json.loads(f['data'][list(demo_keys)[0]].attrs['ep_meta'])['lang']
                    task_description = open_clip.tokenize([task_description]) # returns torch.Size([1, 77])
                    with torch.no_grad():
                        clip_embedding = self.dataset.lang_model(task_description.to(self.dataset.lang_model.device)).cpu() # returns torch.Size([1, 1024])

                        clip_embedding = clip_embedding.unsqueeze(0)  # Shape: [1, 1, 1024]
                    

                    full_autonomous_success = True if 'human' not in actors else False
                    for t in range(len(action_meta_list)):
                        print(f'Processing timestep {t}/{len(action_meta_list)}')
                        elem = action_meta_list[t]
                        actor = elem[1]
                        if actor == 'human':
                            human_idx += 1
                        next_actor = action_meta_list[t+1][1] if t+1 < len(action_meta_list) else None
                        action = elem[0]
                        obs = obs_meta_list[t][0]
                        left_image = obs['robot0_agentview_left_image']
                        right_image = obs['robot0_agentview_right_image']
                        gripper_image = obs['robot0_eye_in_hand_image']

                        left_image = np.stack([self.dataset.convert_frame(frame=frame, size=(round(self.dataset.frame_width/self.dataset.aug['crop']),round(self.dataset.frame_height/self.dataset.aug['crop'])), swap_rgb=self.dataset.swap_rgb) for frame in left_image])
                        right_image = np.stack([self.dataset.convert_frame(frame=frame, size=(round(self.dataset.frame_width/self.dataset.aug['crop']),round(self.dataset.frame_height/self.dataset.aug['crop'])), swap_rgb=self.dataset.swap_rgb) for frame in right_image])
                        gripper_image = np.stack([self.dataset.convert_frame(frame=frame, size=(round(self.dataset.frame_width/self.dataset.aug['crop']),round(self.dataset.frame_height/self.dataset.aug['crop'])), swap_rgb=self.dataset.swap_rgb) for frame in gripper_image])

                        left_image = torch.tensor(left_image, dtype=torch.float32)
                        right_image = torch.tensor(right_image, dtype=torch.float32)
                        gripper_image = torch.tensor(gripper_image, dtype=torch.float32)


                        # Rescale from [-1, 1] to [0, 1] for transforms
                        left_image = (left_image + 1) / 2
                        right_image = (right_image + 1) / 2
                        gripper_image = (gripper_image + 1) / 2

                        left_image = self.dataset.augmentation_transform(left_image, self.dataset.transform_rgb)
                        right_image = self.dataset.augmentation_transform(right_image, self.dataset.transform_rgb)
                        gripper_image = self.dataset.augmentation_transform(gripper_image, self.dataset.transform_rgb)

                        # Key: task_description, Shape: torch.Size([64, 1, 1024])
                        # Key: left_image, Shape: torch.Size([64, 1, 3, 224, 224])
                        # Key: right_image, Shape: torch.Size([64, 1, 3, 224, 224])
                        # Key: gripper_image, Shape: torch.Size([64, 1, 3, 224, 224])
                        # Key: joint_pos, Shape: torch.Size([64, 1, 7])
                        # Key: gripper_pos, Shape: torch.Size([64, 1, 2])
                        # Action shape: torch.Size([64, 32, 7])

                        # unsqueeze first dimension of images
                        left_image = left_image.unsqueeze(0)  # Shape: [1, 1, 3, 224, 224]
                        right_image = right_image.unsqueeze(0)  # Shape: [1, 1, 3, 224, 224]
                        gripper_image = gripper_image.unsqueeze(0)  # Shape: [1, 1, 3, 224, 224]
                        
                        action = torch.tensor(action, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 32, 7]
                        # pdb.set_trace()


                        batch = {
                            "obs": {
                                "task_description": clip_embedding,
                                "left_image": left_image,
                                "right_image": right_image,
                                "gripper_image": gripper_image,

                            },
                            "action": action,
                        }
                        # pdb.set_trace()
                        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                        mod = self.policy # See "policy" folder


                        # Get observations and actions, these are already normalized
                        nobs = batch['obs']
                        nactions = batch['action']
                        batch_size = next(iter(nobs.values())).shape[0]

                        # condition through global feature
                        nobs_features = mod.obs_encoder(nobs)
                        global_cond = nobs_features.reshape(batch_size, -1)

                        # Take actions
                        trajectory = nactions

                        # # flatten actions
                        trajectory = nactions.reshape(batch_size, -1)
                        # pdb.set_trace()

                        if actor == 'robot' and next_actor == 'human':
                            human_idx = 0
                            # If the previous actor was robot and current is human, then this is a success
                            full_dagger_fail_x.append(global_cond.cpu())
                            full_dagger_fail_y.append(trajectory.cpu())
                        elif full_autonomous_success:
                            full_dagger_auton_success_x.append(global_cond.cpu())
                            full_dagger_auton_success_y.append(trajectory.cpu())
                            full_dagger_combined_success_x.append(global_cond.cpu())
                            full_dagger_combined_success_y.append(trajectory.cpu())
                        elif actor == 'human' and human_idx % 16 == 0:
                            # If the current actor is human, then this is a success
                            full_dagger_combined_success_x.append(global_cond.cpu())
                            full_dagger_combined_success_y.append(trajectory.cpu())
                            full_dagger_human_success_x.append(global_cond.cpu())
                            full_dagger_human_success_y.append(trajectory.cpu())

                        # clear variables
                        del batch, nobs, batch_size, nobs_features, global_cond, left_image, right_image, gripper_image, \
                            action, elem, actor, next_actor, obs, trajectory

                    # delete variables to free memory
                    del data, clip_embedding, actors, action_meta_list, obs_meta_list, \
                        full_autonomous_success, demo_keys, task_description

                
                    full_dagger_fail_x = torch.cat(full_dagger_fail_x, dim=0) if full_dagger_fail_x else torch.empty((0, 1024))
                    full_dagger_fail_y = torch.cat(full_dagger_fail_y, dim=0) if full_dagger_fail_y else torch.empty((0, 32*7))
                    full_dagger_auton_success_x = torch.cat(full_dagger_auton_success_x, dim=0) if full_dagger_auton_success_x else torch.empty((0, 1024))
                    full_dagger_auton_success_y = torch.cat(full_dagger_auton_success_y, dim=0) if full_dagger_auton_success_y else torch.empty((0, 32*7))
                    full_dagger_combined_success_x = torch.cat(full_dagger_combined_success_x, dim=0) if full_dagger_combined_success_x else torch.empty((0, 1024))
                    full_dagger_combined_success_y = torch.cat(full_dagger_combined_success_y, dim=0) if full_dagger_combined_success_y else torch.empty((0, 32*7))
                    full_dagger_human_success_x = torch.cat(full_dagger_human_success_x, dim=0) if full_dagger_human_success_x else torch.empty((0, 1024))
                    full_dagger_human_success_y = torch.cat(full_dagger_human_success_y, dim=0) if full_dagger_human_success_y else torch.empty((0, 32*7))
                    # full_dagger_success_x = torch.cat(full_dagger_success_x, dim=0) if full_dagger_success_x else torch.empty((0, 1024))
                    # full_dagger_fail_y = torch.cat(full_dagger_fail_y, dim=0) if full_dagger
                    torch.save({'X': full_dagger_fail_x, 'Y': full_dagger_fail_y}, f'{self.cfg.training.FD_save_dagger_fail_obs_embeds_path}_{demo_num}.pt')
                    torch.save({'X': full_dagger_auton_success_x, 'Y': full_dagger_auton_success_y}, f'{self.cfg.training.FD_save_dagger_auton_succ_obs_embeds_path}_{demo_num}.pt')
                    torch.save({'X': full_dagger_combined_success_x, 'Y': full_dagger_combined_success_y}, f'{self.cfg.training.FD_save_dagger_combined_succ_obs_embeds_path}_{demo_num}.pt')
                    torch.save({'X': full_dagger_human_success_x, 'Y': full_dagger_human_success_y}, f'{self.cfg.training.FD_save_dagger_human_succ_obs_embeds_path}_{demo_num}.pt')


                    full_dagger_fail_x = []; full_dagger_fail_y = []
                    full_dagger_auton_success_x = []; full_dagger_auton_success_y = []
                    full_dagger_combined_success_x = []; full_dagger_combined_success_y = []
                    full_dagger_human_success_x = []; full_dagger_human_success_y = []


                # pdb.set_trace()
        return 


    def run(self):
        cfg = copy.deepcopy(self.cfg)
        self.get_dagger_success_fails()

        # pdb.set_trace()
        
        # open up pt files and concatenate them for success and fail
        # full_dagger_success_x = []; full_dagger_success_y = []
        # full_dagger_fail_x = []; full_dagger_fail_y = []
        # for demo_num in range(self.cfg.dagger.N_dagger_eps):
        #     print(f'Processing demo {demo_num}/{self.cfg.dagger.N_dagger_eps}')
        #     success_data = torch.load(f'{self.cfg.training.FD_save_dagger_succ_obs_embeds_path}_{demo_num}.pt')
        #     fail_data = torch.load(f'{self.cfg.training.FD_save_dagger_fail_obs_embeds_path}_{demo_num}.pt')
        #     if success_data['X'].numel() > 0:
        #         full_dagger_success_x.append(success_data['X'])
        #         full_dagger_success_y.append(success_data['Y'])
        #     if fail_data['X'].numel() > 0:
        #         full_dagger_fail_x.append(fail_data['X'])
        #         full_dagger_fail_y.append(fail_data['Y'])

        # full_dagger_success_x = torch.cat(full_dagger_success_x, dim=0)
        # # full_dagger_success_y = torch.cat(full_dagger_success_y, dim=0)
        # full_dagger_fail_x = torch.cat(full_dagger_fail_x, dim=0)
        # # full_dagger_fail_y = torch.cat(full_dagger_fail_y, dim=0)
        # print(f'Full Dagger Success X: {full_dagger_success_x.shape}')
        # print(f'Full Dagger Fail X: {full_dagger_fail_x.shape}')
        # torch.save({'X': full_dagger_success_x}, self.cfg.training.FD_save_dagger_succ_obs_embeds_path)
        # torch.save({'X': full_dagger_fail_x}, self.cfg.training.FD_save_dagger_fail_obs_embeds_path)

        # open up pt files and concatenate them for success (auton, combined, human) and fail
        full_dagger_auton_success_x = []; full_dagger_auton_success_y = []
        full_dagger_combined_success_x = []; full_dagger_combined_success_y = []
        full_dagger_human_success_x = []; full_dagger_human_success_y = []
        full_dagger_fail_x = []; full_dagger_fail_y = []
        for demo_num in range(self.cfg.dagger.N_dagger_eps):
            print(f'Processing demo {demo_num}/{self.cfg.dagger.N_dagger_eps}')
            auton_success_data = torch.load(f'{self.cfg.training.FD_save_dagger_auton_succ_obs_embeds_path}_{demo_num}.pt')
            combined_success_data = torch.load(f'{self.cfg.training.FD_save_dagger_combined_succ_obs_embeds_path}_{demo_num}.pt')
            human_success_data = torch.load(f'{self.cfg.training.FD_save_dagger_human_succ_obs_embeds_path}_{demo_num}.pt')
            fail_data = torch.load(f'{self.cfg.training.FD_save_dagger_fail_obs_embeds_path}_{demo_num}.pt')

            if auton_success_data['X'].numel() > 0:
                full_dagger_auton_success_x.append(auton_success_data['X'])
                # full_dagger_auton_success_y.append(auton_success_data['Y'])
            if combined_success_data['X'].numel() > 0:
                full_dagger_combined_success_x.append(combined_success_data['X'])
                # full_dagger_combined_success_y.append(combined_success_data['Y'])
            if human_success_data['X'].numel() > 0:
                full_dagger_human_success_x.append(human_success_data['X'])
                # full_dagger_human_success_y.append(human_success_data['Y'])
            if fail_data['X'].numel() > 0:
                full_dagger_fail_x.append(fail_data['X'])
                # full_dagger_fail_y.append(fail_data['Y'])
        full_dagger_auton_success_x = torch.cat(full_dagger_auton_success_x, dim=0)
        # full_dagger_auton_success_y = torch.cat(full_dagger_auton_success_y, dim=0)
        full_dagger_combined_success_x = torch.cat(full_dagger_combined_success_x, dim=0)
        # full_dagger_combined_success_y = torch.cat(full_dagger_combined_success_y, dim=0)
        full_dagger_human_success_x = torch.cat(full_dagger_human_success_x, dim=0)
        # full_dagger_human_success_y = torch.cat(full_dagger_human_success_y, dim=0)
        full_dagger_fail_x = torch.cat(full_dagger_fail_x, dim=0)
        # full_dagger_fail_y = torch.cat(full_dagger_fail_y, dim=0)       
        print(f'Full Dagger Auton Success X: {full_dagger_auton_success_x.shape}')
        # print(f'Full Dagger Auton Success Y: {full_dagger_auton_success_y.shape}')
        print(f'Full Dagger Combined Success X: {full_dagger_combined_success_x.shape}')
        # print(f'Full Dagger Combined Success Y: {full_dagger_combined_success_y.shape}')
        print(f'Full Dagger Human Success X: {full_dagger_human_success_x.shape}')
        # print(f'Full Dagger Human Success Y: {full_dagger_human_success_y.shape}')
        print(f'Full Dagger Fail X: {full_dagger_fail_x.shape}')
        # print(f'Full Dagger Fail Y: {full_dagger_fail_y.shape}')
        torch.save({'X': full_dagger_auton_success_x}, self.cfg.training.FD_save_dagger_auton_succ_obs_embeds_path)
        torch.save({'X': full_dagger_combined_success_x}, self.cfg.training.FD_save_dagger_combined_succ_obs_embeds_path)
        torch.save({'X': full_dagger_human_success_x}, self.cfg.training.FD_save_dagger_human_succ_obs_embeds_path)
        torch.save({'X': full_dagger_fail_x}, self.cfg.training.FD_save_dagger_fail_obs_embeds_path)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
