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
from robocasa.utils.dataset_registry import get_ds_path
# from robocasa.utils.eval_utils import create_eval_env_modified
# import robocasa.utils.eval_utils
from hydra.core.hydra_config import HydraConfig

OmegaConf.register_new_resolver("eval", eval, replace=True)

import pdb
import tqdm
import torch
import os
import matplotlib.pyplot as plt
import sys

from diffusion_policy.failure_detection.UQ_baselines.CFM.net_CFM import get_unet
import diffusion_policy.failure_detection.UQ_baselines.data_loader as data_loader


class FailDetectWorkspace(BaseWorkspace):

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.policy_type = 'diffusion'
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.type = cfg.FD_task_name
        self.is_diffusion = True
        self.fail_detect_data_path = cfg.FD_save_obs_embeds_path
        self.save_score_network_path = cfg.FD_save_score_network_path
        

    def run(self):
        type = self.type
        X, Y = data_loader.get_data(self.fail_detect_data_path, type=type, adjust_shape=True, diffusion=self.is_diffusion)
        global_cond_dim = X.shape[1]; input_dim = Y.reshape(Y.shape[0], 32, -1).shape[-1]

        # pdb.set_trace()

        print(f'Current feature shape: {global_cond_dim}')
        train_data = torch.utils.data.TensorDataset(X)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
        # ckpt_file = f'{type}_{args.policy_type}.ckpt'
        ckpt_file = self.save_score_network_path
        
        # choice of model/method
        net = get_unet(input_dim).to(self.device)
        EPOCHS = 200
        optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)
        # if os.path.exists(ckpt_file):
        #     ckpt = torch.load(ckpt_file)
        #     # import pdb; pdb.set_trace()
        #     net.load_state_dict(ckpt['model'])
        #     optimizer.load_state_dict(ckpt['optimizer'])
        #     starting_epochs = ckpt['epoch']
        #     losses = ckpt['losses']
        # else:
        starting_epochs = 0
        losses = []

        t = tqdm.trange(starting_epochs, EPOCHS)
        for i in t:
            print(f"Training epoch {i+1}/{EPOCHS}")
            # Save checkpoint before to avoid nan loss
            ckpt = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': i+1,
                'losses': losses
            }
            torch.save(ckpt, ckpt_file)

            net.train()
            loss_i = []
            for (x_batch, ) in tqdm.tqdm(train_loader, desc='Training Batches'):
                observation = x_batch.to(self.device)
                optimizer.zero_grad()
                x0, x1 = observation, torch.randn_like(observation).to(self.device)
                vtrue = x1 - x0
                cont_t = torch.rand(len(x1),).to(self.device)
                cont_t = cont_t.view(-1, *[1 for _ in range(len(observation.shape)-1)])
                xnow = x0 + cont_t * vtrue
                time_scale = 100 # In UNet, which takes discrete time steps
                vhat = net(xnow, (cont_t.view(-1)*time_scale).long())
                loss = (vhat - vtrue).pow(2).mean()
                loss.backward()
                # Terminate if loss is NaN
                if torch.isnan(loss):
                    print("Loss is NaN")
                    raise ValueError(f"NaN loss at epoch {i}")
                loss_i += [loss.item()]
                optimizer.step()
            # pdb.set_trace()
            losses += [sum(loss_i)/len(loss_i)]
            print(f"Epoch {i+1} loss: {losses[-1]}")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.set_title("Training loss")
            ax.plot(losses)
            suffix = type
            os.makedirs('images', exist_ok=True)
            plt.savefig(f"images/training_loss_{suffix}.png")
            plt.close('all')

            


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = FailDetectWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
