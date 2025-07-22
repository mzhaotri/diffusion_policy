import os
import sys
import pathlib
import hydra
import torch
import random
import numpy as np
import pickle
import tqdm
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.failure_detection.UQ_baselines.CFM.net_CFM import get_unet
import diffusion_policy.failure_detection.UQ_baselines.data_loader as data_loader
import pdb
from torch.nn.functional import one_hot, log_softmax

OmegaConf.register_new_resolver("eval", eval, replace=True)

class FailDetectWorkspace(BaseWorkspace):

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.type = cfg.FD_task_name
        self.fail_detect_data_path = cfg.FD_save_obs_embeds_path
        self.save_score_network_path = cfg.FD_save_score_network_path

    def joint_energy_forward(self, x, y, model):
        

        y_onehot = torch.nn.functional.one_hot(y, num_classes=2).float().to(x.device)
        y_expanded = y_onehot.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # (B, 2, T)
        x_aug = torch.cat([x, y_expanded], dim=1)  # (B, C+2, T)
        x_aug =x_aug.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)
        print(f"x_aug shape: {x_aug.shape}, y shape: {y.shape}")
        timestep = torch.zeros(x.shape[0], dtype=torch.long).to(x.device)
        energy = model(x_aug, timestep=timestep)
        return energy.squeeze()

    def run(self):
        type = self.type
        X, Y = data_loader.get_data(self.fail_detect_data_path, type=type, adjust_shape=True, diffusion=True)
        Y = torch.ones(len(X), dtype=torch.long)
        X = X.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)

        # X = self.reshape_for_unet1d(X, target_channels=7)  # (N, 7, 477)
        input_channels = X.shape[1]  # 7
        train_data = torch.utils.data.TensorDataset(X, Y)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
        ckpt_file = self.save_score_network_path

        net = get_unet(input_channels + 2).to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

        # if os.path.exists(ckpt_file):
        #     ckpt = torch.load(ckpt_file)
        #     net.load_state_dict(ckpt['model'])
        #     optimizer.load_state_dict(ckpt['optimizer'])
        #     starting_epoch = ckpt['epoch']
        #     losses = ckpt['losses']
        # else:
        starting_epoch = 0
        losses = []

        EPOCHS = 300
        t = tqdm.trange(starting_epoch, EPOCHS)
        for i in t:
            print(f"Training epoch {i+1}/{EPOCHS}")

            net.train()
            loss_i = []
            for (x_batch, y_batch) in tqdm.tqdm(train_loader, desc='Training Batches'):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                # print(f"Batch shape: {x_batch.shape}, Labels shape: {y_batch.shape}")
                
                optimizer.zero_grad()

                energy_pos = self.joint_energy_forward(x_batch, y_batch, net)
                all_labels = torch.tensor([0, 1], device=self.device)
                energy_all = torch.stack([
                    self.joint_energy_forward(x_batch, torch.full_like(y_batch, lbl), net)
                    for lbl in all_labels
                ], dim=1)

                logZ = torch.logsumexp(-energy_all, dim=1)
                loss = (energy_pos + logZ).mean()

                if torch.isnan(loss):
                    raise ValueError(f"NaN loss at epoch {i}")

                loss.backward()
                optimizer.step()

                loss_i.append(loss.item())

            losses.append(sum(loss_i) / len(loss_i))
            print(f"Epoch {i+1} loss: {losses[-1]}")

            ckpt = {
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': i+1,
                'losses': losses
            }
            torch.save(ckpt, ckpt_file)

            fig, ax = plt.subplots(figsize=(10, 3))
            ax.set_title("Training loss")
            ax.plot(losses)
            os.makedirs('images', exist_ok=True)
            plt.savefig(f"images/training_loss_{type}.png")
            plt.close('all')

        self.net = net

    def classify_observation(self, obs):
        self.net.eval()
        obs = self.reshape_for_unet1d(obs.unsqueeze(0), target_channels=7).to(self.device)  # shape (1, C, T)

        with torch.no_grad():
            e0 = self.joint_energy_forward(obs, torch.zeros(1, dtype=torch.long).to(self.device), self.net)
            e1 = self.joint_energy_forward(obs, torch.ones(1, dtype=torch.long).to(self.device), self.net)
            probs = torch.softmax(torch.stack([-e0, -e1], dim=1), dim=1)
            prediction = probs.argmax(dim=1).item()
        return prediction, probs.squeeze().cpu().numpy()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = FailDetectWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()