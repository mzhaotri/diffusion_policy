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

OmegaConf.register_new_resolver("eval", eval, replace=True)

import pdb
import time
import json


def get_finetuning_dataset(taskname, experiment_tag, dagger_episode_folder='dagger_episode_0'):
    TASK_NAME_TO_ORIGINAL_PATH = {'PnPCabToCounter': "../robocasa/datasets_first/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams_im256.hdf5",
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
                            "PnPStoveToCounter": "../robocasa/datasets_first/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo_gentex_im128_randcams_im256.hdf5"
                           }

    TASK_NAME_TO_MERGED_PATH = {'PnPCabToCounter': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_PnPCabToCounter/{dagger_episode_folder}/processed_dagger_data/merged_dagger_data.hdf5',
                            'PnPSinkToCounter': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_PnPSinkToCounter/{dagger_episode_folder}/processed_dagger_data/merged_dagger_data.hdf5',
                            'OpenSingleDoor': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_OpenSingleDoor/{dagger_episode_folder}/processed_dagger_data/merged_dagger_data.hdf5',
                            'OpenDrawer': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_OpenDrawer/{dagger_episode_folder}/processed_dagger_data/merged_dagger_data.hdf5',
                            'CloseDrawer': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_CloseDrawer/{dagger_episode_folder}/processed_dagger_data/merged_dagger_data.hdf5',
                            'TurnOnStove': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_TurnOnStove/{dagger_episode_folder}/processed_dagger_data/merged_dagger_data.hdf5',
                            'TurnOnSinkFaucet': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_TurnOnSinkFaucet/{dagger_episode_folder}/processed_dagger_data/merged_dagger_data.hdf5',
                            'CoffeePressButton': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_CoffeePressButton/{dagger_episode_folder}/processed_dagger_data/merged_dagger_data.hdf5',
                            'CoffeeServeMug': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_CoffeeServeMug/{dagger_episode_folder}/processed_dagger_data/merged_dagger_data.hdf5',
                            'TurnOnMicrowave': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_TurnOnMicrowave/{dagger_episode_folder}/processed_dagger_data/merged_dagger_data.hdf5',
                            'CloseSingleDoor': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_CloseSingleDoor/{dagger_episode_folder}/processed_dagger_data/merged_dagger_data.hdf5',
                            "PnPStoveToCounter": f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_PnPStoveToCounter/{dagger_episode_folder}/processed_dagger_data/merged_dagger_data.hdf5'
                            }

    TASK_NAME_TO_HUMAN_ONLY_PATH = {'PnPCabToCounter': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_PnPCabToCounter/{dagger_episode_folder}/processed_dagger_data/human_only_demo.hdf5',
                            'PnPSinkToCounter': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_PnPSinkToCounter/{dagger_episode_folder}/processed_dagger_data/human_only_demo.hdf5',
                            'OpenSingleDoor': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_OpenSingleDoor/{dagger_episode_folder}/processed_dagger_data/human_only_demo.hdf5',
                            'OpenDrawer': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_OpenDrawer/{dagger_episode_folder}/processed_dagger_data/human_only_demo.hdf5',
                            'CloseDrawer': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_CloseDrawer/{dagger_episode_folder}/processed_dagger_data/human_only_demo.hdf5',
                            'TurnOnStove': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_TurnOnStove/{dagger_episode_folder}/processed_dagger_data/human_only_demo.hdf5',
                            'TurnOnSinkFaucet': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_TurnOnSinkFaucet/{dagger_episode_folder}/processed_dagger_data/human_only_demo.hdf5',
                            'CoffeePressButton': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_CoffeePressButton/{dagger_episode_folder}/processed_dagger_data/human_only_demo.hdf5',
                                'CoffeeServeMug': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_CoffeeServeMug/{dagger_episode_folder}/processed_dagger_data/human_only_demo.hdf5',
                            'TurnOnMicrowave': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_TurnOnMicrowave/{dagger_episode_folder}/processed_dagger_data/human_only_demo.hdf5',
                            'CloseSingleDoor': f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_CloseSingleDoor/{dagger_episode_folder}/processed_dagger_data/human_only_demo.hdf5',
                            "PnPStoveToCounter": f'data/outputs/{experiment_tag}_train_diffusion_unet_clip_PnPStoveToCounter/{dagger_episode_folder}/processed_dagger_data/human_only_demo.hdf5'
                            }
    
    original_dataset = TASK_NAME_TO_ORIGINAL_PATH[taskname]
    merged_dataset = TASK_NAME_TO_MERGED_PATH[taskname]
    human_only_dataset = TASK_NAME_TO_HUMAN_ONLY_PATH[taskname]

    return original_dataset, merged_dataset, human_only_dataset



class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.task_name = cfg.task_name

        # configure model
        self.model: DiffusionUnetTimmPolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetTimmPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())

        obs_encorder_lr = cfg.optimizer.lr
        if cfg.policy.obs_encoder.pretrained:
            obs_encorder_lr *= 0.1
            print('==> reduce pretrained obs_encorder\'s lr')
        obs_encorder_params = list()
        for param in self.model.obs_encoder.parameters():
            if param.requires_grad:
                obs_encorder_params.append(param)
        print(f'obs_encorder params: {len(obs_encorder_params)}')
        param_groups = [
            {'params': self.model.model.parameters()},
            {'params': obs_encorder_params, 'lr': obs_encorder_lr}
        ]
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=param_groups)
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator(log_with='wandb')
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        # if cfg.training.resume:
        # if lastest_ckpt_path.is_file():
        # configure dataset
        # Read task name and configure human_path and tasks
        task_name = cfg.task_name
        self.task_name = task_name
        cfg.task.dataset.tasks = {
            task_name: None,
        }
        cfg.task.dataset.tasks = {task_name: None}

        dagger_episode_folder = cfg.finetuning.dagger_episode_folder

        original_dataset, merged_dataset, human_only_dataset = get_finetuning_dataset(task_name, cfg.experiment_tag, dagger_episode_folder)
        if cfg.finetuning.human_only:
            cfg.task.dataset.human_path = human_only_dataset
        else:
            cfg.task.dataset.human_path = merged_dataset

        if cfg.finetuning.use_only_original:
            cfg.task.dataset.human_path = original_dataset

        dataset = hydra.utils.instantiate(cfg.task.dataset)

        # testing
        # cfg.task.dataset.human_path = original_dataset
        # original_dataset = hydra.utils.instantiate(cfg.task.dataset)

        # og_img = original_dataset[100]['obs']['left_image'][0]
        # # move channels to last dimension
        # og_img = og_img.permute(1, 2, 0).numpy()
        # plt.imshow(og_img)
        

        # cfg.task.dataset.human_path = human_only_dataset
        # human_only_dataset = hydra.utils.instantiate(cfg.task.dataset)

        # ho_img = human_only_dataset[100]['obs']['left_image'][0]
        # # move channels to last dimension
        # ho_img = ho_img.permute(1, 2, 0).numpy()
        # plt.imshow(ho_img)

        self.start_training_time = time.time()
        self.wall_clock_times = {'start': self.start_training_time}


        # pdb.set_trace()
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        print('train dataset:', len(dataset), ', ' 'train dataloader:', len(train_dataloader))

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        if cfg.finetuning.from_scratch is False:
            lastest_ckpt_path = cfg.ckpt_path
            accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
            self.load_checkpoint(path=lastest_ckpt_path)
        else:
            accelerator.print(f"Training from scratch, no checkpoint loaded.")


        # for parameter in self.model.parameters(): freeze all but the obs_encoder
        # pdb.set_trace()
        if cfg.finetuning.freeze_obs_encoder:
            accelerator.print("Freezing obs_encoder parameters.")
            for name, parameter in self.model.named_parameters():
                if 'obs_encoder' in name:
                    parameter.requires_grad = False
                    accelerator.print(f"Freezing parameter: {name}")
                else:
                    accelerator.print(f"Unfreezing parameter: {name}")

        # Reset optimizer
        obs_encorder_lr = cfg.optimizer.lr
        obs_encorder_params = list()
        for param in self.model.obs_encoder.parameters():
            if param.requires_grad:
                obs_encorder_params.append(param)
        print(f'obs_encorder params: {len(obs_encorder_params)}')
        param_groups = [
            {'params': self.model.model.parameters()},
            {'params': obs_encorder_params, 'lr': obs_encorder_lr}
        ]
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=param_groups)
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        # device = torch.device(cfg.training.device)
        # self.model.to(device)
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        # optimizer_to(self.optimizer, device)

        # accelerator
        train_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, self.model, self.optimizer, lr_scheduler
        )
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        # always use the latest batch
                        train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # # run rollout
                # if (self.epoch % cfg.training.rollout_every) == 0:
                #     runner_log = env_runner.run(policy)
                #     # log all
                #     step_log.update(runner_log)

                # run validation
                # if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0 and accelerator.is_main_process:
                #     with torch.no_grad():
                #         val_losses = list()
                #         with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                #                 leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                #             for batch_idx, batch in enumerate(tepoch):
                #                 batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                #                 loss = self.model(batch)
                #                 val_losses.append(loss)
                #                 if (cfg.training.max_val_steps is not None) \
                #                     and batch_idx >= (cfg.training.max_val_steps-1):
                #                     break
                #         if len(val_losses) > 0:
                #             val_loss = torch.mean(torch.tensor(val_losses)).item()
                #             # log epoch average validation loss
                #             step_log['val_loss'] = val_loss
                
                # def log_action_mse(step_log, category, pred_action, gt_action):
                #     B, T, _ = pred_action.shape
                #     pred_action = pred_action.view(B, T, -1, 10)
                #     gt_action = gt_action.view(B, T, -1, 10)
                #     step_log[f'{category}_action_mse_error'] = torch.nn.functional.mse_loss(pred_action, gt_action)
                #     step_log[f'{category}_action_mse_error_pos'] = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
                #     step_log[f'{category}_action_mse_error_rot'] = torch.nn.functional.mse_loss(pred_action[..., 3:9], gt_action[..., 3:9])
                #     step_log[f'{category}_action_mse_error_width'] = torch.nn.functional.mse_loss(pred_action[..., 9], gt_action[..., 9])
                # # run diffusion sampling on a training batch
                # if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                #     with torch.no_grad():
                #         # sample trajectory from training set, and evaluate difference
                #         batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                #         gt_action = batch['action']
                #         pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                #         log_action_mse(step_log, 'train', pred_action, gt_action)

                #         if len(val_dataloader) > 0:
                #             val_sampling_batch = next(iter(val_dataloader))
                #             batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                #             gt_action = batch['action']
                #             pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                #             log_action_mse(step_log, 'val', pred_action, gt_action)

                #         del batch
                #         del gt_action
                #         del pred_action
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    current_time = time.time()
                    self.wall_clock_times[f'epoch_{self.epoch}'] = {'clock': current_time, 'elapsed': current_time - self.start_training_time}
                    # elapsed in minutes
                    elapsed_minutes = (current_time - self.start_training_time) / 60.0
                    self.wall_clock_times[f'epoch_{self.epoch}']['elapsed_minutes'] = elapsed_minutes

                    # save to json
                    wall_clock_times_path = os.path.join(self.output_dir, 'wall_clock_times.json')
                    with open(wall_clock_times_path, 'w') as f:
                        json.dump(self.wall_clock_times, f, indent=4)

                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    # if topk_ckpt_path is not None:
                    #     self.save_checkpoint(path=topk_ckpt_path)

                    checkpoint_filename = f"{topk_manager.save_dir}/epoch_{self.epoch}.ckpt"
                    self.save_checkpoint(path=checkpoint_filename)

                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
                if self.epoch > cfg.finetuning.stop_at_epoch:
                    accelerator.print(f"Stopping training at epoch {self.epoch} as per configuration.")
                    break

        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
