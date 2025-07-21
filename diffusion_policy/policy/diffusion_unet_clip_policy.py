from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D, ConditionalResidualBlock1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)

import pdb

class DiffusionUnetTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        # pdb.set_trace()
        obs_feature_dim = np.prod(obs_encoder.output_shape())
        print(f"obs_feature_dim: {obs_feature_dim}")
        # obs_feature_dim = 5769


        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = obs_dict
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)


        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        
        return nsample
    
    def predict_action_with_infos(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = obs_dict
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        # pdb.set_trace()

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)


        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        assert nsample.shape == (B, self.action_horizon, self.action_dim)

        result = {
            'action': nsample,
            'action_pred': nsample,
            'global_cond': global_cond,
        }
        
        return nsample, result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        # pdb.set_trace()
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        # nobs = self.normalizer.normalize(batch['obs'])
        # nactions = self.normalizer['action'].normalize(batch['action'])
        nobs = batch['obs']
        nactions = batch['action']
        # pdb.set_trace()
        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training 
            # more diffusion steps per each obs encoder forward pass
            global_cond = torch.repeat_interleave(global_cond, 
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions, 
                repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss
    

    def compute_loss_for_feature_given_batch(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        # nobs = self.normalizer.normalize(batch['obs'])
        # nactions = self.normalizer['action'].normalize(batch['action'])
        nobs = batch['obs']
        nactions = batch['action']
        # pdb.set_trace()
        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training 
            # more diffusion steps per each obs encoder forward pass
            global_cond = torch.repeat_interleave(global_cond, 
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions, 
                repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        # loss = loss.type(loss.dtype)
        # loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss.mean()

        return loss
    
    def compute_loss_for_feature(self, obs, action, timesteps):
        # normalize input
        # assert 'valid_mask' not in batch
        # nobs = self.normalizer.normalize(batch['obs'])
        # nactions = self.normalizer['action'].normalize(batch['action'])
        nobs = obs
        nactions = action
        # pdb.set_trace()
        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training 
            # more diffusion steps per each obs encoder forward pass
            global_cond = torch.repeat_interleave(global_cond, 
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions, 
                repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        # timesteps = torch.randint(
        #     0, self.noise_scheduler.config.num_train_timesteps, 
        #     (nactions.shape[0],), device=trajectory.device
        # ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
    

    def is_first_conv1d_block_in_down_modules(self, module_name):
        return module_name.startswith('down_modules.0')

    def get_max_rank_of_noise_pred_net(self):
        return self.get_max_rank_of_network(self.noise_pred_net, layers_to_exclude=self.layers_to_exclude_in_noise_pred_net)

    def get_max_rank_of_network(self, network, layers_to_exclude=[]):
        for name, module in network.named_children():
            # check if it is excluded
            module_full_name = f"{network.__class__.__name__}.{name}"
            #print("layers_to_exclude:", layers_to_exclude)
            if module_full_name in layers_to_exclude:
                #print("module_full_name:", module_full_name)
                continue
            if isinstance(module, nn.Conv1d):
                curr_max_rank = min(module.in_channels, module.out_channels)
                # print("curr_max_rank:", curr_max_rank)
                if curr_max_rank > self.max_rank:
                    self.max_rank = curr_max_rank
            else:
                self.get_max_rank_of_network(module, layers_to_exclude=layers_to_exclude)

    def get_layers_to_exclude_in_noise_pred_net(self):
        layers_to_exclude = []
        network = self.noise_pred_net
        for name, module in network.named_modules():
            if isinstance(module, ConditionalResidualBlock1D):
                # Exclude residual_conv and cond_encoder in ConditionalResidualBlock1D
                layers_to_exclude.append(f"{name}.residual_conv")
                layers_to_exclude.append(f"{name}.cond_encoder")
            elif isinstance(module, (Downsample1d, Upsample1d)):
                # Exclude Downsample1d and Upsample1d layers
                layers_to_exclude.append(name)
            elif isinstance(module, Conv1dBlock):
                # Optionally exclude the first Conv1dBlock in down_modules
                if self.is_first_conv1d_block_in_down_modules(name):
                    layers_to_exclude.append(name)
            elif name == 'final_conv':
                # Exclude final_conv
                layers_to_exclude.append(name)
        return layers_to_exclude