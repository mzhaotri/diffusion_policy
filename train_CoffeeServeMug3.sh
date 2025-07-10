#!/bin/bash

CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 python finetune.py --config-dir=. --config-name=finetune_lora_after_dagger_robocasa.yaml training.seed=42 task.name='CoffeeServeMug' finetuning.dagger_episode_folder='dagger_episode_0' finetuning.apply_lora_on_obs_encoder=False finetuning.stop_at_epoch=601



CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsFalse_loraonobsFalse_after_dagger' task_name='CoffeeServeMug' dagger.ckpt_name='epoch_360'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsFalse_loraonobsFalse_after_dagger' task_name='CoffeeServeMug' dagger.ckpt_name='epoch_420'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsFalse_loraonobsFalse_after_dagger' task_name='CoffeeServeMug' dagger.ckpt_name='epoch_480'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsFalse_loraonobsFalse_after_dagger' task_name='CoffeeServeMug' dagger.ckpt_name='epoch_540'
CUDA_VISIBLE_DEVICES=2 python evaluate.py --config-name=evaluate_lora_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_lora_rank256_scale0.2_freezeobsFalse_loraonobsFalse_after_dagger' task_name='CoffeeServeMug' dagger.ckpt_name='epoch_600'


