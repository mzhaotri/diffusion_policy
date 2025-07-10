# #!/bin/bash
# CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python finetune.py --config-dir=. --config-name=finetune_after_dagger_robocasa.yaml training.seed=42 task.name='CoffeeServeMug' finetuning.dagger_episode_folder='dagger_episode_0' finetuning.human_only=False finetuning.from_scratch=False finetuning.stop_at_epoch=541


CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsFalse_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='CoffeeServeMug' dagger.ckpt_name='epoch_360'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsFalse_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='CoffeeServeMug' dagger.ckpt_name='epoch_420'
CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsFalse_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='CoffeeServeMug' dagger.ckpt_name='epoch_480'
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsFalse_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='CoffeeServeMug' dagger.ckpt_name='epoch_540'
# CUDA_VISIBLE_DEVICES=0 python evaluate.py --config-name=evaluate_after_dagger_finetuning_robocasa dagger.ckpt_folder='dagger_episode_0_freezeobsFalse_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger' task_name='CoffeeServeMug' dagger.ckpt_name='epoch_600'



