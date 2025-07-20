import json
import os
import pathlib
import numpy as np
import pdb

experiment_name = 'train_diffusion_unet_clip'
experiment_tag = 'ST_OOD_DAgger'
task_name = 'CoffeePressButton'
finetune_type1 = 'dagger_episode_0_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger'
finetune_type2 = 'dagger_episode_0_finetune_w_human_onlyTrue_useonlyoriginalTrue_fromscratchFalse_after_dagger'
finetune_type3 = 'dagger_episode_0_frozenobsFalse_loraonobsFalse_after_dagger'
finetune_type4 = 'dagger_episode_0_frozenobsTrue_loraonobsFalse_after_dagger'
finetune_type5 = 'dagger_episode_0_old0.5_new0.5_fromscratchFalse_after_dagger'
finetune_type6 = 'dagger_episode_0_old0.75_new0.25_fromscratchFalse_after_dagger'
finetune_type7 = 'freeze_obs_encoder_dagger_episode_0_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger'

finetune_type_to_name = {
    finetune_type1: 'Finetune D_0+D_H',
    finetune_type2: 'Finetune D_0',
    finetune_type3: 'Finetune Obs, Lora on Noise',
    finetune_type4: 'Frozen Obs, Lora on Noise',
    finetune_type5: 'Finetune w Old 0.5, New 0.5',
    finetune_type6: 'Finetune w Old 0.75, New 0.25',
    finetune_type7: 'Freeze Obs Encoder, Finetune D_0+D_H'
}




list_of_finetune_types = [finetune_type1, finetune_type3, finetune_type4, finetune_type5, finetune_type6, finetune_type7]
list_of_finetune_types = [finetune_type1, finetune_type3, finetune_type4,finetune_type7]

finetune_type_to_wall_clock_times = {}
epochs = [220, 240, 260, 280, 300]
for finetune_type in list_of_finetune_types:

    epoch_to_mean_wall_clock_time = []
    epoch_to_ste_wall_clock_time = []
    result_path = f'data/outputs/{experiment_tag}_{experiment_name}_{task_name}/{finetune_type}/wall_clock_times.json'
    # read the file line by line and look for failure or success, save 1 or 0
    with open(result_path, 'r') as f:
        wall_clock_times = json.load(f)
    
    # wall_clock_times['epoch_220']['elapsed_minutes']
    for epoch in epochs:
        epoch_key = f'epoch_{epoch}'
        mean_wall_clock_time = wall_clock_times[epoch_key]['elapsed_minutes']
        epoch_to_mean_wall_clock_time.append(mean_wall_clock_time)

    finetune_type_to_wall_clock_times[finetune_type] = epoch_to_mean_wall_clock_time



# pdb.set_trace()
# plot the results as a line graph with error bars
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  
# plt.style.use('seaborn-v0_8-grid')
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 5))


for finetune_type in list_of_finetune_types:
    mean_times = finetune_type_to_wall_clock_times[finetune_type]
    plt.plot(
        epochs, 
        mean_times, 
        label=finetune_type_to_name[finetune_type], 
        marker='o', 
        markersize=4, 
        linewidth=1.5
    )
    
plt.xlabel('Finetuning Epoch')
plt.ylabel('Mean Wall Clock Time (minutes)')
plt.title(f'Wall Clock Time for {task_name}')
plt.xticks([220, 240, 260, 280, 300])
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(5))
plt.legend(prop={'size': 7}, loc='lower right')
# plt.ylim(-0.1, 1.1)
# plt.tight_layout()

plt.show()















