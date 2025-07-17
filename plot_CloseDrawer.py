import json
import os
import pathlib
import numpy as np
import pdb

experiment_name = 'train_diffusion_unet_clip'
experiment_tag = 'ST_OOD_DAgger'
task_name = 'CloseDrawer'

base_policy = 'base_policy'
finetune_type1 = 'dagger_episode_0_freezeobsTrue_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger'
finetune_type2 = 'dagger_episode_0_freezeobsFalse_finetune_w_human_onlyFalse_useonlyoriginalFalse_fromscratchFalse_after_dagger'
finetune_type3 = 'dagger_episode_0_lora_rank256_scale0.2_freezeobsFalse_loraonobsFalse_after_dagger'
finetune_type4 = 'dagger_episode_0_lora_rank256_scale0.2_freezeobsTrue_loraonobsFalse_after_dagger'


finetune_type_to_name = {
    finetune_type1: 'Frozen Obs, Finetune D_0+D_H',
    finetune_type2: 'Finetune Obs, Finetune D_0+D_H',
    finetune_type3: 'Finetune Obs, Lora on Noise',
    finetune_type4: 'Frozen Obs, Lora on Noise',
}


# get base policy sr and std error
# base_policy_path = f'data/outputs/{experiment_tag}_{experiment_name}_{task_name}/{base_policy}/eval_compute_rollout_scores_latest/successes.txt'
# successes = []
# with open(base_policy_path, 'r') as f:
#     for line in f:
#         if 'Failure' in line:
#             successes.append(0)
#         elif 'Success' in line:
#             successes.append(1)
#         else:
#             print(f'Unknown line: {line}')
# print("successes", successes)
# base_mean_success = np.mean(successes)
# base_ste_success = np.std(successes) / np.sqrt(len(successes))
base_mean_success = 0.8
base_ste_success = 0.06

print(f'Base Policy Success Rate: {base_mean_success:.2f} ± {base_ste_success:.2f}')

# epochs = [320, 340, 360, 380, 400, 420, 440, 460]
epochs = [660, 680, 700, 720, 740, 760, 780, 800]
epoch_list = [f'eval_compute_rollout_scores_epoch_{epoch}' for epoch in epochs]

list_of_finetune_types = [finetune_type1]

finetune_type_to_mean_success = {}
finetune_type_to_ste_success = {}
for finetune_type in list_of_finetune_types:

    

    epoch_to_mean_success = []
    epoch_to_ste_success = []
    for epoch_folder in epoch_list:
        result_path = f'data/outputs/{experiment_tag}_{experiment_name}_{task_name}/{finetune_type}/{epoch_folder}/successes.txt'
        # read the file line by line and look for failure or success, save 1 or 0
        successes = []
        with open(result_path, 'r') as f:
            for line in f:
                if 'Failure' in line:
                    successes.append(0)
                elif 'Success' in line:
                    successes.append(1)
                else:
                    print(f'Unknown line: {line}')
        print("successes", successes)
        mean_success = np.mean(successes)
        ste_success = np.std(successes) / np.sqrt(len(successes))

        epoch_to_mean_success.append(mean_success)
        epoch_to_ste_success.append(ste_success)

    finetune_type_to_mean_success[finetune_type] = epoch_to_mean_success
    finetune_type_to_ste_success[finetune_type] = epoch_to_ste_success

# pdb.set_trace()
# plot the results as a line graph with error bars
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  
# plt.style.use('seaborn-v0_8-grid')
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=(10, 5))
epochs_plot = [epochs[0]-20]+epochs
epochs_plot = [x-epochs_plot[0] for x in epochs_plot]  # normalize to start at 0

for finetune_type in list_of_finetune_types:
    mean_success = [base_mean_success]+finetune_type_to_mean_success[finetune_type]
    ste_success = [base_ste_success]+finetune_type_to_ste_success[finetune_type]
    plt.errorbar(
        epochs_plot, 
        mean_success, 
        yerr=ste_success, 
        label=finetune_type_to_name[finetune_type],
        marker='o'
    )
plt.xlabel('Finetuning Epoch')
plt.ylabel('Mean Success Rate')
plt.title(f'Success Rate for {task_name}')
plt.xticks(epochs_plot, rotation=45)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(5))
plt.legend(prop={'size': 7}, loc='lower right')
# plt.ylim(-0.1, 1.1)
# plt.tight_layout()

plt.show()




finetune_type_to_wall_clock_times = {}
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
plt.xticks(epochs, rotation=45)
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(5))
plt.legend(prop={'size': 7}, loc='lower right')
# plt.ylim(-0.1, 1.1)
# plt.tight_layout()

plt.show()










