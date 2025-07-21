import h5py
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import scipy.stats
import os

TASK_NAME_TO_HUMAN_PATH = {'PnPCabToCounter': "../robocasa/datasets_first/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams_im256.hdf5",
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


def summarize_h5_files(file1_path, file2_path):
    def count_demos_and_transitions(h5file):
        demo_keys = list(h5file['data'].keys())
        num_demos = len(demo_keys)
        total_transitions = 0
        for key in demo_keys:
            obs = h5file['data'][key]['obs']
            first_obs_key = list(obs.keys())[0]
            num_steps = obs[first_obs_key].shape[0]
            total_transitions += num_steps
        return num_demos, total_transitions

    with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
        demos1, transitions1 = count_demos_and_transitions(f1)
        demos2, transitions2 = count_demos_and_transitions(f2)

    print(f"📁 {file1_path}: {demos1} demos, {transitions1} total transitions")
    print(f"📁 {file2_path}: {demos2} demos, {transitions2} total transitions")
    print(f"📊 Combined total: {demos1 + demos2} demos, {transitions1 + transitions2} transitions")

def summarize_single_h5_file(file1_path):
    def count_demos_and_transitions(h5file):
        demo_keys = list(h5file['data'].keys())
        num_demos = len(demo_keys)
        total_transitions = 0
        for key in demo_keys:
            obs = h5file['data'][key]['obs']
            first_obs_key = list(obs.keys())[0]
            num_steps = obs[first_obs_key].shape[0]
            total_transitions += num_steps
        return num_demos, total_transitions

    with h5py.File(file1_path, 'r') as f1:
        demos1, transitions1 = count_demos_and_transitions(f1)

    print(f"📁 {file1_path}: {demos1} demos, {transitions1} total transitions")

def merge_h5_files_exact_dagger(file1_path, dagger_file2_path, output_path):
    with h5py.File(file1_path, 'r') as f1, \
        h5py.File(dagger_file2_path, 'r') as f2, \
        h5py.File(output_path, 'w') as out:

        # Copy all data from file1 as-is
        out.create_group('data')
        for key in f1['data'].keys():
            f1.copy(f'data/{key}', out['data'], name=key)
        print(f"Copied {len(f1['data'])} demos from {file1_path}")

        # Determine the starting index for file2 demos
        start_idx = len(f1['data'])

        for key in f2['data'].keys():
            new_key = f'demo_{start_idx}'
            f2.copy(f'data/{key}', out['data'], name=new_key)
            start_idx += 1
        print(f"Copied {len(f2['data'])} demos from {dagger_file2_path} with renamed keys")

    print(f"\n✅ Merged files saved to: {output_path}")

def analyze_dagger_experience(dagger_meta_folder, N_dagger_eps):
    global_contribution_dict = {}
    band_folder = 'constant_time_band'
    dagger_folder_raw = 'dagger_data'
    dagger_meta_folder = f'data/outputs/ST_OOD_DAgger_train_diffusion_unet_clip_CoffeePressButton/dagger_episode_0/{dagger_folder_raw}'

    N_dagger_eps = 15
    contribution_dict = {'count_num_human_handovers': [], 'robot_indep': [], 'ts_first_handover':[], 'n_human_timesteps': [], 'n_robot_timesteps':[]}
    for demo_num in range(N_dagger_eps):
        dataset_path_robocasa = f'{dagger_meta_folder}/dagger_episode_meta_{demo_num}.pkl'
        with open(dataset_path_robocasa, "rb") as pickle_file:
            data = pickle.load(pickle_file)
        
        n_human_timesteps = data['n_human_timesteps']
        n_robot_timesteps = data['n_robot_timesteps'] * 16
        
        contribution_dict['n_human_timesteps'].append(n_human_timesteps)
        contribution_dict['n_robot_timesteps'].append(n_robot_timesteps)
        
        count_num_human_handovers = 0
        robot_indep = 1
        actors = [x[-2] for x in data['action_list']]
        ts_first = 0
        for t in range(len(actors)):
            if t > 0:
                if actors[t] == 'human' and actors[t-1] == 'robot':
                    count_num_human_handovers += 1
                    robot_indep = 0
                    if count_num_human_handovers == 1:
                        ts_first = 16 * t
        contribution_dict['count_num_human_handovers'].append(count_num_human_handovers)
        contribution_dict['robot_indep'].append(robot_indep)
        if ts_first > 0:
            contribution_dict['ts_first_handover'].append(ts_first)
        
    global_contribution_dict[band_folder] = contribution_dict
    plot_dagger_experience(global_contribution_dict, dagger_meta_folder)

def plot_dagger_experience(global_contribution_dict, save_folder):
    contribution_dict = global_contribution_dict
    groups = list(contribution_dict.keys())
    metrics = ['n_human_timesteps', 'n_robot_timesteps', 'count_num_human_handovers', 'robot_indep', 'ts_first_handover']

    # Plot config
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    colors = ['skyblue', 'lightgreen']

    # Plot 1: Timesteps (combined)
    ax = axes[0]
    width = 0.35
    x = np.arange(len(groups))  # e.g. group_A, group_B

    # Collect values
    human_means = [np.mean(contribution_dict[g]['n_human_timesteps']) for g in groups]
    human_stds = [np.std(contribution_dict[g]['n_human_timesteps']) for g in groups]
    robot_means = [np.mean(contribution_dict[g]['n_robot_timesteps']) for g in groups]
    robot_stds = [np.std(contribution_dict[g]['n_robot_timesteps']) for g in groups]

    # Plot both
    ax.bar(x - width/2, human_means, width=width, yerr=human_stds, capsize=6, label='Human Timesteps', color='skyblue')
    ax.bar(x + width/2, robot_means, width=width, yerr=robot_stds, capsize=6, label='Robot Timesteps', color='lightgreen')
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_title("Human vs Robot Timesteps", fontsize=14)
    ax.set_ylabel("Timesteps", fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10)

    metric_to_title = {"n_human_timesteps": '# env timesteps human controlling',
                    "n_robot_timesteps": '# env timesteps robot controlling',
                    "count_num_human_handovers": 'Avg # handovers to human per rollout',
                    "robot_indep": '% rollouts robot completely independent',
                    "ts_first_handover": "Env timestep first handover"}

    # Plot 2 & 3: One metric each
    other_metrics = ['count_num_human_handovers', 'robot_indep', 'ts_first_handover']
    for i, metric in enumerate(other_metrics, start=1):
        ax = axes[i]
        means = [np.mean(contribution_dict[g][metric]) for g in groups]
        stds = [scipy.stats.sem(contribution_dict[g][metric]) for g in groups]
        ax.bar(x, means, yerr=stds, capsize=6, color=colors[:len(groups)])
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=12)
        ax.set_title(metric_to_title[metric], fontsize=14)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    fig.suptitle("Group Comparison of Control Contributions", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{save_folder}/dagger_experience_analysis.png", dpi=300)
    plt.close()

def visualize_h5_videos(filename):
    output_fps = 25

    # Get parent folder to save videos
    parent_dir = os.path.dirname(os.path.abspath(filename))

    # make folder in parent_dir for videos
    video_folder = os.path.join(parent_dir, 'videos')
    os.makedirs(video_folder, exist_ok=True)


    with h5py.File(filename, 'r') as file:
        for demo_key in file['data'].keys():
            # Extract observation data
            demos = file['data'][demo_key]['obs']
            print("Demo:", demo_key)
            print("Keys:", demos.keys())

            vid_data_cam0 = demos['robot0_agentview_right_image']  # shape (T, H, W, 3)
            vid_data_cam1 = demos['robot0_agentview_left_image']  # shape (T, H, W, 3)
            vid_data_cam2 = demos['robot0_eye_in_hand_image']     # shape (T, H, W, 3)
            print("vid_data_cam1 shape:", vid_data_cam1.shape)

            # Get video properties
            T, H, W, C = vid_data_cam1.shape
            output_height = H
            output_width = W * 3  # since we're stacking 2 images side-by-side

            # Setup video writer
            video_path = os.path.join(video_folder, f"aggregated_{demo_key}_video.mp4")
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                output_fps,
                (output_width, output_height)
            )

            for frame0, frame1, frame2 in zip(vid_data_cam0, vid_data_cam1, vid_data_cam2):
                # Convert RGB to BGR for OpenCV
                frame0_bgr = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
                frame1_bgr = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)
                frame2_bgr = cv2.cvtColor(frame2, cv2.COLOR_RGB2BGR)

                # Concatenate horizontally
                combined_frame = np.concatenate((frame0_bgr, frame1_bgr, frame2_bgr), axis=1)
                writer.write(combined_frame)

            writer.release()
            print(f"Saved video to {video_path}")



def main():
    parser = argparse.ArgumentParser(description="Merge dagger data for a specific task.")
    parser.add_argument("--task_name", type=str, help="Name of the task (e.g., closedrawer, openfridge)", required=True)
    parser.add_argument("--dataset1", type=str, default='train', required=True)
    parser.add_argument("--dataset2", type=str, default='dagger_episode_0', required=True)

    args = parser.parse_args()

    task_name = args.task_name
    dataset1 = args.dataset1
    dataset2 = args.dataset2

    if dataset1 == 'train':
        dataset1_path = TASK_NAME_TO_HUMAN_PATH[task_name]
    else:
        dagger_meta_folder = f'data/outputs/ST_OOD_DAgger_train_diffusion_unet_clip_{task_name}/{dataset1}/processed_dagger_data'
        dataset1_path = dagger_meta_folder + "/merged_dagger_data.hdf5"

    dagger_meta_folder = f'data/outputs/ST_OOD_DAgger_train_diffusion_unet_clip_{task_name}/{dataset2}/processed_dagger_data'
    dataset2_path = dagger_meta_folder + "/human_only_demo.hdf5"

    merge_h5_files_exact_dagger(dataset1_path, dataset2_path, f"{dagger_meta_folder}/merged_dagger_data.hdf5")
    summarize_h5_files(dataset1_path, dataset2_path)
    summarize_single_h5_file(f"{dagger_meta_folder}/merged_dagger_data.hdf5")
    # analyze_dagger_experience(dagger_meta_folder, 50)
    visualize_h5_videos(f"{dagger_meta_folder}/merged_dagger_data.hdf5")



if __name__ == "__main__":
    main()