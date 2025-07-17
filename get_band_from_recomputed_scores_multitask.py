import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import pandas as pd
import pickle
import argparse
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
import sys
# move out of the diffusion_policy directory
import pdb
import torch
from diffusion_policy.failure_detection.UQ_test.timeseries_cp.utils.data_utils import RegressionType
from diffusion_policy.failure_detection.UQ_test.timeseries_cp.methods.functional_predictor import FunctionalPredictor, ModulationType
import imageio
import os
from PIL import Image


fsize = 12
def adjust_label(ax, factor=8):
    """Adjust x-axis labels by multiplying with a factor."""
    current_ticks = ax.get_xticks()
    new_ticks = [int(i) for i in current_ticks * factor]
    ax.set_xticklabels(new_ticks)

def compute_quantile(value, values):
    values = np.array(values)
    quantile = np.mean(values <= value)
    return quantile

def plot_on_subfig_traj(ax, log_probs, successes, factor=8):
    """Plot trajectories on a subplot, distinguishing between success and failure.
    Args:
        ax: Matplotlib axis object
        log_probs: List of log probability trajectories
        successes: List of binary success indicators
        factor: Scaling factor for x-axis
    """
    failure_plotted = False
    success_plotted = False
    multiplier = 8
    xaxis = np.arange(len(log_probs[0])) * multiplier
    for log_prob, success in zip(log_probs, successes):
        alpha = 0.1 if success == 1 else 0.5
        color = 'blue' if success else 'red'
        label = 'Success' if success else 'Failure'
        if success and not success_plotted:
            ax.plot(xaxis, log_prob, color=color, label=label, alpha=alpha)
            success_plotted = True
        elif not success and not failure_plotted:
            ax.plot(xaxis, log_prob, color=color, label=label, alpha=alpha)
            failure_plotted = True
        else:
            ax.plot(xaxis, log_prob, color=color, alpha=alpha)
    ax.set_xlabel('Time Step', fontsize=fsize)
    ax.legend(fontsize=fsize-4, loc='upper center', bbox_to_anchor=(0.42, -0.16), ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)
    ax.grid()


def plot_on_subfig_traj_new(ax, log_probs, successes, predictions, plot_TP=False):
    """Plot mean and standard error of trajectories grouped by prediction outcomes.
    Args:
        ax: Matplotlib axis object
        log_probs: List of log probability trajectories
        successes: List of binary success indicators
        predictions: List of binary prediction outcomes
        plot_TP: If True, plot True Positives/Negatives; if False, plot False Positives/Negatives
    """
    fp_log_probs, fn_log_probs, tp_log_probs, tn_log_probs = [], [], [], []
    multiplier = 8
    xaxis = np.arange(len(log_probs[0])) * multiplier
    for log_prob, success, prediction in zip(log_probs, successes, predictions):
        if success == 1 and prediction == 1:
            fp_log_probs.append(log_prob)
        elif success == 0 and prediction == 1:
            tp_log_probs.append(log_prob)
        elif success == 0 and prediction == 0:
            fn_log_probs.append(log_prob)
        elif success == 1 and prediction == 0:
            tn_log_probs.append(log_prob)

    def plot_mean_with_error(ax, log_probs, color, label):
        mean_log_prob = np.mean(log_probs, axis=0)
        std_error_log_prob = np.std(log_probs, axis=0) / np.sqrt(len(log_probs))
        ax.plot(xaxis, mean_log_prob, color=color, label=label)
        ax.fill_between(xaxis, mean_log_prob - std_error_log_prob, mean_log_prob + std_error_log_prob, color=color, alpha=0.3)

    if plot_TP:
        if tp_log_probs:
            plot_mean_with_error(ax, tp_log_probs, 'red', 'TP')
        if tn_log_probs:
            plot_mean_with_error(ax, tn_log_probs, 'blue', 'TN')
    else:
        if fp_log_probs:
            plot_mean_with_error(ax, fp_log_probs, 'blue', 'FP')
        if fn_log_probs:
            plot_mean_with_error(ax, fn_log_probs, 'red', 'FN')

    ax.set_xlabel('Time Step', fontsize=fsize)
    ax.legend(fontsize=fsize-4, loc='upper center', bbox_to_anchor=(0.42, -0.16), ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)
    ax.grid()

def plot_on_subfig_traj_failure(ax, scores, n_idx, small=True):
    """Plot specific failure trajectories with custom styling.
    Args:
        ax: Matplotlib axis object
        scores: List of score trajectories
        n_idx: List of indices to plot
        small: If True, plot smallest failures; if False, plot largest failures
    """
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_idx)))
    fsize = 28
    for i, score in enumerate(scores):
        alpha = 1
        if i not in n_idx:
            continue
        kk = n_idx.index(i) + 1
        midfix = 'smallest' if small else 'largest'
        label_dict = {1: f'{midfix} failure', 2: 'median success'}
        label = label_dict[kk]
        ax.plot(range(len(score)), score, '-o', color=colors[kk-1], label=label, alpha=alpha)
    ax.set_xlabel('Time Step', fontsize=fsize)
    ax.set_ylabel('Score', fontsize=fsize)
    # Custom legend
    ax.legend(fontsize=fsize, loc='upper center', 
            bbox_to_anchor=(0.5, -0.2),  ncol=1,
            title_fontsize=fsize-2)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)
    adjust_label(ax)


def get_metric(y_true, y_pred):
    """Calculate various classification metrics.
    Returns:
        List containing [TPR, TNR, accuracy, accuracy_weighted]
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    accuracy = (tpr + tnr) / 2
    tnr_weight = y_true.sum() / len(y_true)
    tpr_weight = 1 - tnr_weight
    accuracy_weighted = tpr_weight * tpr + tnr_weight * tnr
    return [tpr, tnr, accuracy, accuracy_weighted]



def get_detection_with_plot(log_probs, successes, img_frames, save_folder, alpha=0.01, lb=False, CPband=True, suffix=''):
    

    """Detect anomalies using prediction bands and create visualization plots.
    Returns:
        Tuple containing (first_idx_ls, positive_ls, successes_test, amount_exceed_ratio)
    """
    

    num_te = 25 # these are heldout for testing and evaluating failure detection
    max_tr = 25 # these are used in CP construction

    # the training rollouts are further split in D_calibA (of size num_train), and D_calibB (of size num_cal)
    num_train = int(max_tr/2)
    num_cal = int(max_tr/2)
    assert num_train + num_cal <= max_tr

    # these are for constructing the CP band
    log_probs_train = log_probs[:max_tr]
    successes_train = successes[:max_tr]

    # these are heldout for testing 
    log_probs_test = log_probs[max_tr:max_tr+num_te]
    successes_test = successes[max_tr:max_tr+num_te]

    global_indices_of_train = np.arange(len(log_probs_train))
    global_indices_of_test = np.arange(len(log_probs_test)) + len(log_probs_train)


    # count zeros in successes_test
    num_failures_in_test = np.sum(np.array(successes_test) == 0)
    print(f'#### Number of failures in test set: {num_failures_in_test}')


    log_probs_train = np.array([log_probs_train[i] for i, success in enumerate(successes_train) if success])
    ntr = int(len(log_probs_train) * num_train / (num_train + num_cal))
    ncal = len(log_probs_train) - ntr

    print(f'#### Use {ntr} trajectories for training and {ncal} for calibration')
    print(f'#### Number of training trajectories: {len(log_probs_train)}')
    print("#### Number of test trajectories: ", len(log_probs_test))
    print(f'#### Use {len(log_probs_train)} successful trajectories for calibration')
    # predictor = FunctionalPredictor(modulation_type=ModulationType.Const, regression_type=RegressionType.ConstantMean)
    predictor = FunctionalPredictor(modulation_type=ModulationType.Tfunc, regression_type=RegressionType.Mean)
    if CPband:
        print(f'Number of success for mean {ntr} and for band {ncal}')
        # pdb.set_trace()
        target_traj = predictor.get_one_sided_prediction_band(log_probs_train[:ntr], log_probs_train[-ncal:], alpha=alpha, lower_bound=lb).flatten()
    else:
        metric_tr = [np.cumsum(val)[-1] for val in log_probs_train]
        threshold = np.quantile(metric_tr, 1 - alpha)
        # Repeat for each trajectory
        target_traj = np.repeat(threshold, len(log_probs_train[0]))

    train_DcalibA = log_probs_train[:ntr]
    train_DcalibB = log_probs_train[-ncal:]
    test_scores = log_probs_test

    scores_for_faildetect = {}
    scores_for_faildetect['train_DcalibA'] = train_DcalibA
    scores_for_faildetect['train_DcalibB'] = train_DcalibB
    scores_for_faildetect['test_scores'] = test_scores
    # save scores to pickle
    # pdb.set_trace()
    # with open(f'scores_for_faildetect.pkl', 'wb') as f:
    #     pickle.dump(scores_for_faildetect, f)


    print(f'###  Mean of band width: {np.mean(target_traj)}')
    to_plot = min(150, len(log_probs_test))
    # rand_idx = np.random.choice(len(log_probs_test), to_plot, replace=False)
    rand_idx = np.arange(len(log_probs_test)) 
    log_probs_test_plt = [log_probs_test[i] for i in rand_idx]
    successes_test_plt = [successes_test[i] for i in rand_idx]

    # plot CP band relative to train scores
    # plot band
    upper = target_traj
    # lower is all zeros
    lower = np.zeros_like(target_traj)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
    multiplier = 16
    xaxis = np.arange(len(log_probs_test_plt[0])) * multiplier
    ax.fill_between(xaxis, upper, lower, color='blue', alpha=0.25)
    single_cp_value = target_traj[0]

    log_probs_train_plot = log_probs[:max_tr]
    log_probs_train_plot = log_probs_train[:ntr]

    aggregated_single_train_scores = []
    for train_idx in range(len(log_probs_train_plot)):
        log_prob_train_scores = log_probs_train_plot[train_idx]
        success = successes_train[train_idx]
        if success:
            aggregated_single_train_scores.extend(log_prob_train_scores)
        
        # print(f'### Train trajectory {train_idx+1}/{len(log_prob_train_scores)}: Success={success}')
        # plot the CP band - target_traj, and shade underneath
        # pdb.set_trace()
    
        ax.plot(xaxis, log_prob_train_scores, color='red' if success == 0 else 'blue', label='Test Trajectory')
    
    ax.set_title(f'Train Rollouts (Successes used in Calibration)', fontsize=fsize)
    ax.set_xlabel('Time Step', fontsize=fsize)
    ax.set_ylabel('Score', fontsize=fsize)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)
    # ax.legend(fontsize=fsize-4, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.grid()
    fig.tight_layout()
    plt.savefig(os.path.join(save_folder, f'CP_band_train_rollouts{suffix}.png'), dpi=300)
    plt.show()
    plt.close()

    # plot histogram of aggregated_single_train_scores, with line for single_cp_value
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.hist(aggregated_single_train_scores, bins=50, color='blue', alpha=0.5, label='Train Scores')
    ax.axvline(single_cp_value, color='red', linestyle='--', label='CP Band Value')
    ax.set_title('Histogram of Train Scores', fontsize=fsize)
    ax.set_xlabel('Score', fontsize=fsize)
    ax.set_ylabel('Frequency', fontsize=fsize)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)
    ax.legend(fontsize=fsize-4)
    ax.grid()
    fig.tight_layout()
    ax.set_ylim(1000, max(aggregated_single_train_scores) + 100)
    plt.savefig(os.path.join(save_folder, f'CP_band_train_rollouts_histogram{suffix}.png'), dpi=300)
    plt.show()
    plt.close()

    # get quantile of single_cp_value in aggregated_single_train_scores
    quantile_of_cp_value = compute_quantile(single_cp_value, aggregated_single_train_scores)
    print(f'### Quantile of CP value in aggregated train scores: {quantile_of_cp_value}')

    # Plot test alltogether on one plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
    multiplier = 16
    xaxis = np.arange(len(log_probs_test_plt[0])) * multiplier
    ax.fill_between(xaxis, upper, lower, color='blue', alpha=0.25)

    for test_idx in range(len(log_probs_test_plt)):
        log_prob_test_scores = log_probs_test_plt[test_idx]
        success = successes_test_plt[test_idx]
        ax.plot(xaxis, log_prob_test_scores, color='red' if success == 0 else 'blue', label='Test Trajectory')
    
    ax.set_title(f'Test Rollouts', fontsize=fsize)
    ax.set_xlabel('Time Step', fontsize=fsize)
    ax.set_ylabel('Score', fontsize=fsize)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)
    # ax.legend(fontsize=fsize-4, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.grid()
    fig.tight_layout()
    ax.set_ylim(1200, max(aggregated_single_train_scores) + 200)
    plt.savefig(os.path.join(save_folder, f'CP_band_test_rollouts{suffix}.png'), dpi=300)
    plt.show()
    plt.close()

    # get quantile in

    

    num_TP = 0 #True Positives (correctly identified positive cases)
    num_FN = 0 # False Negatives (actual positives incorrectly identified as negative) 
    num_FP = 0 # False Positives (actual negatives incorrectly identified as positive)
    num_TN = 0 # True Negatives (correctly identified negative cases)
    for test_idx in range(len(log_probs_test_plt)):
        log_prob_test_scores = log_probs_test_plt[test_idx]
        success = successes_test_plt[test_idx]

        if np.mean(log_prob_test_scores) > 10000:
            print("high value", global_indices_of_test[test_idx])
        else:
            print("low value", global_indices_of_test[test_idx])
        observation_frames = img_frames[global_indices_of_test[test_idx]]
        
        CP_upper_band = target_traj
        for t in range(len(log_prob_test_scores)):
            if log_prob_test_scores[t] > CP_upper_band[t]:
                if t < len(observation_frames):
                    img= observation_frames[t]
                else:
                    img = observation_frames[-1]
                # pdb.set_trace()
                
                if success == 0: # if failed, then correct detection
                    num_TP += 1
                    # fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
                    # plt.imshow(img)
                    # plt.title("TRUE POSITIVE DETECTION", fontsize=fsize)
                    # plt.show()
                else:
                    num_FP += 1 # if successful demo, then false positive
                    # fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
                    # plt.imshow(img)
                    # plt.title("FALSE POSITIVE DETECTION", fontsize=fsize)
                    # plt.show()
                break

            if t == len(log_prob_test_scores) - 1: # no detection made
                if success == 1: # if successful, then correct detection
                    num_TN += 1
                else:
                    num_FN += 1
    print(f'### Number of True Positives: {num_TP}')
    print(f'### Number of False Negatives: {num_FN}')
    print(f'### Number of False Positives: {num_FP}')
    print(f'### Number of True Negatives: {num_TN}')

    # save CP band to pickle
    with open(f'{save_folder}/CP_band.pkl', 'wb') as f:
        pickle.dump(target_traj, f)

    TPR = num_TP / (num_TP + num_FN) if (num_TP + num_FN) > 0 else 0
    TNR = num_TN / (num_TN + num_FP) if (num_TN + num_FP) > 0 else 0
    accuracy = (TPR + TNR) / 2
    accuracy_weighted = (num_TP + num_TN) / len(log_probs_test_plt)
    print(f'### True Positive Rate (TPR): {TPR}')
    print(f'### True Negative Rate (TNR): {TNR}')
    print(f'### Accuracy: {accuracy}')
    print(f'### Weighted Accuracy: {accuracy_weighted}')
    
    return        


def plot_all_scores_individual(log_probs, successes):
    # plot all scores, color red for failure and blue for success
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)
    # multiplier = 8
    # xaxis = np.arange(len(log_probs[0])) 
    for i in range(len(log_probs)):
        log_prob = log_probs[i]
        color = 'blue' if successes[i]==1 else 'red'
        label = 'Success' if successes[i]==1 else 'Failure'
        # duplicate the log_prob for plotting
        # log_prob = np.array(log_prob)
        # log_prob_mult = np.repeat(log_prob, multiplier)
        ax.plot(np.arange(len(log_prob)), log_prob, color=color, label=label, alpha=0.4)
        # ax[1].plot(xaxis, log_prob, color=color, alpha=0.1)
    ax.set_title('All Trajectories', fontsize=fsize)

    ax.set_xlabel('Time Step', fontsize=fsize)

    ax.set_ylabel('Score', fontsize=fsize)
    # ax.legend(fontsize=fsize-4, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)

    ax.grid()

    fig.tight_layout()
    fig.savefig('all_scores.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_all_scores(log_probs, successes, fsize=14):
    # Separate log_probs by success/failure
    success_scores = [log_probs[i] for i in range(len(log_probs)) if successes[i] == 1]
    fail_scores = [log_probs[i] for i in range(len(log_probs)) if successes[i] == 0]

    def pad_and_stack(trajs):
        max_len = max(len(traj) for traj in trajs)
        padded = np.array([
            np.pad(traj, (0, max_len - len(traj)), constant_values=np.nan)
            for traj in trajs
        ])
        return padded

    success_arr = pad_and_stack(success_scores)
    fail_arr = pad_and_stack(fail_scores)

    success_mean = np.nanmean(success_arr, axis=0)
    success_std = np.nanstd(success_arr, axis=0)
    fail_mean = np.nanmean(fail_arr, axis=0)
    fail_std = np.nanstd(fail_arr, axis=0)

    x = np.arange(max(len(success_mean), len(fail_mean)))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), sharex=True, sharey=True)

    ax.plot(x, success_mean, color='blue', label='Success (mean)')
    ax.fill_between(x, success_mean - success_std, success_mean + success_std, color='blue', alpha=0.2)

    ax.plot(x, fail_mean, color='red', label='Failure (mean)')
    ax.fill_between(x, fail_mean - fail_std, fail_mean + fail_std, color='red', alpha=0.2)

    ax.set_title('Mean Log Scores per Class', fontsize=fsize)
    ax.set_xlabel('Time Step', fontsize=fsize)
    ax.set_ylabel('Score', fontsize=fsize)
    ax.tick_params(axis='both', which='major', labelsize=fsize-4)
    ax.legend(fontsize=fsize-3)
    ax.grid()

    fig.tight_layout()
    fig.savefig('mean_scores.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()



def main():
    parser = argparse.ArgumentParser(description="Run evaluation for a specific task.")
    parser.add_argument("task_name", type=str, help="Name of the task (e.g., closedrawer, openfridge)")
    parser.add_argument("--experiment_name", type=str, default="train_diffusion_unet_clip", required=False)
    parser.add_argument("--experiment_tag", type=str, default="ST_OOD_DAgger", required=False)
    parser.add_argument("--rollout_folder", type=str, default="compute_rollout_scores", required=False)
    parser.add_argument("--score_type_folder", type=str, default="fd_score_network", required=False)

    args = parser.parse_args()

    task_name = args.task_name
    experiment_name = args.experiment_name
    experiment_tag = args.experiment_tag
    rollout_folder = args.rollout_folder
    score_type_folder = args.score_type_folder
    policy_type = 'diffusion'
    data_folder = f'data/outputs/{experiment_tag}_{experiment_name}_{task_name}/{rollout_folder}/{score_type_folder}'
    successes = []
    all_log_probs = []
    all_images = []
    N_rollouts = 50
    print("task_name", task_name)

    for demo_num in range(N_rollouts):
        dataset_path = data_folder + f'/{task_name}_{demo_num}_fd_scores.pkl'
        try:
            with open(dataset_path, "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print(f"File not found: {dataset_path}")
            continue
        # pdb.set_trace()
        experiments = data['tasks'][task_name]['experiments']
        for demo_key in experiments:
            logpzo_scores = experiments[demo_key]['logpzo_scores']
            scores = [elem.detach().cpu().numpy()[0] for elem in logpzo_scores]
            # pdb.set_trace()

            success = experiments[demo_key]['success']
            img_obs = experiments[demo_key]['img_observations']

            scores_filtered = []
            for i in range(len(scores)):
                if i < len(scores) - 1 and scores[i] == scores[i+1]:
                    break
                scores_filtered.append(scores[i])
            scores = scores_filtered

            img_frames = []
            for t in range(len(scores)):
                leftcam, rightcam, grippercam = img_obs[t]
                leftcam = leftcam[0,0,:].detach().cpu().numpy()
                rightcam = rightcam[0,0,:].detach().cpu().numpy()
                grippercam = grippercam[0,0,:].detach().cpu().numpy()

                leftcam = np.rot90(np.swapaxes(leftcam, 0, -1), -1)
                rightcam = np.rot90(np.swapaxes(rightcam, 0, -1), -1)
                grippercam = np.rot90(np.swapaxes(grippercam, 0, -1), -1)

                combined = np.concatenate([leftcam, rightcam, grippercam], axis=1)
                img_frames.append(combined)

            all_log_probs.append(scores)
            successes.append(success)
            all_images.append(img_frames)

    max_length = max(len(lp) for lp in all_log_probs)
    padded_log_probs = [lp + [lp[-1]] * (max_length - len(lp)) for lp in all_log_probs]
    log_probs = np.array(padded_log_probs)
    successes = np.array(successes)
    print("success rate = ", np.mean(successes))
    

    print("log_probs shape:", log_probs.shape)
    print("successes shape:", successes.shape)
    # pdb.set_trace()

    get_detection_with_plot(log_probs, successes, all_images, data_folder, alpha=0.1)
    print("success rate = ", np.mean(successes))

if __name__ == "__main__":
    main()