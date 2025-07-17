import copy
from typing import Dict, Optional

import os
from datetime import datetime
import pathlib
import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits
from tqdm import trange, tqdm
from filelock import FileLock
import shutil

from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
from scipy.spatial.transform import Rotation as R
import open_clip
import numpy as np
import cv2
import random
import h5py
import json

from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS, MULTI_STAGE_TASK_DATASETS
# from robocasa.utils.dataset_registry import get_ds_path, get_new_ds_path

import pdb

def tf_to_axis_angle(T):
    # Extract the rotation matrix (3x3 part of the 4x4 matrix)
    rotation_matrix = T[:3, :3]
    
    # Extract the translation vector (first three elements of the fourth column)
    translation_vector = T[:3, 3]
    
    # Convert the rotation matrix to axis-angle representation
    rotation = R.from_matrix(rotation_matrix)
    axis_angle = rotation.as_rotvec()  # This gives the rotation axis multiplied by the rotation angle
    
    # Concatenate the axis-angle representation and the translation vector
    return np.concatenate((translation_vector, axis_angle))

def axis_angle_to_tf(array):
    # Split the array into translation and axis-angle components
    translation = array[:3]  # First three elements for translation
    axis_angle = array[3:]   # Last three elements for axis-angle
    
    # Convert axis-angle to rotation matrix
    rotation_matrix = R.from_rotvec(axis_angle).as_matrix()
    
    # Create the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation
    
    return transformation_matrix

def euler_to_tf(array):

    # Extract position and Euler angles
    position = array[:3]  # [x, y, z]
    roll, pitch, yaw = array[3:]
    
    # Compute rotation matrix using extrinsic XYZ Euler angles
    rotation = R.from_euler('XYZ', [roll, pitch, yaw]).as_matrix()
    
    # Create homogeneous transformation matrix
    transform = np.eye(4)  # Initialize 4x4 identity matrix
    transform[:3, :3] = rotation  # Add rotation matrix
    transform[:3, 3] = position   # Add translation vector
    
    return transform

def tf_to_euler(transform):

    # Extract the translation vector
    position = transform[:3, 3]
    
    # Extract the rotation matrix
    rotation_matrix = transform[:3, :3]
    
    # Convert rotation matrix to extrinsic XYZ Euler angles
    rotation = R.from_matrix(rotation_matrix)
    roll, pitch, yaw = rotation.as_euler('XYZ', degrees=False)
    
    # Combine position and orientation into a single array
    result = np.hstack((position, [roll, pitch, yaw]))
    
    return result

def inverse_tf_matrix(matrix):

    R = matrix[:3, :3]  # Extract the 3x3 rotation matrix
    t = matrix[:3, 3]   # Extract the translation vector
    
    R_inv = R.T  # Compute the inverse (transpose) of the rotation matrix
    t_inv = -np.dot(R_inv, t)  # Compute the inverse translation vector
    
    # Construct the inverse matrix
    inverse_matrix = np.zeros_like(matrix)
    inverse_matrix[:3, :3] = R_inv
    inverse_matrix[:3, 3] = t_inv
    inverse_matrix[3, 3] = 1  # Set the bottom-right element to 1
    
    return inverse_matrix

class InMemoryVideoDataset(Dataset):
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        tasks: dict,
        skip_demos: dict,
        sample_fps: float,
        video_fps: float,
        pred_horizon: int,
        obs_horizon: int,
        validation_split: int,
        aug: dict,
        action_dim: int,
        swap_rgb: bool,
        lang_model: nn.Module,
        mode: str,
        data_fraction: float,
        human_path: Optional[str] = None,
    ):
        super().__init__()

        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'train', 'val' or 'test'.")
        
        self.mode = mode
        self.validation_split = validation_split    # use the last 2000 subsamples as validation

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.stride = round(video_fps / sample_fps)
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_dim = action_dim
        self.swap_rgb = swap_rgb
        self.aug = aug

        self.lang_model = lang_model

        self.human_path = human_path
        if self.human_path is None:
            self.human_path = 'datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams_im256.hdf5'

        # self.clip_mean = [0.48145466, 0.4578275, 0.40821073]
        # self.clip_std = [0.26862954, 0.26130258, 0.27577711]

        if self.mode == 'train':
            self.transform_rgb = T.Compose([
                T.RandomRotation(degrees=3),  # Rotate within ±3 degrees
                T.RandomCrop(size=(self.frame_height, self.frame_width)), # Crop to 0.9*heightx0.9*width
                T.ColorJitter(brightness=aug['brightness'], contrast=aug['contrast'], saturation=aug['saturation_rgb'], hue=aug['hue_rgb']),  # Random color jitter
                # T.Normalize(mean=self.clip_mean, std=self.clip_std),
            ])

            self.transform_depth = T.Compose([
                T.RandomRotation(degrees=3),  # Rotate within ±3 degrees
                T.RandomCrop(size=(self.frame_height, self.frame_width)), # Crop to 0.9*heightx0.9*width
                T.ColorJitter(brightness=aug['brightness'], contrast=aug['contrast'], saturation=aug['saturation_depth'], hue=aug['hue_depth']),  # Random color jitter
                # T.Normalize(mean=self.clip_mean, std=self.clip_std),
            ])
        elif self.mode == 'val' or self.mode == 'test':
            self.transform_rgb = T.Compose([
                T.CenterCrop(size=(self.frame_height, self.frame_width)),
                # T.Normalize(mean=self.clip_mean, std=self.clip_std),
            ])

            self.transform_depth = T.Compose([
                T.CenterCrop(size=(self.frame_height, self.frame_width)),
                # T.Normalize(mean=self.clip_mean, std=self.clip_std),
            ])

        self.task_list = list(tasks.keys())
        print("Task list:", self.task_list)
        print("loading dataset from path:", self.human_path)
        self.datasets, self.hdf5_datasets = self.get_dataset_file(self.task_list, self.human_path)

        self.indexed_demos = []
        for task_index, task_name in enumerate(self.task_list):

            task_data = self.datasets[task_index]['data']
            demo_keys = list(task_data.keys())
            max_demos = int(len(demo_keys)*data_fraction)  # Modify the number of demo keys
            added_demos = 0  # Counter for how many demos have been added

            for demo_key in demo_keys:
                if (task_name in skip_demos and demo_key in skip_demos[task_name]):     # skip invalid demos with robot base actions
                    continue

                task_description = json.loads(self.hdf5_datasets[task_index]['data'][demo_key].attrs['ep_meta'])['lang']
                task_description = open_clip.tokenize([task_description]) # returns torch.Size([1, 77])
                with torch.no_grad():
                    clip_embedding = self.lang_model(task_description.to(self.lang_model.device)).cpu() # returns torch.Size([1, 1024])

                demo_steps = range(0, task_data[demo_key]['actions_abs'].shape[0])
                for demo_step in demo_steps:
                    self.indexed_demos.append((task_name, task_index, demo_key, demo_step, clip_embedding))

                added_demos += 1
                if added_demos > max_demos:
                    break  # Stop once enough demos are added

        all_relative_actions = []

        for task_index, task_name in enumerate(self.task_list):
            task_data = self.datasets[task_index]['data']
            for demo_key in task_data.keys():
                demo_steps = range(0, int(task_data[demo_key]['actions_abs'].shape[0]))
                for demo_step in demo_steps:
                    all_relative_actions.append(task_data[demo_key]['actions'][demo_step][0:self.action_dim])

        all_relative_actions = np.array(all_relative_actions)

        # self.pose_001 = np.percentile(all_relative_actions, 0.01, axis=(0,))[None, :]
        # self.pose_9999 = np.percentile(all_relative_actions, 99.99, axis=(0,))[None, :]

        # print(self.pose_001)
        # print(self.pose_9999)

        # self.min = np.min(all_relative_actions, axis=(0,))[None, :]
        # self.max = np.max(all_relative_actions, axis=(0,))[None, :]

        self.min = np.ones((1, 7), dtype=np.float32) * -1
        self.max = np.ones((1, 7), dtype=np.float32)

        print(self.min)
        print(self.max)

    def load_hdf5_into_memory(self, h5_file):
        """Load an HDF5 file into memory as a dictionary."""
        def recursive_load(h5_obj):
            if isinstance(h5_obj, h5py.Group):
                # if self.mode == 'train' or self.mode == 'val':
                #     return {key: recursive_load(h5_obj[key]) for key in h5_obj.keys()}
                # elif self.mode == 'test':
                #     return {
                #         key: recursive_load(h5_obj[key])
                #         for key in h5_obj.keys()
                #         if key != 'obs'  # Exclude the 'obs' key
                #     }
                return {
                        key: recursive_load(h5_obj[key])
                        for key in h5_obj.keys()
                        if key != 'obs'  # Exclude the 'obs' key
                    }
            elif isinstance(h5_obj, h5py.Dataset):
                return h5_obj[()]  # Load dataset into memory as a NumPy array
            else:
                raise ValueError(f"Unsupported HDF5 object: {type(h5_obj)}")

        with h5py.File(h5_file, "r") as f:
            return recursive_load(f)
    
    def get_dataset_file(self, task_list, human_path):
        datasets = []
        hdf5_datasets = []

        for task in list(SINGLE_STAGE_TASK_DATASETS):
            if task not in task_list:
                continue

            # human_path = "datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im256_randcams.hdf5"
            # human_path = "datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams_im256.hdf5"
            # human_path = "../robocasa/datasets_first/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-24/demo_gentex_im128_randcams_im256.hdf5"
            # human_path = "dagger_full_constant_time.hdf5"
            # human_path = 'data/experiments/train_diffusion_unet_clip_train_closedrawer_fd_scores_original_env/constant_time_band/processed_dagger_data/combined_demo_im256.hdf5'

            in_memory_data = self.load_hdf5_into_memory(human_path)

            datasets.append(in_memory_data)

            hdf5_file = h5py.File(human_path, 'r')
            hdf5_datasets.append(hdf5_file)

        return datasets, hdf5_datasets

    def undo_operations_and_save(self, frame, output_filename):
        frame = (frame + 1.0) / 2.0

        frame = torch.tensor(frame)
        frame = ToPILImage()(frame)

        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imwrite(output_filename, frame)

    def convert_frame(self, frame, size=None, swap_rgb=False):
        if size is not None:
            original_height, original_width = frame.shape[:2]
            target_width, target_height = size

            if original_width != target_width or original_height != target_height:
                # Calculate aspect ratios
                original_aspect_ratio = original_width / original_height
                target_aspect_ratio = target_width / target_height

                if original_aspect_ratio > target_aspect_ratio:
                    # Crop width (as in the original code)
                    new_width = int(original_height * target_aspect_ratio)
                    crop_start = (original_width - new_width) // 2
                    cropped_image = frame[:, crop_start:crop_start + new_width]
                else:
                    # Crop height
                    new_height = int(original_width / target_aspect_ratio)
                    crop_start = (original_height - new_height) // 2
                    cropped_image = frame[crop_start:crop_start + new_height, :]
                
                # Resize the cropped image to the target size
                frame = cv2.resize(cropped_image, size, interpolation=cv2.INTER_AREA)

        if swap_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = frame.astype(np.float32)
        frame = frame / 255.0
        frame = frame * 2.0 - 1.0
        frame = np.transpose(frame, (2, 0, 1))  # Transpose the frame to have the shape (3, frame_height, frame_width)

        return frame

    def augmentation_transform(self, images, transform):
        transformed_images = []
        seed = random.randint(0, 2**32)
        for frame in images:
            torch.manual_seed(seed)
            transformed_images.append(transform(frame))
        images = torch.stack(transformed_images)

        return images

    def __getitem__(self, i):

        if self.mode == 'val':
            i += len(self.indexed_demos) - self.validation_split

        try:
            task_name, task_index, demo_key, demo_step, clip_embedding = self.indexed_demos[i]
            # relative_actions_abs = self.datasets[task_index]['data'][demo_key]['relative_actions_abs'][demo_step:demo_step+self.pred_horizon*self.stride:self.stride]
            relative_actions_abs = self.hdf5_datasets[task_index]['data'][demo_key]['actions'][demo_step:demo_step+self.pred_horizon*self.stride:self.stride][:, 0:self.action_dim]
            
            pad_size = self.pred_horizon - relative_actions_abs.shape[0]

            if pad_size > 0:
                relative_actions_abs = np.concatenate([relative_actions_abs, np.zeros((pad_size, self.action_dim))], axis=0)
            
            relative_actions_abs_normalized = 2 * ((relative_actions_abs - self.min) / (self.max - self.min)) - 1

            start = demo_step-(self.obs_horizon-1)*self.stride
            indexed_start = max(start, 0)
            end = demo_step+1

            left_image = self.hdf5_datasets[task_index]['data'][demo_key]['obs']['robot0_agentview_left_image'][indexed_start:end:self.stride]
            right_image = self.hdf5_datasets[task_index]['data'][demo_key]['obs']['robot0_agentview_right_image'][indexed_start:end:self.stride]
            gripper_image = self.hdf5_datasets[task_index]['data'][demo_key]['obs']['robot0_eye_in_hand_image'][indexed_start:end:self.stride]

            pad_size = self.obs_horizon - left_image.shape[0]
            if pad_size > 0:
                first_element = np.expand_dims(left_image[0], axis=0)
                padding = np.concatenate([first_element] * pad_size, axis=0)
                left_image = np.concatenate([padding, left_image], axis=0)

                first_element = np.expand_dims(right_image[0], axis=0)
                padding = np.concatenate([first_element] * pad_size, axis=0)
                right_image = np.concatenate([padding, right_image], axis=0)

                first_element = np.expand_dims(gripper_image[0], axis=0)
                padding = np.concatenate([first_element] * pad_size, axis=0)
                gripper_image = np.concatenate([padding, gripper_image], axis=0)

            left_image = np.stack([self.convert_frame(frame=frame, size=(round(self.frame_width/self.aug['crop']),round(self.frame_height/self.aug['crop'])), swap_rgb=self.swap_rgb) for frame in left_image])
            right_image = np.stack([self.convert_frame(frame=frame, size=(round(self.frame_width/self.aug['crop']),round(self.frame_height/self.aug['crop'])), swap_rgb=self.swap_rgb) for frame in right_image])
            gripper_image = np.stack([self.convert_frame(frame=frame, size=(round(self.frame_width/self.aug['crop']),round(self.frame_height/self.aug['crop'])), swap_rgb=self.swap_rgb) for frame in gripper_image])

        except Exception as e:
            print(f'sample retrive exception: {e}')
            
        # pdb.set_trace()
        left_image = torch.tensor(left_image, dtype=torch.float32)
        right_image = torch.tensor(right_image, dtype=torch.float32)
        gripper_image = torch.tensor(gripper_image, dtype=torch.float32)
        relative_actions_abs_normalized = torch.tensor(relative_actions_abs_normalized, dtype=torch.float32)

        # Rescale from [-1, 1] to [0, 1] for transforms
        left_image = (left_image + 1) / 2
        right_image = (right_image + 1) / 2
        gripper_image = (gripper_image + 1) / 2

        left_image = self.augmentation_transform(left_image, self.transform_rgb)
        right_image = self.augmentation_transform(right_image, self.transform_rgb)
        gripper_image = self.augmentation_transform(gripper_image, self.transform_rgb)

        # # Rescale back to [-1, 1]
        # left_image = left_image * 2 - 1
        # right_image = right_image * 2 - 1
        # gripper_image = gripper_image * 2 - 1

        return {
            "obs": {
                "task_description": clip_embedding,
                "left_image": left_image,
                "right_image": right_image,
                "gripper_image": gripper_image,
                "demo_key": demo_key,
            },
            "action": relative_actions_abs_normalized,
        }
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.indexed_demos) - self.validation_split
            # return 3200
        elif self.mode == 'val':
            return self.validation_split
        elif self.mode == 'test':
            return len(self.indexed_demos)
