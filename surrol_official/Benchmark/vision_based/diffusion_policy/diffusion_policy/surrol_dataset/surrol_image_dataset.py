from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

import os
from PIL import Image
import torchvision.transforms as transforms

class SurrolImageDataset(torch.utils.data.Dataset):
    def __init__(self,
            data_root,
            T_o=1,
            T_a=1,
            ):
        
        super().__init__()
        # Assume To=Ta

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
            transforms.ToTensor(),           # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        ])

        self.info = []
        for root, dirs, files in os.walk(data_root):
            
            for name in dirs:
                traj_path = os.path.join(root, name)
                traj_action_path = os.path.join(traj_path, 'action.pth')
                traj_robot_state_path = os.path.join(traj_path, 'robot_state.pth')
                traj_action_info = torch.load(traj_action_path)
                traj_robot_state = torch.load(traj_robot_state_path)

                file_names = os.listdir(traj_path)
                idx = 0
                while f'img_{idx}.npy' in file_names and f'depth_{idx}.npy' in file_names and f'seg_{idx}.npy' in file_names and idx in traj_action_info and idx in traj_robot_state:
                    if (idx + T_a - 1) in traj_action_info and (idx - T_o + 1) in traj_action_info:
                        img_path = [os.path.join(traj_path, f'img_{i}.npy') for i in range(idx - T_o + 1, idx + 1)]
                        # depth_path = [os.path.join(traj_path, f'depth_{i}.npy') for i in range(idx - T_o + 1, idx + 1)]
                        # seg_path = [os.path.join(traj_path, f'seg_{i}.npy') for i in range(idx - T_o + 1, idx + 1)]
                        robot_state = [traj_robot_state[i] for i in range(idx - T_o + 1, idx + 1)]
                        actions = [traj_action_info[i] for i in range(idx, idx+T_a)]

                        cur_data = {'img': img_path, 'robot_state': robot_state,
                                    'actions': actions}
                        # print(f'traj_idx: {name}, idx: {idx}, data: {cur_data}')
                        self.info.append(cur_data)
                    idx += 1
        # self.info = self.info

    def get_validation_dataset(self):
        return self

    def get_normalizer(self, mode='limits', **kwargs):
        raw_action_data = []
        raw_agent_pos_data = []
        import tqdm
        for d in tqdm.tqdm(self):
            raw_action_data.append(d['action'])
            raw_agent_pos_data.append(d['obs']['agent_pos'])
        raw_action_data = torch.stack(raw_action_data)
        raw_agent_pos_data = torch.stack(raw_agent_pos_data)

        data = {
            'action': raw_action_data,
            'agent_pos': raw_agent_pos_data,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.info)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.info[idx]
        img = [Image.fromarray(np.load(data['img'][i]).astype('uint8'), 'RGB') for i in range(0, len(data['img']))]
        img = np.stack([self.transform(ig) for ig in img])

        robot_state = np.stack(data['robot_state'])
        actions = np.stack(data['actions'])

        data = {
            'obs': {
                'image': img, # T, 3, 96, 96
                'agent_pos': robot_state, # T, 2
            },
            'action': actions  # T, 2
        }

        data = dict_apply(data, torch.from_numpy)

        return data


def test():
    dataset = SurrolImageDataset('/research/d1/gds/jwfu/SurRoL_science_robotics_experiment/experiment_data/needlepick_aloha', T_o=4, T_a=4)
    for data in dataset:
        print(f'data keys: {data.keys()}')
        print(f"image shape {data['obs']['image'].shape}")
        print(f"robot state shape {data['obs']['agent_pos'].shape}")
        print(f'action shape: {data["action"].shape}')


if __name__ == '__main__':
    test()
