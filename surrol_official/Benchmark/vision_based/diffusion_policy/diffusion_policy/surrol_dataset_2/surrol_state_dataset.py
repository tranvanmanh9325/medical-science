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

class SurrolStateDataset(torch.utils.data.Dataset):
    def __init__(self,
            data_root,
            T_o=1,
            T_a=1,
            ):
        
        super().__init__()
        # Assume To=Ta

        self.info = []
        for root, dirs, files in os.walk(data_root):
            
            for name in dirs:
                traj_path = os.path.join(root, name)
                traj_action_path = os.path.join(traj_path, 'action.pth')
                traj_observation_path = os.path.join(traj_path, 'observation.pth')
                traj_goal_path = os.path.join(traj_path, 'goal.pth')

                traj_action_info = torch.load(traj_action_path)
                traj_observation_info = torch.load(traj_observation_path)
                traj_goal_info = torch.load(traj_goal_path)


                file_names = os.listdir(traj_path)
                idx = 0
                while idx in traj_action_info and idx in traj_observation_info and idx in traj_goal_info:
                    if (idx + T_a - 1) in traj_action_info and (idx - T_o + 1) in traj_observation_info:
                        # img_path = [os.path.join(traj_path, f'img_{i}.npy') for i in range(idx - T_o + 1, idx + 1)]
                        # depth_path = [os.path.join(traj_path, f'depth_{i}.npy') for i in range(idx - T_o + 1, idx + 1)]
                        # seg_path = [os.path.join(traj_path, f'seg_{i}.npy') for i in range(idx - T_o + 1, idx + 1)]
                        observations = [traj_observation_info[i] for i in range(idx - T_o + 1, idx + 1)]
                        goals = [traj_goal_info[i] for i in range(idx - T_o + 1, idx + 1)]
                        actions = [traj_action_info[i] for i in range(idx, idx+T_a)]

                        cur_data = {'observations': observations, 'goals': goals,
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
            raw_agent_pos_data.append(d['obs'])
        raw_action_data = torch.stack(raw_action_data)
        raw_agent_pos_data = torch.stack(raw_agent_pos_data)

        data = {
            'action': raw_action_data,
            'obs': raw_agent_pos_data,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer
    
    def get_all_actions(self) -> torch.Tensor:
        raw_action_data = []
        import tqdm
        for d in tqdm.tqdm(self):
            raw_action_data.append(d['action'])
        raw_action_data = torch.stack(raw_action_data)
        return raw_action_data

    def __len__(self) -> int:
        return len(self.info)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.info[idx]
        # img = [Image.fromarray(np.load(data['img'][i]).astype('uint8'), 'RGB') for i in range(0, len(data['img']))]
        # img = np.stack([self.transform(ig) for ig in img])

        observation = np.stack(data['observations'])
        goal = np.stack(data['goals'])
        actions = np.stack(data['actions'])

        obs = np.concatenate([observation, goal], axis=1)

        data = {
            'obs': obs,
            'action': actions  # T, 2
        }

        data = dict_apply(data, torch.from_numpy)

        return data


# def test():
#     dataset = SurrolStateDataset('/research/d1/gds/jwfu/SurRoL_science_robotics_experiment/experiment_data/needlepick_aloha', T_o=4, T_a=4)
#     for data in dataset:
#         print(f'data keys: {data.keys()}')
#         print(f"image shape {data['obs']['image'].shape}")
#         print(f"robot state shape {data['obs']['agent_pos'].shape}")
#         print(f'action shape: {data["action"].shape}')


# if __name__ == '__main__':
#     test()
