import torch
from torch.utils.data import Dataset

import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F


class SurrolAlohaDataset(Dataset):

    def __init__(self, data_root):
        super().__init__()

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
                    if (idx + 4) in traj_action_info:
                        img_path = os.path.join(traj_path, f'img_{idx}.npy')
                        depth_path = os.path.join(traj_path, f'depth_{idx}.npy')
                        seg_path = os.path.join(traj_path, f'seg_{idx}.npy')
                        robot_state = traj_robot_state[idx]
                        actions = [traj_action_info[i] for i in range(idx, idx+5)]

                        cur_data = {'img': img_path, 'depth': depth_path, 
                                    'seg': seg_path, 'robot_state': robot_state,
                                    'actions': actions}
                        # print(f'traj_idx: {name}, idx: {idx}, data: {cur_data}')
                        self.info.append(cur_data)
                    idx += 1
    
    def __len__(self):
        return len(self.info)
    
    def __getitem__(self, index):
        data = self.info[index]
        img = np.load(data['img'])
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = self.transform(img)
        depth = np.load(data['depth'])
        seg = np.load(data['seg'])
        robot_state = data['robot_state']
        actions = np.stack(data['actions'])

        return img, robot_state, actions

class SurrolAlohaSegDepthDataset(Dataset):

    def __init__(self, data_root):
        super().__init__()

        # self.transform = transforms.Compose([
        #     transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        #     transforms.ToTensor(),           # Convert the image to a PyTorch tensor
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        # ])


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
                    if (idx + 4) in traj_action_info:
                        img_path = os.path.join(traj_path, f'img_{idx}.npy')
                        depth_path = os.path.join(traj_path, f'depth_{idx}.npy')
                        seg_path = os.path.join(traj_path, f'seg_{idx}.npy')
                        robot_state = traj_robot_state[idx]
                        actions = [traj_action_info[i] for i in range(idx, idx+5)]

                        cur_data = {'img': img_path, 'depth': depth_path, 
                                    'seg': seg_path, 'robot_state': robot_state,
                                    'actions': actions}
                        # print(f'traj_idx: {name}, idx: {idx}, data: {cur_data}')
                        self.info.append(cur_data)
                    idx += 1
    
    def __len__(self):
        return len(self.info)
    
    def __getitem__(self, index):
        data = self.info[index]
        # img = np.load(data['img'])
        # img = Image.fromarray(img.astype('uint8'), 'RGB')
        # img = self.transform(img)
        depth = np.load(data['depth'])
        seg = np.load(data['seg'])
        seg = (seg == 6).astype(np.float32) # For needlepick only
        depth_seg = torch.stack([torch.from_numpy(depth), torch.from_numpy(seg)])
        depth_seg = F.interpolate(depth_seg.unsqueeze(0), size=(300, 400), mode='bilinear', align_corners=False).squeeze(0)

        robot_state = data['robot_state']
        actions = np.stack(data['actions'])

        return depth_seg, robot_state, actions
    

class SurrolAlohaDataset2(Dataset):

    def __init__(self, data_root):
        super().__init__()

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
                    if (idx + 4) in traj_action_info:
                        img_path = os.path.join(traj_path, f'img_{idx}.npy')
                        depth_path = os.path.join(traj_path, f'depth_{idx}.npy')
                        seg_path = os.path.join(traj_path, f'seg_{idx}.npy')
                        robot_state = traj_robot_state[idx]
                        actions = [traj_action_info[i] for i in range(idx, idx+5)]

                        cur_data = {'img': img_path, 'depth': depth_path, 
                                    'seg': seg_path, 'robot_state': robot_state,
                                    'actions': actions, 'action_idx': idx}
                        # print(f'traj_idx: {name}, idx: {idx}, data: {cur_data}')
                        self.info.append(cur_data)
                    idx += 1
    
    def __len__(self):
        return len(self.info)
    
    def __getitem__(self, index):
        data = self.info[index]
        img = np.load(data['img'])
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        img = self.transform(img)
        depth = np.load(data['depth'])
        seg = np.load(data['seg'])
        robot_state = data['robot_state']
        actions = np.stack(data['actions'])
        action_idx = np.array([data['action_idx']])

        return img, robot_state, action_idx, actions


                

if __name__ == '__main__':
    dataset = SurrolAlohaDataset('/media/jwfu/84D609C8D609BC04/science_robotics_experiment_data/needlepick_aloha')
    for img, robot_state, actions in dataset:
        print(f'img size: {img.shape}')
        print(f'robot_state size: {robot_state.shape}')
        print(f'actions: {actions}')





    
    
