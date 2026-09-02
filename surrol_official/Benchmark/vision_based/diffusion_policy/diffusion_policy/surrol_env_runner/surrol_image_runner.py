import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
import gym
from PIL import Image
import torchvision.transforms as transforms


class SurrolImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            env_name,
        ):
        super().__init__(output_dir)


        self.env = gym.make(env_name, render_mode='rgb_array')

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
            transforms.ToTensor(),           # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
        ])

    
    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        is_success_all = []

        print(f'evaluating...')
        for chunk_idx in tqdm.tqdm(range(20)):

            # start rollout
            # obs = env.reset()
            past_action = None
            policy.reset()

            state = env.reset()
            # observation, goal = state['observation'], state['desired_goal']
            img = state['img']
            img = Image.fromarray(img.astype('uint8'), 'RGB')
            img = self.transform(img).numpy()
            robot_state = state['robot_state']
            goal = state['desired_goal']

            # Assuming horizon == 4
            img_input = np.stack([img for _ in range(4)])
            agent_pos_input = np.stack([np.concatenate([robot_state, goal], axis=0) for _ in range(4)])

            episode_return, episode_length = 0, 0
            
            done = False
            is_success = False
            for _ in range(100):
                # create obs dict
                # np_obs_dict = dict(obs)
                # if self.past_action and (past_action is not None):
                #     # TODO: not tested
                #     np_obs_dict['past_action'] = past_action[
                #         :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                img_input = np.concatenate([img_input[1:], np.expand_dims(img, 0)], axis=0)
                agent_pos_input = np.concatenate([agent_pos_input[1:], np.expand_dims(np.concatenate([robot_state, goal], axis=0), 0)], axis=0)


                obs_dict = {'image': img_input, 'agent_pos': agent_pos_input}

                obs_dict = dict_apply(obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device).unsqueeze(0))
                # print(f'device: {device}')
            
                # for k, v in obs_dict.items():
                #     print(f'k: {k}, v shape: {v.shape}')

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)
                
                # print(f'action_dict')
                # for k, v in action_dict.items():
                #     print(f'k: {k}, shape: {v.size()}')
                # exit()

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'][0][0]

                # print(f'action: {action}')

                # step env
                # obs, reward, done, info = env.step(action)
                state, reward, done, info = env.step(action)
                img = state['img']
                img = Image.fromarray(img.astype('uint8'), 'RGB')
                img = self.transform(img).numpy()

                robot_state = state['robot_state']
                goal = state['desired_goal']
                obs_dict = {'image': img, 'agent_pos': np.concatenate([robot_state, goal], axis=0)}

                # done = np.all(done)
                past_action = action
                is_success = info['is_success']


                if done or is_success:
                    break
            
            is_success_all.append(is_success)

        log_data = {'success rate': np.mean(is_success_all)}

        return log_data

