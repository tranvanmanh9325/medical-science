import gym
import torch
import os
import argparse
import imageio
import sys

import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

# Add the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.surrol_data import SurRoLDataset
from agents.goal_conditioned_decision_transformer import GoalConditionedDecisionTransformer



class SequenceEvaluator:
    
    def __init__(self, task, device, observation_mean, \
                 observation_std, goal_mean, goal_std, \
                 target_return=0.0, target_time_to_goal_init=50, \
                 max_ep_len=1000, max_input_len=50, \
                 reward_scale=50):
        self.env = gym.make(task)

        self.device = device

        self.observation_mean = torch.from_numpy(observation_mean).to(device=device)
        self.observation_std = torch.from_numpy(observation_std).to(device=device)
        self.goal_mean = torch.from_numpy(goal_mean).to(device=device)
        self.goal_std = torch.from_numpy(goal_std).to(device=device)

        self.target_return = target_return
        self.target_time_to_goal_init = target_time_to_goal_init

        self.max_ep_len = max_ep_len
        self.max_input_len = max_input_len

        self.reward_scale = reward_scale

    def step(self, model, with_render=False):
        model.eval()
        state = self.env.reset()
        observation, goal = state['observation'], state['desired_goal']

        observation_dim = observation.shape[-1]
        goal_dim = goal.shape[-1]
        action_dim = self.env.action_space.shape[0]

        observations = torch.from_numpy(observation).reshape(1, observation_dim).to(device=self.device, dtype=torch.float32)
        goals = torch.from_numpy(goal).reshape(1, goal_dim).to(device=self.device, dtype=torch.float32)
        actions = torch.zeros((0, action_dim), device=self.device, dtype=torch.float32)
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)

        target_return = torch.tensor(self.target_return, device=self.device, dtype=torch.float32).reshape(1, 1)
        target_time_to_goal = torch.tensor(self.target_time_to_goal_init, device=self.device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)

        episode_return, episode_length = 0, 0
        
        if with_render:
            render_obss = []

        for t in range(self.max_ep_len):

            # Add padding
            actions = torch.cat([actions, torch.zeros((1, action_dim), device=self.device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])

            action = model.get_action(
                (observations.to(dtype=torch.float32) - self.observation_mean) / self.observation_std,
                (goals.to(dtype=torch.float32) - self.goal_mean) / self.goal_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                target_time_to_goal.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )

            actions[-1] = action
            action = action.detach().cpu().numpy()

            if with_render:
                render_obs = self.env.render('rgb_array')
                render_obss.append(render_obs)

            state, reward, done, info = self.env.step(action)

            cur_observation = torch.from_numpy(state['observation']).to(device=self.device).reshape(1, observation_dim)
            cur_goal = torch.from_numpy(state['desired_goal']).to(device=self.device).reshape(1, goal_dim)
            observations = torch.cat([observations, cur_observation], dim=0)
            goals = torch.cat([goals, cur_goal], dim=0)
            rewards[-1] = reward

            # Design for delayed reward
            pred_return = target_return[0, -1] - (reward / self.reward_scale)
            target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

            pred_time_to_goal = target_time_to_goal[0, -1] - 1.
            target_time_to_goal = torch.cat([target_time_to_goal, pred_time_to_goal.reshape(1, 1)], dim=1)

            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=self.device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1

            is_success = info['is_success']

            if done or is_success:
                break
        
        if with_render:
            return episode_return, episode_length, is_success, render_obss
        else:
            return episode_return, episode_length, is_success

    def step_batch(self, model, eval_num_per_iter, print_logs=True, render=False):
        episode_returns, episode_lengthes, is_successes = [], [], []
        success_renders = []
        failed_renders = []

        for _ in tqdm(range(eval_num_per_iter)):

            if render:
                episode_return, episode_length, is_success, render_obss = self.step(model, with_render=render)

                if is_success:
                    success_renders.append(np.array(render_obss))
                else:
                    failed_renders.append(np.array(render_obss))
            else:
                episode_return, episode_length, is_success = self.step(model, with_render=render)


            episode_returns.append(episode_return)
            episode_lengthes.append(episode_length)
            is_successes.append(is_success)
        
        episode_returns = np.array(episode_returns)
        episode_lengthes = np.array(episode_lengthes)
        is_successes = np.array(is_successes)

        if print_logs:
            print('evaluation/episode_returns_mean: {}'.format(np.mean(episode_returns)))
            print('evaluation/episode_returns_std: {}'.format(np.std(episode_returns)))

            print('evaluation/episode_lengthes_mean: {}'.format(np.mean(episode_lengthes)))
            print('evaluation/episode_lengthes_std: {}'.format(np.std(episode_lengthes)))

            print('evaluation/success_rate: {}'.format(np.mean(is_successes)))
        
        if render:
            return episode_returns, episode_lengthes, is_successes, success_renders, failed_renders
        else:
            return episode_returns, episode_lengthes, is_successes

def evaluate(variant):
    device = torch.device(variant['device'])

    # Environment and parameters
    task = variant['env']
    env = gym.make(task)
    obs = env.reset()
    observation_dim = obs['observation'].shape[0]
    achieved_goal_dim = obs['achieved_goal'].shape[0]
    goal_dim = obs['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]
    env.close()

    # Construct dataset and dataloader
    root = os.getcwd() + '/data/success_demo/'
    surrol_dataset = SurRoLDataset(root, seq_max_length=100, task=task, with_relabel=True)
    surrol_dataloader = DataLoader(dataset=surrol_dataset, batch_size=variant['batch_size'], shuffle=True)
    offline_data_obs_mean, offline_data_obs_std, offline_data_goal_mean, offline_data_goal_std = \
    surrol_dataset.pop_normalization_parameters()

    model = GoalConditionedDecisionTransformer(
        state_dim=observation_dim,
        goal_dim=goal_dim,
        act_dim=action_dim,
        max_length=100, 
        max_ep_len=1000,
        hidden_size=variant['hidden_size'],
        n_layer=8,
        n_head=4,
        n_inner=4*128,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
    ).to(device=device)

    checkpoint_path = variant['ckpt_path']
    model.load_state_dict(torch.load(checkpoint_path))

    evaluator = SequenceEvaluator(
        task=task,
        device=device,
        observation_mean=offline_data_obs_mean,
        observation_std=offline_data_obs_std,
        goal_mean=offline_data_goal_mean,
        goal_std=offline_data_goal_std,
        target_return=0.,
        target_time_to_goal_init=variant['expected_step'],
        max_ep_len=100,
        reward_scale=1,
    )

    best_res = 0
    for iter in range(10):
        print('evaluation iter: {}'.format(iter))
        epi_returns, epi_lengthes, is_successes, success_render_obss, failed_render_obss = \
        evaluator.step_batch(model, 10, render=True)
        best_res = max(best_res, np.mean(is_successes))
        print('best result: {}'.format(best_res))

        for idx, epi_success_obss in enumerate(success_render_obss):
            writer = imageio.get_writer('video/{}_success_video/success_epoch_{}_idx_{}.mp4'.format(variant['env'], iter+1, idx), fps=20)
            for img in epi_success_obss:
                writer.append_data(img)
            writer.close()

        
        for idx, epi_failed_obss in enumerate(failed_render_obss):
            if idx % 5 != 0:
                continue
            writer = imageio.get_writer('video/{}_fail_video/fail_epoch_{}_idx_{}.mp4'.format(variant['env'], iter+1, idx), fps=20)
            for img in epi_failed_obss:
                writer.append_data(img)
            writer.close()

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PegTransfer-v0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--expected_step', type=int, default=50)
    parser.add_argument('--ckpt_path', type=str, default='')



    args = parser.parse_args()

    evaluate(vars(args))
