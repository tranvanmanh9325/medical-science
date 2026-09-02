import torch

import time

import numpy as np

from tqdm import tqdm
from abc import abstractmethod



class Trainer:

    def __init__(self, model, optimizer, data_loader, loss_fn, device=None, scheduler=None, eval_fns=None):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.device = device if device != None else torch.device('cuda')
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()

        self.start_time = time.time()

    def train_iteration(self, dataset_repeat_per_iter, iter_num, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        
        for cur_repeat_num in tqdm(range(dataset_repeat_per_iter)):
            for observations, goals, actions, rewards, \
                returns_to_goal, times_to_goal, timesteps, masks in self.data_loader:

                observations = observations.to(device=self.device, dtype=torch.float32)
                goals = goals.to(device=self.device, dtype=torch.float32)
                actions = actions.to(device=self.device, dtype=torch.float32)
                rewards = rewards.to(device=self.device, dtype=torch.float32)
                returns_to_goal = returns_to_goal.to(device=self.device, dtype=torch.float32)
                times_to_goal = times_to_goal.to(device=self.device, dtype=torch.float32)
                timesteps = timesteps.to(device=self.device, dtype=torch.long)
                masks = masks.to(device=self.device, dtype=torch.long)


                train_loss = self.train_step(observations, goals, actions, rewards, \
                                             returns_to_goal, times_to_goal, timesteps, masks)
                
                train_losses.append(train_loss)

                if self.scheduler is not None:
                    self.scheduler.step()




        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            # print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    @abstractmethod
    def train_step(self):
        raise NotImplementedError
