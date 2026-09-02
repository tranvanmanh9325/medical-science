import numpy as np
import torch

import torch.nn as nn

from trainers.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self, observations, goals, actions, rewards, \
                   returns_to_goal, times_to_goal, timesteps, attention_masks):

        action_targets = torch.clone(actions)

        observation_preds, goal_preds, action_preds, return_preds, time_to_goal_preds = self.model(
            observations, goals, actions, rewards, returns_to_goal, times_to_goal, timesteps, attention_masks)


        act_dim = action_preds.shape[-1]
        action_preds = action_preds.reshape(-1, act_dim)[attention_masks.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_masks.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_targets, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_targets)**2).detach().cpu().item()

        return loss.detach().cpu().item()


class MultiObjectiveSequenceTrainer(SequenceTrainer):

    def __init__(self, model, optimizer, data_loader, loss_fn, \
                 device=None, scheduler=None, eval_fns=None, multi_gpu=False, random_mask_reconstruction=False):
        super().__init__(model, optimizer, data_loader, loss_fn, device, scheduler, eval_fns)
        self.multi_gpu = multi_gpu
        self.random_mask_reconstruction = random_mask_reconstruction

    def train_step(self, observations, goals, actions, rewards, \
                   returns_to_goal, times_to_goal, timesteps, attention_masks):
        
        observation_targets = torch.clone(observations)
        goal_targets = torch.clone(goals)
        action_targets = torch.clone(actions)
        return_to_goal_targets = torch.clone(returns_to_goal)
        time_to_goal_targets = torch.clone(times_to_goal)

        # Loss I: action prediction loss
        _, _, action_preds, _, _ = self.model(
            observations, goals, actions, rewards, returns_to_goal, times_to_goal, timesteps, attention_masks)

        act_dim = action_preds.shape[-1]
        action_preds = action_preds.reshape(-1, act_dim)[attention_masks.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_masks.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_targets, None,
        )

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_targets)**2).detach().cpu().item()

        # Loss II: forward dynamics prediction 
        auxiliary_loss_fn = nn.MSELoss()

        if self.multi_gpu:
            observation_preds = self.model.module.forward_dynamics_prediction(
                observations, goals, actions, rewards, returns_to_goal, times_to_goal, timesteps, attention_masks)
        else:
            observation_preds = self.model.forward_dynamics_prediction(
                observations, goals, actions, rewards, returns_to_goal, times_to_goal, timesteps, attention_masks)
        
        observation_dim = observation_preds.shape[-1]
        observation_preds = observation_preds.reshape(-1, observation_dim)[attention_masks.reshape(-1) > 0]
        observation_targets = observation_targets.reshape(-1, observation_dim)[attention_masks.reshape(-1) > 0]

        forward_dynamics_prediction_loss = auxiliary_loss_fn(observation_preds, observation_targets)

        with torch.no_grad():
            self.diagnostics['training/forward_dynamics_error'] = \
                torch.mean((observation_preds-observation_targets)**2).detach().cpu().item()

        # Loss III: time to goal prediction 
        if self.multi_gpu:
            time_to_goal_preds = self.model.module.forward_times_to_goal_prediction(
                observations, goals, actions, rewards, returns_to_goal, times_to_goal, timesteps, attention_masks
            )
        else:
            time_to_goal_preds = self.model.forward_times_to_goal_prediction(
                observations, goals, actions, rewards, returns_to_goal, times_to_goal, timesteps, attention_masks
            )

        time_to_goal_preds = time_to_goal_preds.reshape(-1, 1)[attention_masks.reshape(-1) > 0]
        time_to_goal_targets = time_to_goal_targets.reshape(-1, 1)[attention_masks.reshape(-1) > 0]

        time_to_goal_prediction_loss = auxiliary_loss_fn(time_to_goal_preds, time_to_goal_targets)

        with torch.no_grad():
            self.diagnostics['training/times_to_goal_prediction_error'] = \
                torch.mean((time_to_goal_preds-time_to_goal_targets)**2).detach().cpu().item()
        
        # Loss IV: random mask reconstruction 
        if self.random_mask_reconstruction:
            if self.multi_gpu:
                action_consts = \
                    self.model.module.forward_masked_sequence_reconstruction(
                    observations, goals, actions, rewards, returns_to_goal, times_to_goal, timesteps, attention_masks)
            else:
                action_consts = \
                    self.model.forward_masked_sequence_reconstruction(
                    observations, goals, actions, rewards, returns_to_goal, times_to_goal, timesteps, attention_masks)
            
            # goal_dim = goal_consts.shape[-1]

            # observation_consts = observation_consts.reshape(-1, observation_dim)[attention_masks.reshape(-1) > 0]
            # goal_consts = goal_consts.reshape(-1, goal_dim)[attention_masks.reshape(-1) > 0]
            action_consts = action_consts.reshape(-1, act_dim)[attention_masks.reshape(-1) > 0]
            # return_consts = return_consts.reshape(-1, 1)[attention_masks.reshape(-1) > 0]
            # time_to_goal_consts = time_to_goal_consts.reshape(-1, 1)[attention_masks.reshape(-1) > 0]

            # goal_targets = goal_targets.reshape(-1, goal_dim)[attention_masks.reshape(-1) > 0]
            # return_to_goal_targets = return_to_goal_targets.reshape(-1, 1)[attention_masks.reshape(-1) > 0]
            
            sequence_reconstruction_loss = auxiliary_loss_fn(action_consts, action_targets)
        
        with torch.no_grad():
            self.diagnostics['training/sequence_reconstruction_error'] = \
                torch.mean((time_to_goal_preds-time_to_goal_targets)**2).detach().cpu().item()
        
        loss_final = loss + 0.1*forward_dynamics_prediction_loss + 0.1*time_to_goal_prediction_loss
        
        if self.random_mask_reconstruction:
            loss_final += 0.1*sequence_reconstruction_loss
        
        
        self.optimizer.zero_grad()
        loss_final.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .025)
        self.optimizer.step()

        return loss_final.detach().cpu().item()
