import os
import torch
import cv2
import gym
import argparse
import imageio
import transformers

import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader

from utils.evaluation import SequenceEvaluator
from utils.surrol_data import SurRoLDataset
# from models.decision_transformer import DecisionTransformer
from agents.goal_conditioned_decision_transformer import GoalConditionedDecisionTransformer
from trainers.seq_trainer import SequenceTrainer, MultiObjectiveSequenceTrainer
from utils.model_loader import PretrainedModelBuilder
from modules.trajectory_gpt2 import GPT2Model
from trainers.seq_trainer import SequenceTrainer, MultiObjectiveSequenceTrainer


def experiment(variant):
    device = torch.device(variant['device'])

    # Environment and parameters
    task = variant['env']
    env = gym.make(task)
    obs = env.reset()
    env.close()
    observation_dim = obs['observation'].shape[0]
    achieved_goal_dim = obs['achieved_goal'].shape[0]
    goal_dim = obs['desired_goal'].shape[0]
    action_dim = env.action_space.shape[0]

    # Construct dataset and dataloader
    root = os.getcwd() + '/data/success_demo/'
    surrol_dataset = SurRoLDataset(root, seq_max_length=100, task=task, with_relabel=True)
    surrol_dataloader = DataLoader(dataset=surrol_dataset, batch_size=variant['batch_size'], shuffle=True)
    offline_data_obs_mean, offline_data_obs_std, offline_data_goal_mean, offline_data_goal_std = \
    surrol_dataset.pop_normalization_parameters()

    # Build model
    # TODO: parse environment parameters from environments
    if variant['use_pretrained']:

        # Build embedding and GPT backbone from pretrained parameter for task
        hidden_size = variant['hidden_size']
        max_ep_len = variant['max_ep_len']
        file_path = variant['ckpt_path']
        idx = variant['load_epoch'] # Select a position to load pth

        GPT_backbone, task_modules = PretrainedModelBuilder.build(file_path, task, idx, max_ep_len, hidden_size, \
                                                                  observation_dim, goal_dim, action_dim)

        # Give a "None" to the value invalid in model building from modules
        model = GoalConditionedDecisionTransformer(
            state_dim=observation_dim,
            goal_dim=goal_dim,
            act_dim=action_dim,
            max_length=100, # Change to 100 future
            max_ep_len=1000,
            hidden_size=hidden_size,
            n_layer=None,
            n_head=None,
            n_inner=None,
            activation_function=None,
            n_positions=None,
            resid_pdrop=None,
            attn_pdrop=None,
            GPT_backbone=GPT_backbone,
            # embed_timestep=task_modules['embed_timestep'],
            # embed_return=task_modules['embed_return'],
            # embed_time_to_goal=task_modules['embed_time_to_goal'],
            # embed_state=task_modules['embed_state'],
            # embed_goal=task_modules['embed_goal'],
            # embed_action=task_modules['embed_action'],
            # embed_ln=task_modules['embed_ln'],
            # predict_state=task_modules['predict_state'],
            # predict_goal=task_modules['predict_goal'],
            # predict_action=task_modules['predict_action'],
            # predict_return=task_modules['predict_return'],
            # predict_time_to_goal=task_modules['predict_time_to_goal'],
        )

    else:
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
        )
    
    # # Load ckpt with moe here
    # ckpt_path = '/research/d1/gds/jwfu/SurRoL_science_robotics_experiment_state_based/rl/trained_models/PickAndPlaceRL-v0_153_0.41999998688697815.pth'
    # state_dict = torch.load(ckpt_path, map_location='cpu')
    # model.load_state_dict(state_dict, strict=False)
    # for name, param in model.named_parameters():
    #     if 'experts' in name:
    #         print(f'name: {name}')
    #         name_in_init = name[:20] + name[30:]
    #         print(f'name_in_init: {name_in_init}')
    #         print(f'param before load: {param}')
    #         param = state_dict[name_in_init]
    #         print(f'param after load: {param}')

    model.to(device=device)

    if variant['multi_gpu']:
        model = nn.DataParallel(model)

    # Construct optimizer and scheduler
    warmup_steps = 10000
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    # Construct loss function
    loss_function = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2)

    # Trainer and evaluator
    if variant['multi_objective']:
        trainer = MultiObjectiveSequenceTrainer(
            model=model,
            optimizer=optimizer,
            data_loader=surrol_dataloader,
            loss_fn=loss_function,
            device=device,
            scheduler=scheduler,
        )
    else:
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            data_loader=surrol_dataloader,
            loss_fn=loss_function,
            device=device,
            scheduler=scheduler,
        )

    evaluator = SequenceEvaluator(
        task=task,
        device=device,
        observation_mean=offline_data_obs_mean,
        observation_std=offline_data_obs_std,
        goal_mean=offline_data_goal_mean,
        goal_std=offline_data_goal_std,
        target_return=0.,
        target_time_to_goal_init=50,
        max_ep_len=100,
        reward_scale=1,
    )

    # Training and evaluation
    for iter in range(200):

        print('=' * 80)
        print('Iteration: {}'.format(iter + 1))
        print('training')
        training_outputs = trainer.train_iteration(dataset_repeat_per_iter=50, iter_num=iter+1, print_logs=True)
        
        print('evaluation')
        epi_returns, epi_lengthes, is_successes, success_render_obss, failed_render_obss = \
        evaluator.step_batch(trainer.model.module if variant['multi_gpu'] else trainer.model, 50, render=True)

        torch.save(trainer.model.module.state_dict() if variant['multi_gpu'] else trainer.model.state_dict(), os.getcwd() + '/trained_models/{}_{}_{}.pth'.format(task, iter, np.around(np.mean(is_successes), 2)))


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

def experiment_pretrain(variant):
    # Device
    device = torch.device(variant['device'])

    # Necessary for GPT backbone
    hidden_size = variant['hidden_size']
    max_ep_len = variant['max_ep_len']
    action_tanh = variant['action_tanh']



    # Tasks and corresponding modules of tasks
    tasks_sequence = [
                      'NeedleReachRL-v0',
                      'GauzeRetrieveRL-v0',
                      'NeedlePickRL-v0',
                      'PegTransferRL-v0',
                      'NeedleRegraspRL-v0',
                      'BiPegTransferRL-v0',
                       'BiPegBoardRL-v0',
                       'MatchBoardRL-v0',
                       'MatchBoardPanelRL-v0',
                       'PickAndPlaceRL-v0',
                      ]
    
    task_information = {'NeedleReachRL-v0': {'observation_dim': 7, 'goal_dim': 3, 'action_dim': 5},
                        'GauzeRetrieveRL-v0': {'observation_dim': 19, 'goal_dim': 3, 'action_dim': 5},
                        'NeedlePickRL-v0': {'observation_dim': 19, 'goal_dim': 3, 'action_dim': 5},
                        'PegTransferRL-v0': {'observation_dim': 19, 'goal_dim': 3, 'action_dim': 5},
                        'NeedleRegraspRL-v0': {'observation_dim': 35, 'goal_dim': 3, 'action_dim': 10},
                        'BiPegTransferRL-v0': {'observation_dim': 35, 'goal_dim': 3, 'action_dim': 10},
                        'BiPegBoardRL-v0': {'observation_dim': 35, 'goal_dim': 3, 'action_dim': 10},
                        'MatchBoardRL-v0': {'observation_dim': 20, 'goal_dim': 3, 'action_dim': 5},
                        'MatchBoardPanelRL-v0': {'observation_dim': 31, 'goal_dim': 6, 'action_dim': 5},
                        'PickAndPlaceRL-v0': {'observation_dim': 31, 'goal_dim': 6, 'action_dim': 5}
                        }
    tasks_modules = {}

    for task in tasks_sequence:
        # Initialization task modules 

        observation_dim = task_information[task]['observation_dim']
        goal_dim = task_information[task]['goal_dim']
        action_dim = task_information[task]['action_dim']


        tasks_modules[task] = {'embed_timestep':nn.Embedding(max_ep_len, hidden_size),
                               'embed_return':torch.nn.Linear(1, hidden_size),
                               'embed_time_to_goal':torch.nn.Linear(1, hidden_size),
                               'embed_state':torch.nn.Linear(observation_dim, hidden_size),
                               'embed_goal':torch.nn.Linear(goal_dim, hidden_size),
                               'embed_action':torch.nn.Linear(action_dim, hidden_size),
                               'embed_ln':nn.LayerNorm(hidden_size),
                               'predict_state':torch.nn.Linear(hidden_size, observation_dim),
                               'predict_goal':torch.nn.Linear(hidden_size, goal_dim),
                               'predict_action':nn.Sequential(
                                *([nn.Linear(hidden_size, action_dim)] + ([nn.Tanh()] if action_tanh else []))),
                               'predict_return':torch.nn.Linear(hidden_size, 1),
                               'predict_time_to_goal':torch.nn.Linear(hidden_size, 1)
                               }

    # Build model backbone
    # TODO: parse data from argparser
    config = transformers.GPT2Config(
        vocab_size=1,  # doesn't matter -- we don't use the vocab
        n_embd=128, # Namely hidden_size
        n_layer=8,
        n_head=4,
        n_inner=512,
        activation_function='relu',
        n_positions=1024,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        # **kwargs
    )
    # note: the only difference between this GPT2Model and the default Huggingface version
    # is that the positional embeddings are removed (since we'll add those ourselves)
    GPT_backbone = GPT2Model(config)

    for iter in range(10):
        print('*'*50)
        print('Start training iter: {}'.format(iter+1))

        loss_mean = {}
        loss_std = {}

        np.random.shuffle(tasks_sequence)

        for task in tasks_sequence:

            print('Training iter: {}, task: {}'.format(iter+1, task))

            # Build full model
            # Give a "None" to the value invalid in model building from modules
            model = GoalConditionedDecisionTransformer(
                state_dim=None,
                goal_dim=None,
                act_dim=None,
                max_length=100, 
                max_ep_len=None,
                hidden_size=hidden_size,
                n_layer=None,
                n_head=None,
                n_inner=None,
                activation_function=None,
                n_positions=None,
                resid_pdrop=None,
                attn_pdrop=None,
                GPT_backbone=GPT_backbone,
                embed_timestep=tasks_modules[task]['embed_timestep'],
                embed_return=tasks_modules[task]['embed_return'],
                embed_time_to_goal=tasks_modules[task]['embed_time_to_goal'],
                embed_state=tasks_modules[task]['embed_state'],
                embed_goal=tasks_modules[task]['embed_goal'],
                embed_action=tasks_modules[task]['embed_action'],
                embed_ln=tasks_modules[task]['embed_ln'],
                predict_state=tasks_modules[task]['predict_state'],
                predict_goal=tasks_modules[task]['predict_goal'],
                predict_action=tasks_modules[task]['predict_action'],
                predict_return=tasks_modules[task]['predict_return'],
                predict_time_to_goal=tasks_modules[task]['predict_time_to_goal'],
            )

            model.to(device=device)


            if variant['multi_gpu']:
                model = nn.DataParallel(model)

            # Build dataset
            # Construct dataset and dataloader
            root = os.getcwd() + '/data/success_demo/'
            surrol_dataset = SurRoLDataset(root, seq_max_length=100, task=task, with_relabel=True)
            surrol_dataloader = DataLoader(dataset=surrol_dataset, batch_size=variant['batch_size'], shuffle=True)
            offline_data_obs_mean, offline_data_obs_std, offline_data_goal_mean, offline_data_goal_std = \
            surrol_dataset.pop_normalization_parameters()

            # Construct optimizer and scheduler
            warmup_steps = 10000
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda steps: min((steps+1)/warmup_steps, 1)
            )

            # Construct loss function
            loss_function = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2)

            # Trainer and evaluator
            if variant['multi_objective']:
                trainer = MultiObjectiveSequenceTrainer(
                    model=model,
                    optimizer=optimizer,
                    data_loader=surrol_dataloader,
                    loss_fn=loss_function,
                    device=device,
                    scheduler=scheduler,
                    multi_gpu=variant['multi_gpu'],
                    random_mask_reconstruction=variant['random_reconstruction'],
                )
            else:
                trainer = SequenceTrainer(
                    model=model,
                    optimizer=optimizer,
                    data_loader=surrol_dataloader,
                    loss_fn=loss_function,
                    device=device,
                    scheduler=scheduler,
                )

            training_outputs = trainer.train_iteration(dataset_repeat_per_iter=1, iter_num=iter+1, print_logs=False)
            loss_mean[task] = training_outputs['training/train_loss_mean']
            loss_std[task] = training_outputs['training/train_loss_std']

            # Store parameters of task-specified modules 
            for module_name, module in tasks_modules[task].items():
                if not os.path.exists(os.getcwd() + '/checkpoints/{}/{}'.format(task, module_name)):
                    os.makedirs(os.getcwd() + '/checkpoints/{}/{}'.format(task, module_name))
                torch.save(module.state_dict(), 'checkpoints/{}/{}/{}.pth'.format(task, module_name, iter+1))
        
        # Store GPT backbone
        if not os.path.exists(os.getcwd() + '/checkpoints/gpt_backbone'):
            os.makedirs(os.getcwd() + '/checkpoints/gpt_backbone')
        torch.save(GPT_backbone.state_dict(), 'checkpoints/gpt_backbone/gpt_{}.pth'.format(iter+1))

        # Print loss mean and std for all task in current iter
        for task in tasks_sequence:
            print('iter: {}, task: {}, loss mean: {}, loss std: {}'.format(iter+1, task, loss_mean[task], loss_std[task]))









