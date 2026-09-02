import torch
import transformers

import torch.nn as nn

from modules.trajectory_gpt2 import GPT2Model


class PretrainedModelBuilder:

    @staticmethod
    def build(file_path, task, idx, max_ep_len, hidden_size, \
              observation_dim, goal_dim, action_dim,):
        gpt_backbone = PretrainedModelBuilder.build_backbone(file_path, idx)
        task_modules = PretrainedModelBuilder.build_modules(file_path, task, idx, max_ep_len, \
                                                            hidden_size, observation_dim, goal_dim, \
                                                            action_dim)
        
        return gpt_backbone, task_modules
        
    @staticmethod
    def build_backbone(file_path, idx):
        # Add parse parameters in future
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
        GPT_backbone = GPT2Model(config)
        GPT_backbone.load_state_dict(torch.load(file_path + '/gpt_backbone/gpt_' + str(idx) + '.pth'))

        return GPT_backbone


    @staticmethod
    def build_modules(file_path, task, idx, max_ep_len, hidden_size, observation_dim, \
                        goal_dim, action_dim, action_tanh=True):
        task_modules = {'embed_timestep':nn.Embedding(max_ep_len, hidden_size),
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
        for module_name in task_modules.keys():
            task_modules[module_name].load_state_dict(torch.load(file_path + '/' + task + '/' + \
                                                                 module_name + '/' + str(idx) + '.pth'))

        return task_modules

        