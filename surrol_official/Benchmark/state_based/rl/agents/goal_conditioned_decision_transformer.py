import numpy as np
import torch
import torch.nn as nn

import transformers

from modules.model import TrajectoryModel
from modules.trajectory_gpt2 import GPT2Model


class GoalConditionedDecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model 
    """

    def __init__(
            self,
            state_dim,
            goal_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            # Pretrained modules
            GPT_backbone=None,
            embed_timestep=None,
            embed_return=None,
            embed_time_to_goal=None,
            embed_state=None,
            embed_goal=None,
            embed_action=None,
            embed_ln=None,
            predict_state=None,
            predict_goal=None,
            predict_action=None,
            predict_return=None,
            predict_time_to_goal=None,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)
        self.goal_dim = goal_dim

        self.hidden_size = hidden_size

        if GPT_backbone is not None:
            self.transformer = GPT_backbone
            print('[GoalConditionedDecisionTransformer] load GPT backbone from pretrained module')
        else:
            config = transformers.GPT2Config(
                vocab_size=1,  # doesn't matter -- we don't use the vocab
                n_embd=hidden_size,
                **kwargs
            )
            # note: the only difference between this GPT2Model and the default Huggingface version
            # is that the positional embeddings are removed (since we'll add those ourselves)
            self.transformer = GPT2Model(config)
        
        if embed_timestep is not None:
            self.embed_timestep = embed_timestep
        else:
            self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        
        if embed_return is not None:
            self.embed_return = embed_return
        else:
            self.embed_return = torch.nn.Linear(1, hidden_size)
        
        if embed_time_to_goal is not None:
            self.embed_time_to_goal = embed_time_to_goal
        else:
            self.embed_time_to_goal = torch.nn.Linear(1, hidden_size)
        
        if embed_state is not None:
            self.embed_state = embed_state
        else:
            self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        
        if embed_goal is not None:
            self.embed_goal = embed_goal
        else:
            self.embed_goal = torch.nn.Linear(self.goal_dim, hidden_size)
        
        if embed_action is not None:
            self.embed_action = embed_action
        else:
            self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        
        if embed_ln is not None:
            self.embed_ln = embed_ln
        else:
            self.embed_ln = nn.LayerNorm(hidden_size)

        if predict_state is not None:
            self.predict_state = predict_state
        else:
            self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)

        if predict_goal is not None:
            self.predict_goal = predict_goal
        else:
            self.predict_goal = torch.nn.Linear(hidden_size, self.goal_dim)
        
        if predict_action is not None:
            self.predict_action = predict_action
        else:
            self.predict_action = nn.Sequential(
                *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
            )
        
        if predict_return is not None:
            self.predict_return = predict_return
        else:
            self.predict_return = torch.nn.Linear(hidden_size, 1)
        
        if predict_time_to_goal is not None:
            self.predict_time_to_goal = predict_time_to_goal
        else:
            self.predict_time_to_goal = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, states, goals, actions, rewards, returns_to_go, times_to_goal, timesteps, attention_mask=None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        goal_embeddings = self.embed_goal(goals)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        times_to_goal_embeddings = self.embed_time_to_goal(times_to_goal)
        
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        times_to_goal_embeddings = times_to_goal_embeddings + time_embeddings

        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, times_to_goal_embeddings, state_embeddings, goal_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 5*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 5*seq_length)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), times to goal (1), states (2), goals(3) or actions (4); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        # TODO: try predict action from both decoding of state and goals, even though this operation seems invalid since 
        # casual reasoning of transformer
        return_preds = self.predict_return(x[:,4])  # predict next return given all information
        time_to_goal_preds = self.predict_time_to_goal(x[:,4]) 
        state_preds = self.predict_state(x[:,4])    # predict next state given all information
        goal_preds = self.predict_goal(x[:,4])
        action_preds = self.predict_action(x[:,3])  # predict next action given information without action

        return state_preds, goal_preds, action_preds, return_preds, time_to_goal_preds
    
    def forward_dynamics_prediction(self, states, goals, actions, rewards, returns_to_go, times_to_goal, timesteps, attention_mask):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        goal_embeddings = self.embed_goal(goals)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        times_to_goal_embeddings = self.embed_time_to_goal(times_to_goal)
        
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        times_to_goal_embeddings = times_to_goal_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, times_to_goal_embeddings, state_embeddings, goal_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 5*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 5*seq_length)

        # Mask return to goal, times to goal, and goal
        forward_dynamics_mask = torch.cat([torch.tensor([0., 0., 1., 0., 1.]) for _ in range(seq_length)]).to(device=states.device, dtype=torch.long)
        stacked_attention_mask = stacked_attention_mask & forward_dynamics_mask

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

        state_preds = self.predict_state(x[:,4])    # predict next state given all information (note part of information has been masked)

        return state_preds

    def forward_times_to_goal_prediction(self, states, goals, actions, rewards, returns_to_go, times_to_goal, timesteps, attention_mask):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        goal_embeddings = self.embed_goal(goals)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        times_to_goal_embeddings = self.embed_time_to_goal(times_to_goal)
        
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        times_to_goal_embeddings = times_to_goal_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, times_to_goal_embeddings, state_embeddings, goal_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 5*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 5*seq_length)

        # Mask return to goal
        times_to_goal_mask = torch.cat([torch.tensor([0., 1., 1., 1., 1.]) for _ in range(seq_length)]).to(device=states.device, dtype=torch.long)
        stacked_attention_mask = stacked_attention_mask & times_to_goal_mask

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

        time_to_goal_preds = self.predict_time_to_goal(x[:,4])    # predict times to goal given all information (returns to goal has been masked)

        return time_to_goal_preds

    def forward_masked_sequence_reconstruction(self, states, goals, actions, rewards, returns_to_go, times_to_goal, timesteps, attention_mask):
        batch_size, seq_length = states.shape[0], states.shape[1]

        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        goal_embeddings = self.embed_goal(goals)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        times_to_goal_embeddings = self.embed_time_to_goal(times_to_goal)
        
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        goal_embeddings = goal_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        times_to_goal_embeddings = times_to_goal_embeddings + time_embeddings

        stacked_inputs = torch.stack(
            (returns_embeddings, times_to_goal_embeddings, state_embeddings, goal_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 5*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Mask random
        random_mask = torch.ones((batch_size, seq_length))
        masked_idx = torch.where(torch.rand(batch_size, seq_length) < 0.01)
        random_mask[masked_idx] = 0
        random_mask = random_mask.to(device=states.device, dtype=torch.long)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask, attention_mask, attention_mask&random_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 5*seq_length)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, 5, self.hidden_size).permute(0, 2, 1, 3)

        # return_preds = self.predict_return(x[:,4])  # predict next return given all information
        # time_to_goal_preds = self.predict_time_to_goal(x[:,4]) 
        # state_preds = self.predict_state(x[:,4])    # predict next state given all information
        # goal_preds = self.predict_goal(x[:,4])
        action_recons = self.predict_action(x[:,3])  # predict next action given all information

        return action_recons


    def get_action(self, states, goals, actions, rewards, returns_to_go, times_to_goal, timesteps, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, self.state_dim)
        goals = goals.reshape(1, -1, self.goal_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        times_to_goal = times_to_goal.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            goals = goals[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            times_to_goal = times_to_goal[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            goals = torch.cat(
                [torch.zeros((goals.shape[0], self.max_length-goals.shape[1], self.goal_dim), device=goals.device), goals],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            times_to_goal = torch.cat(
                [torch.zeros((times_to_goal.shape[0], self.max_length-times_to_goal.shape[1], 1), device=times_to_goal.device), times_to_goal],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None
        
        # print('[DEBUG] states shape: {}'.format(states.shape))
        # print('[DEBUG] goals shape: {}'.format(goals.shape))
        # print('[DEBUG] actions shape: {}'.format(actions.shape))
        # print('[DEBUG] returns_to_go shape: {}'.format(returns_to_go.shape))
        # print('[DEBUG] times_to_goal shape: {}'.format(times_to_goal.shape))
        # print('[DEBUG] timesteps shape: {}'.format(timesteps.shape))
        # print('[DEBUG] attention_mask shape: {}'.format(attention_mask.shape))


        _, _, action_preds, _, _ = self.forward(
            states, goals, actions, None, returns_to_go, times_to_goal, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]