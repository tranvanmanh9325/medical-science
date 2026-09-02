import os

import numpy as np
import torch

from .general_utils import AttrDict, RecursiveAverageMeter


def get_env_params(env, cfg):
    #obs = env.reset()
    # For debug
    #env._max_episode_steps=2
    
    env_params = AttrDict(
        #obs=obs['observation'].shape[0],
        #achieved_goal=obs['achieved_goal'].shape[0],
        #goal=obs['desired_goal'].shape[0],
        obs=cfg.agent.obs_num,
        achieved_goal=cfg.agent.achieved_goal_num,
        goal=cfg.agent.goal_num,
        act=env.action_space.shape[0],
        act_rand_sampler=env.action_space.sample,
        max_timesteps=env._max_episode_steps,
        max_action=env.action_space.high[0],
    )
    
    return env_params


class ReplayCache:
    def __init__(self, T):
        self.T = T
        self.reset()

    def reset(self):
        self.t = 0
        self.obs, self.ag, self.g, self.actions, self.dones = [], [], [], [], []
        self.v=[]

    def store_transition(self, obs, action, done):
        #print('observation shape',obs['observation'].shape)
        self.obs.append(obs['observation'])
        
        self.ag.append(obs['achieved_goal'])
        self.g.append(obs['desired_goal'])
        self.actions.append(action)
        self.dones.append(done)
        
        if 'seg_d' in obs:
            self.v.append(obs['seg_d'])
        #else:
        #    exit()

    def store_obs(self, obs):
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])
        if 'seg_d' in obs:
            self.v.append(obs['seg_d'])

    def pop(self):
        #print('debug item: ')
        #print(len(self.obs))
        #print(self.T)
        #print(len(self.actions))
        #print('obs: ',len(self.obs))
        #print('self T: ', self.T+1)
        #print('self action: ', len(self.actions))
        assert len(self.obs) == self.T + 1 and len(self.actions) == self.T
        #obs = np.expand_dims(np.array(self.obs.copy()),axis=0)
        #print(self.obs)
        #print(type(self.obs))
        #print(len(self.obs))
        #if isinstance(self.obs[0],torch.Tensor):
        #    obs=torch.stack(self.obs).unsqueeze(0)
        #else: 
        obs = np.expand_dims(np.array(self.obs.copy()),axis=0)
        #print('obs shape: ', obs.shape)
        
        #print('obs: ',obs.shape)
        
        ag = np.expand_dims(np.array(self.ag.copy()), axis=0)
        #print('ag: ',ag.shape)
        g = np.expand_dims(np.array(self.g.copy()), axis=0)
        actions = np.expand_dims(np.array(self.actions.copy()), axis=0)
        dones = np.expand_dims(np.array(self.dones.copy()), axis=1)
        dones = np.expand_dims(dones, axis=0)
        #print('self obs shape: ', len(self.obs.shape))
        #print('obs shape: ',obs.shape)
        #has_v=False
        if len(self.v)>0:
            v=np.expand_dims(np.array(self.v.copy()),axis=0)
        else:
            v=np.zeros([1,dones.shape[1], 2, 256,256])
        #    has_v=True

        self.reset()
        #if has_v:
        episode = AttrDict(obs=obs, ag=ag, g=g, actions=actions, dones=dones, v=v)
        #else:
        #    episode = AttrDict(obs=obs, ag=ag, g=g, actions=actions, dones=dones)
        return episode

    
def init_buffer(cfg, buffer, agent, normalize=True):
    '''Load demonstrations into buffer and initilaize normalizer'''
    demo_path = cfg.demo_path
    demo = np.load(demo_path, allow_pickle=True)
    demo_obs, demo_acs = demo['obs'], demo['acs']
    #print('buffer init!')
    episode_cache = ReplayCache(buffer.T)
    for epsd in range(cfg.num_demo):
        episode_cache.store_obs(demo_obs[epsd][0])
        for i in range(buffer.T):
            #print(buffer.T)
            #print(demo_obs[epsd][i+1])
            episode_cache.store_transition(
                obs=demo_obs[epsd][i+1],
                action=demo_acs[epsd][i],
                done=i==(buffer.T-1),
            )
        episode = episode_cache.pop()
        buffer.store_episode(episode)
        if normalize:
            agent.update_normalizer(episode)


class RolloutStorage:
    """Can hold multiple rollouts, can compute statistics over these rollouts."""
    def __init__(self, v_action=False):
        self.rollouts = []
        self.v_action=v_action

    def append(self, rollout):
        """Adds rollout to storage."""
        self.rollouts.append(rollout)

    def rollout_stats(self):
        """Returns AttrDict of average statistics over the rollouts."""
        assert self.rollouts    # rollout storage should not be empty
        stats = RecursiveAverageMeter()
        if self.v_action:
            for rollout in self.rollouts:
                stats.update(AttrDict(
                    avg_reward_v=np.stack(rollout.reward).sum(),
                    avg_success_rate_v=rollout.success[-1]
                ))
        else:
            for rollout in self.rollouts:
                stats.update(AttrDict(
                    avg_reward=np.stack(rollout.reward).sum(),
                    avg_success_rate=rollout.success[-1]
                ))
                
        return stats.avg

    def reset(self):
        del self.rollouts
        self.rollouts = []

    def get(self):
        return self.rollouts

    def __contains__(self, key):
        return self.rollouts and key in self.rollouts[0]