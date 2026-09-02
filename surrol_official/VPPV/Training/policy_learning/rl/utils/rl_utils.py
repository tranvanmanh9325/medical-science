import os

import numpy as np
import torch

from .general_utils import AttrDict, RecursiveAverageMeter


def get_env_params(env, cfg):
    #obs = env.reset()
    # For debug
    #env._max_episode_steps=2
    
    env_params = AttrDict(
        obs=19,
        achieved_goal=3,
        goal=3,
        #obs=cfg.agent.obs_num,
        #achieved_goal=cfg.agent.achieved_goal_num,
        #goal=cfg.agent.goal_num,
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
        

    def store_transition(self, obs, action, done):
        #print('observation shape',obs['observation'].shape)
        self.obs.append(obs['observation'])
        
        self.ag.append(obs['achieved_goal'])
        self.g.append(obs['desired_goal'])
        self.actions.append(action)
        self.dones.append(done)
        

    def store_obs(self, obs):
        self.obs.append(obs['observation'])
        self.ag.append(obs['achieved_goal'])
        

    def pop(self):
        
        assert len(self.obs) == self.T + 1 and len(self.actions) == self.T
        
        obs = np.expand_dims(np.array(self.obs.copy()),axis=0)
       
        
        ag = np.expand_dims(np.array(self.ag.copy()), axis=0)
        #print('ag: ',ag.shape)
        g = np.expand_dims(np.array(self.g.copy()), axis=0)
        actions = np.expand_dims(np.array(self.actions.copy()), axis=0)
        dones = np.expand_dims(np.array(self.dones.copy()), axis=1)
        dones = np.expand_dims(dones, axis=0)
       

        self.reset()
        #if has_v:
        episode = AttrDict(obs=obs, ag=ag, g=g, actions=actions, dones=dones)
        
        return episode

    
def world2cam_pos(pos, view_matrix):
    
    point_world = np.array([pos[0], pos[1], pos[2], 1])
    
    cam_pos=view_matrix @ point_world
    return np.array([cam_pos[0],cam_pos[1],cam_pos[2]])
    #return cam_pos
        

def world2cam_rot(euler_world,view_matrix):
    
    rot_matrix_world = np.array([[np.cos(euler_world[1])*np.cos(euler_world[2]), np.sin(euler_world[0])*np.sin(euler_world[1])*np.cos(euler_world[2]) - np.cos(euler_world[0])*np.sin(euler_world[2]), np.cos(euler_world[0])*np.sin(euler_world[1])*np.cos(euler_world[2]) + np.sin(euler_world[0])*np.sin(euler_world[2])],
                                [np.cos(euler_world[1])*np.sin(euler_world[2]), np.sin(euler_world[0])*np.sin(euler_world[1])*np.sin(euler_world[2]) + np.cos(euler_world[0])*np.cos(euler_world[2]), np.cos(euler_world[0])*np.sin(euler_world[1])*np.sin(euler_world[2]) - np.sin(euler_world[0])*np.cos(euler_world[2])],
                                [-np.sin(euler_world[1]), np.sin(euler_world[0])*np.cos(euler_world[1]), np.cos(euler_world[0])*np.cos(euler_world[1])]])
    # Extract rotation matrix from camera extrinsic matrix
    #print(self._view_matrix)
    rot_matrix_camera = view_matrix[:3, :3]
    # Apply camera extrinsic rotation to rotation matrix in world axis
    rot_matrix_camera = rot_matrix_camera @ rot_matrix_world
    # Convert rotation matrix to Euler angles in camera axis
    euler_camera = np.array([np.arctan2(rot_matrix_camera[2, 1], rot_matrix_camera[2, 2]),
                            np.arctan2(-rot_matrix_camera[2, 0], np.sqrt(rot_matrix_camera[2, 1]**2 + rot_matrix_camera[2, 2]**2)),
                            np.arctan2(rot_matrix_camera[1, 0], rot_matrix_camera[0, 0])])
    return euler_camera
    
def world2cam(demo_obs, view_matrix):
    observation=demo_obs['observation']
    robot_state=observation[:7]
    object_pos=observation[7:10]
    waypoint_pos=observation[13:16]
    waypoint_rot=observation[16:]
    robot_pos=world2cam_pos(robot_state[:3],view_matrix)
    robot_rot=world2cam_rot(robot_state[3:6],view_matrix)
    object_pos=world2cam_pos(object_pos, view_matrix)
    waypoint_pos=world2cam_pos(waypoint_pos, view_matrix)
    waypoint_rot=world2cam_rot(waypoint_rot, view_matrix)
    object_rel_pos=object_pos-robot_pos
    #print(robot_pos.shape)
    #print(robot_pos.shape)
    #print()
    new_observation=np.concatenate([
        robot_pos, robot_rot, np.array([robot_state[-1]]),
        object_pos, object_rel_pos, waypoint_pos, waypoint_rot
    ])
    achieved_goal=world2cam_pos(demo_obs['achieved_goal'],view_matrix)
    desired_goal=world2cam_pos(demo_obs['desired_goal'], view_matrix)
    
    demo_obs['observation']=new_observation
    demo_obs['achieved_goal']=achieved_goal
    demo_obs['desired_goal']=desired_goal
    
    return demo_obs    
    
def init_buffer(cfg, buffer, agent, normalize=True, view_matrix=None):
    '''Load demonstrations into buffer and initilaize normalizer'''
    demo_path = cfg.demo_path
    demo = np.load(demo_path, allow_pickle=True)
    demo_obs, demo_acs = demo['obs'], demo['acs']
 
    #print('buffer init!')
    episode_cache = ReplayCache(buffer.T)
    
    if not view_matrix is None:
        print('transfer!!')
        view_matrix=np.array(view_matrix).reshape(4,4)
        for epsd in range(cfg.num_demo):
            
            curr_demo=world2cam(demo_obs[epsd][0], view_matrix)
            #curr_demo['depth']=np.repeat(curr_demo['depth'], 3, axis=0)
            #print(curr_demo['depth'].shape)
            #curr_demo['depth']=np.random.randn(3,256,256)
            episode_cache.store_obs(curr_demo)
            for i in range(buffer.T):
                #print(buffer.T)
                #print(demo_obs[epsd][i+1])
                curr_demo=world2cam(demo_obs[epsd][i+1], view_matrix)
                #curr_demo['depth']=np.random.randn(3,256,256)
                #curr_demo['depth']=np.repeat(curr_demo['depth'][..., np.newaxis], 3, axis=0)
                episode_cache.store_transition(
                    #obs=demo_obs[epsd][i+1],
                    obs=curr_demo,
                    action=demo_acs[epsd][i],
                    done=i==(buffer.T-1),
                )
            episode = episode_cache.pop()
            buffer.store_episode(episode)
            if normalize:
                agent.update_normalizer(episode)
    else:
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
    '''       
    demo_path = '/research/d1/rshr/arlin/SAM-rbt-sim2real/surrol/data/demo/data_PegTransfer-v0_random_200_1.npz'
    demo = np.load(demo_path, allow_pickle=True)
    demo_obs, demo_acs = demo['obs'], demo['acs']
    
    print('the second 200')
    #episode_cache = ReplayCache(buffer.T)
    
    if not view_matrix is None:
        print('transfer!!')
        view_matrix=np.array(view_matrix).reshape(4,4)
        for epsd in range(cfg.num_demo):
            
            curr_demo=world2cam(demo_obs[epsd][0], view_matrix)
            episode_cache.store_obs(curr_demo)
            for i in range(buffer.T):
                #print(buffer.T)
                #print(demo_obs[epsd][i+1])
                curr_demo=world2cam(demo_obs[epsd][i+1], view_matrix)
                episode_cache.store_transition(
                    #obs=demo_obs[epsd][i+1],
                    obs=curr_demo,
                    action=demo_acs[epsd][i],
                    done=i==(buffer.T-1),
                )
            episode = episode_cache.pop()
            buffer.store_episode(episode)
            if normalize:
                agent.update_normalizer(episode)
    '''
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