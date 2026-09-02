import copy
import os

import gym
import torch

from agents.factory import make_agent
from components.checkpointer import CheckpointHandler
from components.logger import Logger, WandBLogger, logger
from modules.replay_buffer import HerReplayBuffer, get_buffer_sampler
from modules.samplers import Sampler
from utils.general_utils import (AverageMeter, Every, Timer, Until,
                                   set_seed_everywhere)
#from utils.mpi import (mpi_gather_experience_episode,
#                         mpi_gather_experience_rollots, mpi_sum,
#                         update_mpi_config)
from utils.rl_utils import RolloutStorage, get_env_params, init_buffer
from .base_trainer import BaseTrainer

from surrol.utils.pybullet_utils import (
    reset_camera,    
)
#from surrol.robots.ecm import Ecm
from modules.visual_obs import VisProcess
import gc


class RLTrainer(BaseTrainer):
    # For visual input
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE=3
    
    def _setup(self):
        self._setup_env()       # Environment
        self._setup_buffer()    # Relay buffer
        self._setup_agent()     # Agent
        
        self._update_vprocessor()
        
        self._setup_sampler()   # Sampler
        self._setup_logger()    # Logger
        self._setup_misc()      # MISC
         

        if self.is_chef:
            self.termlog.info('Setup done')
        
        if self.cfg.resume_training:
            self.resume_training()
        
    
    def _update_vprocessor(self):
        
        self.train_env.update_vp(self.agent.v_processor)
        self.eval_env.update_vp(self.agent.v_processor)

    def _setup_env(self):
        self.train_env = gym.make(self.cfg.task)
        self.eval_env = gym.make(self.cfg.task)
        self.env_params = get_env_params(self.train_env, self.cfg)

    def _setup_buffer(self):
        self.buffer_sampler = get_buffer_sampler(self.train_env, self.cfg.agent.sampler)
        self.buffer = HerReplayBuffer(buffer_size=self.cfg.replay_buffer_capacity, env_params=self.env_params,
                            batch_size=self.cfg.batch_size, sampler=self.buffer_sampler)
        self.demo_buffer = HerReplayBuffer(buffer_size=self.cfg.replay_buffer_capacity, env_params=self.env_params,
                            batch_size=self.cfg.batch_size, sampler=self.buffer_sampler)
        
    def _setup_agent(self):
        self.agent = make_agent(self.env_params, self.buffer_sampler, self.cfg.agent)
        self.agent.cuda()
    def _setup_sampler(self):
        self.train_sampler = Sampler(self.train_env, self.agent, self.env_params['max_timesteps'])
        self.eval_sampler = Sampler(self.eval_env, self.agent, self.env_params['max_timesteps'])

    def _setup_logger(self):
        #update_mpi_config(self.cfg)
        if self.is_chef:
            exp_name = f"{self.cfg.task}_{self.cfg.agent.name}_demo{self.cfg.num_demo}_seed{self.cfg.seed}"
            if self.cfg.postfix is not None:
                exp_name =  exp_name + '_' + str(self.cfg.postfix) 
            if self.cfg.use_wb:
                self.wb = WandBLogger(exp_name=exp_name, project_name=self.cfg.project_name, entity=self.cfg.entity_name, \
                        path=self.work_dir, conf=self.cfg)
            self.logger = Logger(self.work_dir)
            self.termlog = logger
        else: 
            self.wb, self.logger, self.termlog = None, None, None
    
    def _setup_misc(self):
        init_buffer(self.cfg, self.buffer, self.agent, normalize=False) # important for awac, amp
        # TODO: update demonstration
        init_buffer(self.cfg, self.demo_buffer, self.agent, normalize=True)

        if self.is_chef:
            self.model_dir = self.work_dir / 'model'
            self.model_dir.mkdir(exist_ok=True)
            for file in os.listdir(self.model_dir):
                os.remove(self.model_dir / file)

        self.device = torch.device(self.cfg.device)
        print('device: ',self.device)
        self.timer = Timer()
        self._global_step = 0
        self._global_episode = 0
        set_seed_everywhere(self.cfg.seed)
    
    def train(self):
        n_train_episodes = int(self.cfg.n_train_steps / self.env_params['max_timesteps'])
        n_eval_episodes = int(n_train_episodes / self.cfg.n_eval) #* self.cfg.mpi.num_workers
        n_save_episodes = int(n_train_episodes / self.cfg.n_save) #* self.cfg.mpi.num_workers
        n_log_episodes = int(n_train_episodes / self.cfg.n_log) #* self.cfg.mpi.num_workers
        #print(self.cfg.mpi.num_workers)
        
        assert n_save_episodes > n_eval_episodes
        if n_save_episodes % n_eval_episodes != 0:
            n_save_episodes = int(n_save_episodes / n_eval_episodes) * n_eval_episodes
        print('n_train_episode: ', n_train_episodes)
        print('n_save_episodes: ', n_save_episodes)
        print('n_eval_episodes: ',n_eval_episodes)
        print('n_log_episodes: ', n_log_episodes)
        print('n_seed_steps: ', self.cfg.n_seed_steps)

        train_until_episode = Until(n_train_episodes)
        save_every_episodes = Every(n_save_episodes)
        eval_every_episodes = Every(n_eval_episodes)
        log_every_episodes = Every(n_log_episodes)
        seed_until_steps = Until(self.cfg.n_seed_steps)

        if self.is_chef:
            self.termlog.info('Starting training')
        while train_until_episode(self.global_episode):
            self._train_episode(log_every_episodes, seed_until_steps)

            if self.global_step > self.cfg.n_seed_steps and eval_every_episodes(self.global_episode):
                score = self.eval()

            if not self.cfg.dont_save and save_every_episodes(self.global_episode) and self.is_chef:
                filename =  CheckpointHandler.get_ckpt_name(self.global_episode)
                # TODO(tao): expose scoring metric
                CheckpointHandler.save_checkpoint({
                    'episode': self.global_episode,
                    'global_step': self.global_step,
                    'state_dict': self.agent.state_dict(),
                    'o_norm': self.agent.o_norm,
                    'g_norm': self.agent.g_norm,
                    'score': score,
                }, self.model_dir, filename)
                self.termlog.info(f'Save checkpoint to {os.path.join(self.model_dir, filename)}')

    def _train_episode(self, log_every_episodes, seed_until_steps):
        #print('global episode: ', self.global_episode)
        print('global step: ', self.global_step)
        # sync network parameters across workers
        if self.use_multiple_workers > 1:
            self.agent.sync_networks()

        self.timer.reset()
        batch_time = AverageMeter()
        ep_start_step = self.global_step
        metrics = None

        # collect experience
        rollout_storage = RolloutStorage()
        episode, rollouts, env_steps = self.train_sampler.sample_episode(is_train=True, random_act=seed_until_steps(ep_start_step))
        # FOR DEBUG
        # episode, _, env_steps = self.eval_sampler.sample_episode(is_train=False, render=True)
        
        if self.use_multiple_workers:
            transitions_batch = mpi_gather_experience_episode(rollouts)
        else:
            transitions_batch = rollouts

        # update status
        rollout_storage.append(episode)
        rollout_status = rollout_storage.rollout_stats()
        self._global_step += int(env_steps)
        self._global_episode += 1

        # save to buffer
        self.buffer.store_episode(copy.deepcopy(transitions_batch))
        self.agent.update_normalizer(copy.deepcopy(transitions_batch))
        #self.buffer.store_episode(transitions_batch)
        #self.agent.update_normalizer(transitions_batch)

        # update policy
        if not seed_until_steps(ep_start_step):
            if self.is_chef:
                metrics = self.agent.update(self.buffer, self.demo_buffer)
                #pos_loss =self.v_processor.update(self.buffer, self.demo_buffer)
                #self.vis_optimizer.zero_grad()
                #pos_loss.backward()
                #self.vis_optimizer.step()
                #metrics['pos_loss']=pos_loss.item()
                print(metrics)

            if self.use_multiple_workers:
                self.agent.sync_networks()

        # log results
        if metrics is not None and (log_every_episodes(self.global_episode)) and self.is_chef:
            elapsed_time, total_time = self.timer.reset()
            batch_time.update(elapsed_time)
            togo_train_time = batch_time.avg * (self.cfg.n_train_steps - ep_start_step) / env_steps

            self.logger.log_metrics(metrics, self.global_step, ty='train')
            with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                log('fps', env_steps / elapsed_time)
                log('total_time', total_time)
                log('episode_reward', rollout_status.avg_reward)
                log('episode_length', env_steps)
                log('episode_sr', rollout_status.avg_success_rate)
                log('episode', self.global_episode)
                log('step', self.global_step)
                log('ETA', togo_train_time)
            if self.cfg.use_wb:
                self.wb.log_outputs(metrics, None, log_images=False, step=self.global_step, is_train=True)
        # for memory 1
        del rollout_storage
        gc.collect()
        
    def eval(self):
        '''Eval agent.'''
        #self.eval_env.agent.eval()
        #eval_rollout_storage = RolloutStorage()
        eval_rollout_storage_v = RolloutStorage(v_action=True) # use the state predicted by d+seg
        for ii in range(self.cfg.n_eval_episodes):
            
            #episode, _, env_steps = self.train_sampler.sample_episode(is_train=False, render=True)
            #eval_rollout_storage.append(episode)
            episode_v, _, env_steps_v = self.train_sampler.sample_episode(is_train=False, render=True, v_action=False)
            eval_rollout_storage_v.append(episode_v)
        #rollout_status = eval_rollout_storage.rollout_stats()
        rollout_status_v = eval_rollout_storage_v.rollout_stats()
        
        if self.use_multiple_workers:
            #print('gather statas from state..')
            #rollout_status = mpi_gather_experience_rollots(rollout_status)
            #print('gather statas from v..')
            rollout_status_v = mpi_gather_experience_rollots(rollout_status_v)
            for key, value in rollout_status.items():
                #rollout_status[key] = value.mean()
                rollout_status_v[key] = rollout_status_v[key].mean()

        if self.is_chef:
            if self.cfg.use_wb:
                #self.wb.log_outputs(rollout_status, eval_rollout_storage, log_images=True, step=self.global_step)
                self.wb.log_outputs(rollout_status_v, eval_rollout_storage_v, log_images=True, step=self.global_step,video_name="visual_rollouts")
                
            with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
                #log('episode_sr', rollout_status.avg_success_rate)
                #log('episode_reward', rollout_status.avg_reward)
                #log('episode_length', env_steps)
                log('episode', self.global_episode)
                log('step', self.global_step)
                log('V-episode_sr', rollout_status_v.avg_success_rate_v)
                log('V-episode_reward', rollout_status_v.avg_reward_v)
                log('V-episode_length', env_steps_v)
                

        #del eval_rollout_storage
        del eval_rollout_storage_v
        gc.collect()
        #self.eval_env.agent.train()
        return rollout_status_v.avg_success_rate_v

    def resume_training(self):
        CheckpointHandler.load_checkpoint(
                self.cfg.ckpt_dir, self.agent, self.device, self.cfg.ckpt_episode
            )
        self.agent.train()
        self.agent.v_processor.eval()
        
        
    def eval_ckpt(self):
        '''Eval checkpoint.'''
        if self.is_chef:
            CheckpointHandler.load_checkpoint(
                self.cfg.ckpt_dir, self.agent, self.device, self.cfg.ckpt_episode
            )
            avg_success_rate = self.eval()
            self.termlog.info(f'Successful rate: {avg_success_rate}')
                
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def is_chef(self):
        return True
        #return self.cfg.mpi.is_chef

    @property
    def use_multiple_workers(self):
        return False
        #return self.cfg.mpi.num_workers > 1