import numpy as np
import torch
import torch.nn.functional as F

from utils.general_utils import AttrDict
from .ddpgbc import DDPGBC
import sys
sys.path.append('..')
from modules.visual_obs import VisProcess, VisTR, VisRes
from components.normalizer import Normalizer
from torchvision import transforms
import matplotlib.pyplot as plt

#from depth_anything.dpt import DepthAnything
#from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


class VisDEX(DDPGBC):
    def __init__(
        self,
        env_params,
        sampler,
        agent_cfg,
    ):
        super().__init__(env_params, sampler, agent_cfg)
        self.vo_dim=agent_cfg.vo_dim
        self.vis_lr=agent_cfg.vis_lr
        self.k = agent_cfg.k
        self.obj_num=agent_cfg.obj_num
        self.img_size=256
        #self.v_processor=VisProcess(vo_dim=agent_cfg.vo_dim)
        self.v_processor=VisRes()
        #self.v_processor=VisTR()
        self.vis_optimizer=torch.optim.Adam(
            self.v_processor.parameters(), lr=agent_cfg.vis_lr
        )
        self.regress_rbt_staet=agent_cfg.robot_state

        self.depth_norm=Normalizer(
            size=256*256,
            default_clip_range=self.norm_clip,
            eps=agent_cfg.norm_eps
        )
        self.register_buffer("d_norm_mean",torch.zeros(self.depth_norm.size))
        self.register_buffer("d_norm_v",torch.ones(self.depth_norm.size))
        self.register_buffer("g_norm_mean",torch.zeros(self.g_norm.size))
        self.register_buffer("g_norm_v",torch.ones(self.g_norm.size))
        self.register_buffer("o_norm_mean",torch.zeros(self.o_norm.size))
        self.register_buffer("o_norm_v",torch.ones(self.o_norm.size))
        #self._load_dam()
        #self.count=0
        
        # self.dimo

    def _load_dam(self):
        encoder = 'vitb' # can also be 'vitb' or 'vitl'
        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()
    
    def _get_depth_with_dam(self, img):
        '''
        input: rgb image 1xHxW
        '''
        #print('ori img: ',img)
        #img=img/255.0
        #h, w = self.img_size, self.img_size
        
        #img=self.img_transform({'image': img})['image']
        #img=torch.from_numpy(img).unsqueeze(0)
        img=transforms.Resize((518,518))(img)
        with torch.no_grad():
            depth = self.depth_anything(img)
        #print(depth.shape)
        depth = F.interpolate(depth[None], (self.img_size, self.img_size), mode='bilinear', align_corners=False)[0]
        depth_min = torch.amin(depth, dim=(1, 2), keepdim=True)
        depth_max = torch.amax(depth, dim=(1, 2), keepdim=True)
        depth = (depth - depth_min) / (depth_max - depth_min)
        #print(depth.shape)
        #depth = (depth - depth.min()) / (depth.max() - depth.min()) # 0-1
        #exit()
        #print(depth.mean())
        
        #depth = depth.cpu().numpy()

        return depth

    
    def _restart_vis_processor(self):
        self.v_processor=VisProcess(vo_dim=self.vo_dim).cuda()
        self.vis_optimizer=torch.optim.Adam(
            self.v_processor.parameters(), lr=self.vis_lr
        )
        self.v_processor.train()
        print("restart vis!")
    
    def get_samples_v(self, replay_buffer):
        transitions = replay_buffer.sample()
        self.v_processor.eval()
        inputs=self.to_torch(transitions['v'])
        #inputs=torch.tensor(seg_d).unsqueeze(0).float().to(self.device) # B 2 256 256
        #print(inputs.shape)
        
        with torch.no_grad():
            v_output=self.v_processor(inputs).squeeze() # 9
            
        o, g = transitions['obs'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        #obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        
        robot_state=torch.tensor(transforms['obs'][:,:7]).to(self.device)
        #pos=v_output[:3]
        #rel_pos=pos-robot_state[:3]
        rel_pos=v_output[:,:3]
        pos=robot_state[:,:3]+rel_pos
        waypoint_pos=v_output[:,3:]
            
            #o=torch.from_numpy(np.concatenate([o[:,7:10].copy(),o[:,13:19]].copy(),axis=1)).to(self.device)
            
            #o_new=torch.concat((,o),dim=1) # B 19
        o_new=torch.concat((robot_state, pos),dim=-1)
        o_new=torch.concat((o_new, rel_pos),dim=-1)
        o_new=torch.concat((o_new, waypoint_pos),dim=-1)
        o_norm=self.o_norm.normalize(o_new,device=self.device)
        
        input_tensor=torch.concat((o_norm, g_norm), axis=0).to(torch.float32)
        action = self.actor(input_tensor)
        #action = self.to_torch(transitions['actions'])
        next_action = self.to_torch(transitions['next_actions'])
        reward = self.to_torch(transitions['r'])
        done = self.to_torch(transitions['dones'])
        
        return o_norm, action, reward
        
        #return obs, action, reward, done, next_obs, next_action, seg_d, v_gt
    
    def normlize_angles(self, x):
        return np.arctan2(np.sin(x),np.cos(x))

    def get_samples(self, replay_buffer,depth_noise=False):
        '''Addtionally sample next action for guidance propagation'''
        transitions = replay_buffer.sample()

        # preprocess
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        #print("o shape: ",o.shape)
        #v_gt=np.concatenate([o[:,7:10].copy(),o[:,13:19].copy()], axis=-1)
        # regress rel_pos instead of pos
        #v_gt= o[:,10:19].copy()#np.concatenate([o[:,7:10].copy(),o[:,13:19].copy()], axis=-1)
        norm_v_gt_rot=self.normlize_angles(o[:,7+3*self.obj_num+6:])
        #print('norm_v_gt_rot.shape: ',norm_v_gt_rot.shape)
        if not self.regress_rbt_staet:
            v_gt=o[:,7+3*self.obj_num:].copy()
            v_gt[:,6:]=norm_v_gt_rot.copy()
        else:
            v_gt=o.copy()
        #v_gt=o[]
        #print('-->agent.getsample o : ', o.requires_grad)
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        
        #print('get_sampel device: ',transitions['obs'].device)
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        #device=obs_norm.device
        #print('-->agent.getsample o_norm : ', obs_norm.requires_grad)
        #print('-->agent.getsample g_norm : ', g_norm.requires_grad)
        
        g_norm=torch.tensor(g_norm)
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        #obs = torch.concat((obs_norm, g_norm), axis=1).float().to(self.device)

        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        
        #g_next_norm=torch.tensor(g_next_norm)
        #next_obs = torch.concat((obs_next_norm, g_next_norm), axis=1).float().to(self.device)
        
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)

        obs = self.to_torch(inputs_norm)
        next_obs = self.to_torch(inputs_next_norm)
        action = self.to_torch(transitions['actions'])
        next_action = self.to_torch(transitions['next_actions'])
        reward = self.to_torch(transitions['r'])
        done = self.to_torch(transitions['dones'])
        v_gt=self.to_torch(v_gt)
        
        
        # For vis
        #depth=transitions['depth']
        rgb=self.to_torch(transitions['depth'])
        
        depth=self._get_depth_with_dam(rgb) #tensor

        
        '''
        if depth_noise:
            #print("add depth noise")
      
            # Add the Gaussian noise to the depth map
            noise = np.random.normal(0, 0.3, depth.shape)

            # Create a mask of True/False values where True represents the positions where we will add noise
            mask = np.random.choice([False, True], depth.shape, p=[0.9, 0.1])

            # Create a new array where noise is added only at the positions where the mask is True
            depth = np.where(mask, depth + 0.001*noise, depth)
        '''
        depth_norm=self.depth_norm.normalize(depth.reshape(-1,256*256),device=self.device).reshape(-1,1,256,256)
        seg=self.to_torch(transitions['seg'])
        seg_d=torch.concat((seg,depth_norm),dim=1)
        #seg_d=np.concatenate([transitions['seg'], depth_norm],axis=1)
        #seg_d=self.to_torch(seg_d)
        #seg_d=self._preproc_v(transitions['v'])
        
        return obs, action, reward, done, next_obs, next_action, seg_d, v_gt
    
    def _preproc_v(self, v, dim=0, device=None):
        #v=transforms.ToTensor()(v)
        v=torch.tensor(v).to(self.device)
        channel_mean = torch.zeros(2).to(self.device)
        channel_std = torch.zeros(2).to(self.device)
        nb_samples = 0.
        #print(v.shape)
        N, C, H, W = v.shape[:4]
        v = v.view(N, C, -1) 
        channel_mean += v.mean(2).sum(0)  
        channel_std += v.std(2).sum(0)
        nb_samples += N
        channel_mean /= nb_samples
        channel_std /= nb_samples
        v = transforms.Normalize(channel_mean, channel_std)(v)
        #v=torch.tensor(v).to(self.device)
        
        return v
    
    def get_action(self, state, noise=False, v_action=False):
        #print("get_action")
        
        if not v_action:
            return super().get_action(state, noise)
        #self.count+=1

        self.v_processor.eval()
        
        rgb=self.to_torch(state['depth']).unsqueeze(0)
        
        depth=self._get_depth_with_dam(rgb)[0] #tensor
        depth_norm=self.depth_norm.normalize(depth.reshape(-1,256*256),device=self.device).reshape(1,256,256)
        seg=self.to_torch(transitions['seg'])
        seg_d=torch.concat((seg,depth_norm))

        #depth_norm=self.depth_norm.normalize(state['depth'].reshape(-1,256*256),device=self.device).reshape(-1,256,256)
        #plt.imsave('/research/d1/rshr/arlin/SAM-rbt-sim2real/debug/img/depth/depth_ori_2.png',state['depth'][0])
        #plt.imsave('/research/d1/rshr/arlin/SAM-rbt-sim2real/debug/img/depth/depth_norm_2.png',depth_norm[0])
        #exit()
        #seg_d=np.concatenate([state['seg'], depth_norm],axis=0)
        
        inputs=seg_d.unsqueeze(0).float().to(self.device) # B 2 256 256
        #print(inputs.shape)
        
        with torch.no_grad():
            v_output=self.v_processor(inputs).squeeze() # 9

            #v_save=v_output.cpu().data.numpy()
            #np.save('test_record/v_output.npy', v_save)

            o, g = state['observation'], state['desired_goal']
            g=self.g_norm.normalize(g)
            #print("g: ",g)
            g_norm=torch.tensor(g).float().to(self.device)
            #print("g_norm: ",g_norm)
            
            if not self.regress_rbt_staet:
                robot_state=torch.tensor(o[:7]).to(self.device)
                #pos=v_output[:3]
                #rel_pos=pos-robot_state[:3]
                rel_pos=v_output[:3*self.obj_num]
                new_pos=robot_state[:3]+rel_pos[:3]
                
                if self.obj_num>1:
                    for i in range(1, self.obj_num):
                        pos=robot_state[:3]+rel_pos[i*3:3*self.obj_num]
                        new_pos=torch.concat((new_pos,pos))
                
                waypoint_pos_rot=v_output[3*self.obj_num:]
                
                
                #o=torch.from_numpy(np.concatenate([o[:,7:10].copy(),o[:,13:19]].copy(),axis=1)).to(self.device)
                
                #o_new=torch.concat((,o),dim=1) # B 19
                o_new=torch.concat((robot_state, new_pos))
                o_new=torch.concat((o_new, rel_pos))
                o_new=torch.concat((o_new, waypoint_pos_rot))
                o_norm=self.o_norm.normalize(o_new,device=self.device)
            else:
                o_norm=self.o_norm.normalize(v_output,device=self.device)
                o_norm=torch.tensor(o_norm).float().to(self.device)
                
            input_tensor=torch.concat((o_norm, g_norm), axis=0).to(torch.float32)
            #save_input=input_tensor.cpu().data.numpy()
            #np.save('test_record/actor_input.npy', save_input)
            #exit()
            
            action = self.actor(input_tensor).cpu().data.numpy().flatten()

        self.v_processor.train()
        return action
            
            
    def update_vprocesser(self, seg_d, v_gt):
        
        output=self.v_processor(seg_d)
        #print('output: ',output.shape)
        #print('v_gt: ',v_gt.shape)
        
        v_loss=0.
        if not self.regress_rbt_staet:
            #print()
            pos_loss=F.mse_loss(output[:,:3*self.obj_num],v_gt[:,:3*self.obj_num])
            v_loss+=pos_loss
            w_pos_loss=F.mse_loss(output[:,3*self.obj_num: 3*self.obj_num+3],v_gt[:,3*self.obj_num: 3*self.obj_num+3])
            v_loss+=w_pos_loss
            w_rot_loss=F.mse_loss(output[:,3*self.obj_num+3:],v_gt[:,3*self.obj_num+3: ])
            v_loss+=w_rot_loss
            metrics = AttrDict(
                v_pos=pos_loss.item(),
                w_pos_loss_loss=w_pos_loss.item(),
                w_rot_loss=w_rot_loss.item()

            )
        else:
            #robot_loss=F.mse_loss(output[:,:7],v_gt[:,:7])
            #v_loss+=robot_loss
            #pos_loss=F.mse_loss(output[:,7:],v_gt[:,7:])
            pos_loss=F.mse_loss(output, v_gt)
            v_loss+=pos_loss
            metrics = AttrDict(
                #v_robot_loss=robot_loss.item(),
                v_pos_loss=pos_loss.item()
            )
        # optimize v loss
        self.vis_optimizer.zero_grad()
        v_loss.backward()
        self.vis_optimizer.step() 
        # v_loss.backward()
        
        #return v_loss
        return metrics

    def update_critic(self, obs, action, reward, next_obs, next_obs_demo, next_action_demo):
        #print('next_obs_devcie_pre: ', next_obs.device)
        with torch.no_grad():
        #    print('next_obs_devcie: ', next_obs.device)
            next_action_out = self.actor_target(next_obs)
            target_V = self.critic_target(next_obs, next_action_out)
            target_Q = self.reward_scale * reward + (self.discount * target_V).detach()

            # exploration guidance
            topk_actions = self.compute_propagated_actions(next_obs, next_obs_demo, next_action_demo)
            act_dist = self.norm_dist(topk_actions, next_action_out)
            target_Q += self.aux_weight * act_dist 

            clip_return = 5 / (1 - self.discount)
            target_Q = torch.clamp(target_Q, -clip_return, 0).detach()

        Q = self.critic(obs, action)
        critic_loss = F.mse_loss(Q, target_Q)

        # optimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step() 
        
        metrics = AttrDict(
            critic_q=Q.mean().item(),
            critic_target_q=target_Q.mean().item(),
            critic_loss=critic_loss.item(),
            bacth_reward=reward.mean().item()
        )
        return metrics


    def update_actor(self, obs, obs_demo, action_demo):
        
        action_out = self.actor(obs)
        Q_out = self.critic(obs, action_out)

        topk_actions = self.compute_propagated_actions(obs, obs_demo, action_demo)
        act_dist = self.norm_dist(action_out, topk_actions) 
        actor_loss = -(Q_out + self.aux_weight * act_dist).mean()
        actor_loss += action_out.pow(2).mean()

        # optimize actor loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        metrics = AttrDict(
            actor_loss=actor_loss.item(),
            act_dist=act_dist.mean().item()
        )
        return metrics

    def update(self, replay_buffer, demo_buffer):
        metrics = dict()
        v_loss=0.
        #metrics.update(self.update_vprocesser(replay_buffer, demo_buffer))
        for i in range(self.update_epoch):
            #print(i)
            # sample from replay buffer and demo buffer 
            obs, action, reward, done, next_obs, _ , seg_d, v_gt= self.get_samples(replay_buffer, depth_noise=False)
            #print('-->agent.update1: ', obs.requires_grad)
            #print('-->obs_next agent.update1: ', next_obs.requires_grad)
            obs_demo, action_demo, reward_demo, done_demo, next_obs_demo, next_action_demo,_ ,_= self.get_samples(demo_buffer)
            #print('next_obs_devcie: ', next_obs.device)
            #obs_1=obs.clone().detach()
            #next_obs_1=next_obs.clone().detach()
            #print('-->agent.update1: ', obs_1.requires_grad)
            #print('-->obs_next agent.update1: ', next_obs_1.requires_grad)
            #print('-->action grad: ', action.requires_grad)
            #print('-->reward grad: ', reward.requires_grad)
            
            metrics.update(self.update_critic(obs, action, reward, next_obs, next_obs_demo, next_action_demo))
            metrics.update(self.update_actor(obs, obs_demo, action_demo))
            metrics.update(self.update_vprocesser(seg_d, v_gt))
            #v_loss+=self.update_vprocesser(seg_d, obs_demo)
            # Update target critic and actor
            self.update_target()
        
        return metrics


    def compute_propagated_actions(self, obs, obs_demo, action_demo):
        '''Local weighted regression'''
        l2_pair = torch.cdist(obs, obs_demo)
        topk_value, topk_indices = l2_pair.topk(self.k, dim=1, largest=False)
        topk_weight = F.softmin(topk_value, dim=1)

        topk_actions = torch.ones_like(action_demo)
        for i in range(topk_actions.size(0)):
            topk_actions[i] = torch.mm(topk_weight[i].unsqueeze(0), action_demo[topk_indices[i]]).squeeze(0)
        return topk_actions
    
    def update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, dones,depth = episode_batch.obs, episode_batch.ag, episode_batch.g, \
                                                    episode_batch.actions, episode_batch.dones, episode_batch.depth
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       'depth': depth
                       }
        
        transitions = self.sampler.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        
        self.depth_norm.update(transitions["depth"])
        
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()
        self.depth_norm.recompute_stats()

        self.d_norm_mean[:]=torch.tensor(self.depth_norm.mean)
        self.d_norm_v[:]=torch.tensor(self.depth_norm.std)
        self.g_norm_mean[:]=torch.tensor(self.g_norm.mean)
        self.g_norm_v[:]=torch.tensor(self.g_norm.std)
        self.o_norm_mean[:]=torch.tensor(self.o_norm.mean)
        self.o_norm_v[:]=torch.tensor(self.o_norm.std)
    
    def update_norm_test(self):
        self.d_norm_mean[:]=torch.tensor(self.depth_norm.mean)
        self.d_norm_v[:]=torch.tensor(self.depth_norm.std)
        self.g_norm_mean[:]=torch.tensor(self.g_norm.mean)
        self.g_norm_v[:]=torch.tensor(self.g_norm.std)
        self.o_norm_mean[:]=torch.tensor(self.o_norm.mean)
        self.o_norm_v[:]=torch.tensor(self.o_norm.std)


    

        