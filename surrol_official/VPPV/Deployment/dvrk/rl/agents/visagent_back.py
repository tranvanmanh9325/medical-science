'''
 * @author Anran Lin
 * @email [example@mail.com]
 * @create date 2023-07-10 12:00:06
 * @modify date 2023-07-10 12:00:06
 * @desc [description]
 */
'''
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from .ddpg import DDPG
from components.normalizer import Normalizer
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
#from modules.PointNeXt.openpoints.models import build_model_from_cfg
import matplotlib.pyplot as plt
from modules.policies import DeterministicActor
from modules.critics import Critic
import copy

class VisAgent(DDPG):
    def __init__(self, env_params, sampler, agent_cfg):
        super().__init__(env_params, sampler, agent_cfg)
        
        self.use_pcd=env_params.use_pcd
        #print(self.use_pcd)
        self.agent_cfg=agent_cfg
        #print('1')
        self.dimo=7
        self.o_norm=Normalizer(
            size=self.dimo, 
            default_clip_range=self.norm_clip,
            eps=agent_cfg.norm_eps
        )

        self.actor = DeterministicActor(    
            self.dimo, self.dima, agent_cfg.hidden_dim
        ).to(agent_cfg.device)
        self.actor_target = copy.deepcopy(self.actor).to(agent_cfg.device)
        
        self.critic = Critic(
            self.dimo+self.dima, agent_cfg.hidden_dim
        ).to(agent_cfg.device)
        self.critic_target = copy.deepcopy(self.critic).to(agent_cfg.device)
        #print('dima: ',self.dima) #5
        
        # Visual backbone
        if self.use_pcd:
            
            # Point embedding
            self.model = build_model_from_cfg(env_params.model)
            load_checkpoint(self.model.encoder, env_params.model.pretrained_path, env_params.model.get('pretrained_module', None))
            #criterion = build_criterion_from_cfg(cfg.criterion_args)
            
        else:
            # Depth Encoder
            resnet=models.resnet50(pretrained=True)
            #self.d_encoder = nn.Sequential(*list(resnet.children())[:4])
            # image encoder
            self.sam = sam_model_registry["default"](checkpoint=self.agent_cfg.sam_path)
            self.sam.eval()
            
            
            # Perform nn.deconv for image feature from (256,64,64) to (32, 512, 512)
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
            
            self.downsample=nn.Sequential(
                nn.Conv2d(in_channels=34,out_channels=64,kernel_size=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2),
                nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=4),
                nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=4)
            )
            self.pre_flattern=256*7*7
            
            self.mlp=nn.Sequential(
                nn.Linear(in_features=self.pre_flattern,out_features=self.pre_flattern//2),
                nn.ReLU(),
                nn.Linear(in_features=self.pre_flattern//2, out_features=self.pre_flattern//8),
                nn.ReLU(),
                nn.Linear(in_features=self.pre_flattern//8,out_features=self.dimo),
                nn.ReLU()
            )
    def update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, dones = episode_batch.obs, episode_batch.ag, episode_batch.g, \
                                                    episode_batch.actions, episode_batch.dones
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
                       }
        transitions = self.sampler.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        #print('transitions["obs"]: ', transitions['obs'].shape) # max_timesteps, dimo
        #print(transitions['obs'])
        self.o_norm.update(transitions['obs'])
        #self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        #self.g_norm.recompute_stats()

    def get_action(self, state, noise=False, seg_exist=True):
        
        with torch.no_grad():
            if self.use_pcd:
                pcd=state['pcd']
                o=self._feature_emb_pcd(pcd)
            else:
                o_torch=self.prepare_observation(state['v_o'])
                img_feature, d_seg=self._feature_emb_rgbd(o_torch, seg_exist=seg_exist)
                h_t=self._get_ht_rgbd(img_feature, d_seg)
                #print(h_t.shape) #32
                
            
        #g=state['desired_goal']
        #input_tensor=self._preproc_inputs(o, g)
        input_tensor=self.o_norm.normalize(h_t).to(self.device)
        print('input_tensor: ',input_tensor.shape)
        action=self.actor(input_tensor).cpu().data.numpy().flatten()

        if noise:
            action = (action + self.max_action * self.noise_eps * np.random.randn(action.shape[0])).clip(
                -self.max_action, self.max_action)
        #print('action get')
        print('action: ',action.shape) # 5
        return action

    def get_samples(self, replay_buffer):
        # sample from replay buffer 
        transitions = replay_buffer.sample()

        # preprocess
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)

        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        #inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)

        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        #inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        #obs = self.to_torch(inputs_norm)
        obs=self.to_torch(obs_norm)
        next_obs = self.to_torch(obs_next_norm)
        action = self.to_torch(transitions['actions'])
        reward = self.to_torch(transitions['r'])
        done = self.to_torch(transitions['dones'])

        return obs, action, reward, done, next_obs
        
    
    def prepare_observation(self,o):
        o_torch={}
        for key,item in o.items():
            o_torch[key]=torch.from_numpy(item)
        return o_torch
        
    
    def _feature_emb_pcd(self, pcd):
        feature=self.model(pcd)
        
        return feature
    
    def _feature_emb_rgbd(self, o, seg_exist=True):
        
        
        image=o['rgb']#.permute(2,0,1)
        depth=o['depth']
        #self.mask_generator.predictor.set_image(image=image)
        
        # segmentation
        #print(seg_exist)
        if not seg_exist:
            masks=self.mask_generator.generate(image.permute(2,0,1))
            #self._debug_show_seg(image, masks)
        else:
            masks=o['seg']
        
        #print('mask_shape: ',masks.shape) # 256 256
        d_seg=torch.concat((depth.view(256,256,1),masks.view(256,256,1)),dim=-1)
        
        # image feature
        image=self.sam.preprocess(image.permute(2,0,1)) # 3 1024 1024
        #print(image.shape)
        with torch.no_grad():
            img_feature = self.sam.image_encoder(image.unsqueeze(0))
        #print('img_feature: ',img_feature.shape) # 1 256 64 64
        img_feature=img_feature.squeeze()
        
        #mask_generator.predictor.get_image_embedding() # BxCxHxW
        
        # depth feature
        '''
        depth=depth.view(256,256,1).repeat(1,1,3).permute( 2,0,1) # 256 256 3
        print('depth_shape: ',depth.shape)
        d_feature=self.d_encoder(depth.unsqueeze(0))       
        print('depth_feature: ', d_feature.shape)# 1 64 64 64
        d_feature=d_feature.squeeze() 
        '''
        return img_feature, d_seg.permute(2,0,1)#, masks
    
    def _get_ht_rgbd(self,img_feature, d_seg):
        '''
        img_feature: 256 64 64
        d_feature: 256 256
        masks: 256 256
        '''
        
        img_feature = self.deconv(img_feature)
        #print(img_feature.shape)
        img_feature=nn.functional.interpolate(img_feature.unsqueeze(0), scale_factor=0.5).squeeze() # 32 256 256
        #print(img_feature.shape)
        #print(d_seg.shape)
            
        #concat_feature=torch.concat((d_feature.unsqueeze(-1),masks.unsqueeze(-1)),dim=-1)
        concat_feature=torch.concat((d_seg, img_feature),dim=0) # 34, 256 256
        
        down_feature=self.downsample(concat_feature) # 256,7,7
        down_feature=down_feature.view(-1)
        h_t=self.mlp(down_feature) # 32
        
        return h_t
    
    def _debug_show_seg(self,image, masks):
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        self.show_anns(masks)
        plt.axis('off')
        plt.show()     
    
    def show_anns(self,anns):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
        
                
        
    
    
    
    
        
        
        
        
    


