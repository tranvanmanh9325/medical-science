from utils.general_utils import AttrDict, listdict2dictlist
from utils.rl_utils import ReplayCache

from PIL import Image
import cv2
import numpy as np
import os
import open3d as o3d
from visual_obs import VisProcess


# For debug

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

paletee=get_palette(24)

def plot_image(img,is_seg=False, is_depth=False, path='/home/student/code/SAM-rbt-sim2real/debug_result', name='img1.png'):
    if is_depth:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_JET)
        
    i=Image.fromarray(np.asarray(img,dtype=np.uint8))
    
    
    if is_seg:
        np.save(os.path.join(path,'a.npy'),img)
        i.putpalette(paletee)
    
    i.save(os.path.join(path,name))
    
# Debug END
    
    
class Sampler:
    """Collects rollouts from the environment using the given agent."""
    def __init__(self, env, agent, max_episode_len, use_vis=False):
        self._env = env
        self._agent = agent
        self._max_episode_len = max_episode_len
        # For debug
        #max_episode_len=2
        #self._max_episode_len=2

        self._obs = None
        self._episode_step = 0
        self._episode_cache = ReplayCache(max_episode_len)
        
        # For visual input
        #self._camera=camera
        #self._use_pc=use_pc
        self._seg_exist=True
        #if self._camera!=None:
        self.use_vis=use_vis
        #if self.use_vis:
        #    self.v_processor=VisProcess()
        
        '''
        if self._use_pc:
            self.xyz=np.zeros((256*256,3))
            idx=0
            for i in range(256):
                for j in range(256):
                    self.xyz[idx][0]=i
                    self.xyz[idx][1]=j
                idx+=1
        '''

    def init(self):
        """Starts a new rollout. Render indicates whether output should contain image."""
        self._episode_reset()

    def sample_action(self, obs, is_train):
        print('sample action...')
        return self._agent.get_action(obs, noise=is_train, seg_exist=self._seg_exist)
    
    def sample_episode(self, is_train, render=False, random_act=False):
        """Samples one episode from the environment."""
        #render_obs,seg, depth=self._camera.render_image()
        render_obs,seg, depth=self._env.ecm.render_image()
        self.init()
        episode, done = [], False
        #xyz=self.xyz.copy()
        #print('max episode len: ',self._max_episode_len)
        while not done and self._episode_step < self._max_episode_len:
            '''
            print(self._obs)
            
            action = self._env.action_space.sample() if random_act else self.sample_action(self._obs, is_train) 
            if action is None:
                break
            if render:
                render_obs = self._env.render('rgb_array')
            obs, reward, done, info = self._env.step(action)
            episode.append(AttrDict(
                reward=reward,
                success=info['is_success'],
                info=info
            ))
            self._episode_cache.store_transition(obs, action, done)
            if render:
                episode[-1].update(AttrDict(image=render_obs))
            '''
                    
            #render_obs,seg, depth=self._camera.render_image()
            render_obs,seg, depth=self._env.ecm.render_image()
            pos=self.v_processor.get_v_obs(depth, seg)
            '''
            self._obs['v_o']={}
            if self._use_pc:
                
                pcd=self._xyz2pc(depth)
                self.render_pcd(pcd)
                pcd=np.concatenate((pcd, seg.reshape(-1,1)),axis=-1)
                
                self._obs['pcd']=pcd # (256*256,4)
                #print(pcd.shape)
                #exit()
            
            else:
                #print('using rgbd')
                #self._obs['v_o']['rgb']=render_obs # 256, 256, 3 numpy
                self._obs['v_o']['depth']=depth # 256, 256 numpy
                if self._seg_exist:
                    self._obs['v_o']['seg']=seg
                
                #print(seg.shape)  # 256, 256
                #exit()
            '''
            
            #For ddebug
            #random_act=False
            #self._obs['depth'] = depth # [0,1] basically 0.7x-0.9x
            action=self._env.action_space.sample() if random_act else self.sample_action(self._obs, is_train) 
            if action is None:
                break
            obs, reward, done, info = self._env.step(action)
            #print('obs shape: ',obs.keys())
            print('obs: ',obs['observation'].shape) #7
            episode.append(AttrDict(
                reward=reward,
                success=info['is_success'],
                info=info
            ))
            self._episode_cache.store_transition(obs, action, done)
            
            if render:
                episode[-1].update(AttrDict(image=render_obs))
            
            # update stored observation
            self._obs = obs
            self._episode_step += 1

        episode[-1].done = True     # make sure episode is marked as done at final time step
        rollouts = self._episode_cache.pop()
        assert self._episode_step == self._max_episode_len
        return listdict2dictlist(episode), rollouts, self._episode_step

    def _episode_reset(self, global_step=None):
        """Resets sampler at the end of an episode."""
        self._episode_step, self._episode_reward = 0, 0.
        self._obs = self._reset_env()
        # Change to Cam 
        self._obs['desired_goal']=self._w2cam(self._obs['desired_goal'])
        #print(self._obs)
        #exit()
        #self._obs['']
        self._episode_cache.store_obs(self._obs)
        

    def _reset_env(self):
        return self._env.reset()
    
    def _w2cam(self, point):
        #print(self._camera.view_matrix)
        extri_mtx=np.array(self._env.ecm.view_matrix).reshape(4,4)
        R=extri_mtx[:3,:3]
        T=extri_mtx[:3,3]
        pcd=(R@point.T).T+T
        
        return pcd

    def _xyz2pc(self, depth, width=256, height=256):
        
        intri_mtx=np.array(self._env.ecm.proj_matrix).reshape(4,4)
        #extri_mtx=np.array(self._camera.view_matrix).reshape(4,4)
        
        #print(intri_mtx)
        #print(extri_mtx)
        
        intri_inv=np.linalg.inv(intri_mtx)
        fx=intri_inv[0][0]
        fy=intri_inv[1][1]
        u0=intri_inv[0][2]
        v0=intri_inv[1][2]
        
        jj = np.tile(range(width), height)
        ii = np.repeat(range(height), width)
        # Compute constants:
        xx = (jj - u0) / fx
        yy = (ii - v0) / fy
        # transform depth image to vector of z:
        length = height * width
        z = depth.reshape(height * width)
        # compute point cloud
        pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))
        
        
        
        '''
        extri_inv=np.linalg.inv(extri_mtx)
        R=extri_inv[:3,:3]
        T=extri_inv[:3,3]
        pcd=(R@pcd.T).T+T
        '''
        return pcd

    def render_pcd(self, pcd):
        a=o3d.geometry.PointCloud()
        a.points=o3d.utility.Vector3dVector(pcd)
        #o3d.io.write_point_cloud("/home/student/code/SAM-rbt-sim2real/debug_result/test.obj", a)
        o3d.visualization.draw_geometries([a])
    
        
        
        
        