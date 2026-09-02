import os
import time

import numpy as np
import pybullet as p
import pickle

from surrol.const import ASSET_DIR_PATH
from surrol.tasks.psm_env import PsmEnv, goal_distance
from surrol.utils.pybullet_utils import get_link_pose, wrap_angle
from surrol.robots.ecm import Ecm
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

from surrol.utils.robotics import (
    get_euler_from_matrix,
    get_matrix_from_euler
)
import sys
sys.path.append('/home/jwfu/ar_surrol/surrol/tasks')
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from torchvision.transforms import Compose

sys.path.append('/home/jwfu/ar_surrol_datageneration/stateregress')
from vmodel import vismodel
from config import opts

from gym import spaces

re_label={1:0,2:1,3:2,5:3,6:4}

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

def plot_image(img,is_seg=False, is_depth=False, path='/home/student/code/SAM-rbt-sim2real/data/seg_data/peg_transfer', name='img1.png'):
    if is_depth:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_JET)
        
    i=Image.fromarray(np.asarray(img,dtype=np.uint8))
    
    if is_seg:
        np.save(os.path.join(path,'a.npy'),img)
        #i.putpalette(paletee)
    
    i.save(os.path.join(path,name))

    
def seg_with_red(grid_RGB):

    #grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #cv2.imwrite('/home/student/code/SAM-rbt-sim2real/debug_result/rgb.png', grid_RGB)
    
    
    #grid_RGB=img
    grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)
 
    # H、S、V range1：
    lower1 = np.array([0,43,46])
    upper1 = np.array([10,255,255])
    mask1 = cv2.inRange(grid_HSV, lower1, upper1)       # mask: binary
 
    # H、S、V range2：
    lower2 = np.array([156,43,46])
    upper2 = np.array([180,255,255])
    mask2 = cv2.inRange(grid_HSV, lower2, upper2)
    
    mask3 = mask1 + mask2
    return mask3

class PegTransfer(PsmEnv):
    POSE_BOARD = ((0.55, 0, 0.6861), (0, 0, 0))  # 0.675 + 0.011 + 0.001
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.686, 0.745))
    SCALING = 1.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    _cnt=0
    ACTION_SIZE = 7

    # For data collection
    counter=0
    img_list={}

    
    def _env_setup(self):
        self.subtask="grasp"
        #self._load_dam()
        self.action_space = spaces.Box(-1., 1., shape=(self.ACTION_SIZE,), dtype='float32')

        opts.device='cuda:0'
        self.v_model=vismodel(opts)
        ckpt=torch.load('/home/jwfu/ar_surrol_datageneration/stateregress/exp_18again_close_scaling1_exist_depth_corr_d/checkpoints/best_model.pt', map_location=opts.device)
        self.v_model.load_state_dict(ckpt['state_dict'])
        self.v_model.to(opts.device)
        self.v_model.eval()
        # print("load v_output")
        
        self.transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        # TODO: camera position -> change here ecm_view_matrix[14] and ecm_view_matrix[13]
        # TODO: During training perceptual model, should align the camera setting in train.py.
        ecm_view_matrix =[2.7644696427853166e-12, -0.8253368139266968, 0.5646408796310425, 0.0, 1.0, 2.76391192918779e-12, -8.559629784479772e-13, 0.0, -8.541598418149166e-13, 0.5646408796310425, 0.8253368139266968, 0.0, -1.582376590869572e-11, 0.4536721706390381, -5.886332988739014,1.0]
        #ecm_view_matrix[14]=-5.25 #-5.0#-5.25
        ecm_view_matrix[14]=-0.95 #-4.7 #-5.25#-5.0#-5.25
        ecm_view_matrix[13]=0.07 #0.3-0.5
        shape_view_matrix=np.array(ecm_view_matrix).reshape(4,4)
        Tc = np.array([[1,  0,  0,  0],
                        [0, -1,  0,  0],
                        [0,  0, -1,  0],
                        [0,  0,  0,  1]])
        self._view_matrix=Tc@(shape_view_matrix.T)
        self.ecm_view_matrix=np.array(ecm_view_matrix).reshape(4,4)
        super(PegTransfer, self)._env_setup()
        self.has_object = True
        self.img_transform = Compose([
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            ])
        # for subtask
        #self._waypoint_gaol=True

        
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                    scaling=self.SCALING, view_matrix=ecm_view_matrix)
        self.ecm.reset_joint(self.QPOS_ECM)
        self._view_matrix=np.array(self.ecm.view_matrix).reshape(4,4)

        # robot
        
        workspace_limits = self.workspace_limits1

        # TODO: change tip initial position here
        temp=np.random.randint(-5,6)/100
        print('workspace_limits: ',workspace_limits[2])
        print(workspace_limits[2][0] + np.random.rand()/10)
        
        pos = (workspace_limits[0][0],
               temp,
               (workspace_limits[2][1] - np.random.rand()/100))
        
        '''
        pos = (workspace_limits[0][0],
               temp,
               workspace_limits[2][1]/2)
        '''
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        
        # peg board
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'peg_board/peg_board.urdf'),
                            np.array(self.POSE_BOARD[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_BOARD[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1
        self._pegs = np.arange(12)
        
        np.random.shuffle(self._pegs[:6])
        np.random.shuffle(self._pegs[6: 12])
        
        #self._pegs = [ 1 , 2 , 3 , 4 , 5 , 0 , 7 ,11 , 9 ,10 , 6 , 8]

        # blocks
        num_blocks = 1
        # ar: Block can be put in any side
        #np.random.shuffle(self._pegs)
        #self.block_pos_flag=np.random.rand()>0.5
        #if self.block_pos_flag:
        #    peg_for_object_list=self._pegs[6: 6 + num_blocks]
        #else:
        #    peg_for_object_list=self._pegs[:num_blocks]
        #for i in peg_for_object_list:
        
        for i in self._pegs[6: 6 + num_blocks]:
            pos, orn = get_link_pose(self.obj_ids['fixed'][1], i)
            
            #block_pos=np.array(pos) + np.array([0, 0, 0.03+0.045*self.SCALING])
            
            #print('object pos: ',pos)
            yaw = (np.random.rand() - 0.5) * np.deg2rad(60)
            obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'block/block.urdf'),
                                np.array(pos) + np.array([0, 0, 0.03]),
                                p.getQuaternionFromEuler((0, 0, yaw)),
                                useFixedBase=False,
                                globalScaling=self.SCALING)
            self.obj_ids['rigid'].append(obj_id)
        self._blocks = np.array(self.obj_ids['rigid'][-num_blocks:])
        np.random.shuffle(self._blocks)
        #self._blocks = [5,8,7,6]
        for obj_id in self._blocks[:1]:
            # change color to red
            p.changeVisualShape(obj_id, -1, rgbaColor=(255 / 255, 69 / 255, 58 / 255, 1))
        self.obj_id, self.obj_link1 = self._blocks[0], 1
        #print(self._pegs, self.obj_id, self._blocks)
        '''
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)

        workspace_limits = self.workspace_limits1
        
        temp=np.random.randint(-5,6)/20

        #workspace_limits[1][1] if np.random.rand()>0.5 else workspace_limits[1][0]
        #pos = (workspace_limits[0][0],
        #       temp,
        #       workspace_limits[2][1])
        pos=(pos_obj[0], pos_obj[1],pos_obj[2])
                                       
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        '''
    def _load_dam(self):
        encoder = 'vitb' # can also be 'vitb' or 'vitl'
        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()
        self.img_transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',    
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            ])

    
    def _get_depth_with_dam(self, img):
        '''
        input: rgb image 1xHxW
        '''
        #print('ori img: ',img)
        img=img/255.0
        h, w = img.shape[:2]
        
        img=self.img_transform({'image': img})['image']
        img=torch.from_numpy(img).unsqueeze(0)
        with torch.no_grad():
            depth = self.depth_anything(img)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) # 0-1
        #print(depth.mean())
        
        depth = depth.cpu().numpy()

        return depth

        
    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        # TODO: may need to tune parameters
        
        
        #print(achieved_goal)
        #print(desired_goal)
        #print(self.goal)
        #print(self._waypoints[3] )
        is_success_current=np.logical_and(
            goal_distance(achieved_goal[..., :2], desired_goal[..., :2]) < 5e-3 * self.SCALING,
            np.abs(achieved_goal[..., -1] - desired_goal[..., -1]) < 4e-3 * self.SCALING
        ).astype(np.float32)

        #print('is_success_current: ',is_success_current)
        '''
        pos_tip = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        pos_grasp = np.array([pos_obj[0], pos_obj[1],  
                    pos_obj[2]])
        is_success_current=np.logical_and(
            goal_distance(pos_tip[:2],pos_grasp[ :2]) < 5e-3 * self.SCALING,
            np.abs(pos_tip[-1] - pos_grasp[-1]) < 4e-3 * self.SCALING
        ).astype(np.float32)
        print(pos_tip)
        print(pos_grasp)
        print('is_success_current: ',is_success_current)
        '''
        return is_success_current

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        
        # goal = np.array(get_link_pose(self.obj_ids['fixed'][1], self._pegs[0])[0])

        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)

        goal=np.array([pos_obj[0], pos_obj[1],pos_obj[2]+0.005 ])
        #pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        #goal=np.array([pos_obj[0], pos_obj[1],pos_obj[2] + 0.045 * self.SCALING, yaw, -0.5])
        print('goal: ',goal)
        #print('goal2: ',goal2)
        #exit()
        return goal.copy()
    
    def _rot_w2rcm(self, euler):
        
        matrix=np.zeros((3,4))
        matrix=np.concatenate([matrix,np.array([0.,0.,0.,1.]).reshape(1,-1)],axis=0)
        m_rot= get_matrix_from_euler(euler)
        
        matrix[:3,:3]=m_rot
        rcm_pose=self.psm1.pose_world2rcm(matrix)
        rcm_eul=get_euler_from_matrix(rcm_pose[:3,:3])
        
        return rcm_eul

    def _sample_goal_callback_all(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None, None, None, None]  # six waypoints
    
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw
        
        pos_peg = get_link_pose(self.obj_ids['fixed'][1],
                                #self._pegs[self.obj_id - np.min(self._blocks)])[0]
                                self._pegs[6])[0]
        #print('pos_peg: ',pos_peg)
        
        
        
        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, 0.5])  # above object
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, -0.5])  # lift up
        
        pos_place = [self.goal[0] + pos_obj[0] - pos_peg[0],
                     self.goal[1] + pos_obj[1] - pos_peg[1], self._waypoints[0][2]]  # consider offset
        #pos_place=self.goal
        self._waypoints[4] = np.array([pos_place[0], pos_place[1],  self._waypoints[0][2], yaw, -0.5])  # above goal
        self._waypoints[5] = np.array([pos_place[0], pos_place[1], self._waypoints[2][2], yaw, -0.5])  # release
        #self._waypoints[5] = np.array([pos_place[0], pos_place[1], pos_place[2], yaw, 0.5])  # release
        self._waypoints[6] = np.array([pos_place[0], pos_place[1], self._waypoints[2][2], yaw, 0.5])  # release
        
        self.waypoints = self._waypoints.copy()


    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        #self._waypoints = []  
        self._waypoints = [None, None, None, None]  # six waypoints
    
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw
        
        pos_peg = get_link_pose(self.obj_ids['fixed'][1],
                                #self._pegs[self.obj_id - np.min(self._blocks)])[0]
                                self._pegs[6])[0]


        pos_mid1 = [pos_obj[0], 0. + pos_obj[1] - pos_peg[1], pos_obj[2] + 0.043 * self.SCALING]  # consider offset
        
        #-------------------------------Add noise-----------------------------------------
        noise_std = 0.04
        noise = np.clip(noise_std * np.random.random(3), -noise_std, noise_std)
        nsd_pos_mid1 = pos_mid1 + noise

        if self.subtask=="all" or self.subtask=="grasp":
            #----------------------------Subtask 1----------------------------
            #self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2] + 0.015 * self.SCALING, yaw, 0.5]))  # psm2 above object 0
            #self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2] + 0.015 * self.SCALING, yaw, 0.5]))  # psm2 approach 1
            #self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2] + 0.015 * self.SCALING, yaw, 0.5 ]))  # psm2 grasp 2
            #self._waypoints.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2] + 0.015 * self.SCALING, yaw, 0.5]))  # psm2 lift up 3 
            self._waypoints[0] = np.array([pos_obj[0], pos_obj[1]+0.002,
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, 0.5])  # above object
            self._waypoints[1] = np.array([pos_obj[0], pos_obj[1]+0.002,
                                        pos_obj[2] + 0.03 * self.SCALING, yaw, 0.5])  # approach
            self._waypoints[2] = np.array([pos_obj[0], pos_obj[1]+0.002,
                                        pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
            self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, -0.5])  # lift up
            #self.subgoals.append(np.array([nsd_pos_mid1[0], nsd_pos_mid1[1], nsd_pos_mid1[2] + 0.015 * self.SCALING]))
            if self.subtask=="grasp":
                self.goal=np.array([pos_obj[0], pos_obj[1],pos_obj[2] + 0.03 * self.SCALING])


        if self.subtask=="all" or self.subtask=='movegrasp':
        #----------------------------Subtask 2----------------------------        

            pos_place = [self.goal[0] + pos_obj[0] - pos_peg[0],
                        self.goal[1] + pos_obj[1] - pos_peg[1], nsd_pos_mid1[2]]  # consider offset

            self._waypoints.append( np.array([pos_place[0], pos_place[1],  nsd_pos_mid1[2] + 0.015 * self.SCALING, yaw, -0.5]) ) # above goal
            self._waypoints.append(np.array([pos_place[0], pos_place[1], nsd_pos_mid1[2] + 0.015 * self.SCALING ,yaw, -0.5]) ) # release
            #self._waypoints[5] = np.array([pos_place[0], pos_place[1], pos_place[2], yaw, 0.5])  # release
            self._waypoints.append(np.array([pos_place[0], pos_place[1], nsd_pos_mid1[2] + 0.015 * self.SCALING, yaw, 0.5])  )# release
            if self.subtask=="move":
                self.goal=np.array([pos_place[0], pos_place[1], nsd_pos_mid1[2] + 0.015 * self.SCALING, yaw, 0.5])

        self._waypoints_done = [False] * len(self._waypoints)

            #self.subgoals.append(np.array([self.goal[0], self.goal[1], self.goal[2]]))

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        pose = get_link_pose(self.obj_id, -1)
        #print('constraint check--pose: ',pose[0][2])
        #print('constraint check--goal: ',self.goal[2] + 0.01 * self.SCALING)
        #return True
        # return True
        return pose[0][2] > self.goal[2] + 0.01 * self.SCALING
    
    
    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state(idx=0)
        #print('robot cam pos: ',robot_state[:3])
        
        # TODO: may need to modify
       
        pos, _ = get_link_pose(self.obj_id, -1)

        object_pos = np.array(pos)
        #print("ori obejct pose: ",object_pos)
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        waypoint_pos = np.array(pos)
        # rotations
        waypoint_rot = np.array(p.getEulerFromQuaternion(orn))
        
        object_pos=self._world2cam_pos(object_pos)
        #print('obs pos cam: ',object_pos)
        waypoint_pos=self._world2cam_pos(waypoint_pos)
        waypoint_rot=self._world2cam_rot(waypoint_rot)
        #print('waypoint_rot: ', waypoint_rot)
        #print(object_pos)
        #print(waypoint_pos)

        object_rel_pos = object_pos - robot_state[0: 3]
        
        # tip position
        achieved_goal = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])
        achieved_goal =self._world2cam_pos(achieved_goal)

        #print('waypoint_rot: ', waypoint_rot)
        observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_rot.ravel()  # achieved_goal.copy(),
        ])
        goal=self._world2cam_pos(self.goal)
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': goal.copy()
        }
        
        render_obs,seg, depth=self.ecm.render_image()
        #cv2.imwrite('/research/d1/rshr/arlin/data/debug/depth_noise_debug/img.png',cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB))
        #plt.imsave('/research/d1/rshr/arlin/data/debug/depth_noise_debug/img2.png',render_obs)
        #print('depth max: ',np.max(depth))
        #exit()
        render_obs=cv2.resize(render_obs,(320,240))
        
        self.counter+=1
        #print(render_obs[0][0])
        #exit()
        #seg=np.array(seg==6).astype(int)
        
        seg=np.array((seg==6 )| (seg==1)).astype(int)
        #seg=np.array(seg==1).astype(int)
        #seg=np.resize(seg,(320,240))
        
        #plt.imsave('/research/d1/rshr/arlin/data/debug/depth_noise_debug/depth.png',depth)
        #exit()
        seg = cv2.resize(seg, (320,240), interpolation =cv2.INTER_NEAREST)
        #plt.imsave('/research/d1/rshr/arlin/data/debug/seg_debug/noise_{}/seg.png'.format(self.curr_intensity),seg)
        #exit()
        depth = cv2.resize(depth, (320,240), interpolation =cv2.INTER_NEAREST)
        #print(np.max(depth))
        #depth = cv2.resize(depth, (320,240),interpolation=cv2.INTER_LANCZOS4)

        
        #image=cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB) / 255.0
        #plt.imsave('/home/student/code/regress_data7/seg/seg_{}.png'.format(self.counter),seg)
        #image = self.transform({'image': image})['image']
        #image=torch.from_numpy(image).to("cuda:0").float()

         # test depth noise
        
        #if np.random.randn()<0.5:
        #    instensity=np.random.randint(3,15)/100
        #instensity=0.1
        #    depth = add_gaussian_noise(depth, instensity)
        '''
        if self.counter==10:
            cv2.imwrite('/research/d1/rshr/arlin/data/debug/depth_noise_debug/gaussian/img.png',cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB))
            plt.imsave('/research/d1/rshr/arlin/data/debug/depth_noise_debug/gaussian/depth.png',depth)
            for i in [0.01,0.05,0.1,0.15,0.2]:
                noisy_depth_map = add_random_noise(depth, i)
                plt.imsave('/research/d1/rshr/arlin/data/debug/depth_noise_debug/gaussian/noise_{}.png'.format(i),noisy_depth_map)

            exit()
        '''

        #noisy_segmentation_map = add_noise_to_segmentation(seg, self.seg_noise_intensity)
        #noisy_depth_map = add_gaussian_noise(depth, self.curr_intensity)
        #if self.counter==10:
        #    plt.imsave('/research/d1/rshr/arlin/data/debug/seg_debug/noise_{}/img.png'.format(self.curr_intensity),render_obs)
        #    plt.imsave('/research/d1/rshr/arlin/data/debug/seg_debug/noise_{}/seg.png'.format(self.curr_intensity),seg)
        #    plt.imsave('/research/d1/rshr/arlin/data/debug/seg_debug/noise_{}/noise_seg.png'.format(self.curr_intensity),noisy_segmentation_map)

        seg=torch.from_numpy(seg).to("cuda:0").float()
        depth=torch.from_numpy(depth).to("cuda:0").float()

        
        with torch.no_grad():
            v_output=self.v_model.get_obs(seg.unsqueeze(0), depth.unsqueeze(0))[0]#.cpu().data().numpy()
        #print(v_output.shape)
        v_output=v_output.cpu().numpy()
        #print("get v_output")
        o=obs['observation']
        #print("ori obs: ", o)
        robot_state=o[:7]
        rel_pos=v_output[:3]
        new_pos=robot_state[:3]+rel_pos[:3]
        waypoint_pos_rot=v_output[3:]
        o_new=np.concatenate([robot_state, new_pos, rel_pos, waypoint_pos_rot])
        #print('new observation: ',o_new)
        #with open("/research/d1/rshr/arlin/SAM-rbt-sim2real/traj.txt","a") as f:
        #    f.write('obs:\n')
        #    f.write(str(o_new))

        #print("new obs: ", o_new)
        obs['observation']=o_new
        
        return obs
    
    
    def calculate_ecm_rotation(self, world2ecm_matrix, yaw_change):
        # 计算世界坐标系中的yaw角度改变对应的旋转矩阵
        world_rotation = R.from_euler('z', yaw_change)

        # 获取RCM坐标系的旋转矩阵
        ecm_rotation_matrix = world2ecm_matrix[:3, :3]

        # 将世界坐标系中的旋转矩阵转换到RCM坐标系中
        ecm_rotation = np.matmul(ecm_rotation_matrix, world_rotation.as_matrix())

        # 将得到的旋转矩阵转换为欧拉角
        ecm_euler = R.from_matrix(ecm_rotation).as_euler('xyz')

        return ecm_euler
    

    def get_oracle_action_subgoal(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        obss=obs['observation'][:6]
        action = np.zeros(7)

        for i, waypoint in enumerate(self._waypoints):
            if self._waypoints_done[i]:
                continue
            #print("current: ",i)

            ecm_waypoint_pos=self._world2cam_pos(waypoint[:3])
            ecm_obs_pos=self._world2cam_pos(obss[:3].copy())
            delta_pos=(ecm_waypoint_pos-ecm_obs_pos)/0.01/self.SCALING
            
            delta_yaw=(waypoint[3]-obss[-1])
           
            while abs(delta_yaw+np.pi/2)<abs(delta_yaw):
                delta_yaw=delta_yaw+np.pi/2
          
            delta_yaw=delta_yaw.clip(-0.4,0.4)
            
            delta_rot=self.calculate_ecm_rotation(self._view_matrix, delta_yaw)
           
            # TODO: current assume delta_yaw is positive
            
            while abs(delta_rot[2]-np.pi/2)<abs(delta_rot[2]):
                delta_rot[2]=delta_rot[2]-np.pi/2
          
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.7
            delta_pos *= scale_factor
            
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_rot[0], delta_rot[1],delta_rot[2],waypoint[4]])
            
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 2e-3 and np.abs(delta_yaw) < np.deg2rad(2.):
                
                self._waypoints_done[i] = True
            break

        return action
    
    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # six waypoints executed in sequential order
        obss=obs['observation'][:6]
        action = np.zeros(7)
        
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            print("current: ",i)
            
            #print(obss)
            '''
            ecm_waypoint_pos=self._world2cam_pos(waypoint[:3])
            ecm_waypoint_euler=self._world2cam_rot(np.array([0.,0.,waypoint[3]]))
            ecm_obs_pos=self._world2cam_pos(obss[:3].copy())
            ecm_obs_euler=self._world2cam_rot(obss[3:].copy())
            print('goal yaw: ',ecm_waypoint_euler[-1])
            print('robot yaw: ',ecm_obs_euler[-1])
            
            delta_pos=(ecm_waypoint_pos-ecm_obs_pos)/0.01/self.SCALING
            delta_yaw=(ecm_waypoint_euler[-1]-ecm_obs_euler[-1])#.clip(-0.4,0.4)
            print("delta_yaw: ",delta_yaw)
            '''
            #print('world rot: ',obss[3:])
            #print(waypoint[:3])
            ecm_waypoint_pos=self._world2cam_pos(waypoint[:3])
            #print(ecm_waypoint_pos)
            #print(np.dot(self._view_matrix[:3,:3], waypoint[:3])+self._view_matrix[:3,3])
            #exit()
            ecm_obs_pos=self._world2cam_pos(obss[:3].copy())
            delta_pos=(ecm_waypoint_pos-ecm_obs_pos)/0.01/self.SCALING
            #delta_pos=(waypoint[:3]-obss[:3])/0.01/self.SCALING
            delta_yaw=(waypoint[3]-obss[-1])
            #print("delta_yaw: ",delta_yaw)
            
            while abs(delta_yaw+np.pi/2)<abs(delta_yaw):
                delta_yaw=delta_yaw+np.pi/2
            
            #print("delta_yaw: ",delta_yaw)
            delta_yaw=delta_yaw.clip(-0.4,0.4)
            
            delta_rot=self.calculate_ecm_rotation(self._view_matrix, delta_yaw)
            #print('delta_rot: ',delta_rot)
            #print("delta_yaw: ",delta_yaw)
            
            # TODO: current assume delta_yaw is positive
            #while delta_rot[2]<0:
            #    delta_rot[2]=delta_rot[2]+np.pi/2
            while abs(delta_rot[2]-np.pi/2)<abs(delta_rot[2]):
                delta_rot[2]=delta_rot[2]-np.pi/2
            #print("delta_rot: ",delta_rot)
            #if i>3:
            #    delta_yaw=0
            #else:
            #delta_yaw=delta_yaw.clip(-0.4,0.4)
            
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.7
            delta_pos *= scale_factor
            
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_rot[0], delta_rot[1],delta_rot[2],waypoint[4]])
            #print("delta_yaw: ",delta_yaw)
            #print("delta_pos: ", delta_pos)
            
            #action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint_rot[1]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 2e-3 and np.abs(delta_yaw) < np.deg2rad(2.):
                self._waypoints[i] = None
                print('solve ',i)
                
            break
        
        return action
    # get action in RCM axis
    def get_oracle_action_RCM(self, obs) -> np.ndarray: 
        """
        Define a human expert strategy
        """
        # six waypoints executed in sequential order
        obss=obs['observation'][:6]
        action = np.zeros(5)
        action1=np.zeros(5)
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            print("current: ",i)
        
            new_waypoint=np.append(waypoint[:3],[0.,0.])
            new_waypoint=np.append(new_waypoint,waypoint[3])
            
            #print(obss)
            rcm_waypoint=self.psm1.pose_world2rcm((new_waypoint[:3],new_waypoint[3:]),option = 'tuple')
            rcm_observation=self.psm1.pose_world2rcm((obss[:3].copy(),obss[3:].copy()),option = 'tuple')
            rcm_wp_euler=np.array(p.getEulerFromQuaternion(rcm_waypoint[1]))
            rcm_obs_euler=np.array(p.getEulerFromQuaternion(rcm_observation[1]))
            
            #rcm_obs_euler=(rcm_obs_euler+np.deg2rad(np.pi/2))
            
            rcm_wp_pos=np.array(rcm_waypoint[0])
            rcm_obs_pos=np.array(rcm_observation[0])
            #print("rcm_obs_pos: ",rcm_obs_pos)
            #rcm_obs_euler[-1]=(rcm_obs_euler[-1]-np.deg2rad(np.pi))
            print("goal yaw: ",rcm_wp_euler[-1])
            print('obs yaw: ',rcm_obs_euler[-1])
            #rcm_obs_euler=(rcm_obs_euler-np.deg2rad(np.pi))
            
            delta_pos=(rcm_wp_pos-rcm_obs_pos)/0.01/self.SCALING
            delta_yaw=(rcm_wp_euler[-1]-rcm_obs_euler[-1])#.clip(-0.4,0.4)
            print("delta_yaw: ",delta_yaw)
            
            # TODO: current assume delta_yaw is positive
            while abs(delta_yaw+np.pi/2)<abs(delta_yaw):
                delta_yaw=delta_yaw+np.pi/2
            print("delta_yaw: ",delta_yaw)
            #if i>3:
            #    delta_yaw=0
            #else:
            delta_yaw=delta_yaw.clip(-0.4,0.4)
            #print("delta_yaw: ",delta_yaw)
            #delta_yaw=delta_yaw%(np.pi/2)#.clip(-0.4,0.4)
            
            #if delta_yaw>np.deg2rad(np.pi/2):
            #    rcm_obs_euler=(rcm_obs_euler+np.deg2rad(np.pi/2))
            #    delta_yaw=rcm_wp_euler[-1]-rcm_obs_euler[-1]
                
            #delta_pos=(waypoint[:3]-rcm_obs_pos)/0.01/self.SCALING
            #delta_yaw=(waypoint[3]-rcm_obs_euler[-1])#.clip(-0.4,0.4)
            
            #delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.7
            delta_pos *= scale_factor
            
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            print("delta_yaw: ",delta_yaw)
            print("delta_pos: ", delta_pos)
            
            '''
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.7
            delta_pos *= scale_factor
            
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            '''
            #action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint_rot[1]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 2e-3 and np.abs(delta_yaw) < np.deg2rad(1.):
                self._waypoints[i] = None
                print('solve ',i)
                
            break
        
        return action
    
    def subgoal_grasp(self):
        scale_factor = 0.7
        # robot_state
        robot_state = self._get_robot_state(idx=0)
        pos_robot = robot_state[:3]
        yaw_robot = robot_state[5]
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw_angle = robot_state[-1]

        pos_tip = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])

        # object pose
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        orn = p.getEulerFromQuaternion(orn_obj)
        yaw_grasp = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        pos_grasp = np.array([pos_obj[0], pos_obj[1],  
                    pos_obj[2]])  # grasp

        # is_success
        is_grasp = yaw_angle < 0.2  
        is_success = self._is_success(pos_grasp, pos_tip) #and (is_grasp or self._meet_contact_constraint_requirement())
        return is_success.astype(np.float32)

    def is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        return self._is_success(achieved_goal, desired_goal)
        

    @property
    def pegs(self):
        return self._pegs


if __name__ == "__main__":
    env = PegTransfer(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
