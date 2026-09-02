import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera,
    wrap_angle
)
from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm
import cv2
import torch
from gym import spaces
from scipy.spatial.transform import Rotation as R


import sys
sys.path.append('/home/jwfu/ar_surrol/surrol/tasks')
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from torchvision.transforms import Compose

sys.path.append('/media/jwfu/84D609C8D609BC04/ar_surrol_datageneration/stateregress')
from vmodel import vismodel
from config import opts

from surrol.utils.pybullet_utils import (
    step,
    render_image,
)


class MatchBoard(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    # RPOT_POS = (0.55, -0.025, 0.6781)
    # GPOT_POS = (0.55, 0.03, 0.6781)
    # POT_ORN = (1.57,0,0)
    BOARD_POS = (0.55, 0, 0.6781)
    BOARD_ORN= (1.57,0,0)
    BOARD_SCALING = 0.03
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 1.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE=7
    ACTION_SIZE = 7
    haptic = False

    # For data collection
    counter=0
    img_list={}
    

    # TODO: grasp is sometimes not stable; check how to fix it
    def __init__(self, render_mode=None, cid = -1):
        super(MatchBoard, self).__init__(render_mode, cid)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )


    def _env_setup(self):

        # self.subtask="all"
        #self._load_dam()
        self.action_space = spaces.Box(-1., 1., shape=(self.ACTION_SIZE,), dtype='float32')

        opts.device='cuda:0'
        self.v_model=vismodel(opts)
        ckpt=torch.load('/media/jwfu/84D609C8D609BC04/ar_surrol_datageneration/stateregress/exp_18again_close_scaling1_exist_depth_corr_d_matchboard/checkpoints/best_model.pt', map_location=opts.device)
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

        super(MatchBoard, self)._env_setup()
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True
 
        # camera
        if self._render_mode == 'human':
            # reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
            #              target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
            reset_camera(yaw=89.60, pitch=-56, dist=5.98,
                         target=(-0.13, 0.03,-0.94))
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING,
                       view_matrix=ecm_view_matrix)
        self.ecm.reset_joint(self.QPOS_ECM)
        # p.setPhysicsEngineParameter(enableFileCaching=0,numSolverIterations=10,numSubSteps=128,contactBreakingThreshold=2)


        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # physical interaction
        self._contact_approx = False



        self.distance_threshold = 0.005

        # metal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/dot_metal_min.jpg"))
        # newmetal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/metal2.jpg"))
        # tray pad
        board = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=0.2*0.03,
                            useFixedBase=1)
        self.obj_ids['fixed'].append(board)  # 1

        match_board = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'match_board_rl/match_board.urdf'),
                            np.array(self.BOARD_POS) * self.SCALING,
                            p.getQuaternionFromEuler(self.BOARD_ORN),
                            globalScaling=0.2*0.03,
                            useFixedBase=1)
        self.obj_ids['fixed'].append(match_board)

        # Set candidate object pose
        obj_pose_offset = []
        for i in range(7):
            if i==3 or i==2 or i==6:
                continue
            if i<3:
                if i%3==0:
                    pos_offset = [-0.15*0.2,(-0.075*(i%3+1)-0.15)*0.2,0.06]
                else:
                    pos_offset = [-0.15*0.2,(0.075*i+0.15)*0.2,0.06]
            else:
                # pos_offset = [0,-0.075*(7-i)-0.15,0.06]
                if i==4:
                    pos_offset = [0*0.2,(-0.075*(i%3)-0.15)*0.2,0.06]
                else:
                    pos_offset = [0*0.2,(0.075*(i-4)+0.15)*0.2,0.06]
            obj_pose_offset.append(pos_offset)
        for i in range(3):
            if i==2:
                continue
            if i%3==0:
                pos_offset = [0.15*0.2,(-0.075*(i%3+1)-0.15)*0.2,0.06]
            else:
                pos_offset = [0.15*0.2,(0.075*i+0.15)*0.2,0.06]
            obj_pose_offset.append(pos_offset)
        
        # Object library
        object_library = ['0', '1', '2', '4', '5', '6', 'A', 'B', 'C']
        self.object_encoding = {'0':0, '1':1, '2':2, '4':3, '5':4, '6':5, 'A':6, 'B':7, 'C':8}
        self.object_manipulation_offset = {'0': [0.*0.2, 0.025*0.2, 0., 0.],
                                           '1': [0.*0.2, 0.*0.2, 0., 0.],
                                           '2': [0.*0.2, 0.*0.2, 0., -np.pi/6.], # todo
                                           '4': [0.02*0.2, 0.015*0.2, 0., 0.],
                                           '5': [0.*0.2, 0.*0.2, 0., -np.pi/2.], 
                                           '6': [0.035*0.2, 0.*0.2, 0., -np.pi/2.], 
                                           'A': [0.01*0.2, 0.*0.2, 0., 0.],
                                           'B': [0.*0.2, 0.*0.2, 0., 0.],
                                           'C': [0.*0.2, -0.025*0.2, 0., -np.pi/2], 
                                           }
        self.object = np.random.choice(object_library, 1)[0]

        # self.object = '6'

        pos_offset = obj_pose_offset[np.random.randint(0, len(obj_pose_offset)-1)]

        while self.object == 'C' and pos_offset[1] > 0.0:
            pos_offset = obj_pose_offset[np.random.randint(0, len(obj_pose_offset)-1)]
        
        # pos_offset[0] *= 0.2
        # pos_offset[1] *= 0.2

        if self.object in ['A', 'B', 'C']:
            fn = 'match_board_rl/'+chr(ord(self.object)-ord('A')+ord('a'))+'.urdf'
        else:
            fn = 'match_board_rl/'+str(self.object)+'.urdf'
        urdf_path = os.path.join(ASSET_DIR_PATH, fn)


        obj= p.loadURDF(urdf_path,np.array(self.BOARD_POS) * self.SCALING+pos_offset,p.getQuaternionFromEuler(self.BOARD_ORN),globalScaling=0.2*0.03)
        self.obj_ids['rigid'].append(obj)

        self.obj_target_offset = {'0': [-0.125, -0.125+0.03, 0.0, 0.],
                            '1': [-0.125, 0.0, 0.0, 0.],
                            '2': [-0.125+0.01, 0.125, 0.0, 0.],
                            '4': [0.02+0.005, -0.125+0.015, 0.0, 0.],
                            '5': [0.0, 0.0, 0.0, 0.],
                            '6': [0.035, 0.125, 0.0, 0.],
                            'A': [0.125, -0.125, 0.0, 0.],
                            'B': [0.125, 0.0, 0.0, 0.],
                            'C': [0.125, 0.125-0.025, 0., 0.]}
        self.obj_target = {'0': [-0.125, -0.125, 0.0, 0.],
                            '1': [-0.125, 0.0, 0.0, 0.],
                            '2': [-0.125, 0.125, 0.0, 0.],
                            '4': [0.0, -0.125, 0.0, 0.],
                            '5': [0.0, 0.0, 0.0, 0.],
                            '6': [0., 0.125, 0.0, 0.],
                            'A': [0.125, -0.125, 0.0, 0.],
                            'B': [0.125, 0.0, 0.0, 0.],
                            'C': [0.125, 0.125, 0., 0.]}
        self.target_pose = np.array(self.BOARD_POS) * self.SCALING + self.obj_target[self.object][:3]

        self.target_pose_with_offset = np.array(self.BOARD_POS) * self.SCALING + self.obj_target_offset[self.object][:3]


        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1
        




    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        # workspace_limits = self.workspace_limits1
        # goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
        #                  workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
        #                  workspace_limits[2][1] - 0.04 * self.SCALING])
        # goal_1 = np.array(self.RPOT_POS) * self.SCALING
        # #goal_2 = np.array(self.GPOT_POS) * self.SCALING
        # #return np.concatenate([goal_1.copy(), goal_2.copy()])
        # return goal_1.copy()
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)

        goal=np.array([pos_obj[0], pos_obj[1],pos_obj[2] + 0.01 * self.SCALING])

        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        # super()._sample_goal_callback()
        self._waypoints = [None] * 7  # four waypoints

        # TODO: refine the specific manipulating point for different object with different shape
        # First object waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = orn[2] if abs(wrap_angle(orn[2] - orn_eef[2])) < abs(wrap_angle(orn[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn[2] + np.pi)  # minimize the delta yaw

        # # for physical deployment only
        # print(" -> Needle pose: {}, {}".format(np.round(pos_obj, 4), np.round(orn_obj, 4)))
        # qs = self.psm1.get_current_joint_position()
        # joint_positions = self.psm1.inverse_kinematics(
        #     (np.array(pos_obj) + np.array([0, 0, (-0.0007 + 0.0102)]) * self.SCALING,
        #      p.getQuaternionFromEuler([-90 / 180 * np.pi, -0 / 180 * np.pi, yaw])),
        #     self.psm1.EEF_LINK_INDEX)
        # self.psm1.reset_joint(joint_positions)
        # print("qs: {}".format(joint_positions))
        # print("Cartesian: {}".format(self.psm1.get_current_position()))
        # self.psm1.reset_joint(qs)
        cur_goal = self.target_pose_with_offset[:3]
        object_catch_offset = self.object_manipulation_offset[self.object]
        self._waypoints[0] = np.array([pos_obj[0]+object_catch_offset[0], pos_obj[1]+object_catch_offset[1],
                                       pos_obj[2]+object_catch_offset[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING,
                                       yaw+object_catch_offset[3], 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0]+object_catch_offset[0], pos_obj[1]+object_catch_offset[1],
                                       pos_obj[2]+object_catch_offset[2] + (-0.0007 + 0.0102) * self.SCALING, 
                                       yaw+object_catch_offset[3], 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0]+object_catch_offset[0], pos_obj[1]+object_catch_offset[1],
                                       pos_obj[2]+object_catch_offset[2] + (-0.0007 + 0.0102) * self.SCALING,
                                       yaw+object_catch_offset[3], -0.5])  # grasp
        self._waypoints[3] = np.array([pos_obj[0]+object_catch_offset[0], pos_obj[1]+object_catch_offset[1],
                                       cur_goal[2] + 0.0302 * self.SCALING + (-0.0007 + 0.0102) * self.SCALING,
                                       yaw+object_catch_offset[3], -0.5])  # lift up
        self._waypoints[4] = np.array([cur_goal[0], cur_goal[1],
                                       cur_goal[2] + 0.0302 * self.SCALING, 
                                       yaw+object_catch_offset[3], -0.5])  # move
        self._waypoints[5] = np.array([cur_goal[0], cur_goal[1],
                                       cur_goal[2] + 0.0302 * self.SCALING, 
                                       yaw+object_catch_offset[3], 0.5])  # release
        self._waypoints[6] = np.array([cur_goal[0], cur_goal[1],
                                       cur_goal[2] + 0.0302 * self.SCALING, 
                                       yaw+object_catch_offset[3], 0.5])  # release


        


    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        # return True
        if self._contact_approx or self.haptic is True:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.008 * self.SCALING


    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
        # action = np.zeros(5)
        # action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            # print('*'*50)
            # print('[DEBUG] waypoints index: {}'.format(i))
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            if obs['observation'][5] < -np.pi + 0.1:
                obs['observation'][5] += 2 * np.pi
            # print('[DEBUG] target yaw: {}, current yaw: {}'.format(waypoint[3], obs['observation'][5]))
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.5
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            # print('[DEBUG] delta_pos norm: {}'.format(np.linalg.norm(delta_pos)))
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-3 and np.abs(delta_yaw) < 1e-2:
                self._waypoints[i] = None
            break

        return action
    def _set_action_ecm(self, action):
        action *= 0.01 * self.SCALING
        pose_rcm = self.ecm.get_current_position()
        pose_rcm[:3, 3] += action
        pos, _ = self.ecm.pose_rcm2world(pose_rcm, 'tuple')
        joint_positions = self.ecm.inverse_kinematics((pos, None), self.ecm.EEF_LINK_INDEX)  # do not consider orn
        self.ecm.move_joint(joint_positions[:self.ecm.DoF])

    def _reset_ecm_pos(self):
        self.ecm.reset_joint(self.QPOS_ECM)
    
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
        
        # self.counter+=1
        #print(render_obs[0][0])
        #exit()
        #seg=np.array(seg==6).astype(int)
        
        seg=np.array((seg==7 )| (seg==10000)).astype(int)
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
        v_output = np.random.uniform(low=-1, high=1, size=9)

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
    
    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal, desired_goal)
        # print('[DEBUG] d: {}'.format(d))


        return (d < self.distance_threshold).astype(np.float32)



    
    def _step_callback(self):
        """ Remove the contact constraint if no contacts
        """
        if self.block_gripper or not self.has_object:
            return
        if self._contact_constraint is None:
            # the grippers activate; to check if they can grasp the object
            # TODO: check whether the constraint may cause side effects


            if self._meet_contact_constraint_requirement():
                psm = self.psm1 
                body_pose = p.getLinkState(psm.body, psm.EEF_LINK_INDEX)
                obj_pose = p.getBasePositionAndOrientation(self.obj_id)
                world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                   world_to_body[1],
                                                   obj_pose[0], obj_pose[1])

                self._contact_constraint = p.createConstraint(
                    parentBodyUniqueId=psm.body,
                    parentLinkIndex=psm.EEF_LINK_INDEX,
                    childBodyUniqueId=self.obj_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=obj_to_body[0],
                    parentFrameOrientation=obj_to_body[1],
                    childFramePosition=(0, 0, 0),
                    childFrameOrientation=(0, 0, 0))
                # TODO: check the maxForce; very subtle
                p.changeConstraint(self._contact_constraint, maxForce=20)

        else:

            # self._contact_constraint is not None
            # the gripper grasp the object; to check if they remain contact
            psm = self.psm1
            points = p.getContactPoints(bodyA=psm.body, linkIndexA=6) \
                        + p.getContactPoints(bodyA=psm.body, linkIndexA=7)
            points = [point for point in points if point[2] == self.obj_id]
            remain_contact_1 = len(points) > 0

            # print('[DEBUG] remain_contact_1: {}'.format(remain_contact_1))


            if not remain_contact_1 and not self._contact_approx:
                # release the previously grasped object because there is no contact any more
                self._release_1()
    
    def _release_1(self):
        # release the object
        if self.block_gripper:
            return


        if self._contact_constraint is not None:
            try:
                p.removeConstraint(self._contact_constraint)
                self._contact_constraint = None
                # enable collision
                psm = self.psm1
                p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id,
                                            linkIndexA=6, linkIndexB=-1, enableCollision=1)
                p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id,
                                            linkIndexA=7, linkIndexB=-1, enableCollision=1)
            except:
                pass

        

if __name__ == "__main__":
    env = MatchBoard(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
