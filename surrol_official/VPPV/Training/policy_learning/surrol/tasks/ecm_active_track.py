import os
import time

import pybullet as p
from surrol.tasks.ecm_env_movecp import EcmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_body_pose,
)
import random
import cv2
import pickle
from PIL import Image
from surrol.utils.robotics import (
    get_euler_from_matrix,
    get_matrix_from_euler
)
import torch
from surrol.utils.utils import RGB_COLOR_255, Boundary, Trajectory, get_centroid
from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
import numpy as np
from surrol.robots.psm import Psm1, Psm2
import sys
sys.path.append('/home/kejianshi/Desktop/Surgical_Robot/science_robotics/stateregress_back')
sys.path.append('/home/kejianshi/Desktop/Surgical_Robot/science_robotics/stateregress_back/utils')
from general_utils import AttrDict
sys.path.append('/home/kejianshi/Desktop/Surgical_Robot/science_robotics/ar_surrol/surrol_datagen/tasks')
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

from vmodel import vismodel
from config import opts

class ActiveTrack(EcmEnv):
    """
    Active track is not a GoalEnv since the environment changes internally.
    The reward is shaped.
    """
    ACTION_MODE = 'cVc'
    # RCM_ACTION_MODE = 'yaw'
    QPOS_ECM = (0, 0, 0.02, 0)
    WORKSPACE_LIMITS = ((-0.3, 0.6), (-0.4, 0.4), (0.05, 0.05))

    CUBE_NUMBER = 18

    def __init__(self, render_mode=None, bg_img_path='/home/kejianshi/Desktop/Surgical_Robot/science_robotics/to_KJ/to_KJ/background/0.png'):
        # to control the step
        self._step = 0
        self.counter=0
        self.img_list={}
        super(ActiveTrack, self).__init__(render_mode)
        self.background = np.array(Image.open(bg_img_path))[...,:3]
        self.bg_height, self.bg_width, _ = self.background.shape
        self.debug=False

    def step(self, action: np.ndarray):
        obs, reward, done, info = super().step(action)
        centroid = obs['observation'][-3: -1]
        if not (-1.2 < centroid[0] < 1.1 and -1.1 < centroid[1] < 1.1):
            # early stop if out of view
            done = True
        info['achieved_goal'] = centroid
        return obs, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ Dense reward."""
        centroid, wz = achieved_goal, self.ecm.wz
        d = goal_distance(centroid, desired_goal) / 2
        reward = 1 - (d + np.linalg.norm(wz) * 0.1)  # maximum reward is 1, important for baseline DDPG
        return reward

    def _env_setup(self):
        super(ActiveTrack, self)._env_setup()
        opts.device='cuda:0'
        self.v_model=vismodel(opts)
        ckpt=torch.load(opts.ckpt_dir, map_location=opts.device)
        self.v_model.load_state_dict(ckpt['state_dict'])
        self.v_model.to(opts.device)
        self.v_model.eval()

        self.use_camera = True

        # robot
        self.ecm.reset_joint(self.QPOS_ECM)
        pos_x = random.uniform(0.18, 0.24)
        pos_y = random.uniform(0.21, 0.24)
        pos_z = random.uniform(0.5, 0.6)
        left_right = random.choice([-1, 1])

        self.POSE_PSM1 = ((pos_x, left_right*pos_y, pos_z), (0, 0, -(90+ left_right*20 ) / 180 * np.pi)) #(x:0.18-0.25, y:0.21-0.24, z:0.5)
        self.QPOS_PSM1 = (0, 0, 0.10, 0, 0, 0)
        self.PSM_WORLSPACE_LIMITS = ((0.18+0.45,0.18+0.55), (0.24-0.29,0.24-0.19), (0.5-0.1774,0.5-0.1074))
        self.PSM_WORLSPACE_LIMITS = np.asarray(self.PSM_WORLSPACE_LIMITS) \
                           + np.array([0., 0., 0.0102]).reshape((3, 1))
        # trajectory
        traj = Trajectory(self.PSM_WORLSPACE_LIMITS, seed=None)
        self.traj = traj
        self.traj.set_step(self._step)
        self.psm1 = Psm1(self.POSE_PSM1[0], p.getQuaternionFromEuler(self.POSE_PSM1[1]),
                         scaling=self.SCALING)
        self.psm_id = self.psm1.body
        if left_right == 1:
            self.psm1.move_joint([0.4516922970194888, -0.11590085534517788, 0.1920614431341014, -0.275713630305575, -0.025332969748983816, -0.44957632598600145])
        else:
            self.psm1.move_joint([0.4516922970194888, -0.11590085534517788, 0.1920614431341014, -0.275713630305575, -0.025332969748983816, -0.44957632598600145])
        # target cube
        init_psm_Pose  = self.psm1.get_current_position(frame='world')
        # if left_right == 1:
        #     self.psm1.move_joint([0.4516922970194888, -0.11590085534517788, 0.1920614431341014, -0.275713630305575, -0.025332969748983816, -0.44957632598600145])
        # else:
            # self.psm1.move_joint([0.4516922970194888, -0.11590085534517788, 0.1920614431341014, -0.275713630305575, -0.025332969748983816, -0.44957632598600145])
            # target cube
        reset_psm_joint = False
            # print(init_psm_Pose[:3, 3])
        while not reset_psm_joint:
            print('resetting psm joint by randomly sampling')
            x = random.uniform(0.18+0.45,0.18+0.55)
            y = random.uniform(0.24-0.29,0.24-0.19)
            z = random.uniform(0.5-0.1774,0.5-0.1074)
            pos = (x, y, z)

            #pos = (workspace_limits[0][0],
            #       workspace_limits[1][1],
            #       (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
            orn = (0.5, 0.5, -0.5, -0.5)
            joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
            result = self.psm1.reset_joint(joint_positions)
            if result is not None:
                if result is not False:
                    reset_psm_joint = True
        # print(init_psm_Pose[:3, 3])
        # exit()
        b = Boundary(self.PSM_WORLSPACE_LIMITS)
        x, y = self.traj.step()
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'cube/cube.urdf'),
                            (init_psm_Pose[0, 3], init_psm_Pose[1, 3], init_psm_Pose[2, 3]),
                            p.getQuaternionFromEuler(np.random.uniform(np.deg2rad([0, 0, -90]),
                                                                       np.deg2rad([0, 0, 90]))),
                            globalScaling=0.001 * self.SCALING)
        # print('psm_eef:', self.psm1.get_joint_number())
        color = RGB_COLOR_255[0]
        p.changeVisualShape(obj_id, -1,
                            rgbaColor=(color[0] / 255, color[1] / 255, color[2] / 255, 0),
                            specularColor=(0.1, 0.1, 0.1))
        self.obj_ids['fixed'].append(obj_id)  # 0 (target)
        self.obj_id = obj_id
        b.add(obj_id, sample=False, min_distance=0.12)
        # self._cid = p.createConstraint(obj_id, -1, -1, -1,
        #                                p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [x, y, 0.05 * self.SCALING])
        self._cid = p.createConstraint(
            parentBodyUniqueId=self.psm1.body,
            parentLinkIndex=5,
            childBodyUniqueId=self.obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0]
        )

    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state()
        # robot_state = self.ecm.get_current_joint_position()
        robot_state = np.array(robot_state)
        # assert(robot_state.shape == (4,))
        render_obs,seg, depth=self.ecm.render_image()

        render_obs=cv2.resize(render_obs,(320,240))
        # plt.imshow(seg)
        # print(seg.shape)
        # plt.show()
        # self.counter+=1
        seg = cv2.resize(seg, (320,240), interpolation =cv2.INTER_NEAREST)
        # plt.imsave('/home/kejianshi/Desktop/Surgical_Robot/science_robotics/ar_surrol/surrol/data/test_data/seg_{}.png'.format(self.counter), seg)
        # seg=np.array((seg==6 )| (seg==1)).astype(int)
        depth = cv2.resize(depth, (320,240), interpolation =cv2.INTER_NEAREST)

        seg=torch.from_numpy(seg).to("cuda:0").float()

        # depth=torch.from_numpy(depth).to("cuda:0").float()

        # if self.success_info:
        #     v_output = self.v_output.cpu().numpy()
        # else:
        with torch.no_grad():
            self.v_output=self.v_model.get_obs(seg.unsqueeze(0))[0]#.cpu().data().numpy()
        #print(v_output.shape)
        v_output=self.v_output.cpu().numpy()
        # print('v_output', v_output)
        # exit()
        achieved_goal = np.array([
            v_output[0], v_output[1], 0.0
        ])
        # print('wz',self.ecm.wz)

        observation = np.concatenate([
            robot_state, np.array([0.0]).astype(np.float).ravel(),
            v_output.ravel(), np.array([0.0]).astype(np.float)  # achieved_goal.copy(),
        ])
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        # print('homo_delta: ', self.ecm.homo_delta)
        # print('ecm wz: ', self.ecm.wz)
        
        return obs
        
    # def _get_obs(self) -> dict:
    #     robot_state = self._get_robot_state()
        
    #     render_obs,seg, depth=self.ecm.render_image()
    #     #cv2.imwrite('/research/d1/rshr/arlin/data/debug/depth_noise_debug/img.png',cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB))
    #     #plt.imsave('/research/d1/rshr/arlin/data/debug/depth_noise_debug/img2.png',render_obs)
    #     #print('depth max: ',np.max(depth))
    #     #exit()
    #     render_obs=cv2.resize(render_obs,(320,240))
        
    #     self.counter+=1
    #     #print(render_obs[0][0])
    #     #exit()
    #     #seg=np.array(seg==6).astype(int)
        
    #     # seg=np.array((seg==6 )| (seg==1)).astype(int)
    #     #seg=np.array(seg==1).astype(int)
    #     #seg=np.resize(seg,(320,240))
        
    #     #plt.imsave('/research/d1/rshr/arlin/data/debug/depth_noise_debug/depth.png',depth)
    #     #exit()
    #     seg = cv2.resize(seg, (320,240), interpolation =cv2.INTER_NEAREST)
    #     #plt.imsave('/research/d1/rshr/arlin/data/debug/seg_debug/noise_{}/seg.png'.format(self.curr_intensity),seg)
    #     #exit()
    #     depth = cv2.resize(depth, (320,240), interpolation =cv2.INTER_NEAREST)
    #     #print(np.max(depth))
    #     #depth = cv2.resize(depth, (320,240),interpolation=cv2.INTER_LANCZOS4)

        
    #     #image=cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB) / 255.0
    #     #plt.imsave('/home/student/code/regress_data7/seg/seg_{}.png'.format(self.counter),seg)
    #     #image = self.transform({'image': image})['image']
    #     #image=torch.from_numpy(image).to("cuda:0").float()

    #      # test depth noise
        
    #     #if np.random.randn()<0.5:
    #     #    instensity=np.random.randint(3,15)/100
    #     #instensity=0.1
    #     #    depth = add_gaussian_noise(depth, instensity)
    #     '''
    #     if self.counter==10:
    #         cv2.imwrite('/research/d1/rshr/arlin/data/debug/depth_noise_debug/gaussian/img.png',cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB))
    #         plt.imsave('/research/d1/rshr/arlin/data/debug/depth_noise_debug/gaussian/depth.png',depth)
    #         for i in [0.01,0.05,0.1,0.15,0.2]:
    #             noisy_depth_map = add_random_noise(depth, i)
    #             plt.imsave('/research/d1/rshr/arlin/data/debug/depth_noise_debug/gaussian/noise_{}.png'.format(i),noisy_depth_map)

    #         exit()
    #     '''

    #     #noisy_segmentation_map = add_noise_to_segmentation(seg, self.seg_noise_intensity)
    #     #noisy_depth_map = add_gaussian_noise(depth, self.curr_intensity)
    #     #if self.counter==10:
    #     #    plt.imsave('/research/d1/rshr/arlin/data/debug/seg_debug/noise_{}/img.png'.format(self.curr_intensity),render_obs)
    #     #    plt.imsave('/research/d1/rshr/arlin/data/debug/seg_debug/noise_{}/seg.png'.format(self.curr_intensity),seg)
    #     #    plt.imsave('/research/d1/rshr/arlin/data/debug/seg_debug/noise_{}/noise_seg.png'.format(self.curr_intensity),noisy_segmentation_map)

    #     seg=torch.from_numpy(seg).to("cuda:0").float()
    #     depth=torch.from_numpy(depth).to("cuda:0").float()

        
    #     with torch.no_grad():
    #         v_output=self.v_model.get_obs(seg.unsqueeze(0), depth.unsqueeze(0))[0]#.cpu().data().numpy()
    #     #print(v_output.shape)
    #     v_output=v_output.cpu().numpy()

    #     achieved_goal = np.array([
    #         v_output[0], v_output[1], self.ecm.wz
    #     ])

    #     observation = np.concatenate([
    #         robot_state, np.array([0.0]).astype(np.float).ravel(),
    #         v_output.ravel(), np.array(self.ecm.wz).ravel()  # achieved_goal.copy(),
    #     ])
    #     obs = {
    #         'observation': observation.copy(),
    #         'achieved_goal': achieved_goal.copy(),
    #         'desired_goal': np.array([0., 0., 0.]).copy()
    #     }
    #     return obs
    

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array([0., 0., 0.])
        return goal.copy()

    def _step_callback(self):
        """ Move the target along the trajectory
        """
        for _ in range(10):
            x, y = self.traj.step()
            self._step = self.traj.get_step()
            current_PSM_position = self.psm1.get_current_position(frame='world')
            new_PSM_position = current_PSM_position.copy()

            new_PSM_position[0, 3] =x
            new_PSM_position[1, 3] =y
            new_PSM_position = self.psm1.pose_world2rcm(new_PSM_position)
            self.psm1.move(new_PSM_position)
            # pivot = [x, y, 0.05 * self.SCALING]
            # p.changeConstraint(self._cid, pivot, maxForce=50)
            p.stepSimulation()

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        centroid = obs['observation'][-3: -1]
        cam_u = centroid[0] * RENDER_WIDTH
        cam_v = centroid[1] * RENDER_HEIGHT
        self.ecm.homo_delta = np.array([cam_u, cam_v]).reshape((2, 1))
        if np.linalg.norm(self.ecm.homo_delta) < 10 and np.linalg.norm(self.ecm.wz) < 0.1:
            # e difference is small enough
            action = np.zeros(4)
        else:
            # print("Pixel error: {:.4f}".format(np.linalg.norm(self.ecm.homo_delta)))
            # controller
            fov = np.deg2rad(FoV)
            fx = (RENDER_WIDTH / 2) / np.tan(fov / 2)
            fy = (RENDER_HEIGHT / 2) / np.tan(fov / 2)  # TODO: not sure
            cz = 1.0
            Lmatrix = np.array([[-fx / cz, 0., cam_u / cz],
                                [0., -fy / cz, cam_v / cz]])
            action = 0.5 * np.dot(np.linalg.pinv(Lmatrix), self.ecm.homo_delta).flatten() / 0.01
            if np.abs(action).max() > 1:
                action /= np.abs(action).max()
            action *= 0.8
            action *= 0.01 * self.SCALING  # velocity (HeadPitch, HeadYaw), limit maximum change in velocity
            action = 0.05 * self.ecm.cVc_to_dq(action)
        return action

    def render_background(self):
        render_obs,seg, depth=self.ecm.render_image(width=self.bg_width,height=self.bg_height)
        ### for background rendering:
        mask=seg==self.psm_id
        mask = np.logical_not(mask)
        background_rgb = render_obs.copy()
        background_rgb[mask,:] = self.background[mask,:]
        return background_rgb
    

if __name__ == "__main__":
    env = ActiveTrack(render_mode='human',bg_img_path= '/home/kejianshi/Desktop/Surgical_Robot/science_robotics/to_KJ/to_KJ/background/0.png')  # create one process and corresponding env

    while 1:
        p.stepSimulation()
        img=env.render_background()
        cv2.imshow('surrol',img[...,[2,1,0]])# RGB-->BGR: fetures of openCV
        cv2.waitKey(1)
