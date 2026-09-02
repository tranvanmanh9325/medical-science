import os
import time

import pybullet as p
from surrol.tasks.ecm_env import EcmEnv, goal_distance
import matplotlib.pyplot as plt
from surrol.utils.pybullet_utils import (
    get_body_pose,
)
import copy
import random
import cv2
import pickle
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
from PIL import Image

class StaticTrack(EcmEnv):
    ACTION_MODE = 'dmove'
    DISTANCE_THRESHOLD = 0.05
    QPOS_ECM = (0, 0, 0.02, 0)
    WORKSPACE_LIMITS = ((-0.3, 0.6), (-0.4, 0.4), (0.05, 0.05))
    CUBE_NUMBER = 18
    
    # QPOS_ECM = (0, 0.6, 0.04, 0)
    # POSE_TABLE = ((0.5, 0, 0.001), (0, 0, 0))

    ACTION_ECM_SIZE=3
    def __init__(self, render_mode=None, cid = -1, bg_img_path = None):
        super(StaticTrack, self).__init__(render_mode, cid)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0.27, -0.2, 0.55),
            distance=2.3,
            yaw=150,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        self.debug=False
        # self.background = np.array(Image.open(bg_img_path))[...,:3]
        # self.bg_height, self.bg_width, _ = self.background.shape


    def _env_setup(self):
        super(StaticTrack, self)._env_setup()
        self.counter = 0
        opts.device='cuda:0'
        self.v_model=vismodel(opts)
        ckpt=torch.load(opts.ckpt_dir, map_location=opts.device)
        self.v_model.load_state_dict(ckpt['state_dict'])
        self.v_model.to(opts.device)
        self.v_model.eval()
        self.misorientation_threshold = 0.01
        self.use_camera = True

        self.success_info = False
        # robot
        self.ecm.reset_joint(self.QPOS_ECM)
        pos_x = random.uniform(0.18, 0.24)
        pos_y = random.uniform(0.21, 0.24)
        pos_z = random.uniform(0.5, 0.6)
        left_right = random.choice([-1, 1])

        self.POSE_PSM1 = ((pos_x, left_right*pos_y, pos_z), (0, 0, -(90+ left_right*20 ) / 180 * np.pi)) #(x:0.18-0.25, y:0.21-0.24, z:0.5)
        self.QPOS_PSM1 = (0, 0, 0.10, 0, 0, 0)
        self.PSM_WORLSPACE_LIMITS = ((0.18+0.45,0.18+0.55), (0.24-0.29,0.24-0.19), (0.5-0.1774,0.5-0.1074))
        # self.PSM_WORLSPACE_LIMITS = np.asarray(self.PSM_WORLSPACE_LIMITS) \
        #                    + np.array([0., 0., 0.0102]).reshape((3, 1))
        
        self.psm1 = Psm1(self.POSE_PSM1[0], p.getQuaternionFromEuler(self.POSE_PSM1[1]),
                         scaling=self.SCALING)
        self.psm_id = self.psm1.body
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

        # exit()
        b = Boundary(self.PSM_WORLSPACE_LIMITS)
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
        # self.ecm.reset_joint([-0.057295, -0.45844, 0.056096, 0.25349])


    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state()[:3]
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

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = goal_distance(achieved_goal[..., :2], desired_goal[..., :2])
        # misori = np.abs(achieved_goal[..., 2] - achieved_goal[..., 2])
        # print(f"ECM static track: {d} {self.distance_threshold} {d < self.distance_threshold}")
        # print(np.logical_and(
        #     d < self.distance_threshold,
        #     misori < self.misorientation_threshold
        # ).astype(np.float32))
        return (d < self.distance_threshold).astype(np.float32)
        #     misori < self.misorientation_threshold
        # ).astype(np.float32)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array([0., 0., 0.])
        return goal.copy()

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        cam_u = obs['achieved_goal'][0] * RENDER_WIDTH
        cam_v = obs['achieved_goal'][1] * RENDER_HEIGHT
        self.ecm.homo_delta = np.array([cam_u, cam_v]).reshape((2, 1))
        if np.linalg.norm(self.ecm.homo_delta) < 8 and np.linalg.norm(self.ecm.wz) < 0.1:
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

    def _set_action_ecm(self, action):
        # print(action.shap)
        # action *= 0.01 * self.SCALING
        pose_rcm = self.ecm.get_current_position()
        pose_rcm[:3, 3] += action
        pos, _ = self.ecm.pose_rcm2world(pose_rcm, 'tuple')
        joint_positions = self.ecm.inverse_kinematics((pos, None), self.ecm.EEF_LINK_INDEX)  # do not consider orn
        self.ecm.move_joint(joint_positions[:self.ecm.DoF])
    def _reset_ecm_pos(self):
        self.ecm.reset_joint(self.QPOS_ECM)
    
    def render_background(self):
        render_obs,seg, depth=self.ecm.render_image(width=self.bg_width,height=self.bg_height)
        ### for background rendering:
        mask=seg==self.psm_id
        mask = np.logical_not(mask)
        background_rgb = render_obs.copy()
        background_rgb[mask,:] = self.background[mask,:]
        return background_rgb

if __name__ == "__main__":
    env = StaticTrack(render_mode='human',bg_img_path= '/home/kejianshi/Desktop/Surgical_Robot/science_robotics/to_KJ/to_KJ/background/0.png')  # create one process and corresponding env

    while 1:
        p.stepSimulation()
        img=env.render_background()
        cv2.imshow('surrol',img[...,[2,1,0]])# RGB-->BGR: fetures of openCV
        cv2.waitKey(1)
