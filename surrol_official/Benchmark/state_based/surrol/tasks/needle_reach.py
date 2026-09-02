import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env_full import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera,    
    wrap_angle
)
from surrol.const import ASSET_DIR_PATH

from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm
# import dvrk
# import rospy
# import math
# # move in cartesian space
# import PyKDL
# from dvrk import mtm
class NeedleReachFullDof(PsmEnv):
    """
    Refer to Gym FetchReach
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/reach.py
    """
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE=3
    def __init__(self, render_mode=None, cid = -1):
        super(NeedleReachFullDof, self).__init__(render_mode, cid)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
    def _env_setup(self):
        super(NeedleReachFullDof, self)._env_setup()
        self.has_object = False

        # camera
        if self._render_mode == 'human':
            # reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
            #              target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
            reset_camera(yaw=89.60, pitch=-56, dist=5.98,
                         target=(-0.13, 0.03,-0.94))
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(10, 10, 10))
        self.obj_ids['fixed'].append(obj_id)  # 1

        # needle
        yaw = (np.random.rand() - 0.5) * np.pi
        # obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
        #                     (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,
        #                      workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
        #                      workspace_limits[2][0] + 0.01),
        #                     p.getQuaternionFromEuler((0, 0, yaw)),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        # p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        # self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['fixed'][0], -1

        # self.m = mtm('MTMR')

        # # turn gravity compensation on/off
        # self.m.use_gravity_compensation(True)
        # self.m.body.servo_cf(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


        # psm_pose = self.psm1.get_current_position()
        # print(f"PSM pose RCM: {psm_pose}")
        # psm_pose_ori = psm_pose.copy()

        # # psm_measured_cp = np.matmul(np.linalg.inv(ecm_pose), psm_pose_ori)#over ecm's rcm
        # psm_measured_cp=psm_pose_ori #not over ecm

        # goal = PyKDL.Frame()
        # goal.p = self.m.setpoint_cp().p
        # # # goal.p[0] += 0.05
        # goal.M= self.m.setpoint_cp().M

        # # psm_measured_cp = np.matmul(mapping_mat,psm_measured_cp)
        # for i in range(3):
        #     print(f"previous goal:{goal.M}")
        #     for j in range(3):
        #         goal.M[i,j]=psm_measured_cp[i][j]
        #         # if j==1:
        #         #     goal.M[i,j]*=-1
        #         # goal.M[i,j]=psm_pose[i][j]
        #     print(f"modified goal:{goal.M}")
        # print(goal.M.GetEulerZYX())
        # # print(rotationMatrixToEulerAngles(psm_measured_cp[:3,:3]))
        # self.m.move_cp(goal).wait() #align

    # def _set_action(self, action: np.ndarray):
    #     # action[3] = 0  # no yaw change
    #     print('action here')
    #     super(NeedleReachFullDof, self)._set_action(action)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        goal = np.array([pos[0], pos[1], pos[2] + 0.005 * self.SCALING])
        return goal.copy()

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        delta_pos = (obs['desired_goal'] - obs['achieved_goal']) / 0.01
        if np.linalg.norm(delta_pos) < 1.5:
            delta_pos.fill(0)
        if np.abs(delta_pos).max() > 1:
            delta_pos /= np.abs(delta_pos).max()
        delta_pos *= 0.3

        action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], 0., 0.])
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

if __name__ == "__main__":
    env = NeedleReachFullDof(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
