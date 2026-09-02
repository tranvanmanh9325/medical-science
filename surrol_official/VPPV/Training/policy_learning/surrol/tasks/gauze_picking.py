import os
import time
import numpy as np
import torch

import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera,    
    wrap_angle
)
from surrol.const import ASSET_DIR_PATH

from surrol.tasks.psm_env import PsmEnv

from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm
import pickle 
import cv2
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import sys

sys.path.append('../../../state_regress')
from vmodel import vismodel
from config import opts

class GauzePick(PsmEnv):
    """
    Refer to Gym FetchPickAndPlace
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch/pick_and_place.py
    """
    POSE_TRAY = ((0.55, 0, 0.6781), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.681, 0.745))
    SCALING = 1.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE=3
    ACTION_SIZE=4
    counter=0
    img_list={}
    
    def _env_setup(self):

        opts.device='cuda:0'
        self.v_model=vismodel(opts)
        ckpt=torch.load(opts.ckpt_dir, map_location=opts.device)
        self.v_model.load_state_dict(ckpt['state_dict'])
        self.v_model.to(opts.device)
        self.v_model.eval()
        print("load v_output")
        

        ecm_view_matrix =[2.7644696427853166e-12, -0.8253368139266968, 0.5646408796310425, 0.0, 1.0, 2.76391192918779e-12, -8.559629784479772e-13, 0.0, -8.541598418149166e-13, 0.5646408796310425, 0.8253368139266968, 0.0, -1.582376590869572e-11, 0.4536721706390381, -5.886332988739014,1.0]
        #ecm_view_matrix[14]=-5.25 #-5.0#-5.25
        ecm_view_matrix[14]=-0.97 #-4.7 #-5.25#-5.0#-5.25
        ecm_view_matrix[13]=0.07 #0.3-0.5
        # ecm_view_matrix[12]=0.008 #0.3-0.5
        shape_view_matrix=np.array(ecm_view_matrix).reshape(4,4)
        Tc = np.array([[1,  0,  0,  0],
                        [0, -1,  0,  0],
                        [0,  0, -1,  0],
                        [0,  0,  0,  1]])
        self._view_matrix=Tc@(shape_view_matrix.T)
        self.ecm_view_matrix=np.array(ecm_view_matrix).reshape(4,4)

        super(GauzePick, self)._env_setup(goal_plot=False)
        
        self.has_object = True
        self._waypoint_goal = True
        # self._contact_approx = True  # mimic the dVRL setting, prove nothing?
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                    scaling=self.SCALING, view_matrix=ecm_view_matrix)
        self.ecm.reset_joint(self.QPOS_ECM)
        #self._view_matrix=np.array(self.ecm.view_matrix).reshape(4,4)

        # robot
        workspace_limits = self.workspace_limits1
        temp_x=np.random.randint(50,61)/100
        temp_y=np.random.randint(-5,6)/100
        temp_z=np.random.rand()*(workspace_limits[2][1]-workspace_limits[2][0])+workspace_limits[2][0]
        

        #workspace_limits[1][1] if np.random.rand()>0.5 else workspace_limits[1][0]
        #print('workspace_limits: ',workspace_limits)
        
        pos = (temp_x,
               temp_y,
               temp_z)

        #pos = (workspace_limits[0][0],
        #       workspace_limits[1][1],
        #       (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1
        p.changeVisualShape(obj_id, -1, rgbaColor=(225 / 255, 225 / 255, 225 / 255, 1))

        # gauze
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'gauze/gauze.urdf'),
                            (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.02,  # TODO: scaling
                             workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.02,
                             workspace_limits[2][0] + 0.01),
                            (0, 0, 0, 1),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        #print('gauze id: ',obj_id)
        p.changeVisualShape(obj_id, -1, specularColor=(0, 0, 0))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], -1
    
    def get_camera_matrix(self):
        return self._view_matrix

    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state(idx=0)
        # print('jaw: ',robot_state[-1])
        
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
        render_obs=cv2.resize(render_obs,(320,240))
        
        self.counter+=1
        #print(render_obs[0][0])
        #exit()
        #seg=np.array(seg==6).astype(int)
        
        seg=np.array((seg==6)).astype(int)
        #seg=np.array(seg==1).astype(int)
        #seg=np.resize(seg,(320,240))
        
        seg = cv2.resize(seg, (320,240), interpolation =cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (320,240), interpolation =cv2.INTER_NEAREST)

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

        obs['observation']=o_new
        
        return obs   

    def _set_action(self, action: np.ndarray):
        """
        delta_position (3), delta_theta (1) and open/close the gripper (1)
        in the world frame
        """
        assert len(action) == self.ACTION_SIZE, "The action should have the save dim with the ACTION_SIZE"
        # time0 = time.time()
        action = action.copy()  # ensure that we don't change the action outside of this scope
        action[:3] *= 0.01 * self.SCALING  # position, limit maximum change in position
        
        # ECM action

        curr_rcm_pose=self.psm1.get_current_position()
        #print('pre set action: ',curr_rcm_pose)

        rcm_action=curr_rcm_pose.copy()

        pose_world = self.psm1.pose_rcm2world(curr_rcm_pose, 'tuple')
        euler=np.array(p.getEulerFromQuaternion(pose_world[1]))
        
        pos=np.array(pose_world[0])
        # world2ecm
        euler=self._world2cam_rot(euler)
        pos=self._world2cam_pos(pos)

        pos=pos+action[:3]
        world_pos=self._camera2world_pos(pos)
        curr_rcm_pose[:3,3]=world_pos
        
        world_action_pos=self.psm1.pose_world2rcm(curr_rcm_pose)
        rcm_action[:3,3]=world_action_pos[:3,3]
        #print('aft set action: ',rcm_action)
        
        self.psm1.move(rcm_action)
        
        # jaw
        #print(action[6])
        #print('jaw: ',action[3])
        if self.block_gripper:
            action[3] = -1
        if action[3] < 0 or action[3]==0:
            self.psm1.close_jaw()
            
            self._activate(0)
        else:
            self.psm1.move_jaw(np.deg2rad(40))  # open jaw angle; can tune
            self._release(0)
    
    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        
        # goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
        #                  workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
        #                  workspace_limits[2][1] - 0.07 * self.SCALING])
        '''
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        goal=np.array([pos_obj[0], pos_obj[1],pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING])
        '''
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        goal=np.array([pos_obj[0], pos_obj[1],pos_obj[2]+0.005 ])

        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None]  # two waypoints
        #self._waypoints = [None, None]
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, 0., 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., 0.5])  # approach
        # self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
        #                                pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., -0.5])  # grasp
        # self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
        #                                pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, 0., -0.5])  # grasp
        # self._waypoints[4] = np.array([self.goal[0], self.goal[1],
        #                                self.goal[2] + 0.0102 * self.SCALING, 0., -0.5])  # lift up

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped object to make it stable
        pose = get_link_pose(self.obj_id, self.obj_link1)
        return pose[0][2] > self._waypoint_z_init + 0.0025 * self.SCALING
        # return True  # mimic the dVRL setting

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
    
    
    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
        #action = np.zeros(5)
        obss=obs['observation'][:6]
        action = np.zeros(4)
        action[3] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            #delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            ecm_waypoint_pos=self._world2cam_pos(waypoint[:3])
            
            ecm_obs_pos=self._world2cam_pos(obss[:3].copy())
            delta_pos=(ecm_waypoint_pos-ecm_obs_pos)/0.01/self.SCALING
            #delta_pos=(waypoint[:3]-obss[:3])/0.01/self.SCALING
            delta_yaw=(waypoint[3]-obss[-1])
            #print("delta_yaw: ",delta_yaw)
            #while abs(delta_yaw+np.pi/2)<abs(delta_yaw):
            #    delta_yaw=delta_yaw+np.pi/2
            #print("delta_yaw: ",delta_yaw)
            delta_yaw=delta_yaw.clip(-0.4,0.4)
            
            delta_rot=self.calculate_ecm_rotation(self._view_matrix, delta_yaw)
           
            #while abs(delta_rot[2]-np.pi/2)<abs(delta_rot[2]):
            #    delta_rot[2]=delta_rot[2]-np.pi/2
            
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.6
            delta_pos *= scale_factor
            #action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_rot[0], delta_rot[1],delta_rot[2],waypoint[4]])
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2],waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4:
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

if __name__ == "__main__":
    env = GauzePick(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
