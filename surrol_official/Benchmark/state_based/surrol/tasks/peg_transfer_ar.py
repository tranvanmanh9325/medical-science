import os
import time
from turtle import pos

import numpy as np
import pybullet as p

from surrol.const import ASSET_DIR_PATH
from surrol.tasks.psm_env import PsmEnv, goal_distance
from surrol.utils.pybullet_utils import get_link_pose, wrap_angle
from surrol.robots.ecm import Ecm
import cv2

def seg_with_red(grid_RGB):

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
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_SIZE=7

    # TODO: grasp is sometimes not stable; check how to fix it

    def _env_setup(self):
        super(PegTransfer, self)._env_setup()
        self.has_object = True
        
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                    scaling=self.SCALING)
        self.ecm.reset_joint(self.QPOS_ECM)
        self._view_matrix=np.array(self.ecm.view_matrix).reshape(4,4)

        # robot
        workspace_limits = self.workspace_limits1
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               workspace_limits[2][1])
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
        # for i in range(6, 6 + num_blocks):
        for i in self._pegs[6: 6 + num_blocks]:
            pos, orn = get_link_pose(self.obj_ids['fixed'][1], i)
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


    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        # TODO: may need to tune parameters
        return np.logical_and(
            goal_distance(achieved_goal[..., :2], desired_goal[..., :2]) < 5e-3 * self.SCALING,
            np.abs(achieved_goal[..., -1] - desired_goal[..., -1]) < 4e-3 * self.SCALING
        ).astype(np.float32)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array(get_link_pose(self.obj_ids['fixed'][1], self._pegs[0])[0])
        return goal.copy()

    def _sample_goal_callback(self):
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

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, 0.5])  # above object
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (0.003 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + 0.045 * self.SCALING, yaw, -0.5])  # lift up

       
        pos_peg = get_link_pose(self.obj_ids['fixed'][1],
                                self._pegs[self.obj_id - np.min(self._blocks) + 6])[0]  # 6 pegs
        pos_place = [self.goal[0] + pos_obj[0] - pos_peg[0],
                     self.goal[1] + pos_obj[1] - pos_peg[1], self._waypoints[0][2]]  # consider offset
        self._waypoints[4] = np.array([pos_place[0], pos_place[1], pos_place[2], yaw, -0.5])  # above goal
        self._waypoints[5] = np.array([pos_place[0], pos_place[1], self._waypoints[2][2], yaw, -0.5])  # release
        
        self._waypoints[6] = np.array([pos_place[0], pos_place[1], self._waypoints[2][2], yaw, 0.5])  # release
        self.waypoints = self._waypoints.copy()

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        pose = get_link_pose(self.obj_id, -1)
        return pose[0][2] > self.goal[2] + 0.01 * self.SCALING
    
    def _get_obs(self, use_v=False) -> dict:
        
        obs=super()._get_obs()
        render_obs,_, depth=self.ecm.render_image()
        
        depth=cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX) #0,1

        seg=seg_with_red(render_obs)

        img_size=seg.shape[0]
        seg_d=np.concatenate([seg.reshape(1, img_size, img_size), \
                              depth.reshape(1, img_size, img_size)],axis=0)
        obs['seg_d']=seg_d.copy()
        return obs
    
    def calculate_ecm_rotation(self, world2ecm_matrix, yaw_change):
        
        world_rotation = R.from_euler('z', yaw_change)

        ecm_rotation_matrix = world2ecm_matrix[:3, :3]

        ecm_rotation = np.dot(ecm_rotation_matrix, world_rotation.as_matrix())

        ecm_euler = R.from_matrix(ecm_rotation).as_euler('xyz')

        return ecm_euler

    # get action in ECM axis
    def get_oracle_action_ecm(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # six waypoints executed in sequential order
        obss=obs['observation'][:6]
        action = np.zeros(7)
        
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
           
            ecm_waypoint_pos=self._world2cam_pos(waypoint[:3])
            ecm_obs_pos=self._world2cam_pos(obss[:3].copy())
            delta_pos=(ecm_waypoint_pos-ecm_obs_pos)/0.01/self.SCALING
            #delta_pos=(waypoint[:3]-obss[:3])/0.01/self.SCALING
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
                self._waypoints[i] = None
                print('solve ',i)
                
            break
        
        return action

    # get action in RCM axis
    def get_oracle_action(self, obs) -> np.ndarray: 
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
            
            rcm_waypoint=self.psm1.pose_world2rcm((new_waypoint[:3],new_waypoint[3:]),option = 'tuple')
            rcm_observation=self.psm1.pose_world2rcm((obss[:3].copy(),obss[3:].copy()),option = 'tuple')
            rcm_wp_euler=np.array(p.getEulerFromQuaternion(rcm_waypoint[1]))
            rcm_obs_euler=np.array(p.getEulerFromQuaternion(rcm_observation[1]))
            
            rcm_wp_pos=np.array(rcm_waypoint[0])
            rcm_obs_pos=np.array(rcm_observation[0])
            
            delta_pos=(rcm_wp_pos-rcm_obs_pos)/0.01/self.SCALING
            delta_yaw=(rcm_wp_euler[-1]-rcm_obs_euler[-1])
            
            # TODO: current assume delta_yaw is positive
            while abs(delta_yaw+np.pi/2)<abs(delta_yaw):
                delta_yaw=delta_yaw+np.pi/2
           
            delta_yaw=delta_yaw.clip(-0.4,0.4)
           
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.7
            delta_pos *= scale_factor
            
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
           
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
        is_grasp = yaw_angle < 0.2  # TODO: fine-tune
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
