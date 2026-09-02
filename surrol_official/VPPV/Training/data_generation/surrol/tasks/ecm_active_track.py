import os
import time

import pybullet as p
from surrol.tasks.ecm_env import EcmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_body_pose,
)
import random
import cv2
import pickle
from surrol.utils.robotics import (
    get_euler_from_matrix,
    get_matrix_from_euler
)
import matplotlib.pyplot as plt
from surrol.utils.utils import RGB_COLOR_255, Boundary, Trajectory, get_centroid
from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
import numpy as np
from surrol.robots.psm import Psm1, Psm2

class ActiveTrack(EcmEnv):
    """
    Active track is not a GoalEnv since the environment changes internally.
    The reward is shaped.
    """
    ACTION_MODE = 'cVc'
    # RCM_ACTION_MODE = 'yaw'
    QPOS_ECM = (0, 0, 0.04, 0)
    WORKSPACE_LIMITS = ((-0.3, 0.6), (-0.4, 0.4), (0.05, 0.05))

    CUBE_NUMBER = 18

    def __init__(self, render_mode=None):
        # to control the step
        self._step = 0
        self.counter=0
        self.img_list={}
        super(ActiveTrack, self).__init__(render_mode)

    def step(self, action: np.ndarray):
        obs, reward, done, info = super().step(action)
        centroid = obs[-3: -1]
        if not (-1.2 < centroid[0] < 1.1 and -1.1 < centroid[1] < 1.1):
            # early stop if out of view
            done = True
        info['achieved_goal'] = centroid
        return obs, reward, done, info

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ Dense reward."""
        centroid, wz = achieved_goal[-3: -1], achieved_goal[-1]
        d = goal_distance(centroid, desired_goal) / 2
        reward = 1 - (d + np.linalg.norm(wz) * 0.1)  # maximum reward is 1, important for baseline DDPG
        return reward

    def _env_setup(self):
        super(ActiveTrack, self)._env_setup()
        self.use_camera = True

        # robot
        self.ecm.reset_joint(self.QPOS_ECM)
        pos_x = random.choice([0.3, 0.34, 0.37])#random.uniform(0.18, 0.24)
        pos_y = 0.3#random.uniform(0.21, 0.24)
        pos_z = 0.5
        left_right = random.random()
        if left_right > 0.5:
            left_right = 1
        else:
            left_right = -1

        self.POSE_PSM1 = ((pos_x, left_right*pos_y, pos_z), (0, 0, -(90+ left_right*20 ) / 180 * np.pi)) #(x:0.18-0.25, y:0.21-0.24, z:0.5)
        # self.QPOS_PSM1 = (0, 0, 0.10, 0, 0, 0)
        self.PSM_WORLSPACE_LIMITS = np.asarray(((0.65, 0.75), (-0.05, 0.05), (0.35, 0.4)))
        # self.PSM_WORLSPACE_LIMITS = np.asarray(self.PSM_WORLSPACE_LIMITS) \
        #                    + np.array([0., 0., 0.0102]).reshape((3, 1))
        # trajectory
        traj = Trajectory(self.PSM_WORLSPACE_LIMITS, seed=None)
        self.traj = traj
        self.traj.set_step(self._step)
        self.psm1 = Psm1(self.POSE_PSM1[0], p.getQuaternionFromEuler(self.POSE_PSM1[1]),
                         scaling=self.SCALING)
        # if left_right == 1:
        #     self.psm1.move_joint([0.4516922970194888, -0.11590085534517788, 0.1920614431341014, -0.275713630305575, -0.025332969748983816, -0.44957632598600145])
        # else:
        #     self.psm1.move_joint([0.4516922970194888, -0.11590085534517788, 0.1920614431341014, -0.275713630305575, -0.025332969748983816, -0.44957632598600145])
        # target cube
        reset_psm_joint = False
            # print(init_psm_Pose[:3, 3])
        while not reset_psm_joint:
            print('resetting psm joint by randomly sampling')
            x = random.uniform(0.65, 0.71)# + left_right * random.uniform(0, 0.05) #random.uniform(0.18+0.47,0.18+0.53)   0.68 is the middle
            # if left_right == 1:
            #     y = random.uniform(0.02, 0.03)
            # else:
            #     y = random.uniform(-0.03, -0.02)
            if random.random() > 0.5:
                y = random.uniform(0.02, 0.03)
            else:
                y = random.uniform(-0.03, -0.02)
            z = 0.4
            pos = (x, y, z)
            print(pos)

            #pos = (workspace_limits[0][0],
            #       workspace_limits[1][1],
            #       (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
            orn = (0,0.5,-0.5,-0.5)
            joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)

            # joint_positions[-4:] = [-0.9702065773308277,
            #     -0.15254750348145063,
            #     -0.04118536503790979,
            #     0.5236686757349086]
            # joint_positions[-2] = 0.0
            # joint_positions[-1] = 1.0
            result = self.psm1.reset_joint(joint_positions)
            if result is not None:
                if result is not False:
                    reset_psm_joint = True

        init_psm_Pose  = self.psm1.get_current_position(frame='world')
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
        

        # '''
        # Set up initial env
        # '''
        # self.psm1_eul = np.array(p.getEulerFromQuaternion(
        #     self.psm1.pose_rcm2world(self.psm1.get_current_position(), 'tuple')[1]))  # in the world frame
        
        # # robot 
        # #self.psm1_eul = np.array(p.getEulerFromQuaternion(
        # #    self.psm1.pose_rcm2world(self.psm1.get_current_position(), 'tuple')[1]))  # in the world frame
        
        # if self.RCM_ACTION_MODE == 'yaw':
        #     #self.psm1_eul = np.array([np.deg2rad(-90), 0., self.psm1_eul[2]])
        #     '''
        #     # RCM init
        #     #eul=np.array([np.deg2rad(-90), 0., 0.])
        #     print(self.psm1.wTr)
        #     print(self.psm1.tool_T_tip)
        #     init_pose=self.psm1.get_current_position()
            
        #     eul=np.array([0, 0.,np.deg2rad(-50)])
        #     rcm_eul=get_matrix_from_euler(eul)
        #     init_pose[:3,:3]=rcm_eul
            
        #     rcm_pose=self.psm1.pose_world2rcm(init_pose)
        #     rcm_eul=get_euler_from_matrix(rcm_pose[:3,:3])
        #     print('from [0, 0.,np.deg2rad(-50)] to ',rcm_eul)
        #     #exit()
        #     eul=np.array([0, 0.,np.deg2rad(-90)])
        #     rcm_eul=get_matrix_from_euler(eul)
        #     init_pose[:3,:3]=rcm_eul
            
        #     rcm_pose=self.psm1.pose_world2rcm(init_pose)
        #     rcm_eul=get_euler_from_matrix(rcm_pose[:3,:3])
        #     print('from [0, 0.,np.deg2rad(-90)] to ',rcm_eul)
            
        #     m=np.array([[ 0.93969262 ,-0.34202014 , 0.         , 1.21313615],
        #                 [ 0.34202014 , 0.93969262 , 0.         ,-2.25649898],
        #                 [ 0.         , 0.         , 1.         ,-4.25550013],
        #                 [ 0.         , 0.         , 0.         , 1.        ]])
        #     #print(m.shape)
        
        #     m=get_euler_from_matrix(m[:3,:3])
        #     print('m1: ',m)
            
        #     m=np.array([[ 0.         ,-0.93969262 ,-0.34202014 , 1.21313615],
        #                 [ 0.         ,-0.34202014 , 0.93969262 ,-2.25649898],
        #                 [-1.         , 0.         , 0.         ,-4.25550013],
        #                 [ 0.         , 0.         , 0.         , 1.        ],])
        #     m=get_euler_from_matrix(m[:3,:3])
        #     print('m2: ',m)
        #     exit()
        #     '''
        #     # RCM init
        #     eul=np.array([np.deg2rad(-90), 0., 0.])
        #     eul= get_matrix_from_euler(eul)
        #     init_pose=self.psm1.get_current_position()
        #     self.rcm_init_eul=np.array(get_euler_from_matrix(init_pose[:3, :3]))
        #     init_pose[:3,:3]=eul
        #     rcm_pose=self.psm1.pose_world2rcm_no_tip(init_pose)
        #     rcm_eul=get_euler_from_matrix(rcm_pose[:3,:3])
        #     #print('rcm eul: ',rcm_eul)
        #     #exit()
        #     self.rcm_init_eul[0]=rcm_eul[0]
        #     self.rcm_init_eul[1]=rcm_eul[1]
        #     print(self.rcm_init_eul)
        #     #exit()
            
            

            
            
        # elif self.RCM_ACTION_MODE == 'pitch':
        #     self.psm1_eul = np.array([np.deg2rad(0), self.psm1_eul[1], np.deg2rad(-90)])
        # else:
        #     raise NotImplementedError
        # self.psm2 = None
        # self._contact_constraint = None
        # self._contact_approx = False
        # other cubes
        # b.set_boundary(self.workspace_limits + np.array([[-0.2, 0], [0, 0], [0, 0]]))
        # for i in range(self.CUBE_NUMBER):
        #     obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'cube/cube.urdf'),
        #                         (0, 0, 0.05), (0, 0, 0, 1),
        #                         globalScaling=0.8 * self.SCALING)
        #     color = RGB_COLOR_255[1 + i // 2]
        #     p.changeVisualShape(obj_id, -1,
        #                         rgbaColor=(color[0] / 255, color[1] / 255, color[2] / 255, 1),
        #                         specularColor=(0.1, 0.1, 0.1))
        #     # p.changeDynamics(obj_id, -1, mass=0.01)
        #     b.add(obj_id, min_distance=0.12)

    # def _get_obs(self) -> np.ndarray:
    #     robot_state = self._get_robot_state()
    #     # may need to modify
    #     rgb_array, mask, depth = self.ecm.render_image()
    #     in_view, centroids = get_centroid(mask, self.obj_id)

    #     if not in_view:
    #         # out of view; differ when the object is on the boundary.
    #         pos, _ = get_body_pose(self.obj_id)
    #         centroids = self.ecm.get_centroid_proj(pos)
    #         # print(" -> Out of view! {}".format(np.round(centroids, 4)))

    #     observation = np.concatenate([
    #         robot_state, np.array(in_view).astype(np.float).ravel(),
    #         centroids.ravel(), np.array(self.ecm.wz).ravel()  # achieved_goal.copy(),
    #     ])
    #     return observation
    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state()
        
        # may need to modify
        render_obs,seg,depth=self.ecm.render_image()
        in_view, centroids = get_centroid(seg, self.obj_id)
        if not in_view:
            # out of view; differ when the object is on the boundary.
            pos, _ = get_body_pose(self.obj_id)
            centroids = self.ecm.get_centroid_proj(pos)
            
            # print(" -> Out of view! {}".format(np.round(centroids, 4)))
        in_view, _ = get_centroid(seg, self.psm1.body)
        print(centroids)
        observation = np.concatenate([
            robot_state, np.array(in_view).astype(np.float).ravel(),
            centroids.ravel(), np.array(self.ecm.wz).ravel()  # achieved_goal.copy(),
        ])

        achieved_goal = np.array([
            centroids[0], centroids[1], self.ecm.wz
        ])

        
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': np.array([0., 0., 0.]).copy()
        }

        
        if in_view:
            # print(centroids)    
            render_obs=cv2.resize(render_obs,(320,240))
            #seg=np.resize(seg,(320,240))
            
            #print('depth : ', np.max(depth))
            
            seg = cv2.resize(seg, (320,240), interpolation =cv2.INTER_NEAREST)
            # seg=np.array(seg==2).astype(int)
            depth = cv2.resize(depth, (320,240), interpolation =cv2.INTER_NEAREST)
        
            # plt.imsave('/home/kejianshi/Desktop/Surgical_Robot/science_robotics/ar_surrol/surrol/data/seg_{}.png'.format(self.counter),seg)
            # exit()
            np.save('/home/kejianshi/Desktop/Surgical_Robot/science_robotics/data_collected/seg_npy/seg_{}.npy'.format(self.counter),seg)
            np.save('/home/kejianshi/Desktop/Surgical_Robot/science_robotics/data_collected/depth/depth_{}.npy'.format(self.counter),depth)
            # cv2.imwrite('/home/kejianshi/Desktop/Surgical_Robot/science_robotics/data_collected/img_{}.png'.format(self.counter),cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB))
            

            #img_size=seg.shape[0]
            #obs['depth']=depth.reshape(1, img_size, img_size).copy()
            #obs['seg']=seg.reshape(1, img_size, img_size).copy()
            
            #cv2.imwrite('/home/student/code/regress_data7/img/img_{}.png'.format(self.counter),cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB))
            
            #exit()
            self.img_list[self.counter]={}
            self.img_list[self.counter]['obs']=obs
            self.counter+=1
            #self.img_list[self.counter]=obs['observation']

        
        if self.counter>20000:
            with open('/home/kejianshi/Desktop/Surgical_Robot/science_robotics/data_collected/img_obs.pkl',"wb") as f:
                pickle.dump(self.img_list,f)
            exit()
        
        return observation
    

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        goal = np.array([0., 0.])
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
        centroid = obs[-3: -1]
        cam_u = centroid[0] * RENDER_WIDTH
        cam_v = centroid[1] * RENDER_HEIGHT
        self.ecm.homo_delta = np.array([cam_u, cam_v]).reshape((2, 1))
        if np.linalg.norm(self.ecm.homo_delta) < 8 and np.linalg.norm(self.ecm.wz) < 0.1:
            # e difference is small enough
            action = np.zeros(3)
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
        return action


if __name__ == "__main__":
    env = ActiveTrack(render_mode='human')  # create one process and corresponding env

    env.test(horizon=200)
    env.close()
    time.sleep(2)
