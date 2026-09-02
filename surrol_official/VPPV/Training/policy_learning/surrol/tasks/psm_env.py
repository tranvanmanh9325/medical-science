import os
import numpy as np
import pybullet as p
from surrol.gym.surrol_goalenv import SurRoLGoalEnv
from surrol.robots.psm import Psm1, Psm2
from surrol.utils.pybullet_utils import (
    get_link_pose,
    wrap_angle,
    reset_camera
)
from surrol.utils.robotics import (
    get_euler_from_matrix,
    get_matrix_from_euler
)
from surrol.const import ROOT_DIR_PATH, ASSET_DIR_PATH

# only for demo
import time
import pandas as pd

# soft body
import MPM.mpm3d as mpm3d
import matplotlib.pyplot as plt
import trimesh


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class PsmEnv(SurRoLGoalEnv):
    """
    Single arm env using PSM1
    Refer to Gym fetch
    https://github.com/openai/gym/blob/master/gym/envs/robotics/fetch_env.py
    ravens
    https://github.com/google-research/ravens/blob/master/ravens/environments/environment.py
    """
    ACTION_SIZE = 5  # (dx, dy, dz, dyaw/dpitch, open/close)
    ACTION_MODE = 'yaw'
    DISTANCE_THRESHOLD = 0.005
    POSE_PSM1 = ((0.05, 0.24, 0.8524), (0, 0, -(90 + 20) / 180 * np.pi))
    # POSE_PSM1 = ((0.96, 0.1, 0.38), ((90 - 25) / 180 * np.pi, 0, 1/2 * np.pi))

    QPOS_PSM1 = (0, 0, 0.10, 0, 0, 0)
    POSE_TABLE = ((0.5, 0, 0.001), (0, 0, 0))
    # original limits
    WORKSPACE_LIMITS1 = ((0.50, 0.60), (-0.05, 0.05), (0.675, 0.745))
    #modified limits for peg_board
    # WORKSPACE_LIMITS1 = ((0.50, 0.60), (-0.05, 0.05), (0.675, 0.785))
    SCALING = 1.

    def __init__(self,render_mode=None, cid = -1):
        self.STEP_COUNT=0
        self.particles_list = []
        self.meshes_list = []
        self.colormap = plt.get_cmap("YlGnBu")
        self.rendering_cost = []
        self.rigid_sim_cost = []
        self.soft_sim_cost=[]
        # workspace
        workspace_limits = np.asarray(self.WORKSPACE_LIMITS1) \
                           + np.array([0., 0., 0.0102]).reshape((3, 1))  # tip-eef offset with collision margin
        workspace_limits *= self.SCALING  # use scaling for more stable collistion simulation
        self.workspace_limits1 = workspace_limits
        # plot goal
        self._goal_plot = True
        # has_object
        self.has_object = False
        self._waypoint_goal = False
        self.obj_id, self.obj_link1, self.obj_link2 = None, None, None  # obj_link: waypoint link

        # gripper
        self.block_gripper = True
        self._activated = -1
        self._max_constraint_id = 1
        super(PsmEnv, self).__init__(render_mode, cid)

        # distance_threshold
        self.distance_threshold = self.DISTANCE_THRESHOLD * self.SCALING

        # render related setting
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=0.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        # self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
        #     cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.345 * self.SCALING),
        #     distance=0.77 * self.SCALING,
        #     yaw=90,
        #     pitch=-30,
        #     roll=0,
        #     upAxisIndex=2
        # )
    

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        """ All sparse reward.
        The reward is 0 or -1.
        """
        # d = goal_distance(achieved_goal, desired_goal)
        # return - (d > self.distance_threshold).astype(np.float32)
        return self._is_success(achieved_goal, desired_goal).astype(np.float32) - 1.

    def _env_setup(self,goal_plot=True):
        # camera
        if self._render_mode == 'human':
            reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
                         target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))

        # robot
        self.psm1 = Psm1(self.POSE_PSM1[0], p.getQuaternionFromEuler(self.POSE_PSM1[1]),
                         scaling=self.SCALING)
        self.psm1_eul = np.array(p.getEulerFromQuaternion(
            self.psm1.pose_rcm2world(self.psm1.get_current_position(), 'tuple')[1]))  # in the world frame
        
               
        if self.ACTION_MODE == 'yaw':
            #self.psm1_eul = np.array([np.deg2rad(-90), 0., self.psm1_eul[2]])
            '''
            eul=np.array([np.deg2rad(-90), 0., 0.])
            eul= get_matrix_from_euler(eul)
            init_pose=self.psm1.get_current_position()
            self.psm1_eul=np.array(get_euler_from_matrix(init_pose[:3, :3]))
            init_pose[:3,:3]=eul
            rcm_pose=self.psm1.pose_world2rcm(init_pose)
            rcm_eul=get_euler_from_matrix(rcm_pose[:3,:3])
            self.psm1_eul[0]=rcm_eul[0]
            self.psm1_eul[1]=rcm_eul[1]
            '''
            # RCM init
            eul=np.array([np.deg2rad(-90), 0., 0.])
            eul= get_matrix_from_euler(eul)
            init_pose=self.psm1.get_current_position()
            self.rcm_init_eul=np.array(get_euler_from_matrix(init_pose[:3, :3]))
            init_pose[:3,:3]=eul
            rcm_pose=self.psm1.pose_world2rcm_no_tip(init_pose)
            rcm_eul=get_euler_from_matrix(rcm_pose[:3,:3])
            
            self.rcm_init_eul[0]=rcm_eul[0]
            self.rcm_init_eul[1]=rcm_eul[1]
            
        elif self.ACTION_MODE == 'pitch':
            self.psm1_eul = np.array([np.deg2rad(0), self.psm1_eul[1], np.deg2rad(-90)])
        else:
            raise NotImplementedError
        self.psm2 = None
        self._contact_constraint = None
        self._contact_approx = False

        p.loadURDF(os.path.join(ASSET_DIR_PATH, 'table/table.urdf'),
                   np.array(self.POSE_TABLE[0]) * self.SCALING,
                   p.getQuaternionFromEuler(self.POSE_TABLE[1]),
                   globalScaling=self.SCALING)

        # for goal plotting
        if goal_plot:
            obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                                globalScaling=self.SCALING)
            self.obj_ids['fixed'].append(obj_id)  # 0
        else: 
            obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'sphere/sphere.urdf'),
                                globalScaling=0.001) #visually remove
            self.obj_ids['fixed'].append(obj_id)  # 0  
        p.changeVisualShape(obj_id, -1, rgbaColor=(0 / 255, 0 / 255, 0 / 255, 0))  
        #print(f'goal:{obj_id}')
        pass  # need to implement based on every task
        # self.obj_ids

        # # only for demo
        # if len(self.actions) > 0:
        #     # record the actions
        #     folder_path = os.path.join(ROOT_DIR_PATH, "../logs/actions/{}".format(self.__class__.__name__))
        #     os.makedirs(folder_path, exist_ok=True)
        #     pd.DataFrame(np.array(self.actions)).to_csv(
        #         os.path.join(folder_path, "{}.csv".format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))))
        # self.actions = []
    
    def _camera2world_pos(self, position):
        
        # Create a 4x1 homogenous position vector
        position_homogeneous = np.append(position, 1)

        # Invert the extrinsic matrix
        extrinsic_matrix_inv = np.linalg.inv(self._view_matrix)

        # Perform the transformation
        position_transformed = np.dot(extrinsic_matrix_inv, position_homogeneous)

        return position_transformed[:3]
    
    
    def _camera2world_rot(self,euler_angles):
       
        # Convert Euler angles to rotation matrix
        # return: matrix
        roll, pitch, yaw = euler_angles
        R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))

        # Invert the extrinsic matrix
        extrinsic_matrix_inv = np.linalg.inv(self._view_matrix)

        # Extract the rotation part from the inverted extrinsic matrix
        rotation_matrix_inv = extrinsic_matrix_inv[:3, :3]

        # Perform the rotation
        position_rotated = np.dot(rotation_matrix_inv, rotation_matrix)

        return position_rotated
    
    def _world2cam_pos(self, pos):
    
        point_world = np.array([pos[0], pos[1], pos[2], 1])
        
        cam_pos=self._view_matrix @ point_world
        return np.array([cam_pos[0],cam_pos[1],cam_pos[2]])
        #return cam_pos
        

    def _world2cam_rot(self,euler_world):
        
        rot_matrix_world = np.array([[np.cos(euler_world[1])*np.cos(euler_world[2]), np.sin(euler_world[0])*np.sin(euler_world[1])*np.cos(euler_world[2]) - np.cos(euler_world[0])*np.sin(euler_world[2]), np.cos(euler_world[0])*np.sin(euler_world[1])*np.cos(euler_world[2]) + np.sin(euler_world[0])*np.sin(euler_world[2])],
                                    [np.cos(euler_world[1])*np.sin(euler_world[2]), np.sin(euler_world[0])*np.sin(euler_world[1])*np.sin(euler_world[2]) + np.cos(euler_world[0])*np.cos(euler_world[2]), np.cos(euler_world[0])*np.sin(euler_world[1])*np.sin(euler_world[2]) - np.sin(euler_world[0])*np.cos(euler_world[2])],
                                    [-np.sin(euler_world[1]), np.sin(euler_world[0])*np.cos(euler_world[1]), np.cos(euler_world[0])*np.cos(euler_world[1])]])
        # Extract rotation matrix from camera extrinsic matrix
        #print(self._view_matrix)
        rot_matrix_camera = self._view_matrix[:3, :3]
        # Apply camera extrinsic rotation to rotation matrix in world axis
        rot_matrix_camera = rot_matrix_camera @ rot_matrix_world
        # Convert rotation matrix to Euler angles in camera axis
        euler_camera = np.array([np.arctan2(rot_matrix_camera[2, 1], rot_matrix_camera[2, 2]),
                                np.arctan2(-rot_matrix_camera[2, 0], np.sqrt(rot_matrix_camera[2, 1]**2 + rot_matrix_camera[2, 2]**2)),
                                np.arctan2(rot_matrix_camera[1, 0], rot_matrix_camera[0, 0])])
        return euler_camera
    
    def _get_robot_state(self, idx: int) -> np.ndarray:
        # robot state: tip pose in the world coordinate
        psm = self.psm1 if idx == 0 else self.psm2
        #print('rcm posi: ',psm.get_current_position())
        pose_world = psm.pose_rcm2world(psm.get_current_position(), 'tuple')
        euler=np.array(p.getEulerFromQuaternion(pose_world[1]))
        pos=np.array(pose_world[0])
        # add world2cam
        #if hasattr(self, 'ecm'):
        euler=self._world2cam_rot(euler)
        pos=self._world2cam_pos(pos)
        #else:
        #    euler=world_euler
        jaw_angle = psm.get_current_jaw_position()
        return np.concatenate([
            pos, euler,np.array(jaw_angle).ravel()
        ])  # 3 + 3 + 1 = 7
    
    def _get_robot_state_world(self, idx: int) -> np.ndarray:
        # robot state: tip pose in the world coordinate
        psm = self.psm1 if idx == 0 else self.psm2
        #print('rcm posi: ',psm.get_current_position())
        pose_world = psm.pose_rcm2world(psm.get_current_position(), 'tuple')
        euler=np.array(p.getEulerFromQuaternion(pose_world[1]))
        pos=np.array(pose_world[0])
        # add world2cam
        #if hasattr(self, 'ecm'):
        # euler=self._world2cam_rot(euler)
        # pos=self._world2cam_pos(pos)
        #else:
        #    euler=world_euler
        jaw_angle = psm.get_current_jaw_position()
        return np.concatenate([
            pos, euler,np.array(jaw_angle).ravel()
        ])  # 3 + 3 + 1 = 7

    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state(idx=0)
        
        # TODO: may need to modify
        if self.has_object:
            pos, _ = get_link_pose(self.obj_id, -1)
            object_pos = np.array(pos)
            #print("object pos: ",object_pos)
            pos, orn = get_link_pose(self.obj_id, self.obj_link1)
            waypoint_pos = np.array(pos)
            # rotations
            waypoint_rot = np.array(p.getEulerFromQuaternion(orn))
            # relative position state
            
            # world2cam
            #object_pos=np.dot(self._view_matrix[:3,:3], object_pos)+self._view_matrix[:3,3]
            object_pos=self._world2cam_pos(object_pos)
            waypoint_pos=self._world2cam_pos(waypoint_pos)
            waypoint_rot=self._world2cam_rot(waypoint_rot)
            
            object_rel_pos = object_pos - robot_state[0: 3]
        else:
            # TODO: can have a same-length state representation
            object_pos = waypoint_pos = waypoint_rot = object_rel_pos = np.zeros(0)
        #print('object_pos: ',object_pos)
        if self.has_object:
           
            achieved_goal = object_pos.copy() if not self._waypoint_goal else waypoint_pos.copy()
        else:
            # tip position
            achieved_goal = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])
            achieved_goal =self._world2cam_pos(achieved_goal)
            
        
        observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_rot.ravel()  # achieved_goal.copy(),
        ])
        
        '''
        observation = np.concatenate([
            object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_rot.ravel()  # achieved_goal.copy(),
        ])
        '''
        #print("goal: ",self.goal)
        #exit()
        goal=self._world2cam_pos(self.goal)
        #print("goal ecm: ",goal)
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': goal
        }
        
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
        '''
        # World Action
        pose_world = self.psm1.pose_rcm2world(self.psm1.get_current_position())
        #print(pose_world)
        
        workspace_limits = self.workspace_limits1
        pose_world[:3, 3] = np.clip(pose_world[:3, 3] + action[:3],
                                    workspace_limits[:, 0] - [0.02, 0.02, 0.],
                                    workspace_limits[:, 1] + [0.02, 0.02, 0.08])  # clip to ensure convergence
        rot = get_euler_from_matrix(pose_world[:3, :3])
        if self.ACTION_MODE == 'yaw':
            action[3] *= np.deg2rad(20)  # yaw, limit maximum change in rotation
            rot = (self.psm1_eul[0], self.psm1_eul[1], wrap_angle(rot[2] + action[3]))  # only change yaw
        elif self.ACTION_MODE == 'pitch':
            action[3] *= np.deg2rad(15)  # pitch, limit maximum change in rotation
            pitch = np.clip(wrap_angle(rot[1] + action[3]), np.deg2rad(-90), np.deg2rad(90))
            rot = (self.psm1_eul[0], pitch, self.psm1_eul[2])  # only change pitch
        else:
            raise NotImplementedError
        pose_world[:3, :3] = get_matrix_from_euler(rot)
        action_rcm = self.psm1.pose_world2rcm(pose_world)
        # time1 = time.time()
        self.psm1.move(action_rcm)
        
        
        # RCM action
        pose_rcm=self.psm1.get_current_position()
        
        pose_rcm[:3,3]=pose_rcm[:3,3]+action[:3]
        #print("r: ",pose_rcm[:3,3])
        pose_world=self.psm1.pose_rcm2world(pose_rcm)
        workspace_limits = self.workspace_limits1
        
        pose_world[:3,3]=np.clip(pose_world[:3, 3],
                                    workspace_limits[:, 0] - [0.02, 0.02, 0.],
                                    workspace_limits[:, 1] + [0.02, 0.02, 0.08])
        action_rcm=self.psm1.pose_world2rcm(pose_world)
        #print("action_rcm: ",action_rcm[:3,3])
        rot=get_euler_from_matrix(action_rcm[:3, :3])
       

        if self.ACTION_MODE == 'yaw':
           
            action[3] *= np.deg2rad(20)  # yaw, limit maximum change in rotation
            rot = (self.psm1_eul[0], self.psm1_eul[1], wrap_angle(rot[2]+action[3]))
           
        elif self.ACTION_MODE == 'pitch':
            action[3] *= np.deg2rad(15)  # pitch, limit maximum change in rotation
            pitch = np.clip(wrap_angle(rot[1] + action[3]), np.deg2rad(-90), np.deg2rad(90))
            rot = (self.psm1_eul[0], pitch, self.psm1_eul[2])  # only change pitch
        else:
            raise NotImplementedError
        action_rcm[:3, :3] = get_matrix_from_euler(rot)
        self.psm1.move(action_rcm)
        '''
        # ECM action
        curr_rcm_pose=self.psm1.get_current_position()
        pose_world = self.psm1.pose_rcm2world(curr_rcm_pose, 'tuple')
        euler=np.array(p.getEulerFromQuaternion(pose_world[1]))
        
        pos=np.array(pose_world[0])
        # world2ecm
        euler=self._world2cam_rot(euler)
        pos=self._world2cam_pos(pos)
      
        pos=pos+action[:3]
       
        # Change (pitch, roll, yaw) in ECM axis
        action[3:6] *= np.deg2rad(20)  
        
        rot =(wrap_angle(euler[0]+action[3]),wrap_angle(euler[1]+action[4]),wrap_angle(euler[2]+action[5]))
        
        world_pos=self._camera2world_pos(pos)
        
        world_rot=self._camera2world_rot(rot) # return matrix
        world_rot_euler=get_euler_from_matrix(world_rot)
        
        curr_rcm_pose[:3,:3]=world_rot
        curr_rcm_pose[:3,3]=world_pos
        
        rcm_action=self.psm1.pose_world2rcm_no_tip(curr_rcm_pose)
        rcm_action_eul=get_euler_from_matrix(rcm_action[:3,:3])
        rcm_action_eul=(self.rcm_init_eul[0], self.rcm_init_eul[1], rcm_action_eul[2])
        
        rcm_action_matrix=get_matrix_from_euler(rcm_action_eul)
        rcm_action[:3,:3]=rcm_action_matrix
        rcm_action=np.matmul(rcm_action, self.psm1.tool_T_tip)
        
        self.psm1.move(rcm_action)
        
        # jaw
        if self.block_gripper:
            action[4] = -1
        if action[4] < 0:
            self.psm1.close_jaw()
            
            self._activate(0)
        else:
            self.psm1.move_jaw(np.deg2rad(40))  # open jaw angle; can tune
            self._release(0)
            
    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        #print('achieved_goal: ',achieved_goal)
        #print('desired_goal: ',desired_goal)
        d = goal_distance(achieved_goal, desired_goal)
        #print('d: ',d)
        return (d < self.distance_threshold).astype(np.float32)

    def _step_callback(self,demo=1):
        """ Remove the contact constraint if no contacts
        """
        if self.block_gripper or not self.has_object or self._activated < 0:
            # print(f'skip{self.block_gripper} {self.has_object} {self._activated}')
            return
        # if demo:
        if self._contact_constraint is None:
            # the grippers activate; to check if they can grasp the object
            # TODO: check whether the constraint may cause side effects
            # pass
            psm = self.psm1 if self._activated == 0 else self.psm2
            print(self._meet_contact_constraint_requirement())
            if self._meet_contact_constraint_requirement():
                # print(self.obj_id)
                points_1 = p.getContactPoints(bodyA=psm.body, linkIndexA=6)
                points_2 = p.getContactPoints(bodyA=psm.body, linkIndexA=7)
                points_1 = [point[2] for point in points_1 if point[2] in self.obj_ids['rigid']]
                points_2 = [point[2] for point in points_2 if point[2] in self.obj_ids['rigid']]
                contact_List = list(set(points_1)&set(points_2))
                # print(f'contact{contact_List}')
                print(f'contact item id:{contact_List}')
                if len(contact_List)>0:
                    contact_Id=contact_List[-1]
                    body_pose = p.getLinkState(psm.body, psm.EEF_LINK_INDEX)
                    obj_pose = p.getBasePositionAndOrientation(contact_Id)
                    world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                    obj_to_body = p.multiplyTransforms(world_to_body[0],
                                                    world_to_body[1],
                                                    obj_pose[0], obj_pose[1])                

                    self._contact_constraint = p.createConstraint(
                        parentBodyUniqueId=psm.body,
                        parentLinkIndex=psm.EEF_LINK_INDEX,
                        childBodyUniqueId=contact_Id,
                        childLinkIndex=-1,
                        jointType=p.JOINT_FIXED,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=obj_to_body[0],
                        parentFrameOrientation=obj_to_body[1],
                        childFramePosition=(0, 0, 0),
                        childFrameOrientation=(0, 0, 0))
                    # TODO: check the maxForce; very subtle
                    print(f'contact with {contact_Id}!create constraint id{self._contact_constraint}!')
                    p.changeConstraint(self._contact_constraint, maxForce=20)
        else:
            # self._contact_constraint is not None
            # the gripper grasp the object; to check if they remain contact
            # pass
            psm = self.psm1 if self._activated == 0 else self.psm2
            # points = p.getContactPoints(bodyA=psm.body, linkIndexA=6) \
            #          + p.getContactPoints(bodyA=psm.body, linkIndexA=7)
            points_1 = p.getContactPoints(bodyA=psm.body, linkIndexA=6)
            points_2 = p.getContactPoints(bodyA=psm.body, linkIndexA=7)
            points_1 = [point[2] for point in points_1 if point[2] in self.obj_ids['rigid']]
            points_2 = [point[2] for point in points_2 if point[2] in self.obj_ids['rigid']]
            intersect = list(set(points_1)&set(points_2))
            # if len(intersect) > 0:
            #     contact_Id = intersect[-1]
            # else:
            #     contact_Id = -100
            # points = [point for point in points if point[2] == contact_Id]
            # remain_contact = len(points) > 0
            remain_contact = len(intersect) > 0

            if not remain_contact and not self._contact_approx:
                # release the previously grasped object because there is no contact any more
                # print("no contact!remove constraint!")
                self._release(self._activated)
        print(f'num of constraints for stepcallback: {p.getNumConstraints()}')

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        raise NotImplementedError

    def _sample_goal_callback(self):
        """ Set the red sphere pose for goal visualization
        """
        p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], self.goal, (0, 0, 0, 1))

    def _activate(self, idx: int):
        """
        :param idx: 0 for psm1 and 1 for psm2
        :return:
        """
        # check if the gripper closed and grasped something
        if self.block_gripper:
            return
        if self._activated < 0:
            # only activate one psm
            psm = self.psm1 if idx == 0 else self.psm2
            if self._contact_approx:
                # activate if the distance between the object and the tip below a threshold
                pos_tip, _ = get_link_pose(psm.body, psm.TIP_LINK_INDEX)
                if not self._waypoint_goal:
                    link_id = -1
                else:
                    link_id = self.obj_link1 if idx == 0 else self.obj_link2  # TODO: check
                pos_obj, _ = get_link_pose(self.obj_id, link_id)
                if np.linalg.norm(np.array(pos_tip) - np.array(pos_obj)) < 2e-3 * self.SCALING:
                    self._activated = idx
                    # disable collision
                    p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id,
                                             linkIndexA=6, linkIndexB=-1, enableCollision=0)
                    p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id,
                                             linkIndexA=7, linkIndexB=-1, enableCollision=0)
            else:
                # activate if a physical contact happens between grippers and object items
                points_1 = p.getContactPoints(bodyA=psm.body, linkIndexA=6)
                points_2 = p.getContactPoints(bodyA=psm.body, linkIndexA=7)
                points_1 = [point[2] for point in points_1 if point[2] in self.obj_ids['rigid']]
                points_2 = [point[2] for point in points_2 if point[2] in self.obj_ids['rigid']]
                intersect = list(set(points_1)&set(points_2))
                # print(f'left gripper: {points_1}')
                # print(f'right gripper: {points_2}')
                if len(intersect)>0:
                    self._activated = idx

    def _release(self, idx: int,demo=1):
        # release the object
        if self.block_gripper:
            return

        if self._activated == idx:
            self._activated = -1
            if self._contact_constraint is not None:
                try:
                    print(f"no contact!to remove constraint id{self._contact_constraint}!")
                    for i in range(1,self._contact_constraint+1):
                        p.changeConstraint(i, maxForce=0)
                        p.removeConstraint(i)
                    if not p.getNumConstraints():
                        self._contact_constraint = None
                        # print(f"constraint status: {p.getConstraintInfo(i)}")
                        # print(f"constraint state:{p.getConstraintState(i)}")
                    # p.changeConstraint(self._contact_constraint, maxForce=0)
                    # self._contact_constraint = None
                    print(f"#(constraint)?{p.getNumConstraints()}!")


                    # enable collision
                    # psm = self.psm1 if idx == 0 else self.psm2
                    # p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id,
                    #                          linkIndexA=6, linkIndexB=-1, enableCollision=1)
                    # p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id,
                    #                          linkIndexA=7, linkIndexB=-1, enableCollision=1)
                except:
                    print("unable to get constraint info since removed")
                    pass

    def _meet_contact_constraint_requirement(self) -> bool:
        # check if meeting the contact constraint
        if self.block_gripper or self.has_object is None:
            return False
        return False

    def get_pts_color(self):
        return self.colormap(self.mpm_solver.F_x.to_numpy()[:,2]*2)[:,:3]
    
    def get_goal_idx(self, pos, threshold = 0.01):
        return self.mpm_solver.choose_idx_given_position(pos, threshold)
    
    def init_soft_body(
        self,
        collision_obj_list,
        model_filename,
        collision_sdf_filename,
        soft_body_base_position,
        mpm_solver,
        gs_model
    ):
        mpm_solver.init(collision_obj_list, model_filename, gs_model)
        # load needle's sdf
        mpm_solver.init_pos()
        mpm_solver.init_sdf(np.load(collision_sdf_filename))
        print("Soft Body Base Position:")
        print(soft_body_base_position)
        self.soft_body_base_position = soft_body_base_position
        mpm_solver.set_base_position(soft_body_base_position)
        self.mpm_solver = mpm_solver

    def get_soft_body_goal_position(self, idx, scaling):
        if self.mpm_solver is None:
            return None
        else:
            return np.array(self.mpm_solver.F_x[idx])*scaling + self.soft_body_base_position
        
    def get_ecm_image(self,image_width=640,image_height=512):
        self.ecm.render_image()
        _, _, rgb_image, depth_image, mask = p.getCameraImage(
            width=image_width,
            height=image_height,
            viewMatrix=self.ecm.view_matrix,
            projectionMatrix=self.ecm.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
        near,far = 0.01,1000 #default value in pybullet
        depth = far * near / (far - (far - near) * depth_image)
        return rgb_image, depth, mask
    
    def soft_body_step(
        self,
        gs_model,
        i_rot_list,
        i_pos_list,
        USE_POINTS=True,
        scale=1.0,
        use_debug=False,
        external_forces=False,
        texture_id=-1,
        add_collision_shape=False,
        visualize_deformation=False,
        points_color=None,
        draw_soft_body = True
    ):
        """
        One simulation step of soft body.

        i_rot: inverse rotation matrix

        i_pos: inverse advection offset

        USE_POINTS: draw points or meshes

        base_position: the start position of the soft body

        scale: scale soft body

        use_debug: whether visualize euler grid and output inverse transform matrix

        external_forces: (beta) apply forces on pybullet rigid body

        texture_id: the id of texture which will be applied on soft body

        add_collision_shape: add collision bounding shape on soft body

        visualize_deformation: visualize the deformation gradient

        points_color: specify the color of each point

        draw_soft_body: visualize soft body in pybullet or not
        """
        t0 = time.time()
        base_position = self.soft_body_base_position
        self.mpm_solver.step(
            scale,
            inverse_pos_list=i_pos_list,
            inverse_rot_list=i_rot_list,
            apply_force=external_forces,
            threshold=self.threshold,
            visualize_deformation=visualize_deformation,
            debug=use_debug
        )

        for i in self.particles_list:
            p.removeUserDebugItem(i)
        self.particles_list.clear()
        for i in self.meshes_list:
            p.removeBody(i)
        self.meshes_list.clear()
        
        t2 = time.time()
        if use_debug:
            print(f"head time: {(t2-t0)*1000}ms")

        if draw_soft_body:
            if USE_POINTS:
                if gs_model is not None:
                    particles_array = gs_model.get_xyz.cpu().detach().numpy()
                else:
                    particles_array = self.mpm_solver.F_x.to_numpy()*scale + base_position

                num_particles = len(particles_array)
                idx = 0
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                if points_color is None:
                    points_color = [self.mpm_solver.ORIANGE] * num_particles
                pld_id = p.addUserDebugPoints(
                    particles_array,
                    points_color,
                    pointSize=3,
                )
                self.particles_list.append(pld_id)
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            else:
                try:
                    vtx, idx, _ = self.mpm_solver.get_mesh(smooth_scale=0.2, debug = use_debug)
                except:
                    print("get mesh error")
                    self.is_softSim_error=True
                    return
                t3 = time.time()
                if use_debug:
                    print(f"outer MC time: {(t3-t2)*1000}ms")

                if visualize_deformation:
                    vtx_deformation = np.zeros(shape=(len(vtx),))
                    # Interpolate vertex color from grid nodes
                    for i, v in enumerate(vtx):
                        base = v.astype("int")
                        dx, dy, dz = v - base
                        vtx_deformation[i] = self.mpm_solver.interpolate_deformation(
                            base, dx, dy, dz
                        )

                    # normalize
                    print(
                        f"max deformation: {vtx_deformation.max()}, min deformation: {vtx_deformation.min()}"
                    )

                    vtx_deformation = (vtx_deformation - 300.0) / 3300.0
                    # #compute color
                    vtx_colors = self.colormap(1 - vtx_deformation)

                vtx = vtx / (self.mpm_solver.n_grid * 1.0)
                vertices = vtx * scale + base_position
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

                if visualize_deformation:
                    # bake texture and compute uv coordinates
                    import trimesh
                    import pymeshlab
                    reshape_idx = idx.reshape((-1, 3))
                    m = pymeshlab.Mesh(vtx, reshape_idx, v_color_matrix=vtx_colors)
                    ms = pymeshlab.MeshSet()
                    ms.add_mesh(m, f"level_set{self.STEP_COUNT}")
                    ms.compute_texcoord_by_function_per_wedge()
                    ms.compute_texmap_from_color(
                        textname=f"./tex_tmp{self.STEP_COUNT}"
                    )  # file path?
                    ms.save_current_mesh("./visual/tmp.obj")
                    mesh = trimesh.load("./visual/tmp.obj", force="mesh")
                    # *****************************************

                # oriange=[0.93,0.33,0.23]
                if visualize_deformation:
                    vid = p.createVisualShape(
                        shapeType=p.GEOM_MESH,
                        vertices=vertices,
                        indices=idx,
                        uvs=mesh.visual.uv,
                    )
                elif texture_id != -1:
                    # dynamic position based texture mapping
                    anchor = self.mpm_solver.F_x[0].to_numpy()
                    UV = vtx[:, 1:] + vtx[:, :2] - (anchor[1:] + anchor[:2])
                    UV = vtx[:, :2]

                    vid = p.createVisualShape(
                        shapeType=p.GEOM_MESH,
                        vertices=vertices,
                        indices=idx,
                        uvs=UV,
                    )
                else:
                    vid = p.createVisualShape(
                        shapeType=p.GEOM_MESH,
                        vertices=vertices,
                        indices=idx,
                        rgbaColor=[0.93, 0.33, 0.23, 1.0],
                    )

                cid = -1
                if add_collision_shape:
                    cid = p.createCollisionShape(
                        shapeType=p.GEOM_MESH, vertices=vertices, indices=idx
                    )

                body_id = p.createMultiBody(
                    baseVisualShapeIndex=vid, baseCollisionShapeIndex=cid
                )

                if visualize_deformation:
                    # texture mapping
                    defo_tex_id = p.loadTexture(f"./visual/tex_tmp{self.STEP_COUNT}.png")
                    self.STEP_COUNT += 1
                    p.changeVisualShape(
                        objectUniqueId=body_id, linkIndex=-1, textureUniqueId=defo_tex_id
                    )
                elif texture_id != -1:  #!: may lead to crash in v2
                    p.changeVisualShape(
                        objectUniqueId=body_id, linkIndex=-1, textureUniqueId=texture_id
                    )
                self.meshes_list.append(body_id)
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

                t4 = time.time()
                if use_debug:
                    print(f"Rendering time: {(t4-t3)*1000}ms")
                    print(f"inner soft body step time: {(t4-t0)*1000}ms")
                    # self.rendering_cost.append((t4 - t3) * 1000)

    def sim_step(
        self,
        gs_model,
        collision_obj_list,
        use_points=False,
        external_forces=False,
        texture_id=-1,
        add_collision_shape=False,
        threshold=0.05,
        scale=1.0,
        visualize_deformation=False,
        automate=False,
        debug=False,
        points_color=None,
        draw_soft_body = True
    ):
        inv_pos_list = []
        inv_rot_list = []
        tb = time.time()

        assert (
            len(collision_obj_list) == self.mpm_solver.MAX_COLLISION_OBJECTS
        ), f"Please edit MAX_COLLISION_OBJECTS in mpm3d.py to {len(collision_obj_list)}."

        for co_obj in collision_obj_list:
            if co_obj[1] == -1:  # object but no link
                t_p, t_q = p.getBasePositionAndOrientation(co_obj[0])
            else:
                t_p, t_q = p.getLinkState(co_obj[0], co_obj[1])[:2]
            s_p, s_q = self.soft_body_base_position, (0, 0, 0, 1.0)
            inv_tp, inv_tq = p.invertTransform(t_p, t_q)
            # pay attention to the multiply sequence
            i_pos, i_quaternion = p.multiplyTransforms(inv_tp, inv_tq, s_p, s_q)
            i_rot = (
                np.array(p.getMatrixFromQuaternion(i_quaternion))
                .reshape(3, 3)
                .astype(np.float32)
            )
            i_pos = np.array(i_pos).astype(np.float32)

            inv_pos_list.append(i_pos)
            inv_rot_list.append(i_rot)
        t0 = time.time()
        if debug:
            print(f'reverse transformation:{(t0-tb)*1000}')

        # TODO need to be modified if self._duration != 0.2
        # for _ in range(int(80*self._duration)):
        # soft body simulation step
            # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        self.soft_body_step(
            gs_model=gs_model,
            i_rot_list=inv_rot_list,
            i_pos_list=inv_pos_list,
            USE_POINTS=use_points,
            scale=scale,
            use_debug=debug,
            external_forces=external_forces,
            texture_id=texture_id,
            add_collision_shape=add_collision_shape,
            visualize_deformation=visualize_deformation,
            points_color=points_color,
            draw_soft_body=draw_soft_body
        )
            # t1 = time.time()
            # if debug:
            #     print(f"Whole Soft Simulation:{(t1-t0)*1000}ms")
            # self.soft_sim_cost.append((t1 - t0) * 1000)

        # rigid body simulation step need to be consistent with the soft body simulation step (timestep * FPS)
        num_step = int(self.mpm_solver.timestep * 240)
        num_step = int(240*self._duration)
        for _ in range(num_step):
            if automate:
                action = self.get_oracle_action(self.obs)
                self.obs, _, _, _ = self.step(action)
                if action[4] == -0.5:
                    # print("I am here")
                    self.threshold = 0.03
                else:
                    self.threshold = 0.01
            else:
                p.stepSimulation()
                # step(self._duration)
        t2 = time.time()
        if debug:
            print(f"Whole Rigid Simulation:{(t2-t0)*1000}ms")
        # self.rigid_sim_cost.append((t2 - t1) * 1000)

    @property
    def action_size(self):
        return self.ACTION_SIZE

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a scripted oracle strategy
        """
        raise NotImplementedError


class PsmsEnv(PsmEnv):
    """
    Dual arm env using PSM1 and PSM2
    """
    ACTION_SIZE = 5 * 2  # (dx, dy, dz, dyaw/dpitch, open/close) * 2
    ACTION_MODE = 'yaw'
    DISTANCE_THRESHOLD = 0.005
    POSE_PSM1 = ((0.05, 0.24, 0.8524), (0, 0, -(90 + 20) / 180 * np.pi))
    QPOS_PSM1 = (0, 0, 0.10, 0, 0, 0)
    POSE_PSM2 = ((0.05, -0.24, 0.8524), (0, 0, -(90 - 20) / 180 * np.pi))
    QPOS_PSM2 = (0, 0, 0.10, 0, 0, 0)
    POSE_TABLE = ((0.5, 0, 0.001), (0, 0, 0))
    WORKSPACE_LIMITS1 = ((0.50, 0.60), (-0., 0.05), (0.675, 0.745))
    WORKSPACE_LIMITS2 = ((0.50, 0.60), (-0.05, 0.), (0.675, 0.745))
    SCALING = 1.

    def __init__(self,
                 render_mode=None, cid = -1):
        # workspace
        workspace_limits = np.asarray(self.WORKSPACE_LIMITS2) \
                           + np.array([0., 0., 0.0102]).reshape((3, 1))  # tip-eef offset with collision margin
        workspace_limits *= self.SCALING
        self.workspace_limits2 = workspace_limits

        super(PsmsEnv, self).__init__(render_mode, cid)

    def _env_setup(self):
        super(PsmsEnv, self)._env_setup()

        # robot
        self.psm2 = Psm2(self.POSE_PSM2[0], p.getQuaternionFromEuler(self.POSE_PSM2[1]),
                         scaling=self.SCALING)
        self.psm2_eul = np.array(p.getEulerFromQuaternion(
            self.psm2.pose_rcm2world(self.psm2.get_current_position(), 'tuple')[1]))
        if self.ACTION_MODE == 'yaw':
            self.psm2_eul = np.array([np.deg2rad(-90), 0., self.psm2_eul[2]])
        elif self.ACTION_MODE == 'pitch':
            self.psm2_eul = np.array([np.deg2rad(-180), self.psm2_eul[1], np.deg2rad(90)])
        else:
            raise NotImplementedError
        self._contact_constraint2 = None

        pass  # need to implement based on every task
        # self.obj_ids

    def _get_obs(self) -> dict:
        psm1_state = self._get_robot_state(0)
        psm2_state = self._get_robot_state(1)
        robot_state = np.concatenate([psm1_state, psm2_state])
        # may need to modify
        if self.has_object:
            pos, _ = get_link_pose(self.obj_id, -1)
            object_pos = np.array(pos)
            # waypoint1
            pos, orn = get_link_pose(self.obj_id, self.obj_link1)
            waypoint_pos1 = np.array(pos)
            waypoint_rot1 = np.array(p.getEulerFromQuaternion(orn))
            # waypoint2
            pos, orn = get_link_pose(self.obj_id, self.obj_link2)
            waypoint_pos2 = np.array(pos)
            waypoint_rot2 = np.array(p.getEulerFromQuaternion(orn))
            # gripper state
            object_rel_pos1 = object_pos - robot_state[0: 3]
            object_rel_pos2 = object_pos - robot_state[7: 10]
        else:
            object_pos = waypoint_pos1 = waypoint_rot1 = waypoint_pos2 = waypoint_rot2 = \
                object_rel_pos1 = object_rel_pos2 = np.zeros(0)

        if self.has_object:
            achieved_goal = object_pos.copy() if not self._waypoint_goal else waypoint_pos1.copy()
        else:
            # tip position
            pos1 = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])
            pos2 = np.array(get_link_pose(self.psm2.body, self.psm2.TIP_LINK_INDEX)[0])
            achieved_goal = np.concatenate([pos1, pos2])

        observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos1.ravel(), object_rel_pos2.ravel(),
            waypoint_pos1.ravel(), waypoint_rot1.ravel(),
            waypoint_pos2.ravel(), waypoint_rot2.ravel()  # achieved_goal.copy(),
        ])
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        return obs

    def _set_action(self, action: np.ndarray):
        """
        delta_position (3), delta_theta (1) and open/ close the gripper (1); in world coordinate
        *2 for PSM1 [0: 5] and PSM2 [5: 10]
        """
        assert len(action) == self.ACTION_SIZE
        action = action.copy()  # ensure that we don't change the action outside of this scope
        action[0: 3] *= 0.01 * self.SCALING  # position, limit maximum change in position
        action[5: 8] *= 0.01 * self.SCALING
        for i, psm in enumerate((self.psm1, self.psm2)):
            # set the action for PSM1 and PSM2
            pose_world = psm.pose_rcm2world(psm.get_current_position())
            idx = i * 5
            workspace_limits = self.workspace_limits1 if i == 0 else self.workspace_limits2
            pose_world[:3, 3] = np.clip(pose_world[:3, 3] + action[idx: idx + 3],
                                        workspace_limits[:, 0] - [0.02, 0.02, 0.],
                                        workspace_limits[:, 1] + [0.02, 0.02, 0.08])  # clip to ensure convergence
            rot = get_euler_from_matrix(pose_world[:3, :3])
            psm_eul = self.psm1_eul if i == 0 else self.psm2_eul
            if self.ACTION_MODE == 'yaw':
                action[idx + 3] *= np.deg2rad(20)  # limit maximum change in rotation
                rot = (psm_eul[0], psm_eul[1], wrap_angle(rot[2] + action[idx + 3]))  # only change yaw
            elif self.ACTION_MODE == 'pitch':
                action[idx + 3] *= np.deg2rad(15)  # limit maximum change in rotation
                pitch = np.clip(wrap_angle(rot[1] + action[idx + 3]), np.deg2rad(-90), np.deg2rad(90))
                rot = (psm_eul[0], pitch, psm_eul[2])  # only change pitch
            else:
                raise NotImplementedError
            pose_world[:3, :3] = get_matrix_from_euler(rot)
            action_rcm = psm.pose_world2rcm(pose_world)
            psm.move(action_rcm)

            # jaw
            if self.block_gripper:
                action[idx + 4] = -1
            if action[idx + 4] < 0:
                psm.close_jaw()
                self._activate(i)
            else:
                psm.move_jaw(np.deg2rad(40))
                self._release(i)

        # if self.block_gripper:
        #     action[4] = action[9] = -1
        # if action[4] < 0:
        #     self.psm1.close_jaw()
        #     self._activate(0)
        # else:
        #     self.psm1.move_jaw(np.deg2rad(40))
        #     self._release(0)
        #
        # if action[9] < 0:
        #     self.psm2.close_jaw()
        #     self._activate(1)
        # else:
        #     self.psm2.move_jaw(np.deg2rad(40))
        #     self._release(1)
