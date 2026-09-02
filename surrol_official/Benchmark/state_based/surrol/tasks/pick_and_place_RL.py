import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env_RL_2 import PsmEnv, goal_distance
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera,
    wrap_angle
)
from surrol.utils.robotics import (
    get_euler_from_matrix,
    get_matrix_from_euler
)
from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm


class PickAndPlace(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    RPOT_POS = (0.55, -0.025, 0.6781)
    GPOT_POS = (0.55, 0.03, 0.6781)
    POT_ORN = (1.57,0,0)
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE=3
    haptic=False
    
    # TODO: grasp is sometimes not stable; check how to fix it
    def __init__(self, render_mode=None, cid = -1):
        super(PickAndPlace, self).__init__(render_mode, cid)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        self.distance_threshold = 0.053
        self._contact_constraint_2 = None


    def _env_setup(self):
        super(PickAndPlace, self)._env_setup()
        # np.random.seed(4)  # for experiment reproduce
        
        self.has_object = True
        self._waypoint_goal = True
        self._current_action = None
 
        # camera
        if self._render_mode == 'human':
            reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
                         target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
            # reset_camera(yaw=89.60, pitch=-56, dist=5.98,
            #              target=(-0.13, 0.03,-0.94))
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                       scaling=self.SCALING)
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

        scaling=0.15
        # metal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/dot_metal_min.jpg"))
        # newmetal = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/metal2.jpg"))
        # tray pad
        board = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING,
                            useFixedBase=1)
        self.obj_ids['fixed'].append(board)  # 1
        # p.changeVisualShape(board, -1, textureUniqueId=metal)
        red_pot = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'pot/pot.urdf'),
                            np.array(self.RPOT_POS) * self.SCALING,
                            p.getQuaternionFromEuler(self.POT_ORN),
                            useFixedBase=1)
        self.obj_ids['fixed'].append(red_pot) # 2

        green_pot = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'pot/pot.urdf'),
                            np.array(self.GPOT_POS) * self.SCALING,
                            p.getQuaternionFromEuler(self.POT_ORN),
                            useFixedBase=1)
        p.changeVisualShape(green_pot,-1,rgbaColor=(0,1,0,1),specularColor=(80,80,80))
        self.obj_ids['fixed'].append(green_pot) # 2

        pos_1_x, pos_1_y, pos_2_x, pos_2_y = workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,\
                                             workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,\
                                             workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,\
                                             workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1
        while np.linalg.norm([pos_1_x - pos_2_x, pos_1_y - pos_2_y]) < 0.1:
            pos_1_x, pos_1_y, pos_2_x, pos_2_y = workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,\
                                                 workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,\
                                                 workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,\
                                                 workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1

        # ch4
        yaw = (np.random.rand() - 0.5) * np.pi
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'CH4/CH4_waypoints_rl.urdf'),
                            (pos_1_x,  # TODO: scaling
                             pos_1_y,
                             workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1

        # ch4
        yaw = (np.random.rand() - 0.5) * np.pi
        obj_id_2 = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'CH4/CH4_waypoints_rl.urdf'),
                            (pos_2_x,  # TODO: scaling
                             pos_2_y,
                             workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id_2, -1, rgbaColor=(0,1,0,1), specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id_2)  # 0
        self.obj_id_2, self.obj_link2 = self.obj_ids['rigid'][1], 1

        # for i in range(1,6):
            
        #     if(i<=2):
        #         pick_item = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'CH4/CH4_waypoints.urdf'),
        #                     (workspace_limits[0].mean() - i/3 * 0.25,  # TODO: scaling
        #                      workspace_limits[1].mean() -i/3*0.15,
        #                      workspace_limits[2][0] + 0.05),
        #                     p.getQuaternionFromEuler((0, 0, yaw)),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        #         p.changeVisualShape(pick_item, -1, specularColor=(80, 80, 80))
        #         self.obj_ids['rigid'].append(pick_item)
        #     else:
        #         pick_item = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'CH4/CH4_waypoints.urdf'),
        #                     (workspace_limits[0].mean() - i/3 * 0.1,  # TODO: scaling
        #                      workspace_limits[1].mean() +i/3*0.15,
        #                      workspace_limits[2][0] + 0.05),
        #                     p.getQuaternionFromEuler((0, 0, yaw)),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        #         p.changeVisualShape(pick_item,-1,rgbaColor=(0,1,0,1),specularColor=(80,80,80))
        #         self.obj_ids['rigid'].append(pick_item)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        # workspace_limits = self.workspace_limits1
        # goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
        #                  workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
        #                  workspace_limits[2][1] - 0.04 * self.SCALING])
        goal_1 = np.array(self.RPOT_POS) * self.SCALING
        goal_2 = np.array(self.GPOT_POS) * self.SCALING
        return np.concatenate([goal_1.copy(), goal_2.copy()])

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        # super()._sample_goal_callback()
        # print('[DEBUG]: goal: {}'.format(self.goal))
        # p.resetBasePositionAndOrientation(self.obj_ids['fixed'][0], self.goal[:3], (0, 0, 0, 1))
        # p.resetBasePositionAndOrientation(self.obj_ids['fixed'][1], self.goal[3:], (0, 0, 0, 1))

        self._waypoints = [None, None, None, None, None, None, None, None, None, None]  # four waypoints
        
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
        goal_1 = self.goal[:3]

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0027 + 0.0082) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0027 + 0.0082) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([goal_1[0], goal_1[1],
                                       goal_1[2] + 0.0502 * self.SCALING, yaw, -0.5])  # lift up
        self._waypoints[4] = np.array([goal_1[0], goal_1[1],
                                       goal_1[2] + 0.0502 * self.SCALING, yaw, 0.5])  # release
    

        pos_obj_2, orn_obj_2 = get_link_pose(self.obj_id_2, self.obj_link2)
        self._waypoint_z_init_2 = pos_obj_2[2]
        orn_2 = p.getEulerFromQuaternion(orn_obj_2)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw_2 = orn_2[2] if abs(wrap_angle(orn_2[2] - orn_eef[2])) < abs(wrap_angle(orn_2[2] + np.pi - orn_eef[2])) \
            else wrap_angle(orn_2[2] + np.pi)  # minimize the delta yaw
        
        goal_2 = self.goal[3:]


        self._waypoints[5] = np.array([pos_obj_2[0], pos_obj_2[1],
                                       pos_obj_2[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw_2, 0.5])  # approach
        self._waypoints[6] = np.array([pos_obj_2[0], pos_obj_2[1],
                                       pos_obj_2[2] + (-0.0027 + 0.0082) * self.SCALING, yaw_2, 0.5])  # approach
        self._waypoints[7] = np.array([pos_obj_2[0], pos_obj_2[1],
                                       pos_obj_2[2] + (-0.0027 + 0.0082) * self.SCALING, yaw_2, -0.5])  # grasp
        self._waypoints[8] = np.array([goal_2[0], goal_2[1], 
                                       goal_2[2] + 0.0502 * self.SCALING, yaw_2, -0.5])  # lift up
        self._waypoints[9] = np.array([goal_2[0], goal_2[1], 
                                       goal_2[2] + 0.0502 * self.SCALING, yaw_2, 0.5])  # release
        
        # green_waypoints = self._waypoints[5:]
        # red_waypoints = self._waypoints[:5]
        # self._waypoints = green_waypoints + red_waypoints


    def _meet_contact_constraint_requirement(self):
        # print('[DEBUG] z_init_1: {}'.format(self._waypoint_z_init))
        # add a contact constraint to the grasped block to make it stable
        # return False
        if self._current_action is not None and self._current_action[4] >= 0.0:
            return False
        if self._contact_approx or self.haptic is True:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            # print(pose[0][2])
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING
    
    def _meet_contact_constraint_requirement_2(self):
        # print('[DEBUG] z_init_2: {}'.format(self._waypoint_z_init_2))
        # add a contact constraint to the grasped block to make it stable
        # return False
        if self._current_action is not None and self._current_action[4] >= 0.0:
            return False
        if self._contact_approx or self.haptic is True:
            return True  # mimic the dVRL setting
        else:
            pose_2 = get_link_pose(self.obj_id_2, self.obj_link2)

            return pose_2[0][2] > self._waypoint_z_init_2 + 0.005 * self.SCALING

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = 0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.45
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            # print('[DEBUG] waypoint idx: {}, delta_pos: {}, delta_yaw: {}'.format(i, delta_pos, delta_yaw))
            if (np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2) or \
               (np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and (i == 3 or i == 8)):
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
    
    def _set_action(self, action: np.ndarray):
        """
        delta_position (3), delta_theta (1) and open/close the gripper (1)
        in the world frame
        """
        assert len(action) == self.ACTION_SIZE, "The action should have the save dim with the ACTION_SIZE"
        self._current_action = action
        # time0 = time.time()
        action = action.copy()  # ensure that we don't change the action outside of this scope
        action[:3] *= 0.01 * self.SCALING  # position, limit maximum change in position
        pose_world = self.psm1.pose_rcm2world(self.psm1.get_current_position())
        workspace_limits = self.workspace_limits1
        pose_world[:3, 3] = np.clip(pose_world[:3, 3] + action[:3],
                                    workspace_limits[:, 0] - [0.02, 0.02, 0.],
                                    workspace_limits[:, 1] + [0.02, 0.02, 0.08])  # clip to ensure convergence
        rot = get_euler_from_matrix(pose_world[:3, :3])
        if self.ACTION_MODE == 'yaw':
            action[3] *= np.deg2rad(30)  # yaw, limit maximum change in rotation
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
        # time2 = time.time()

        # jaw
        if self.block_gripper:
            action[4] = -1
        if action[4] < 0:
            self.psm1.close_jaw()
            self._activate(0)
        else:
            self.psm1.move_jaw(np.deg2rad(40))  # open jaw angle; can tune
            self._release_1()
            self._release_2()
        # time3 = time.time()
        # print("transform time: {:.4f}, IK time: {:.4f}, jaw time: {:.4f}, total time: {:.4f}"
        #       .format(time1 - time0, time2 - time1, time3 - time2, time3 - time0))

        # # only for demo
        # act = self.psm1.get_current_position().reshape(-1)
        # act = np.append(act, int(action[4] < 0))
        # self.actions.append(act)
    


    def _reset_ecm_pos(self):
        self.ecm.reset_joint(self.QPOS_ECM)

    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state(idx=0)

        pos, _ = get_link_pose(self.obj_id, -1)
        pos_2, _ = get_link_pose(self.obj_id_2, -1)
        object_pos = np.array(pos)
        object_pos_2 = np.array(pos_2)
        pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        pos_2, orn_2 = get_link_pose(self.obj_id_2, self.obj_link2)
        waypoint_pos = np.array(pos)
        waypoint_pos_2 = np.array(pos_2)
        # rotations
        waypoint_rot = np.array(p.getEulerFromQuaternion(orn))
        waypoint_rot_2 = np.array(p.getEulerFromQuaternion(orn_2))

        # relative position state
        object_rel_pos = object_pos - robot_state[0: 3]
        object_rel_pos_2 = object_pos_2 - robot_state[0: 3] 

        object_rel_pos = np.concatenate([object_rel_pos, object_rel_pos_2])
        object_pos = np.concatenate([object_pos, object_pos_2])
        waypoint_pos = np.concatenate([waypoint_pos, waypoint_pos_2])
        waypoint_rot = np.concatenate([waypoint_rot, waypoint_rot_2])

        # object/waypoint position
        achieved_goal = object_pos.copy() if not self._waypoint_goal else waypoint_pos.copy()


        observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_rot.ravel()  # achieved_goal.copy(),
        ])
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        return obs

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d_1 = goal_distance(achieved_goal[..., :3], desired_goal[..., :3])
        d_2 = goal_distance(achieved_goal[..., 3:], desired_goal[..., 3:])

        # print('[DEBUG]: d_1: {}, d_2: {}'.format(d_1, d_2))
        return np.logical_and(d_1 < self.distance_threshold, d_2 < self.distance_threshold).astype(np.float32)
    
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

    def _release_2(self):
        # release the object
        if self.block_gripper:
            return



        if self._contact_constraint_2 is not None:
            try:
                # p.removeConstraint(self._contact_constraint_2)
                # self._contact_constraint_2 = None
                # # enable collision
                # psm = self.psm1 
                # p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id_2,
                #                             linkIndexA=6, linkIndexB=-1, enableCollision=1)
                # p.setCollisionFilterPair(bodyUniqueIdA=psm.body, bodyUniqueIdB=self.obj_id_2,
                #                             linkIndexA=7, linkIndexB=-1, enableCollision=1)
                for i in range(1,self._contact_constraint_2+1):
                    p.changeConstraint(i, maxForce=0)
                    p.removeConstraint(i)
                if not p.getNumConstraints():
                    self._contact_constraint_2 = None
            except:
                pass
    
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

        if self._contact_constraint_2 is None:
            # the grippers activate; to check if they can grasp the object
            # TODO: check whether the constraint may cause side effects


            if self._meet_contact_constraint_requirement_2():
                psm = self.psm1 
                body_pose = p.getLinkState(psm.body, psm.EEF_LINK_INDEX)
                obj_pose_2 = p.getBasePositionAndOrientation(self.obj_id_2)
                world_to_body = p.invertTransform(body_pose[0], body_pose[1])
                obj_2_to_body = p.multiplyTransforms(world_to_body[0],
                                                   world_to_body[1],
                                                   obj_pose_2[0], obj_pose_2[1])

                self._contact_constraint_2 = p.createConstraint(
                    parentBodyUniqueId=psm.body,
                    parentLinkIndex=psm.EEF_LINK_INDEX,
                    childBodyUniqueId=self.obj_id_2,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=obj_2_to_body[0],
                    parentFrameOrientation=obj_2_to_body[1],
                    childFramePosition=(0, 0, 0),
                    childFrameOrientation=(0, 0, 0))
                # TODO: check the maxForce; very subtle
                p.changeConstraint(self._contact_constraint_2, maxForce=20)

        else:

            # self._contact_constraint is not None
            # the gripper grasp the object; to check if they remain contact
            psm = self.psm1
            points = p.getContactPoints(bodyA=psm.body, linkIndexA=6) \
                        + p.getContactPoints(bodyA=psm.body, linkIndexA=7)
            points = [point for point in points if point[2] == self.obj_id_2]
            remain_contact_2 = len(points) > 0

            # print('[DEBUG] remain_contact_2: {}'.format(remain_contact_2))


            if not remain_contact_2 and not self._contact_approx:
                # release the previously grasped object because there is no contact any more
                self._release_2()
        
        

        
        


if __name__ == "__main__":
    env = PickAndPlace(render_mode='human')  # create one process and corresponding env

    print(env.reset())

    # env.test()
    # env.close()
    time.sleep(20)
