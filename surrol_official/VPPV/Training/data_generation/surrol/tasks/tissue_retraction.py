import os
import time
import numpy as np
import pybullet as p
from MPM.mpm3d import MPM_Solver
from surrol.robots.ecm import Ecm
from surrol.const import ASSET_DIR_PATH
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import get_link_pose, reset_camera, step
from surrol.utils.robotics import get_matrix_from_pose_2d, add_value_to_tensor, set_value_of_tensor
import pickle
import cv2
from scipy.spatial.transform import Rotation as R

KEYS = {char: ord(char) for char in "rtqeadzxcuojlikanpm"}

class TissueRetract(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS1 = ((0.50, 0.60), (-0.05, 0.05), (0.675, 0.745))
    SCALING = 1.0  # sdf need to be recomputed if scaling is changed
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE = 3
    ACTION_SIZE=4
    counter=0
    haptic = True
    img_list={}
    # STEP_COUNT = 0

    use_soft_body = True
    mpm_solver = None

    # for evaluation with visulization
    # def __init__(self, render_mode='human'):
    #     super(BluntDissection, self).__init__(render_mode)


    def _env_setup(self):
        ecm_view_matrix =[2.7644696427853166e-12, -0.8253368139266968, 0.5646408796310425, 0.0, 1.0, 2.76391192918779e-12, -8.559629784479772e-13, 0.0, -8.541598418149166e-13, 0.5646408796310425, 0.8253368139266968, 0.0, -1.582376590869572e-11, 0.4536721706390381, -5.886332988739014,1.0]
        #ecm_view_matrix[14]=-5.25 #-5.0#-5.25
        ecm_view_matrix[14]=-0.97 #-4.7 #-5.25#-5.0#-5.25
        ecm_view_matrix[13]=0.07 #0.3-0.5
        # ecm_view_matrix[12]=0.004 #0.3-0.5
        shape_view_matrix=np.array(ecm_view_matrix).reshape(4,4)
        Tc = np.array([[1,  0,  0,  0],
                        [0, -1,  0,  0],
                        [0,  0, -1,  0],
                        [0,  0,  0,  1]]) 
        self._view_matrix=Tc@(shape_view_matrix.T)

        super(TissueRetract, self)._env_setup(goal_plot=False)

        self.has_object = False
        # self._waypoint_goal = True
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                    scaling=self.SCALING, view_matrix=ecm_view_matrix)
        self.ecm.reset_joint(self.QPOS_ECM)

        self.threshold=0.01
        # self.tex_id = p.loadTexture(
        #     os.path.join(ASSET_DIR_PATH, "texture/tissue_512.jpg")
        # )

        # p.setPhysicsEngineParameter(enableFileCaching=0,numSolverIterations=10,numSubSteps=128,contactBreakingThreshold=2)

        # robot
        # for psm, workspace_limits in ((self.psm1, self.workspace_limits1), (self.psm2, self.workspace_limits2)):
        #     pos = (workspace_limits[0].mean(),
        #            workspace_limits[1].mean(),
        #            workspace_limits[2].mean())
        #     # orn = p.getQuaternionFromEuler(np.deg2rad([0, np.random.uniform(-45, -135), -90]))
        #     orn = p.getQuaternionFromEuler(np.deg2rad([0, -90, -90]))  # reduce difficulty

        #     # psm.reset_joint(self.QPOS_PSM1)
        #     joint_positions = psm.inverse_kinematics((pos, orn), psm.EEF_LINK_INDEX)
        #     psm.reset_joint(joint_positions)

        workspace_limits = self.workspace_limits1
        temp=np.random.randint(-5,6)/100

        pos = (workspace_limits[0][0],
               temp,
               (workspace_limits[2][1]+ workspace_limits[2][1])/2 )
        
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)


        self.block_gripper = False  # set the constraint

        # needle
        # limits_span = (workspace_limits[:, 1] - workspace_limits[:, 0]) / 3
        # sample_space = workspace_limits.copy()
        # sample_space[:, 0] += limits_span
        # sample_space[:, 1] -= limits_span
        # obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm_RL.urdf'),
        #                     (0.01 * self.SCALING, 0, 0),
        #                     (0, 0, 0, 1),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        # p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        # self.obj_ids['rigid'].append(obj_id)  # 0
        # self.obj_id, self.obj_link1, self.obj_link2 = self.obj_ids['rigid'][0], 4, 5



        # while True:
        #     # open the jaw
        #     psm.open_jaw()
        #     # TODO: strange thing that if we use --num_env=1 with openai baselines, the qs vary before and after step!
        #     step(0.5)

        #     # set the position until the psm can grasp it
        #     pos_needle = np.random.uniform(low=sample_space[:, 0], high=sample_space[:, 1])
        #     pitch = np.random.uniform(low=-105., high=-75.)  # reduce difficulty
        #     orn_needle = p.getQuaternionFromEuler(np.deg2rad([-90, pitch, 90]))
        #     p.resetBasePositionAndOrientation(obj_id, pos_needle, orn_needle)

        #     # record the needle pose and move the psm to grasp the needle
        #     pos_waypoint, orn_waypoint = get_link_pose(obj_id, self.obj_link2)  # the right side waypoint
        #     orn_waypoint = np.rad2deg(p.getEulerFromQuaternion(orn_waypoint))
        #     p.resetBasePositionAndOrientation(obj_id, (0, 0, 0.01 * self.SCALING), (0, 0, 0, 1))

        #     # get the eef pose according to the needle pose
        #     orn_tip = p.getQuaternionFromEuler(np.deg2rad([90, -90 - orn_waypoint[1], 90]))
        #     pose_tip = [pos_waypoint + np.array([0.0015 * self.SCALING, 0, 0]), orn_tip]
        #     pose_eef = psm.pose_tip2eef(pose_tip)

        #     # move the psm
        #     pose_world = get_matrix_from_pose_2d(pose_eef)
        #     action_rcm = psm.pose_world2rcm(pose_world)
        #     success = psm.move(action_rcm)
        #     if success is False:
        #         continue
        #     step(1)
        #     p.resetBasePositionAndOrientation(obj_id, pos_needle, orn_needle)
        #     cid = p.createConstraint(obj_id, -1, -1, -1,
        #                              p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos_needle,
        #                              childFrameOrientation=orn_needle)
        #     psm.close_jaw()
        #     step(0.5)
        #     p.removeConstraint(cid)
        #     self._activate(0)
        #     self._step_callback()
        #     step(1)
        #     self._step_callback()
        #     if self._activated >= 0:
        #         break
        self._env_setup_soft()



    def _env_setup_soft(self):

        self.base_position = (
            self.workspace_limits1[0].mean()-0.092172 + (np.random.rand() - 0.5) * 0.02,
            self.workspace_limits1[1].mean()-0.028737 + (np.random.rand() - 0.5) * 0.02,
            self.workspace_limits1[2][0] - 0.01 ,
        )
        # self.needle_id = obj_id
        self.sdf_filename = os.path.join(ASSET_DIR_PATH, "needle/needle_sdf256.npy")
        # self.tex_id = p.loadTexture(
        #     os.path.join(ASSET_DIR_PATH, "texture/tissue_512.jpg")
        # )

        self.contact_point_index = 93
        soft_bpos = list(self.base_position)

        print(f"\033[91mSoft base position: {soft_bpos}\033[0m]")
        p.setRealTimeSimulation(0)
        # p.resetBasePositionAndOrientation(self.needle_id, [100,100,0],[0,0,0,1]) # remove needle from scene
        self.collision_obj_list = [[self.psm1.body, 6], [self.psm1.body, 7]]
    
        if self.mpm_solver is None:
            self.mpm_solver = MPM_Solver()
            self.mpm_solver.init_kernel(n_particles=80000, MAX_COLLISION_OBJECTS = len(self.collision_obj_list), gravity=9.8)
        else:
            self.mpm_solver.reset()
        # self.mpm_solver = MPM_Solver()
        # self.mpm_solver.init_kernel(n_particles=80000, MAX_COLLISION_OBJECTS = len(self.collision_obj_list), gravity=9.8)
        model_filename = "/research/dept7/yhlong/Science_Robotics/ar_surrol_datageneration/MPM/dissection_phantom_scale.ply"
        self.init_soft_body(
            collision_obj_list=self.collision_obj_list,
            model_filename= model_filename,
            collision_sdf_filename=self.sdf_filename,
            soft_body_base_position=soft_bpos,
            mpm_solver=self.mpm_solver,
            gs_model= False
        )# load model in MPM solver

        # self.pts_color = self.get_pts_color() #compute the color of each point
        # goal_idxs = self.get_goal_idx(pos = (0.44, 0.14, 0.16), threshold = 0.03) #select points using given position
        # self.pts_color[goal_idxs] = [1.0,.0,.0]
        # goal_idx = goal_idxs[0] #choose the first point as goal point

        E = 300000

        self.mpm_solver.set_parameters(s_E=E)

        self.soft_scale = 0.2
        self.use_point = False
        self.watch_defo = False
        self.tex_id = -1


    def reset(self):
        # reset scene in the corresponding file
        for i in self.particles_list:
            p.removeUserDebugItem(i)
        self.particles_list.clear()
        for i in self.meshes_list:
            p.removeBody(i)
        self.meshes_list.clear()

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))

        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)

        plane = p.loadURDF(os.path.join(ASSET_DIR_PATH, "plane/plane.urdf"), (0, 0, -0.001))
        wood = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/wood.jpg"))
        p.changeVisualShape(plane, -1, textureUniqueId=wood)
        self._env_setup()
        self.sim_step(
            gs_model=None,
            collision_obj_list=self.collision_obj_list,
            use_points=self.use_point,
            external_forces=False,
            texture_id=self.tex_id,
            add_collision_shape=False,
            threshold=0.05,
            scale=self.soft_scale,
            visualize_deformation=self.watch_defo,
            automate=False,
            debug=False,
            points_color=None,
            draw_soft_body= True
        )
        self.goal = self._sample_goal().copy()
        self._sample_goal_callback()

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        pos = self._waypoints[0][:3]
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)

        obs = self._get_obs(store_obs=False)

        steps, done = 0, False
        # obs_new = obs.copy()
        # obs_new['observation'][0:3] = self._camera2world_pos(obs['observation'][0:3])
        while not done and steps <= 9999900:
            tic = time.time()
            action = self.get_oracle_action(obs)
            print('\n -> step: {}, action: {}'.format(steps, np.round(action, 4)))
            # print('action:', action)
            obs, reward, done, info = self.step(action,store_obs=False)
            # obs_new['observation'][0:3] = self._camera2world_pos(obs['observation'][0:3])
            if action[3] == -0.5:
                break
            # time.sleep(0.05)
        print('\n -> Done initalize\n')
        self.block_gripper = True
        
        obs = self._get_obs()

        #print('-->surrol_env.reset: ', obs['observation'].requires_grad)
        return obs        

    
    def step(self, action: np.ndarray, store_obs=True):
        # action should have a shape of (action_size, )
        if len(action.shape) > 1:
            action = action.squeeze(axis=-1)
        #print(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        #print(action)
        # time0 = time.time()
        self._set_action(action, store_obs=store_obs)
        # time1 = time.time()
        # TODO: check the best way to step simulation
        # step(self._duration)
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        t0 = time.time()
        self.sim_step(
            gs_model=None,
            collision_obj_list=self.collision_obj_list,
            use_points=self.use_point,
            external_forces=False,
            texture_id=self.tex_id,
            add_collision_shape=False,
            threshold=0.05,
            scale=self.soft_scale,
            visualize_deformation=self.watch_defo,
            automate=False,
            debug=False,
            points_color=None,
            draw_soft_body= True
        )
        t1 = time.time()
        # p_loc = self.get_soft_body_goal_position(idx=0, scaling=self.soft_scale) #tracking point position using index
        # print(f"\033[91mFPS: {int(1/(t1-t0))}\t Key Point Location: {p_loc}\033[0m]")

        # time2 = time.time()
        # print(" -> robot action time: {:.6f}, simulation time: {:.4f}".format(time1 - time0, time2 - time1))
        # self._step_callback()
        obs = self._get_obs(store_obs)
        #print('-->surrol_env.setup: ', obs['observation'].requires_grad)
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),#self.goal),
        } if isinstance(obs, dict) else {'achieved_goal': None}
        
        if isinstance(obs, dict):
            #reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
            reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
        else:
            reward = self.compute_reward(obs, self.goal, info)
            
        # if len(self.actions) > 0:
        #     self.actions[-1] = np.append(self.actions[-1], [reward])  # only for demo
        return obs, reward, done, info

    def get_camera_matrix(self):
        return self._view_matrix

    def _get_obs(self, store_obs=True) -> dict:
        robot_state = self._get_robot_state(idx=0)
        
        
        # TODO: may need to modify
       
        pos = self.get_soft_body_goal_position(idx=self.contact_point_index, scaling=self.soft_scale) #tracking point position using index
        object_pos = np.array(pos)
        #print("ori obejct pose: ",object_pos)
        # pos, orn = get_link_pose(self.obj_id, self.obj_link1)
        waypoint_pos = np.array(pos)
        # rotations
        # waypoint_rot = np.array(p.getEulerFromQuaternion(orn))
        waypoint_rot = np.array([0.0,0.0,0.0])
        object_rel_pos = object_pos - robot_state[0: 3]
        
        # tip position
        # achieved_goal = np.array(get_link_pose(self.psm1.body, self.psm1.TIP_LINK_INDEX)[0])
        achieved_goal = object_pos.copy()    
        #print('waypoint_rot: ', waypoint_rot)
        observation = np.concatenate([
            robot_state, object_pos.ravel(), object_rel_pos.ravel(),
            waypoint_pos.ravel(), waypoint_rot.ravel()  # achieved_goal.copy(),
        ])
        
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        if not store_obs:
            return obs
        
        if self.counter==0:
            self.counter+=1                 
            return obs
        
        render_obs,seg,depth=self.ecm.render_image()
        render_obs=cv2.resize(render_obs,(320,240))
        seg=np.array((seg==5)).astype(int)

        #seg=np.resize(seg,(320,240))
        
        #print('depth : ', np.max(depth))
        
        seg = cv2.resize(seg, (320,240), interpolation =cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (320,240), interpolation =cv2.INTER_NEAREST)
        

        np.save('/research/dept7/yhlong/Science_Robotics/data/regress_data_soft_p2/test/seg_npy/seg_{}.npy'.format(self.counter),seg)
        np.save('/research/dept7/yhlong/Science_Robotics/data/regress_data_soft_p2/test/depth/depth_{}.npy'.format(self.counter),depth)
        cv2.imwrite('/research/dept7/yhlong/Science_Robotics/data/regress_data_soft_p2/test/img/img_{}.png'.format(self.counter),cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB))
        

        #img_size=seg.shape[0]
        #obs['depth']=depth.reshape(1, img_size, img_size).copy()
        #obs['seg']=seg.reshape(1, img_size, img_size).copy()
        
        #cv2.imwrite('/home/student/code/regress_data7/img/img_{}.png'.format(self.counter),cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB))
        
        #exit()
        self.img_list[self.counter]={}
        self.img_list[self.counter]['obs']=obs['observation']
        
        #self.img_list[self.counter]=obs['observation']

        
        if self.counter>=20001:
            with open('/research/dept7/yhlong/Science_Robotics/data/regress_data_soft_p2/test/img_obs.pkl',"wb") as f:
                pickle.dump(self.img_list,f)
            exit()

        return obs

    def _set_action(self, action: np.ndarray, store_obs=True):
        """
        delta_position (3), delta_theta (1) and open/close the gripper (1)
        in the world frame
        """
        assert len(action) == self.ACTION_SIZE, "The action should have the save dim with the ACTION_SIZE"
        # time0 = time.time()
        action = action.copy()  # ensure that we don't change the action outside of this scope

        action[:3] *= 0.01 * self.SCALING  # position, limit maximum change in position
        
        if store_obs:
            self.img_list[self.counter]['acs']=action
            self.counter+=1
        
        #print('action: ',action)
        # ECM action
 
        curr_rcm_pose=self.psm1.get_current_position()
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
        
        self.psm1.move(rcm_action)
        
        # jaw
        #print(action[6])
        #print('jaw: ',action[3])
        if self.block_gripper:
            action[3] = -1
        if action[3] < 0: #or action[3]==0:
            self.psm1.close_jaw()
            
            self._activate(0)
        else:
            self.psm1.move_jaw(np.deg2rad(40))  # open jaw angle; can tune
            self._release(0)

    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        # !!!! To find the index of the point within certain area !!!!
        # goal_idxs = self.get_goal_idx(pos = (0.44, 0.14, 0.16), threshold = 0.01)
        # print(goal_idxs[0])
        # print(goal_idxs)
        # self.contact_point_index = goal_idxs[0] #choose the first point as goal point

        goal = self.get_soft_body_goal_position(idx=self.contact_point_index, scaling=self.soft_scale) #tracking point position using index
        # goal=np.array([goal[0], goal[1],goal[2]+0.025* self.SCALING ])
        goal=np.array([goal[0], goal[1],goal[2]+0.02* self.SCALING ])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        #self._waypoints = [None, None, None, None, None]  # five waypoints
        self._waypoints = [None, None, None, None]  # four waypoints
        # self._waypoints = [None, None]
        # pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        pos_obj = self.get_soft_body_goal_position(idx=self.contact_point_index, scaling=self.soft_scale)
        self._waypoint_z_init = pos_obj[2]

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.008) * self.SCALING, 0., 0.5])  # approach
        # self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
        #                                pos_obj[2] + (-0.0007 + 0.0102+ 0.01) * self.SCALING, 0., 0.5])  # approach

        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., 0.5])  # approach    
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, 0., -0.5])  # grasp
        self._waypoints[3] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.02 + 0.006) * self.SCALING, 0., -0.5])  # grasp

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx or self.haptic is True:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING
        

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
        # action[3] = -0.5
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
            scale_factor = 0.5
            delta_pos *= scale_factor
            #action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_rot[0], delta_rot[1],delta_rot[2],waypoint[4]])
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2],waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4:
                self._waypoints[i] = None
            break

        # action = np.zeros(4)
        return action

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
        for index in range(int(80*self._duration)):
        # soft body simulation step
            # p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            if index == int(80*self._duration)-1:
                draw_soft_body = True
            else:
                draw_soft_body = False
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
            # num_step = int(240*self._duration)
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

    def _set_action_ecm(self, action):
        action *= 0.01 * self.SCALING
        pose_rcm = self.ecm.get_current_position()
        pose_rcm[:3, 3] += action
        pos, _ = self.ecm.pose_rcm2world(pose_rcm, "tuple")
        joint_positions = self.ecm.inverse_kinematics(
            (pos, None), self.ecm.EEF_LINK_INDEX
        )  # do not consider orn
        self.ecm.move_joint(joint_positions[: self.ecm.DoF])

    def _reset_ecm_pos(self):
        self.ecm.reset_joint(self.QPOS_ECM)

    # def get_ecm_image(self,image_width=640,image_height=512):
    #     self.ecm.render_image()
    #     _, _, rgb_image, depth_image, mask = p.getCameraImage(
    #         width=image_width,
    #         height=image_height,
    #         viewMatrix=self.ecm.view_matrix,
    #         projectionMatrix=self.ecm.proj_matrix,
    #         renderer=p.ER_BULLET_HARDWARE_OPENGL,
    #         )
    #     near,far = 0.01,1000 #default value in pybullet
    #     depth = far * near / (far - (far - near) * depth_image)
    #     return rgb_image, depth, mask



if __name__ == "__main__":
    env = TissueRetract(render_mode="human")  # create one process and corresponding env
    env.test()
    env.close()
    time.sleep(2)