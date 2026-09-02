import os
import sys
import time
import math
import argparse
import numpy as np
import taichi as ti
import pybullet as p
from MPM.mpm3d import MPM_Solver
from MPM.utils import bbox3d
from surrol.robots.ecm import Ecm
from surrol.const import ASSET_DIR_PATH
from surrol.tasks.psm_env_mpm import PsmEnvMPM
from surrol.utils.pybullet_utils import get_link_pose, reset_camera, step, wrap_angle
from surrol.utils.robotics import get_matrix_from_pose_2d, add_value_to_tensor, set_value_of_tensor
from haptic_src.touch_haptic import initTouch_right, closeTouch_right, getDeviceAction_right,startScheduler, stopScheduler

KEYS = {char: ord(char) for char in "rtqeadzxcuojlikanpm"}

class StomachRetraction(PsmEnvMPM):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = (
        (0.50, 0.60),
        (-0.05, 0.05),
        (0.685, 0.745),
    )  # reduce tip pad contact
    SCALING = 5.0  # sdf need to be recomputed if scaling is changed
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE = 3
    haptic = True
    # STEP_COUNT = 0

    # TODO: grasp is sometimes not stable; check how to fix it
    def __init__(self, render_mode=None, cid=-1, use_soft_body=False, use_touch = False):
        super(StomachRetraction, self).__init__(render_mode, cid)
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
        )
        self.use_soft_body = use_soft_body
        self.obs = self._get_obs()
        self.use_touch = use_touch
        if use_touch:
            """===initialize haptic==="""
            initTouch_right()
            startScheduler()
            """======================="""

    def __del__(self):
        if self.use_touch:
            stopScheduler()
            closeTouch_right()


    def _env_setup(self):
        super(StomachRetraction, self)._env_setup()
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True
        self.threshold = 0.01
        # camera
        if self._render_mode == "human":
            # reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
            #              target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
            reset_camera(yaw=89.60, pitch=-56, dist=5.98, target=(-0.13, 0.03, -0.94))
        self.ecm = Ecm(
            (0.15, 0.0, 0.8524),  # p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
            scaling=self.SCALING,
        )
        self.ecm.reset_joint(self.QPOS_ECM)
        # p.setPhysicsEngineParameter(enableFileCaching=0,numSolverIterations=10,numSubSteps=128,contactBreakingThreshold=2)

        # robot
        workspace_limits = self.workspace_limits1
        pos = (
            workspace_limits[0][0],
            workspace_limits[1][1],
            (workspace_limits[2][1] + workspace_limits[2][0]) / 2,
        )
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics(
            (pos, orn), self.psm1.EEF_LINK_INDEX
        )
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # physical interaction
        self._contact_approx = False

        # tray pad
        board = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            globalScaling=self.SCALING,
                            useFixedBase=1)
        self.obj_ids['fixed'].append(board)  # 1

        # needle
        yaw = (np.random.rand() - 0.5) * np.pi
        self.needle_base_position = (
            workspace_limits[0].mean()
            + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
            workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1 + 0.1,
            workspace_limits[2][0] + 0.01,
        )
        needle_base_quaternion = p.getQuaternionFromEuler((0, 0, yaw))

        # suturing pad
        # color = np.array([2.8, 1.0, 0.7])
        # pad_id = p.createVisualShape(
        #     shapeType=p.GEOM_MESH,
        #     fileName=os.path.join(ASSET_DIR_PATH, 'suturing_pad/pad_t.obj'),
        #     rgbaColor=[color[0], color[1], color[2], 1.0],
        #     meshScale=2.0)
        # pad_pos = np.array(self.needle_base_position)
        # pad_pos[0] += 0.48
        # pad_pos[1] += 0.05 - 0.3
        # pad_pos[2] -= 0.035
        # p.createMultiBody(baseVisualShapeIndex=pad_id, basePosition=pad_pos)

        # for debug
        # self.needle_base_position = [1.0, 1.0, 0.0]
        # needle_base_quaternion = [0.0, 0.0, 0.0, 1.0]
        obj_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, "needle/needle_40mm_RL.urdf"),
            basePosition=self.needle_base_position,
            baseOrientation=needle_base_quaternion,
            useFixedBase=False,
            globalScaling=self.SCALING,
        )

        # test
        # p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm_RL.urdf'),
        #            basePosition=self.needle_base_position,
        #            baseOrientation=(0, 0, 0, 1.0),
        #            useFixedBase=False,
        #            globalScaling=10)

        # self.sdf_filename = os.path.join(ASSET_DIR_PATH,
        #                                  'needle/needle_sdf64_scale5.npy')
        self.sdf_filename = os.path.join(ASSET_DIR_PATH, "needle/needle_sdf256.npy")
        #!! why output twice?
        # print(obj_id)
        p.resetBasePositionAndOrientation(
            obj_id, self.needle_base_position, needle_base_quaternion
        )
        # print(p.getBasePositionAndOrientation(obj_id))
        # print(self.needle_base_position)
        # exit(0)
        self.needle_id = obj_id
        self.base_position = (
            workspace_limits[0].mean()-0.05,
            workspace_limits[1].mean(),
            workspace_limits[2][0] + 0.01,
        )

        # load texture
        self.tex_id = p.loadTexture(
            os.path.join(ASSET_DIR_PATH, "texture/tissue_512.jpg")
        )

        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids["rigid"].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids["rigid"][0], 1

    def _sample_goal(self) -> np.ndarray:
        """Samples a new goal and returns it."""
        workspace_limits = self.workspace_limits1
        # goal = np.array([
        #     workspace_limits[0].mean() +
        #     0.01 * np.random.randn() * self.SCALING,
        #     workspace_limits[1].mean() +
        #     0.01 * np.random.randn() * self.SCALING,
        #     workspace_limits[2][1] - 0.04 * self.SCALING
        # ])

        # fixed goal position
        goal = np.array(
            [
                workspace_limits[0].mean() - 0.1,
                workspace_limits[1].mean() + 0.2,
                workspace_limits[2][1] - 0.05,
            ]
        )
        return goal.copy()

    def _sample_goal_callback(self):
        """Define waypoints"""
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None]  # four waypoints
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        self._waypoint_z_init = pos_obj[2]
        orn = p.getEulerFromQuaternion(orn_obj)
        orn_eef = get_link_pose(self.psm1.body, self.psm1.EEF_LINK_INDEX)[1]
        orn_eef = p.getEulerFromQuaternion(orn_eef)
        yaw = (
            orn[2]
            if abs(wrap_angle(orn[2] - orn_eef[2]))
            < abs(wrap_angle(orn[2] + np.pi - orn_eef[2]))
            else wrap_angle(orn[2] + np.pi)
        )  # minimize the delta yaw
        # press and grasp
        goal_x = self.goal[0] + 0.07
        goal_y = self.goal[1] - 0.08
        self._waypoints[0] = np.array(
            [goal_x, goal_y, self.goal[2] - 0.28, yaw, 0.5]
        )  # approach

        self._waypoints[1] = np.array(
            [goal_x, goal_y, self.goal[2] - 0.28, yaw, -0.5]
        )  # approach

        self._waypoints[1] = np.array(
            [goal_x, goal_y, self.goal[2] - 0.12, yaw, -0.5]
        )  # grasp

        self._waypoints[3] = np.array(
            [goal_x, goal_y, self.goal[2] - 0.12, yaw, -0.5]
        )  # pull

    def _meet_contact_constraint_requirement(self):
        # add a contact constraint to the grasped block to make it stable
        if self._contact_approx or self.haptic is True:
            return True  # mimic the dVRL setting
        else:
            pose = get_link_pose(self.obj_id, self.obj_link1)
            return pose[0][2] > self._waypoint_z_init + 0.005 * self.SCALING

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # four waypoints executed in sequential order
        action = np.zeros(5)
        action[4] = -0.5
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            delta_pos = (waypoint[:3] - obs["observation"][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs["observation"][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array(
                [delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]]
            )
            if (
                np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4
                and np.abs(delta_yaw) < 1e-2
            ):
                self._waypoints[i] = None
            break

        return action

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

    def get_ecm_image(self,image_width=640,image_height=512):
        self.ecm.render_image()
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=0.8* self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2,
        )
        # self.ecm.view_matrix,
        _, _, rgb_image, depth_image, mask = p.getCameraImage(
            width=image_width,
            height=image_height,
            viewMatrix= view_matrix,
            projectionMatrix=self.ecm.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
        near,far = 0.01,1000 #default value in pybullet
        depth = far * near / (far - (far - near) * depth_image)
        return rgb_image, depth, mask



def keyboard_control(keys, action, env):
    global tex_id
    global swap_tex_id
    global use_point
    if KEYS['r'] in keys and keys[KEYS['r']] & p.KEY_WAS_TRIGGERED:
        p.resetBasePositionAndOrientation(
            env.needle_id, env.needle_base_position, (0, 0, 0, 1.0)
        )
    # reset soft body
    if KEYS['t'] in keys and keys[KEYS['t']] & p.KEY_WAS_TRIGGERED:
        mpm3d.reset(filename=model_filename)

    # psm1 keyboard control
    if KEYS['q'] in keys and keys[KEYS['q']] & p.KEY_IS_DOWN:
        add_value_to_tensor(action, 0, step_size)
    if KEYS['e'] in keys and keys[KEYS['e']] & p.KEY_IS_DOWN:
        add_value_to_tensor(action, 0, -step_size)
    if KEYS['a'] in keys and keys[KEYS['a']] & p.KEY_IS_DOWN:
        add_value_to_tensor(action, 1, -step_size)
    if KEYS['d'] in keys and keys[KEYS['d']] & p.KEY_IS_DOWN:
        add_value_to_tensor(action, 1, step_size)
    if KEYS['z'] in keys and keys[KEYS['z']] & p.KEY_IS_DOWN:
        add_value_to_tensor(action, 2, step_size)
    if KEYS['x'] in keys and keys[KEYS['x']] & p.KEY_IS_DOWN:
        add_value_to_tensor(action, 2, -step_size)
    # psm2 keyboard control
    if KEYS['u'] in keys and keys[KEYS['u']]&p.KEY_IS_DOWN:
        add_value_to_tensor(action,0+5,step_size)
    if KEYS['o'] in keys and keys[KEYS['o']]&p.KEY_IS_DOWN:
        add_value_to_tensor(action,0+5,-step_size)
    if KEYS['j'] in keys and keys[KEYS['j']]&p.KEY_IS_DOWN:
        add_value_to_tensor(action,1+5,-step_size)
    if KEYS['l'] in keys and keys[KEYS['l']]&p.KEY_IS_DOWN:
        add_value_to_tensor(action,1+5,step_size)
    if KEYS['i'] in keys and keys[KEYS['i']]&p.KEY_IS_DOWN:
        add_value_to_tensor(action,2+5,step_size)
    if KEYS['k'] in keys and keys[KEYS['k']]&p.KEY_IS_DOWN:
        add_value_to_tensor(action,2+5,-step_size)

    # control visualization
    # if KEYS['p'] in keys and keys[KEYS['p']] & p.KEY_IS_DOWN:
    #     use_point = not use_point
    # if KEYS['o'] in keys and keys[KEYS['o']] & p.KEY_IS_DOWN:
    #     # watch_defo = not watch_defo
    #     t = tex_id
    #     tex_id = swap_tex_id
    #     swap_tex_id = t

    # gripper control
    env.threshold = 0.01

    if KEYS['c'] in keys:
        if keys[KEYS['c']] & p.KEY_IS_DOWN:
            set_value_of_tensor(action, 4, -0.5)
            env.threshold = 0.03
        elif keys[KEYS['c']] & p.KEY_WAS_RELEASED:
            set_value_of_tensor(action, 4, 1.0)
    
    if KEYS['n'] in keys:
        if keys[KEYS['n']]&p.KEY_IS_DOWN:
            set_value_of_tensor(action,4+5,-0.5)
        elif keys[KEYS['n']]&p.KEY_WAS_RELEASED:
            set_value_of_tensor(action,4+5,1.0)

    env._set_action(action)
    action.fill(0)

def touch_control(action, env):
    retrived_action = np.array([0, 0, 0, 0, 0], dtype = np.float32)
    getDeviceAction_right(retrived_action)
    """=====haptic right====="""
    """retrived_action -> x,y,z, angle, buttonState(0,1,2)"""
    if retrived_action[4] == 2:
        '''Clutch'''
        action[0] = 0
        action[1] = 0
        action[2] = 0
        action[3] = 0            
    else:
        '''Control PSM'''
        action[0] = retrived_action[2]*0.2
        action[1] = retrived_action[0]*0.2
        action[2] = retrived_action[1]*0.2
        action[3] = -retrived_action[3]/math.pi*180*0.1

    '''Grasping'''
    env.threshold=0.01
    if retrived_action[4] == 0:
        action[4] = 1
    if retrived_action[4] == 1:
        action[4] = -0.1
        env.threshold=0.03

    env._set_action(action)
    action.fill(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='I am a dog.')
    parser.add_argument('-E', '--youngs', type=float, default=1000)
    parser.add_argument('-v', '--visual_way', type=str, default="deformation")
    parser.add_argument('--view1', action='store_true')
    args = parser.parse_args()

    scale = 1.0
    use_touch = False
    env = StomachRetraction(render_mode="human", use_touch = use_touch)  # create one process and corresponding env
    soft_bpos = list(env.base_position)
    soft_bpos[0] -= 0.3
    soft_bpos[1] -= 0.15
    # soft_bpos[2] -= 0.1
    print(f"\033[91mSoft base position: {soft_bpos}\033[0m]")
    p.setRealTimeSimulation(0)
    p.resetBasePositionAndOrientation(env.needle_id, [100,100,0],[0,0,0,1]) # remove needle from scene
    collision_obj_list = [[env.psm1.body, 6], [env.psm1.body, 7]]

    particle_number = 100_000

    mpm_solver = MPM_Solver()
    mpm_solver.init_kernel(n_particles = particle_number, MAX_COLLISION_OBJECTS = len(collision_obj_list), gravity=9.8, p_rho=1000)
    bbox = bbox3d(padding=3, x0=0, x1=100, y0=0, y1=100, z0=0, z1=100)
    mpm_solver.set_boundary_box(bbox=bbox)
    env.init_soft_body(
        collision_obj_list=collision_obj_list,
        model_filename= None,
        collision_sdf_filename=env.sdf_filename,
        mpm_solver=mpm_solver,
        gs_model= False,
        scale=scale
    )# load model in MPM solver

    select_point = ti.Vector([0.3, 0.4, 0.02])
    select_scale = 0.003
    # given_velocity = ti.Vector([0.5, 0.5, 0.0])

    ## Apply Velocity on Selected Area
    # given_velocity = ti.Vector([0.0, 0.0, 1.0])
    # mpm_solver.set_velocity(given_velocity)
    # mpm_solver.select(select_point, select_scale)

    # pts_color = env.get_pts_color() #compute the color of each point
    # goal_idxs = env.get_goal_idx(pos = (0.44, 0.14, 0.16), threshold = 0.03) #select points using given position
    # pts_color[goal_idxs] = [1.0,.0,.0]
    # goal_idx = goal_idxs[0] #choose the first point as goal point

    step_size = 0.1
    
    use_point = False
    watch_defo = False
    save_screen = False
    visualize_grid = False
    tex_id = env.tex_id
    swap_tex_id = -1
    action = np.zeros(env.ACTION_SIZE)

    E = args.youngs
    visualize_way = args.visual_way
    # use_view1 = args.view1
    # print(use_view1)
    # exit(0)

    # Deformation Visualization Using Taichi
    # window = ti.ui.Window("3D Render", (1024, 1024), vsync=True)
    # canvas = window.get_canvas()
    # canvas.set_background_color((1, 1, 1))
    # scene = ti.ui.Scene()
    # camera = ti.ui.Camera()

    # if args.view1:
    #     camera.up(0,-1,0)
    #     camera.position(0.2, 0.6, 0.8) #view 1
    #     view = "view1"
    # else:
    #     camera.up(0,0,1)
    #     camera.position(0.2, 1.0, 0.5) #view 2
    #     view = "view2"
    
    # folder_path = os.path.join('./figures', f'{int(E)}', view, visualize_way)
    # os.makedirs(folder_path, exist_ok= True)
    # camera.lookat(0.2, 0.3, 0.2)
    # scene.set_camera(camera)
    # scene.ambient_light((0.8, 0.8, 0.8))

    mpm_solver.set_parameters(s_E=E)
    save_img = False
    pKey = ord("p")
    import matplotlib.pyplot as plt
    # Simulation Loop
    cnt = 0
    while 1:
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        keys = p.getKeyboardEvents()
        keyboard_control(keys, action, env)
        if use_touch:
            touch_control(action, env)
        t0 = time.time()
        J_step = env.sim_step(
            gs_model=None,
            collision_obj_list=collision_obj_list,
            soft_bpos=soft_bpos,
            use_points=use_point,
            external_forces=False,
            texture_id=tex_id,
            add_collision_shape=False,
            threshold=0.05,
            scale=scale,
            visualize_deformation = True,
            automate=False,
            debug=True,
            draw_soft_body=True,
            scene= None,
            visualize_way=visualize_way
        )
        t1 = time.time()
        # p_loc = env.get_soft_body_goal_position(idx=goal_idx, soft_bpos=soft_bpos) #tracking point position using index
        rgb, _, _ = env.get_ecm_image()

        # point = ti.Vector.field(3, float, 1)
        # point[0] = ti.Vector([0,0,0])
        # scene.particles(point, radius=0.1, color=(0.934, 0.33, 0.23))

        # canvas.scene(scene)
        # img = np.ascontiguousarray(np.transpose(window.get_image_buffer_as_numpy(), (1,0,2))[::-1,:,:])

        
        # if cnt>30:
        #     exit(0)
        #     pass
        # elif cnt%1==0:
        #     # plt.imsave(os.path.join(folder_path,f"points_{particle_number}_{cnt:06d}.png"), img)
        #     plt.imsave(os.path.join(folder_path,f"{cnt:06d}.png"), img)
        #     pass

        # window.show()

        if pKey in keys and keys[pKey] & p.KEY_IS_DOWN:
            save_img = not save_img
        if save_img:
            plt.imsave(f'./images/retraction/{cnt:06d}.png',rgb)
        cnt += 1
        sys.stdout.write(f"\rFPS: {int(1/(t1-t0))}")
        sys.stdout.flush()