import os
import time
import trimesh
import pymeshlab
import numpy as np
import pybullet as p
import MPM.mpm3d as mpm3d
import matplotlib.pyplot as plt

from surrol.robots.ecm import Ecm
from surrol.tasks.psm_env import PsmEnv,PsmsEnv
from surrol.const import ASSET_DIR_PATH
from surrol.tasks.ecm_env import EcmEnv, goal_distance
from surrol.utils.robotics import add_value_to_tensor,set_value_of_tensor
from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.utils.pybullet_utils import get_link_pose, reset_camera, wrap_angle, step

rKey = ord("r")
tKey = ord("t")
qKey = ord("q")
eKey = ord("e")
aKey = ord("a")
zKey = ord("z")
dKey = ord("d")
xKey = ord("x")
cKey = ord("c")
pKey = ord("p")
oKey = ord("o")
nKey = ord("n")
mKey = ord("m")


class NeedlePick(PsmEnv):
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
    def __init__(self, render_mode=None, cid=-1, use_soft_body=False):
        super(NeedlePick, self).__init__(render_mode, cid)
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

    def _env_setup(self):
        super(NeedlePick, self)._env_setup()
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
        # obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH,
        #                                  'suturing_pad/pad.urdf'),
        #                     np.array(self.POSE_TRAY[0]) * self.SCALING,
        #                     baseOrientation=(0, 1, 0, 0),
        #                     globalScaling=5.0)
        # self.obj_ids['fixed'].append(obj_id)  # 1

        # needle
        yaw = (np.random.rand() - 0.5) * np.pi
        self.needle_base_position = (
            workspace_limits[0].mean()
            + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
            workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1 + 0.1,
            workspace_limits[2][0] + 0.01,
        )
        needle_base_quaternion = p.getQuaternionFromEuler((0, 0, yaw))

        obj_id = p.loadURDF(
            os.path.join(ASSET_DIR_PATH, "needle/needle_40mm_RL.urdf"),
            basePosition=self.needle_base_position,
            baseOrientation=needle_base_quaternion,
            useFixedBase=False,
            globalScaling=self.SCALING,
        )
        self.sdf_filename = os.path.join(ASSET_DIR_PATH, "needle/needle_sdf256.npy")
        p.resetBasePositionAndOrientation(
            obj_id, self.needle_base_position, needle_base_quaternion
        )
        self.needle_id = obj_id

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

        # self._waypoints[0] = np.array([
        #     pos_obj[0], pos_obj[1],
        #     pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5
        # ])  # approach
        # self._waypoints[1] = np.array([
        #     pos_obj[0], pos_obj[1],
        #     pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, 0.5
        # ])  # approach
        # self._waypoints[2] = np.array([
        #     pos_obj[0], pos_obj[1],
        #     pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, -0.5
        # ])  # grasp
        # self._waypoints[3] = np.array([
        #     self.goal[0], self.goal[1], self.goal[2] + 0.0102 * self.SCALING,
        #     yaw, -0.5
        # ])  # lift up

        # press and grasp
        goal_x=self.goal[0]+0.07
        goal_y=self.goal[1]-0.08
        self._waypoints[0] = np.array(
            [goal_x, goal_y, self.goal[2]-0.28, yaw, 0.5]
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

def grid_visualization(scale,base_position):
    print(base_position)
    p_array=mpm3d.pos.to_numpy().reshape((-1,3))
    print(p_array.shape)
    num=len(p_array)
    p.addUserDebugPoints(
                p_array[:] * scale + base_position,
                [mpm3d.ORIANGE] * num,
                pointSize=0.5,
            )
    
def screenshot(watch_defo):
    distance = 1.0
    camera_pos = np.array(soft_bpos) + np.array(
        [0.8 * distance, 0.3, 0.5 * distance]
    )
    tar_pos = np.array(soft_bpos) + np.array([0.0, 0.3, 0.0])
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_pos,
        cameraTargetPosition=tar_pos,
        cameraUpVector=[-1.0, 0.0, 0.5],
    )
    proj_matrix = p.computeProjectionMatrixFOV(
        fov=50.0, aspect=1.0, nearVal=0.01, farVal=20
    )
    _, _, rgb_image, _, _ = p.getCameraImage(
        width=1024,
        height=1024,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )  # to add light: renderer=p.ER_BULLET_HARDWARE_OPENGL
    if watch_defo:
        if not os.path.exists(f"./screenshots/{mpm3d.E}"):
            os.mkdir(f"./screenshots/{mpm3d.E}")
        plt.imsave(f"./screenshots/{mpm3d.E}/{cnt}.png", rgb_image)
    else:
        if not os.path.exists(f"./screenshots1/{mpm3d.E}"):
            os.mkdir(f"./screenshots1/{mpm3d.E}")
        plt.imsave(f"./screenshots1/{mpm3d.E}/{cnt}.png", rgb_image)

def keyboard_control(keys,action,env):
    if rKey in keys and keys[rKey] & p.KEY_WAS_TRIGGERED:
                p.resetBasePositionAndOrientation(
                    env.needle_id, env.needle_base_position, (0, 0, 0, 1.0)
                )
            # reset soft body
    if tKey in keys and keys[tKey] & p.KEY_WAS_TRIGGERED:
        mpm3d.reset(filename=model_filename)
    # psm keyboard control
    if qKey in keys and keys[qKey] & p.KEY_IS_DOWN:
        add_value_to_tensor(action, 0, step_size)
    if eKey in keys and keys[eKey] & p.KEY_IS_DOWN:
        add_value_to_tensor(action, 0, -step_size)
    if aKey in keys and keys[aKey] & p.KEY_IS_DOWN:
        add_value_to_tensor(action, 1, -step_size)
    if dKey in keys and keys[dKey] & p.KEY_IS_DOWN:
        add_value_to_tensor(action, 1, step_size)
    if zKey in keys and keys[zKey] & p.KEY_IS_DOWN:
        add_value_to_tensor(action, 2, step_size)
    if xKey in keys and keys[xKey] & p.KEY_IS_DOWN:
        add_value_to_tensor(action, 2, -step_size)

    if nKey in keys and keys[nKey] & p.KEY_IS_DOWN:
        E -= 1000
    if mKey in keys and keys[mKey] & p.KEY_IS_DOWN:
        E += 1000

    # control visualization
    if pKey in keys and keys[pKey] & p.KEY_IS_DOWN:
        use_point = not use_point
    if oKey in keys and keys[oKey] & p.KEY_IS_DOWN:
        # watch_defo = not watch_defo
        t=tex_id
        tex_id=swap_tex_id
        swap_tex_id=t

    # gripper control
    env.threshold=0.01
    if cKey in keys:
        if keys[cKey] & p.KEY_IS_DOWN:
            set_value_of_tensor(action, 4, -0.5)
            env.threshold=0.03
        elif keys[cKey] & p.KEY_WAS_RELEASED:
            set_value_of_tensor(action, 4, 1.0)

    env._set_action(action)
    action.fill(0)

if __name__ == "__main__":
    scale=1.0
    env = NeedlePick(render_mode="human")  # create one process and corresponding env
    soft_bpos = list(env.needle_base_position)
    soft_bpos[2] -= 0.1
    soft_bpos[0] -= 0.4
    soft_bpos[1] -= 0.2

    # change a way to organize soft body simulation
    p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # needle,gripper_1,gripper_2
    collision_obj_list = [[env.needle_id, -1], [env.psm1.body, 6], [env.psm1.body, 7]]
    model_filename =None# os.path.join(ASSET_DIR_PATH, "./phantom/pad.npy")
    env.init_soft_body(
        collision_obj_list=collision_obj_list,
        model_filename=model_filename,
        collision_sdf_filename=env.sdf_filename,
        soft_body_base_position=soft_bpos,
    )
    
    step_size = 0.1
    use_point = False
    watch_defo = False
    save_screen = False
    visualize_grid=False
    tex_id=-1
    swap_tex_id=env.tex_id
    action = np.zeros(env.ACTION_SIZE)

    # for visualization
    p.resetBasePositionAndOrientation(env.needle_id, (0.0, 0.0, 0.0), (0, 0, 0, 1.0))
    # remove surgical robot
    # p.resetBasePositionAndOrientation(1, (0.0, 0.0, 0.0), (0, 0, 0, 1.0))

    if visualize_grid:
        grid_visualization(scale,soft_bpos)
    # test_E = [1000,8000,15000,100000]

    test_E = [10000]
    for E in test_E:
        cnt = 0
        J_list = []
        mpm3d.set_parameters(s_E=E)
        while 1:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            keys = p.getKeyboardEvents()
            keyboard_control(keys,action,env)

            t0 = time.time()
            J_step = env.sim_step(
                collision_obj_list=collision_obj_list,
                soft_body_base_position=soft_bpos,
                use_points=use_point,
                external_forces=False,
                texture_id=tex_id,
                add_collision_shape=False,
                threshold=0.05,
                scale=scale,
                visualize_deformation=watch_defo,
                automate=False,
            )
            t1 = time.time()
            print(f"FPS: {int(1/(t1-t0))}")

            if save_screen:
                screenshot(watch_defo)

            cnt += 1

        print(f"total steps: {cnt}")
        print(f"Rigid Body Simulation: {np.array(env.rigid_sim_cost[1:]).mean()}")
        print(f"Whole Soft Simulation: {np.array(env.soft_sim_cost[1:]).mean()}")

        print(f"Soft Body Simulation: {np.array(mpm3d.co_st_list[1:]).mean()}")
        print(f"Collision: {np.array(mpm3d.co_d_list[1:]).mean()}")
        print(f"Marching Cubes: {np.array(mpm3d.mc[1:]).mean()}")
        print(f"Rendering: {np.array(env.rendering_cost[1:]).mean()}")
        print(f"GPU to CPU: {np.array(mpm3d.g2c[1:]).mean()}ms")

        mpm3d.reset(filename=model_filename)
