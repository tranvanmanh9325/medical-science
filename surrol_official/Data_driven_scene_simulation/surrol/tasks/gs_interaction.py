import os
import cv2
import time
import math
import torch
# import cudacanvas
import numpy as np
import pybullet as p
from PIL import Image
from MPM.mpm3d import MPM_Solver
from MPM.utils import get_op_map, padding_from_opmap, bbox3d
import matplotlib.pyplot as plt

from gs.gaussian_renderer import render
from gs.scene.gaussian_model import GaussianModel

from surrol.robots.ecm import Ecm
from surrol.tasks.psm_env_mpm import PsmEnvMPM, PsmsEnvMPM
from surrol.const import ASSET_DIR_PATH
from surrol.tasks.ecm_env import EcmEnv, goal_distance
from surrol.utils.robotics import add_value_to_tensor, set_value_of_tensor
from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.utils.pybullet_utils import get_link_pose, reset_camera, wrap_angle, step
from haptic_src.touch_haptic import initTouch_right, closeTouch_right, getDeviceAction_right,startScheduler, stopScheduler

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


class Pipe:
    def __init__(
        self,
        convert_SHs_python: bool,
        compute_cov3D_python: bool,
        debug: bool,
        use_deformed_cov: bool,
    ):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug
        self.use_deformed_cov = use_deformed_cov


class tiny_camera:
    def __init__(
        self,
        FoVx,
        FoVy,
        image_height,
        image_width,
        world_view_transform,
        full_proj_transform,
        camera_center,
    ):
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_height = image_height
        self.image_width = image_width
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center
        self.zfar = 100.0
        self.znear = 0.01


class DebugCamera:

    def __init__(
        self, w, h, view, proj, up, forward, hori, verti, yaw, pitch, dist, target
    ):
        # self.width = w
        # self.height = h
        # self.view = np.array(view).reshape((4, 4))
        # self.projection = np.array(proj).reshape((4, 4))
        # self.cameraUp = np.array(up)
        # self.cameraForward = np.array(forward)
        # self.horizontal = np.array(hori)
        # self.vertical = np.array(verti)
        # self.yaw = yaw
        # self.pitch = pitch
        # self.dist = dist
        # self.target = np.array(target)

        self.image_height = h
        self.image_width = w
        self.world_view_transform = torch.tensor(
            np.array(view).reshape((4, 4)), dtype=torch.float32, device="cuda"
        )
        self.full_proj_transform = torch.tensor(
            np.array(proj).reshape((4, 4)), dtype=torch.float32, device="cuda"
        )
        self.full_proj_transform[2][3] = 1.0

        self.FoVx = 2 * math.atan(1.0 / self.full_proj_transform[0][0])
        self.FoVy = 2 * math.atan(1.0 / self.full_proj_transform[1][1])
        self.camera_center = torch.linalg.inv(self.world_view_transform)[3][:3]

        self.zfar = self.full_proj_transform[3][2] / (
            self.full_proj_transform[2][2] + 1.0
        )
        self.znear = self.full_proj_transform[3][2] / (
            self.full_proj_transform[2][2] - 1.0
        )

    def set_img_size(self, width, height):
        self.image_width = width
        self.image_height = height


class NeedlePick(PsmEnvMPM):
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
    def __init__(self, render_mode=None, cid=-1, use_soft_body=False, use_touch_device = False):
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
        self.use_touch_device = use_touch_device
        if self.use_touch_device:
            """===initialize haptic==="""
            initTouch_right()
            startScheduler()
            """======================="""
    
    def __del__(self):
        if self.use_touch_device:
            stopScheduler()
            closeTouch_right()

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

    def get_ecm_image(self,image_width=640,image_height=512, offset = None, soft_bpos = [0,0,0]):
        self.ecm.render_image(width=image_width,height=image_height, offset = offset)
        soft_bpos = np.array((1.9,0.1,3.0))
        eye = np.array((0.8, 0, 1.1)) + soft_bpos + np.array((0.3, 0.0, 0.0))
        target = np.array((0,0,0)) + soft_bpos
        view_mat = p.computeViewMatrix(cameraEyePosition=eye,
                                       cameraTargetPosition=target,
                                       cameraUpVector=(-1.7,0,1))
        _, _, rgb_image, depth_image, mask = p.getCameraImage(
            width=image_width,
            height=image_height,
            viewMatrix= view_mat,
            projectionMatrix=self.ecm.proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
        # print(self.ecm.view_matrix)
        near,far = 0.01,1000 #default value in pybullet
        depth = far * near / (far - (far - near) * depth_image)
        return rgb_image, depth, mask
    
    def reset(self,):
        self.mpm_solver.reset(self.backups)
        # self._set_action()



def grid_visualization(scale, base_position):
    print(base_position)
    p_array = mpm3d.pos.to_numpy().reshape((-1, 3))
    print(p_array.shape)
    num = len(p_array)
    p.addUserDebugPoints(
        p_array[:] * scale + base_position,
        [mpm3d.ORIANGE] * num,
        pointSize=0.5,
    )


def screenshot(cam: tiny_camera):
    view = tuple(cam.world_view_transform.detach().cpu().numpy().reshape((16,)))
    proj = tuple(cam.full_proj_transform.detach().cpu().numpy().reshape((16,)))
    # the view and proj matrix in getCameraImage is a list with 16 elements
    _, _, rgb_image, depth_image, _ = p.getCameraImage(
        width=cam.image_width,
        height=cam.image_height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )  # to add light: renderer=p.ER_BULLET_HARDWARE_OPENGL
    near = cam.znear
    far = cam.zfar
    depth = far * near / (far - (far - near) * depth_image)
    return rgb_image, depth


# def screenshot():
#     distance = 1.0
#     camera_pos = np.array(soft_bpos) + np.array([0.8 * distance, 0.3, 0.5 * distance])
#     tar_pos = np.array(soft_bpos) + np.array([0.0, 0.3, 0.0])
#     view_matrix = p.computeViewMatrix(
#         cameraEyePosition=camera_pos,
#         cameraTargetPosition=tar_pos,
#         cameraUpVector=[-1.0, 0.0, 0.5],
#     )
#     proj_matrix = p.computeProjectionMatrixFOV(
#         fov=50.0, aspect=1.0, nearVal=0.01, farVal=20
#     )
#     _, _, rgb_image, depth_map, _ = p.getCameraImage(
#         width=1024,
#         height=1024,
#         viewMatrix=view_matrix,
#         projectionMatrix=proj_matrix,
#         renderer=p.ER_BULLET_HARDWARE_OPENGL,
#     )  # to add light: renderer=p.ER_BULLET_HARDWARE_OPENGL
#     return rgb_image, depth_map


def keyboard_control(keys, action, env):
    global tex_id
    global swap_tex_id
    global use_point
    if rKey in keys and keys[rKey] & p.KEY_WAS_TRIGGERED:
        p.resetBasePositionAndOrientation(
            env.needle_id, env.needle_base_position, (0, 0, 0, 1.0)
        )
        env.reset()
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
    # if pKey in keys and keys[pKey] & p.KEY_IS_DOWN:
    #     use_point = not use_point
    if oKey in keys and keys[oKey] & p.KEY_IS_DOWN:
        # watch_defo = not watch_defo
        t = tex_id
        tex_id = swap_tex_id
        swap_tex_id = t

    # gripper control
    env.threshold = 0.001
    if cKey in keys:
        if keys[cKey] & p.KEY_IS_DOWN:
            set_value_of_tensor(action, 4, -0.5)
            env.threshold = 0.03
        elif keys[cKey] & p.KEY_WAS_RELEASED:
            set_value_of_tensor(action, 4, 1.0)

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

def col_mat(x):
    mat=np.array(x).reshape(4,4)
    mat[:,1:3]*=-1.0
    return torch.tensor(mat,dtype=torch.float32,device="cuda")

if __name__ == "__main__":
    scale = 3.0
    use_touch = False
    save_state = False
    save_img = False
    env = NeedlePick(render_mode="human", use_touch_device= use_touch)  # create one process and corresponding env

    soft_bposX_slider = p.addUserDebugParameter("Soft Body Base Position X", -5, 5, 1.579)
    soft_bposY_slider = p.addUserDebugParameter("Soft Body Base Position Y", -5, 5, -1.316)
    soft_bposZ_slider = p.addUserDebugParameter("Soft Body Base Position Z", -5, 5, 1.632)
    sliders = [soft_bposX_slider,soft_bposY_slider,soft_bposZ_slider]
    camX_slider = p.addUserDebugParameter("Cam Offset X", -0.1, 0.1, 0)
    camY_slider = p.addUserDebugParameter("Cam Offset Y", -0.1, 0.1, 0)
    camZ_slider = p.addUserDebugParameter("Cam Offset Z", -0.1, 0.1, 0)
    cam_sliders = [camX_slider, camY_slider, camZ_slider]
    p.setRealTimeSimulation(0)
    collision_obj_list = [[env.needle_id, -1], [env.psm1.body, 6], [env.psm1.body, 7]] # needle,gripper_1,gripper_2
    model_filename = "../../MPM/tissue.ply"
    gs_model = GaussianModel(sh_degree=3)
    gs_model.load_ply(model_filename)
    env.get_ecm_image()

    view_mat = col_mat(env.ecm.view_matrix)
    rot_mat = view_mat[:3,:3] + 0 #torch.tensor([[ 0.7071068,-0.7071068,0],[ 0.7071068,0.7071068,0],[0,0,1]],dtype= torch.float32,device="cuda") #rotate 45 degree along z-axis
    view_mat[:,0:2]*=-1.0
    gs_model.apply_rotation(rot_mat)
    # gs_model.apply_scaling(scale=0.5)

    pre_solver = MPM_Solver()
    pre_solver.initialize_from_gs_model(gs_model)

    # Gaussian Padding
    pre_solver.compute_grid_opacity()
    points = padding_from_opmap(
        op_map=get_op_map(pre_solver.opacity_field.to_numpy(), threshold=1.0), bottom=0.65
    )
    print(f"\033[91mPadding points shape: {points.shape}\033[0m]") #padding num is zero, not so right.
    padded_xyz = torch.cat([pre_solver.F_x.to_torch(device="cuda"), points], dim=0)
    pad_cov = torch.zeros((points.shape[0], 6), dtype=torch.float32, device="cuda")
    padded_cov = torch.cat([gs_model.get_covariance(), pad_cov], dim=0)

    #!!!!: attention
    # padded_xyz=pre_solver.F_x.to_torch(device="cuda")
    # padded_cov=gs_model.get_covariance()

    # del mpm_solver
    mpm_solver = MPM_Solver()
    mpm_solver.initialize_from_torch(xyz=padded_xyz, cov=padded_cov)
    normalize_base, normalize_scale = pre_solver.get_normalize_parameter()
    mpm_solver.set_normalize_parameter(normalize_base, normalize_scale)
    del pre_solver
    bbox = bbox3d(padding=3, x0=10, x1=70, y0=30, y1=70, z0=0, z1=100)
    mpm_solver.set_boundary_box(bbox=bbox)

    env.init_soft_body(
        collision_obj_list=collision_obj_list,
        model_filename=model_filename,
        collision_sdf_filename=env.sdf_filename,
        mpm_solver=mpm_solver,
        gs_model= True,
        scale=scale,
        backups={'xyz':padded_xyz,'cov':padded_cov,'normalize_base':normalize_base, 'normalize_scale':normalize_scale}
    )

    step_size = 0.1
    use_point = True
    watch_defo = False
    save_screen = False
    visualize_grid = False
    tex_id = -1
    swap_tex_id = env.tex_id
    action = np.zeros(env.ACTION_SIZE)

    if visualize_grid:
        grid_visualization(scale, soft_bpos)

    gs_cam = tiny_camera(
        FoVx=1.0239093368021417,
        FoVy=1.0239093368021417,
        image_height=512,
        image_width=640,
        world_view_transform=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device="cuda:0",
        ),
        full_proj_transform = None, #TIPS: full_proj_transform = proj_transform @ world_view_transform
        camera_center=torch.tensor([0.0, 0.0, 0.0], device="cuda:0"),
    )

    #[[1.7796, 0.0000, 0.0000, 0.0000],[0.0000, 2.2245, 0.0000, 0.0000],[0.0000, 0.0000, 1.0001, 1.0000],[0.0000, 0.0000, -0.0100, 0.0000]] #default value for full_proj_transform
    gs_cam.world_view_transform[:3,:3] = rot_mat #view matrix is placed in column order
    # proj = torch.tensor(np.array(env.ecm.proj_matrix).reshape(4,4), dtype=torch.float32, device="cuda")
    proj = torch.tensor([[ 1.7796,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  2.2245,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  1.0001,  1.0000],
        [ 0.0000,  0.0000, -0.0100,  0.0000]], device='cuda:0')
    gs_cam.full_proj_transform = proj @ gs_cam.world_view_transform
    # gs_cam.full_proj_transform[:2,:]*=-1.0

    pipe = Pipe(
        convert_SHs_python=False,
        compute_cov3D_python=True,
        debug=False,
        use_deformed_cov=True,
    )

    test_E = [4000]
    for E in test_E:
        cnt = 0
        J_list = []
        mpm_solver.set_parameters(s_E=E)
        # Simulation Loop
        while 1:
            # mpm_solver.save_ply(folder_path="./sim_results")
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
            keys = p.getKeyboardEvents()
            keyboard_control(keys, action, env)
            if use_touch:
                touch_control(action, env)
            if save_state:
                env.save_ply('./000.ply')
                exit(0)
            soft_bpos=[0,0,0]
            for i,v in enumerate(sliders):
                soft_bpos[i]=p.readUserDebugParameter(v)
            cam_offset = [.0,.0,.0]
            for i,v in enumerate(cam_sliders):
                cam_offset[i]=p.readUserDebugParameter(v)
            t0 = time.time()
            J_step = env.sim_step(
                gs_model=gs_model,
                collision_obj_list=collision_obj_list,
                soft_bpos=soft_bpos,
                use_points=use_point,
                external_forces=False,
                texture_id=tex_id,
                add_collision_shape=False,
                threshold=0.05,
                scale=scale,
                visualize_deformation=watch_defo,
                automate= False,
                draw_soft_body= True
            )
            surrol_rgb, surrol_depth, surrol_mask=env.get_ecm_image(offset=cam_offset,soft_bpos = soft_bpos)
            # 2. update gaussian according to simulation
            gs_model.set_xyz(
                mpm_solver.F_x.to_torch(device="cuda")[: gs_model.get_gs_number]
                / mpm_solver.normalize_scale
                + mpm_solver.normalize_base
            )  # inverse normalization
            gs_model.set_deformed_covariance(
                mpm_solver.Cov_deformed.to_torch(device="cuda")[
                    : gs_model.get_gs_number
                ]
            )
            # 3. Gaussian Rasterizer
            view_mat = col_mat(env.ecm.view_matrix)
            view_mat[:,0:2]*=-1.0
            rot_mat = view_mat[:3,:3] + 0
            gs_cam.world_view_transform[:3,:3] = rot_mat #view matrix is placed in column order
            # proj = torch.tensor(np.array(env.ecm.proj_matrix).reshape(4,4), dtype=torch.float32, device="cuda")
            gs_cam.full_proj_transform = proj @ view_mat
            # gs_cam.full_proj_transform[:2,:]*=-1.0
            with torch.no_grad():
                results = render(
                    gs_cam,
                    gs_model,
                    pipe=pipe,
                    bg_color=torch.tensor(
                        [0, 0, 0], dtype=torch.float32, device="cuda"
                    ),
                    scaling_modifier=1.0,
                )
            gs_image, gs_depth = results["render"], results["depth"]
            # 4. Get final rendering results
            surrol_rgb = (
                torch.tensor(surrol_rgb, device="cuda").permute(2, 0, 1)[:3, :, :]
                / 255.0
            )
            t1 = time.time()
            mask=torch.tensor(surrol_mask,device='cuda')
            mask= mask ==1 #torch.logical_or(mask==1, mask==5) #gripper and needle
            gs_image = torch.clamp(gs_image, min=0.0, max=1.0)
            gs_image[:,mask] = surrol_rgb[:,mask]
            cpu_img=np.transpose(gs_image.cpu().detach().numpy(),(1,2,0))
            

            keys = p.getKeyboardEvents()
            if pKey in keys and keys[pKey] & p.KEY_IS_DOWN:
                save_img = not save_img

            # save_path = f'./images/retraction/{cnt:06d}.png'
            # plt.imsave(save_path ,cpu_img)
            # print(f"{save_path} is saved.")
            cnt += 1

            cpu_img=cpu_img[...,[2,1,0]].copy() # RGB-->BGR
            # cv2.putText(cpu_img, f"FPS:{int(1/(t1-t0))}", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('surrol',cpu_img)
            cv2.waitKey(1)

        print(f"total steps: {cnt}")
        print(f"Rigid Body Simulation: {np.array(env.rigid_sim_cost[1:]).mean()}")
        print(f"Whole Soft Simulation: {np.array(env.soft_sim_cost[1:]).mean()}")
        print(f"Soft Body Simulation: {np.array(mpm_solver.co_st_list[1:]).mean()}")
        print(f"Collision: {np.array(mpm_solver.co_d_list[1:]).mean()}")
        print(f"Marching Cubes: {np.array(mpm_solver.mc[1:]).mean()}")
        print(f"Rendering: {np.array(env.rendering_cost[1:]).mean()}")
        print(f"GPU to CPU: {np.array(mpm_solver.g2c[1:]).mean()}ms")
