import time, os
import socket

import gym
from gym import spaces
from gym.utils import seeding

import pybullet as p
import pybullet_data
import pkgutil
from surrol.utils.pybullet_utils import (
    step,
    render_image,
)
import numpy as np
from surrol.const import ROOT_DIR_PATH, ASSET_DIR_PATH

import MPM.mpm3d as mpm3d

RENDER_HEIGHT = 480  # train
RENDER_WIDTH = 640
# RENDER_HEIGHT = 1080  # record
# RENDER_WIDTH = 1920

particles_list = []
meshes_list = []


def soft_body_step(i_rot,
                   i_pos,
                   USE_POINTS=False,
                   base_position=(0, 0, 0),
                   scale=1.0,
                   use_debug=False):
    '''
    One simulation step of soft body.

    i_rot: inverse rotation matrix

    i_pos: inverse advection offset

    USE_POINTS: draw points or meshes

    base_position: the start position of the soft body

    scale: scale soft body

    use_debug: whether visualize euler grid and output inverse transform matrix
    '''
    if use_debug:
        grid_pos, grid_sdf = mpm3d.debug_step(scale, i_rot, i_pos)
        grid_pos = grid_pos[:8000]
        grid_sdf = grid_sdf[:8000]
        inverse_trans_pos = np.moveaxis(
            i_rot @ np.moveaxis(grid_pos, 0, -1) * scale, 0, -1) + i_pos
    else:
        mpm3d.step(scale, i_rot, i_pos)

    if use_debug:
        print("DEBUG:#################################")
        print(grid_sdf.max(), grid_sdf.min())
        print(i_rot)
        print(i_pos)
        print("DEBUG:#################################")

    for i in particles_list:
        p.removeUserDebugItem(i)
    particles_list.clear()

    if USE_POINTS:
        particles_array = mpm3d.F_x.to_numpy()
        num_particles = len(particles_array)
        max_vertices = 8000  #the maximum vertices can be drawn on apple mac
        idx = 0
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        while idx < num_particles:
            upper_bound = -1
            if idx + max_vertices > num_particles:
                upper_bound = num_particles
            else:
                upper_bound = idx + max_vertices
            pld_id = p.addUserDebugPoints(
                particles_array[idx:upper_bound] * scale + base_position,
                [mpm3d.ORIANGE for _ in range(upper_bound - idx)],
                pointSize=10)
            particles_list.append(pld_id)
            idx = upper_bound
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        # On other platform(Linux, Windows) ,I think it's ok to directly invoke the instruction below.
        # pld_id = p.addUserDebugPoints(
        #     mpm3d.F_x.to_numpy() * scale + base_position,
        #     [mpm3d.ORIANGE for _ in range(mpm3d.n_particles)],
        #     pointSize=10)
        # particles_list.append(pld_id)
    else:
        for i in meshes_list:
            p.removeBody(i)
        meshes_list.clear()
        vtx, idx = mpm3d.get_mesh(smooth_scale=0.5)
        vtx = vtx / (mpm3d.n_grid * 1.0)
        vertices = vtx * scale + base_position
        vid = p.createVisualShape(shapeType=p.GEOM_MESH,
                                  vertices=vertices,
                                  indices=idx,
                                  rgbaColor=[0.93, 0.33, 0.23, 1.0])
        body_id = p.createMultiBody(baseVisualShapeIndex=vid)
        meshes_list.append(body_id)

    #debug: visualize euler grid
    if use_debug:
        pld_id_0 = p.addUserDebugPoints(grid_pos * scale + base_position,
                                        [(i * 1.0, 0.0, 0.0)
                                         for i in grid_sdf],
                                        pointSize=10)
        pld_id_1 = p.addUserDebugPoints(inverse_trans_pos,
                                        [(1.0, 0.0, 0.0) for i in grid_sdf],
                                        pointSize=10)
        particles_list.append(pld_id_0)
        particles_list.append(pld_id_1)


class SurRoLEnv(gym.Env):
    """
    A gym Env wrapper for SurRoL.
    refer to: https://github.com/openai/gym/blob/master/gym/core.py
    """

    metadata = {'render.modes': ['human', 'rgb_array', 'img_array']}

    def __init__(self, render_mode: str = None, cid: int = -1):
        # rendering and connection options
        self._render_mode = render_mode
        # render_mode = 'human'
        # if render_mode == 'human':
        #     self.cid = p.connect(p.SHARED_MEMORY)
        #     if self.cid < 0:
        if render_mode == 'human':
            if cid < 0:
                self.cid = p.connect(p.GUI)
            else:
                self.cid = cid
            # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            if cid < 0:
                self.cid = p.connect(p.DIRECT)
            else:
                self.cid = cid
            # See PyBullet Quickstart Guide Synthetic Camera Rendering
            # TODO: no light when using direct without egl
            if socket.gethostname().startswith('pc') or True:
                # TODO: not able to run on remote server
                egl = pkgutil.get_loader('eglRenderer')
                plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
        # camera related setting
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(0, 0, 0.2),
            distance=1.5,
            yaw=90,
            pitch=-36,
            roll=0,
            upAxisIndex=2)
        self._proj_matrix = p.computeProjectionMatrixFOV(
            fov=45,
            aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1,
            farVal=20.0)
        # additional settings
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))
        p.setGravity(0, 0, -9.81)
        plane = p.loadURDF(os.path.join(ASSET_DIR_PATH, "plane/plane.urdf"),
                           (0, 0, -0.001))
        wood = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/wood.jpg"))
        p.changeVisualShape(plane, -1, textureUniqueId=wood)

        self.obj_ids = {'fixed': [], 'rigid': [], 'deformable': []}

        self.seed()

        # self.actions = []  # only for demo
        self._env_setup()
        step(0.25)
        self.goal = self._sample_goal()  # tasks are all implicitly goal-based
        self._sample_goal_callback()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1.,
                                       1.,
                                       shape=(self.action_size, ),
                                       dtype='float32')
        if isinstance(obs, np.ndarray):
            # gym.Env
            self.observation_space = spaces.Box(-np.inf,
                                                np.inf,
                                                shape=obs.shape,
                                                dtype='float32')
        elif isinstance(obs, dict):
            # gym.GoalEnv
            self.observation_space = spaces.Dict(
                dict(
                    desired_goal=spaces.Box(-np.inf,
                                            np.inf,
                                            shape=obs['achieved_goal'].shape,
                                            dtype='float32'),
                    achieved_goal=spaces.Box(-np.inf,
                                             np.inf,
                                             shape=obs['achieved_goal'].shape,
                                             dtype='float32'),
                    observation=spaces.Box(-np.inf,
                                           np.inf,
                                           shape=obs['observation'].shape,
                                           dtype='float32'),
                ))
        else:
            raise NotImplementedError

        # @taohuang
        self._duration = 0.2  # important for mini-steps

    def step(self, action: np.ndarray):
        # action should have a shape of (action_size, )
        if len(action.shape) > 1:
            action = action.squeeze(axis=-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # time0 = time.time()
        self._set_action(action)
        # time1 = time.time()
        # TODO: check the best way to step simulation

        #Zhenya:specially designed for soft body simulation
        # step(self._duration)
        p.stepSimulation()

        # time2 = time.time()
        # print(" -> robot action time: {:.6f}, simulation time: {:.4f}".format(time1 - time0, time2 - time1))
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        } if isinstance(obs, dict) else {
            'achieved_goal': None
        }
        if isinstance(obs, dict):
            reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        else:
            reward = self.compute_reward(obs, self.goal, info)
        # if len(self.actions) > 0:
        #     self.actions[-1] = np.append(self.actions[-1], [reward])  # only for demo
        return obs, reward, done, info

    def reset(self):
        # reset scene in the corresponding file
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.configureDebugVisualizer(lightPosition=(10.0, 0.0, 10.0))

        # Temporarily disable rendering to load scene faster.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        # p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)

        plane = p.loadURDF(os.path.join(ASSET_DIR_PATH, "plane/plane.urdf"),
                           (0, 0, -0.001))
        wood = p.loadTexture(os.path.join(ASSET_DIR_PATH, "texture/wood.jpg"))
        p.changeVisualShape(plane, -1, textureUniqueId=wood)
        self._env_setup()
        step(0.25)
        self.goal = self._sample_goal().copy()
        self._sample_goal_callback()

        # Re-enable rendering.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        obs = self._get_obs()
        return obs

    def close(self):
        if self.cid >= 0:
            p.disconnect()
            self.cid = -1

    def render(self, mode='rgb_array'):
        self._render_callback(mode)
        if mode == "human":
            return np.array([])
        # TODO: check the way to render image
        rgb_array, mask = render_image(RENDER_WIDTH, RENDER_HEIGHT,
                                       self._view_matrix, self._proj_matrix)
        if mode == 'rgb_array':
            return rgb_array
        else:
            return rgb_array, mask

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_reward(self, achieved_goal, desired_goal, info):
        raise NotImplementedError

    def _env_setup(self):
        pass

    def _get_obs(self):
        raise NotImplementedError

    def _set_action(self, action):
        """ Applies the given action to the simulation.
        """
        raise NotImplementedError

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError

    def _sample_goal(self):
        """ Samples a new goal and returns it.
        """
        raise NotImplementedError()

    def _sample_goal_callback(self):
        """ For goal visualization, etc.
        """
        pass

    def _render_callback(self, mode):
        """ A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """ A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    @property
    def action_size(self):
        raise NotImplementedError

    def get_oracle_action(self, obs) -> np.ndarray:
        """
        Define a scripted oracle strategy
        """
        raise NotImplementedError

    # def test(self, horizon=100):
    # original 100 steps; change to 200 for bimanual need to be reverted
    def test(self,
             horizon=200,
             use_soft_body=False,
             soft_body_base_position=(0, 0, 0),
             collision_obj_id=-1,
             collision_sdf_filename='',
             model_filename=None,
             reset_position=(0, 0, 0),
             reset_orientation=(0, 0, 0, 1)):
        """
        Run the test simulation without any learning algorithm for debugging purposes

        use_soft_body: whether add soft body in simulation scene

        soft_body_base_position: the base position of soft body

        collision_obj_id: the id of object which need to be detected for collision on soft body

        collision_sdf_filename: the path of precomputed Signed Distance Field(SDF)  
        """
        if use_soft_body:
            mpm3d.init(collision_obj_id, model_filename)
            mpm3d.init_sdf(np.load(collision_sdf_filename))
            mpm3d.set_base_position(soft_body_base_position)

        steps, done = 0, False
        obs = self.reset()
        #!!!: for debug
        # while not done and steps <= 9999900:
        rKey = ord('r')
        while 1:
            keys = p.getKeyboardEvents()
            if rKey in keys and keys[rKey] & p.KEY_WAS_TRIGGERED:
                p.resetBasePositionAndOrientation(collision_obj_id,
                                                  reset_position,
                                                  reset_orientation)

            if use_soft_body:
                #switch reference frame to reduce sdf computation
                t_p, t_q = p.getBasePositionAndOrientation(collision_obj_id)
                s_p, s_q = soft_body_base_position, (0, 0, 0, 1.0)
                inv_tp, inv_tq = p.invertTransform(t_p, t_q)
                #pay attention to the multiply sequence
                i_pos, i_quaternion = p.multiplyTransforms(
                    inv_tp, inv_tq, s_p, s_q)

                i_rot = np.array(
                    p.getMatrixFromQuaternion(i_quaternion)).reshape(
                        3, 3).astype(np.float32)
                i_pos = np.array(i_pos).astype(np.float32)

                soft_body_step(i_rot,
                               i_pos,
                               USE_POINTS=False,
                               base_position=soft_body_base_position,
                               scale=1.0,
                               use_debug=False)

            # p.getCameraImage(width=128, height=128)
            tic = time.time()
            action = self.get_oracle_action(obs)
            print('\n -> step: {}, action: {}'.format(steps,
                                                      np.round(action, 4)))
            # print('action:', action)

            obs, reward, done, info = self.step(action)

            if isinstance(obs, dict):
                print(" -> achieved goal: {}".format(
                    np.round(obs['achieved_goal'], 4)))
                print(" -> desired goal: {}".format(
                    np.round(obs['desired_goal'], 4)))
            else:
                print(" -> achieved goal: {}".format(
                    np.round(info['achieved_goal'], 4)))

            done = info['is_success'] if isinstance(obs, dict) else done
            steps += 1
            toc = time.time()
            print(" -> step time: {:.4f}".format(toc - tic))

            if use_soft_body:
                #To avoid the flicker when simulate the soft body
                p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING,
                                           1)
                #clean gui
                # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

            # time.sleep(0.05)
        print('\n -> Done: {}\n'.format(done > 0))

    def __del__(self):
        self.close()
