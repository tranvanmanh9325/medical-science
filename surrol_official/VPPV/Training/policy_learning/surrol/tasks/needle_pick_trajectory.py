import os
import time
import numpy as np

import pybullet as p
from surrol.tasks.psm_env import PsmEnv
from surrol.utils.pybullet_utils import (
    get_link_pose,
    reset_camera,    
    wrap_angle
)
from surrol.tasks.ecm_env import EcmEnv, goal_distance

from surrol.robots.ecm import RENDER_HEIGHT, RENDER_WIDTH, FoV
from surrol.const import ASSET_DIR_PATH
from surrol.robots.ecm import Ecm
import torch
import cv2
from PIL import Image
#from rl.modules.visual_obs import VisProcess

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

paletee=get_palette(24)

def plot_image(img,is_seg=False, is_depth=False, path='/research/d1/rshr/arlin/SAM-rbt-sim2real/debug', name='img1.png'):
    if is_depth:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_JET)
        
    i=Image.fromarray(np.asarray(img,dtype=np.uint8))
    
    
    if is_seg:
        np.save(os.path.join(path,'seg_npy',name.split('/')[-1]),img)
        i.putpalette(paletee)
    
    i.save(os.path.join(path,name))


class NeedlePick(PsmEnv):
    POSE_TRAY = ((0.55, 0, 0.6751), (0, 0, 0))
    WORKSPACE_LIMITS = ((0.50, 0.60), (-0.05, 0.05), (0.685, 0.745))  # reduce tip pad contact
    SCALING = 5.
    QPOS_ECM = (0, 0.6, 0.04, 0)
    ACTION_ECM_SIZE=3
    ACTION_SIZE=7
    haptic=True

    # TODO: grasp is sometimes not stable; check how to fix it
    def __init__(self, render_mode=None, cid = -1):
        self._cnt=0
        self.v_processor=None
        super(NeedlePick, self).__init__(render_mode, cid)
        
        print('Needle Pick env init!')
        
        self._view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=(-0.05 * self.SCALING, 0, 0.375 * self.SCALING),
            distance=1.81 * self.SCALING,
            yaw=90,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        
        # For vis
        self._render_mode='human'
        self.goal_plot=False
        #self.v_processor=VisProcess()

    def update_vp(self, vp):
        self.v_processor=vp
    
    def _env_setup(self):
        ecm_view_matrix =[2.7644696427853166e-12, -0.8253368139266968, 0.5646408796310425, 0.0, 1.0, 2.76391192918779e-12, -8.559629784479772e-13, 0.0, -8.541598418149166e-13, 0.5646408796310425, 0.8253368139266968, 0.0, -1.582376590869572e-11, 0.4536721706390381, -5.886332988739014,1.0]
        ecm_view_matrix[14]=-5.0
        self._view_matrix=np.array(ecm_view_matrix).reshape(4,4)
        super(NeedlePick, self)._env_setup()
        # np.random.seed(4)  # for experiment reproduce
        self.has_object = True
        self._waypoint_goal = True
 
        # camera
        #if self._render_mode == 'human':
            # reset_camera(yaw=90.0, pitch=-30.0, dist=0.82 * self.SCALING,
            #              target=(-0.05 * self.SCALING, 0, 0.36 * self.SCALING))
        #    reset_camera(yaw=89.60, pitch=-56, dist=5.98,
        #                 target=(-0.13, 0.03,-0.94))
        #if self._render_mode == 'human':
        self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
                    scaling=self.SCALING, view_matrix=ecm_view_matrix)
        self.ecm.reset_joint(self.QPOS_ECM)
        #self.ecm = Ecm((0.15, 0.0, 0.8524), #p.getQuaternionFromEuler((0, 30 / 180 * np.pi, 0)),
        #               scaling=self.SCALING)
        #self.ecm.reset_joint(self.QPOS_ECM)
        # p.setPhysicsEngineParameter(enableFileCaching=0,numSolverIterations=10,numSubSteps=128,contactBreakingThreshold=2)


        # robot
        workspace_limits = self.workspace_limits1

        temp=np.random.randint(-5,6)/20

        #workspace_limits[1][1] if np.random.rand()>0.5 else workspace_limits[1][0]
        pos = (workspace_limits[0][0],
               temp,
               workspace_limits[2][1])
        '''
        pos = (workspace_limits[0][0],
               workspace_limits[1][1],
               (workspace_limits[2][1] + workspace_limits[2][0]) / 2)
        '''
        orn = (0.5, 0.5, -0.5, -0.5)
        joint_positions = self.psm1.inverse_kinematics((pos, orn), self.psm1.EEF_LINK_INDEX)
        self.psm1.reset_joint(joint_positions)
        self.block_gripper = False
        # physical interaction
        self._contact_approx = False

        # tray pad
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'tray/tray_pad.urdf'),
                            np.array(self.POSE_TRAY[0]) * self.SCALING,
                            p.getQuaternionFromEuler(self.POSE_TRAY[1]),
                            #globalScaling=0.000001)
                            globalScaling=self.SCALING)
        self.obj_ids['fixed'].append(obj_id)  # 1

        # needle
        # yaw = (np.random.rand() - 0.5) * np.pi
        # obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
        #                     (workspace_limits[0].mean() + (np.random.rand() - 0.5) * 0.1,  # TODO: scaling
        #                      workspace_limits[1].mean() + (np.random.rand() - 0.5) * 0.1,
        #                      workspace_limits[2][0] + 0.01),
        #                     p.getQuaternionFromEuler((0, 0, yaw)),
        #                     useFixedBase=False,
        #                     globalScaling=self.SCALING)
        yaw = (np.random.rand() - 0.5) * np.pi
        obj_id = p.loadURDF(os.path.join(ASSET_DIR_PATH, 'needle/needle_40mm.urdf'),
                            (workspace_limits[0].mean() ,  # TODO: scaling
                             workspace_limits[1].mean(),
                             workspace_limits[2][0] + 0.01),
                            p.getQuaternionFromEuler((0, 0, yaw)),
                            useFixedBase=False,
                            #globalScaling=0.000001)
                            globalScaling=self.SCALING)
        p.changeVisualShape(obj_id, -1, specularColor=(80, 80, 80))
        self.obj_ids['rigid'].append(obj_id)  # 0
        self.obj_id, self.obj_link1 = self.obj_ids['rigid'][0], 1
    
    def _get_obs(self) -> dict:
        robot_state = self._get_robot_state(idx=0)
        #print('robot cam pos: ',robot_state[:3])
        
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
        self.goal = np.array([0.54, 0.013, 0.708])
        print(f'goal world: {self.goal}')
        print(f'view_matrix: {self._view_matrix}')
        goal=self._world2cam_pos(self.goal)
        print(f'goal: {goal}')
        exit()
        obs = {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': goal.copy()
        }
        
        render_obs,seg, depth=self.ecm.render_image()
        #cv2.imwrite('/research/d1/rshr/arlin/data/debug/depth_noise_debug/img.png',cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB))
        #plt.imsave('/research/d1/rshr/arlin/data/debug/depth_noise_debug/img2.png',render_obs)
        #print('depth max: ',np.max(depth))
        #exit()
        render_obs=cv2.resize(render_obs,(320,240))
        
        self.counter+=1
        #print(render_obs[0][0])
        #exit()
        #seg=np.array(seg==6).astype(int)
        
        seg=np.array((seg==6 )| (seg==1)).astype(int)
        #seg=np.array(seg==1).astype(int)
        #seg=np.resize(seg,(320,240))
        
        #plt.imsave('/research/d1/rshr/arlin/data/debug/depth_noise_debug/depth.png',depth)
        #exit()
        seg = cv2.resize(seg, (320,240), interpolation =cv2.INTER_NEAREST)
        #plt.imsave('/research/d1/rshr/arlin/data/debug/seg_debug/noise_{}/seg.png'.format(self.curr_intensity),seg)
        #exit()
        depth = cv2.resize(depth, (320,240), interpolation =cv2.INTER_NEAREST)
        #print(np.max(depth))
        #depth = cv2.resize(depth, (320,240),interpolation=cv2.INTER_LANCZOS4)

        
        #image=cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB) / 255.0
        #plt.imsave('/home/student/code/regress_data7/seg/seg_{}.png'.format(self.counter),seg)
        #image = self.transform({'image': image})['image']
        #image=torch.from_numpy(image).to("cuda:0").float()

         # test depth noise
        
        #if np.random.randn()<0.5:
        #    instensity=np.random.randint(3,15)/100
        #instensity=0.1
        #    depth = add_gaussian_noise(depth, instensity)
        '''
        if self.counter==10:
            cv2.imwrite('/research/d1/rshr/arlin/data/debug/depth_noise_debug/gaussian/img.png',cv2.cvtColor(render_obs, cv2.COLOR_BGR2RGB))
            plt.imsave('/research/d1/rshr/arlin/data/debug/depth_noise_debug/gaussian/depth.png',depth)
            for i in [0.01,0.05,0.1,0.15,0.2]:
                noisy_depth_map = add_random_noise(depth, i)
                plt.imsave('/research/d1/rshr/arlin/data/debug/depth_noise_debug/gaussian/noise_{}.png'.format(i),noisy_depth_map)

            exit()
        '''

        #noisy_segmentation_map = add_noise_to_segmentation(seg, self.seg_noise_intensity)
        #noisy_depth_map = add_gaussian_noise(depth, self.curr_intensity)
        #if self.counter==10:
        #    plt.imsave('/research/d1/rshr/arlin/data/debug/seg_debug/noise_{}/img.png'.format(self.curr_intensity),render_obs)
        #    plt.imsave('/research/d1/rshr/arlin/data/debug/seg_debug/noise_{}/seg.png'.format(self.curr_intensity),seg)
        #    plt.imsave('/research/d1/rshr/arlin/data/debug/seg_debug/noise_{}/noise_seg.png'.format(self.curr_intensity),noisy_segmentation_map)

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
        #print('new observation: ',o_new)
        #with open("/research/d1/rshr/arlin/SAM-rbt-sim2real/traj.txt","a") as f:
        #    f.write('obs:\n')
        #    f.write(str(o_new))

        #print("new obs: ", o_new)
        obs['observation']=o_new
        
        return obs
    
    '''
    def _get_obs(self) -> dict:
        robot_state=self._get_robot_state(idx=0)
        #robot_state=torch.tensor(robot_state).cuda()
        
        # For vis_input
        render_obs,seg, depth=self.ecm.render_image()
        observation, achieved_goal=self.v_processor.get_v_obs(depth, seg, robot_state) # object_pos, waypoint_pos, waypoint_rot
        #print(type(observation))
        #observation=torch.concat((observation, robot_state), dim=0)
        #print(observation.shape) 19
        obs={
            'observation': observation,
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy()
        }
        return obs
    '''
    def _sample_goal(self) -> np.ndarray:
        """ Samples a new goal and returns it.
        """
        workspace_limits = self.workspace_limits1
        '''
        goal = np.array([workspace_limits[0].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[1].mean() + 0.01 * np.random.randn() * self.SCALING,
                         workspace_limits[2][1] - 0.07 * self.SCALING])
        '''
        pos_obj, orn_obj = get_link_pose(self.obj_id, self.obj_link1)
        goal=np.array([pos_obj[0], pos_obj[1],pos_obj[2]+0.005 ])
        return goal.copy()

    def _sample_goal_callback(self):
        """ Define waypoints
        """
        super()._sample_goal_callback()
        self._waypoints = [None, None, None, None]  # four waypoints
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

        self._waypoints[0] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102 + 0.005) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[1] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, 0.5])  # approach
        self._waypoints[2] = np.array([pos_obj[0], pos_obj[1],
                                       pos_obj[2] + (-0.0007 + 0.0102) * self.SCALING, yaw, -0.5])  # grasp
        self._waypoints[3] = np.array([self.goal[0], self.goal[1],
                                       self.goal[2] + 0.0102 * self.SCALING, yaw, -0.5])  # lift up

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
            delta_pos = (waypoint[:3] - obs['observation'][:3]) / 0.01 / self.SCALING
            delta_yaw = (waypoint[3] - obs['observation'][5]).clip(-0.4, 0.4)
            if np.abs(delta_pos).max() > 1:
                delta_pos /= np.abs(delta_pos).max()
            scale_factor = 0.4
            delta_pos *= scale_factor
            action = np.array([delta_pos[0], delta_pos[1], delta_pos[2], delta_yaw, waypoint[4]])
            if np.linalg.norm(delta_pos) * 0.01 / scale_factor < 1e-4 and np.abs(delta_yaw) < 1e-2:
                self._waypoints[i] = None
            break

        return action
    
    def get_oracle_action_ecm(self, obs) -> np.ndarray:
        """
        Define a human expert strategy
        """
        # six waypoints executed in sequential order
        obss=obs['observation'][:6]
        action = np.zeros(7)
        action[4] = -0.5
        
        for i, waypoint in enumerate(self._waypoints):
            if waypoint is None:
                continue
            print("current: ",i)
            
            ecm_waypoint_pos=self._world2cam_pos(waypoint[:3])
            
            ecm_obs_pos=self._world2cam_pos(obss[:3].copy())
            delta_pos=(ecm_waypoint_pos-ecm_obs_pos)/0.01/self.SCALING
            #delta_pos=(waypoint[:3]-obss[:3])/0.01/self.SCALING
            delta_yaw=(waypoint[3]-obss[-1])
            print("delta_yaw: ",delta_yaw)
            #while abs(delta_yaw+np.pi/2)<abs(delta_yaw):
            #    delta_yaw=delta_yaw+np.pi/2
            #print("delta_yaw: ",delta_yaw)
            delta_yaw=delta_yaw.clip(-0.4,0.4)
            
            delta_rot=self.calculate_ecm_rotation(self._view_matrix, delta_yaw)
           
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
    env = NeedlePick(render_mode='human')  # create one process and corresponding env

    env.test()
    env.close()
    time.sleep(2)
