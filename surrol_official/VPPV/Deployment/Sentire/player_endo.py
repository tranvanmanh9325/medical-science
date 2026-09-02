### define the working station:
workstation='CSR'

import time
import torch
import torch.nn as nn
import numpy as np
import os
import cv2

if workstation=='dvrk':
    import dvrk
    import PyKDL

elif workstation=='CSR':
    from csrk.arm_proxy import ArmProxy
    from csrk.node import Node
    import PyCSR

from PIL import Image
import matplotlib.pyplot as plt
import yaml
import math
from scipy.spatial.transform import Rotation as R
from easydict import EasyDict as edict
import sys
import copy
sys.path.append('IGEV/core')
sys.path.append('IGEV')
from igev_stereo import IGEVStereo
from IGEV.core.utils.utils import InputPadder
from rl.agents.ddpg import DDPG
import rl.components as components

import argparse
from FastSAM.fastsam import FastSAM, FastSAMPrompt 
import ast
from PIL import Image
from FastSAM.utils.tools import convert_box_xywh_to_xyxy
from gym import spaces
import torch.nn.functional as F
import queue, threading

from vmodel import vismodel
from config import opts

from cotracker.co_tracker_api import CoTrackerVisualSurrolAPI
from rectify import my_rectify

# edit for csr
if workstation=='CSR':  
    node_ = Node("/home/student/csr_test/NDDS_QOS_PROFILES.CSROS.xml") # NOTE: path Where you put the ndds xml file
# end edit for csr

def SetPoints(windowname, img):
    
    points = []

    def onMouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(temp_img, (x, y), 10, (102, 217, 239), -1)
            points.append([x, y])
            cv2.imshow(windowname, temp_img)

    temp_img = img.copy()
    cv2.namedWindow(windowname)
    cv2.imshow(windowname, temp_img)
    cv2.setMouseCallback(windowname, onMouse)
    key = cv2.waitKey(0)
    if key == 13:  # Enter
        print('select point: ', points)
        del temp_img
        cv2.destroyAllWindows()
        return points
    elif key == 27:  # ESC
        print('quit!')
        del temp_img
        cv2.destroyAllWindows()
        return
    else:
        
        print('retry')
        return SetPoints(windowname, img)

def crop_img(img):
    crop_img = img[:,100: ]
    crop_img = crop_img[:,: -100]
    #print(crop_img.shape)
    crop_img=cv2.resize(crop_img ,(256,256))
    return crop_img

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    video_name='test_record/{}.mp4'.format(name.split('/')[-1])
    self.output_video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (800, 600))

    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()
    #t.join()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      self.output_video.write(frame)
      if not self.q.empty():
        try:
          self.q.get_nowait()   
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()
  
  def release(self):

      self.cap.release()
      self.output_video.release()


class VisPlayer(nn.Module):
    def __init__(self):
        super().__init__()
        # depth estimation
        self.device='cuda:0'
        #self._load_depth_model()
        #self._load_policy_model()
        self._init_rcm()
        self.img_size=(320,240)
        self.scaling=1. # for peg transfer
        self.calibration_data = {
            'baseline': 0.004671,
            'focal_length_left': 788.96950318,
            'focal_length_right': 788.96950318
        }
        # self.threshold=0.013
        #self.init_run()
        self.co_tracker = CoTrackerVisualSurrolAPI()

        
    def _init_rcm(self):
        # TODO check matrix
        self.tool_T_tip=np.array([[0.0, 1.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])

        eul=np.array([np.deg2rad(-90), 0., 0.])
        eul= self.get_matrix_from_euler(eul)
        self.rcm_init_eul=np.array([-2.94573084 , 0.15808114 , 1.1354972])
        self.rcm_init_pos=np.array([ -0.0617016, -0.00715477,  -0.0661465])

    def _load_depth_model(self, checkpoint_path='pretrained_models/sceneflow.pth'):
        args=edict()
        args.restore_ckpt=checkpoint_path
        args.save_numpy=False
        args.mixed_precision=False
        args.valid_iters=32
        args.hidden_dims=[128]*3
        args.corr_implementation="reg"
        args.shared_backbone=False
        args.corr_levels=2
        args.corr_radius=4
        args.n_downsample=2
        args.slow_fast_gru=False
        args.n_gru_layers=3
        args.max_disp=192

        self.depth_model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
        #self.depth_model=IGEVStereo(args)
        self.depth_model.load_state_dict(torch.load(args.restore_ckpt))

        self.depth_model = self.depth_model.module
        self.depth_model.to("cuda")
        self.depth_model.eval()
    
    
    def _load_policy_model(self, vmodel_file, filepath='./pretrained_models/state_dict.pt'):
        with open('rl/configs/agent/ddpg.yaml',"r") as f:
                agent_params=yaml.load(f.read(),Loader=yaml.FullLoader)
        agent_params=edict(agent_params)
        env_params = edict(
            obs=3,
            achieved_goal=3,
            goal=3,
            act=3,
            max_timesteps=10,
            max_action=1,
            act_rand_sampler=None,
        )
        

        self.agent=DDPG(env_params=env_params,agent_cfg=agent_params)
        checkpt_path=filepath
        checkpt = torch.load(checkpt_path, map_location=self.device)
        self.agent.load_state_dict(checkpt)
        self.agent.g_norm.std=self.agent.g_norm_v.numpy()
        self.agent.g_norm.mean=self.agent.g_norm_mean.numpy()
        self.agent.o_norm.std=self.agent.o_norm_v.numpy()
        self.agent.o_norm.mean=self.agent.o_norm_mean.numpy()
        self.agent.cuda()
        self.agent.eval()

        opts.device='cuda:0'
        self.v_model=vismodel(opts)
        ckpt=torch.load(vmodel_file, map_location=opts.device)
        self.v_model.load_state_dict(ckpt['state_dict'])
        self.v_model.to(opts.device)
        self.v_model.eval()

    def convert_disparity_to_depth(self, disparity, baseline, focal_length):
        depth = baseline * focal_length/ disparity
        return depth


    def _get_depth(self, limg, rimg):
        # input image should be RGB(Image.open().convert('RGB')); numpy.array
        '''
        img = np.array(Image.open(imfile)).astype(np.uint8)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img[None].to(DEVICE)
        '''
        limg=torch.from_numpy(limg).permute(2, 0, 1).float().to(self.device).unsqueeze(0)
        rimg=torch.from_numpy(rimg).permute(2, 0, 1).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            padder = InputPadder(limg.shape, divis_by=32)
            image1, image2 = padder.pad(limg, rimg)
            disp = self.depth_model(image1, image2, iters=32, test_mode=True)
            disp = disp.cpu().numpy()
        
            disp = padder.unpad(disp).squeeze()
            depth_map = self.convert_disparity_to_depth(disp, self.calibration_data['baseline'], self.calibration_data['focal_length_left'])
        return depth_map
    
    def _load_fastsam(self, model_path="./FastSAM/weights/FastSAM-x.pt"):
        
        self.seg_model=FastSAM(model_path)
        
    
    def _seg_with_fastsam(self, input, object_point):
        point_prompt=str([object_point,[200,200]])
        point_prompt = ast.literal_eval(point_prompt)
        point_label = ast.literal_eval("[1,0]")
        everything_results = self.seg_model(
            input,
            device=self.device,
            retina_masks=True,
            imgsz=608,
            conf=0.25,
            iou=0.7    
            )
        
        prompt_process = FastSAMPrompt(input, everything_results, device=self.device)
        ann = prompt_process.point_prompt(
            points=point_prompt, pointlabel=point_label
        )
        
        return ann[0]
    
    def _get_action(self, seg, robot_state, ecm_wz, goal):
        # the pos should be in ecm space
        '''
        input: seg (h,w); depth(h,w); robot_pos 3; robot_rot 3; jaw 1; goal 3
        '''
       
        seg=torch.from_numpy(seg).to("cuda:0").float()
        robot_state = torch.tensor(robot_state).to(self.device)
        goal=torch.tensor(goal).to(self.device)

        with torch.no_grad():

            v_output=self.v_model.get_obs(seg.unsqueeze(0))[0]
            assert v_output.shape == (2,)
            print('voutput_origin: ',v_output)
            print('v_output: ', v_output)
            
            o_new=torch.cat([
                v_output, ecm_wz.to(self.device)
            ])
            o_norm=self.agent.o_norm.normalize(o_new,device=self.device)

            g_norm=self.agent.g_norm.normalize(goal, device=self.device)

            input_tensor=torch.cat((o_norm, g_norm), axis=0).to(torch.float32)

            action = self.agent.actor(input_tensor).cpu().data.numpy().flatten()
        print('action no clip:', action)
        action *= 0.005
        return o_new, action

    def get_euler_from_matrix(self, mat):
        """
        :param mat: rotation matrix (3*3)
        :return: rotation in 'xyz' euler
        """
        rot = R.from_matrix(mat)
        return rot.as_euler('xyz', degrees=False)
    
    def get_matrix_from_euler(self, ori):
        """
        :param ori: rotation in 'xyz' euler
        :return: rotation matrix (3*3)
        """
        rot = R.from_euler('xyz', ori)
        return rot.as_matrix()
    
    
    
    def _set_action(self, action, action_mode='dmove', workstation=workstation):
        if workstation == 'dvrk':
            delta_pos = np.array(action)
            print(delta_pos)
            abs_pos = np.array(self.ecm.measured_jp())
            new_abs_pos = np.array([abs_pos[0] + delta_pos[0],
                                    abs_pos[1] + delta_pos[1],
                                    abs_pos[2] + delta_pos[2],
                                    abs_pos[3] + delta_pos[3]])
            self.ecm.move_jp(new_abs_pos)
        elif workstation=='CSR':
            if action_mode == 'dmove':
                current_pose = self.ecm.measured_cp()
                current_pose.p[0] -= action[0]
                current_pose.p[1] -= action[1]
                current_pose.p[2] += action[2]
                self.ecm.move_cp(current_pose, 5, 3, 0)

            

    
    def goal_distance(self,goal_a, goal_b):
        assert goal_a.shape==goal_b.shape
        return np.linalg.norm(goal_a-goal_b,axis=-1)

    def is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        d = self.goal_distance(achieved_goal[..., :2], desired_goal[..., :2])
        misori = np.abs(achieved_goal[..., 2] - achieved_goal[..., 2])
        print(f"ECM static track: {d} {self.distance_threshold} {d < self.distance_threshold} {misori} {misori < self.misorientation_threshold}")

        return np.logical_and(
            d < self.distance_threshold,
            misori < self.misorientation_threshold
        ).astype(np.float32)
    


    def init_run(self, zoom = 'zoom_in'):
        intrinsics_matrix=np.array([[916.367081, 1.849829, 381.430393], [0.000000, 918.730361, 322.704614], [0.000000, 0.000000, 1.000000]])
        if workstation=='dvrk':
            self.ecm = dvrk.ecm('ECM')
            joints = self.ecm.measured_jp()
            print('joints:', joints)
            print('loaded_ecm')
        elif workstation == 'CSR':
            self.ecm= ArmProxy(node_, "psa2")
            while(not self.ecm.is_connected):
                self.ecm.measured_cp()
            # To check if the arm is connected
            self.ecm.read_rtrk_arm_state()
            print("connection: ",self.ecm.is_connected)


        self.limits ={
                    'lower':[-1.57079633, -0.78539816, 0.   , -1.57079633],
                    'upper':[ 1.57079633,  1.15889862, 0.254,  1.57079633]
                    }
        upper = np.array([0.003, 0.003, 0.0000])
        lower = np.array([-0.003, -0.003, -0.0000])
        self.action_space = spaces.Box(low=lower, high=upper, dtype='float32')
        self._finished=False
        self._load_depth_model()
        self._load_policy_model(vmodel_file='./models/vmodel.pt',filepath='./models/policy_model.pt')
        self._load_fastsam()

        self.cap_0=VideoCapture("/dev/video0") # left 5.23
        self.cap_2=VideoCapture("/dev/video2") # right 5.23

        for i in range(10):
            frame1=self.cap_0.read()
            frame2=self.cap_2.read()
        
        if workstation=='CSR':
            # edit for csr camera
            self.fs = cv2.FileStorage("/home/student/csr_test/endoscope_calibration.yaml", cv2.FILE_STORAGE_READ)
        else:
            self.fs = cv2.FileStorage("/home/kj/ar/EndoscopeCalibration/endoscope_calibration_csr_0degree.yaml", cv2.FILE_STORAGE_READ)

        frame1=cv2.resize(frame1, self.img_size)
        frame2=cv2.resize(frame2, self.img_size)      

        point=SetPoints("Goal Selection", frame1)
        self.object_point=point[0]
        self.object_point_init=copy.deepcopy(self.object_point)

        frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        goal= np.array([0,0,0.0])

        self.goal=goal
        self.distance_threshold = 0.1
        self.misorientation_threshold = 0.1
        self.count=0

    def run_step(self, mode = 'zoom'):
        if self._finished:
            return True
        
        #time.sleep(.5)
        self.count+=1
        print("--------step {}----------".format(self.count))

        frame1=self.cap_0.read()
        frame2=self.cap_2.read()


        frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        frame1=cv2.resize(frame1, self.img_size)
        frame2=cv2.resize(frame2, self.img_size)   

        new_w, new_h = self.co_tracker.step(frame1, self.object_point_init)
        self.object_point = [int(new_w), int(new_h)]

        print(f'self.object point: {self.object_point}')
            
        seg=self._seg_with_fastsam(frame1,self.object_point)       
        seg=np.array(seg==True).astype(int)
        if len(np.unique(seg)) == 1:
            point=SetPoints("Goal Selection", frame1)

            self.object_point=point[0]
            self.co_tracker = CoTrackerVisualSurrolAPI()
            self.object_point_init=copy.deepcopy(self.object_point)
            seg=self._seg_with_fastsam(frame1,self.object_point)
            
            seg=np.array(seg==True).astype(int)
        plt.imsave('/home/student/kj_demo/test_record/seg_{}.png'.format(self.count),seg)
        print("finish seg")
        plt.imsave( '/home/student/kj_demo/test_record/frame1_{}.png'.format(self.count),frame1)
        ##################
        ### get action ###
        ##################     
        robot_state = self.ecm.measured_cp()
        robot_state = np.array([-robot_state.p[0], robot_state.p[1], robot_state.p[2]])

        new_obs, action=self._get_action(seg ,robot_state, torch.tensor([0.0]), self.goal)
        print("finish get action")
        print("action: ",action)

        if self.is_success(new_obs[-3:].cpu().numpy(), self.goal):
            if mode == 'zoom':
                while not (tool_tip_final_position < 0.08 and tool_tip_final_position > 0.06):
                    frame1=self.cap_0.read()
                    frame2=self.cap_2.read()


                    frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

                    frame1=cv2.resize(frame1, self.img_size)
                    frame2=cv2.resize(frame2, self.img_size)  

                    new_w, new_h = self.co_tracker.step(frame1, self.object_point_init)
                    self.object_point = [int(new_w), int(new_h)]

                    print(f'self.object point: {self.object_point}')
                         
                    seg=self._seg_with_fastsam(frame1,self.object_point)       
                    seg=np.array(seg==True).astype(int)
                    if len(np.unique(seg)) == 1:
                        point=SetPoints("Goal Selection", frame1)

                        self.object_point=point[0]
                        self.co_tracker = CoTrackerVisualSurrolAPI()
                        self.object_point_init=copy.deepcopy(self.object_point)
                        seg=self._seg_with_fastsam(frame1,self.object_point)
                        
                        seg=np.array(seg==True).astype(int)

                    print(f'self.object point: {self.object_point}')
                    tool_tip_final_position = depth[self.object_point[1]][self.object_point[0]]
                    print('tool tip pisition is:', tool_tip_final_position)
                
                    robot_state = self.ecm.measured_cp()
                    if tool_tip_final_position > 0.08:
                        robot_state.p[2] -= 0.005
                    if tool_tip_final_position < 0.06:
                        robot_state.p[2] += 0.005
                    self.ecm.move_cp(robot_state, 1, 5 ,0)   
                    time.sleep(0.5)

            self._finished=True
            print('success')

            return self._finished
        else:
            action = np.array(action)
            action[2] = 0
            self._set_action(action)
            print("finish set action")
        
    def record_video(self, out1, out2):
        for i in range(10):
            frame1=self.cap_0.read()
            frame2=self.cap_2.read()
            out1.write(frame1)
            out2.write(frame2)
        return 

        
import threading

if __name__=="__main__":
    #lock = threading.Lock()
    zoom = 'zoom_in'
    
    ecm= ArmProxy(node_, "psa2")
    while(not ecm.is_connected):
        ecm.measured_cp()
    # To check if the arm is connected
    ecm.read_rtrk_arm_state()
    print("connection: ",ecm.is_connected)
    joints = ecm.measured_jp()
    if zoom == 'zoom_in':
        joints[2] += 0.03
        ecm.move_jp(joints, 0.015, 0.1)
    if zoom == 'zoom_out':
        joints[2] -= 0.03
        ecm.move_jp(joints, 0.015, 0.1)

    time.sleep(1)
    player1=VisPlayer()

    player1.init_run('zoom_out')
    p1_finished=False
    while not p1_finished:
        p1_finished=player1.run_step('no_zoom')
    player1.cap_0.release()
    player1.cap_2.release()
