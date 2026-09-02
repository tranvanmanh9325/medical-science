import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
# edit for csr
from csrk.arm_proxy import ArmProxy
from csrk.node import Node
import PyCSR
# end edit for csr

import matplotlib.pyplot as plt
import yaml
from scipy.spatial.transform import Rotation as R
from easydict import EasyDict as edict
import sys
import ast
import os
import time
import argparse

sys.path.append('IGEV/core')
sys.path.append('IGEV')
from igev_stereo import IGEVStereo
from IGEV.core.utils.utils import InputPadder
from rl.agents.ddpg import DDPG
from FastSAM.fastsam import FastSAM, FastSAMPrompt 
from torchvision.transforms import Compose
from vmodel import vismodel
from config import opts
from manipulation_csr import Manipulator, calculate_average_gradient
from player_utils import SetPoints, VideoCapture, my_rectify, add_gaussian_noise, gaussian_blur


# edit for csr
node_ = Node("/home/student/csr_test/NDDS_QOS_PROFILES.CSROS.xml") # NOTE: path Where 



class VisPlayer(nn.Module):
    def __init__(self,player_opts):
        super().__init__()
        self.task=player_opts.task
        self.player_opts=player_opts
        self.action_len=player_opts.action_len
        self.device='cuda:0'
        
        self.img_size=(320,240)
        self.scaling=1. # for peg transfer
        # edit for csr camera
        self.calibration_data = player_opts.calibration_data
        # edit for csr camera end
        self.threshold=player_opts.threshold[self.task]
        self.intrinsics_matrix=player_opts.intrinsics_matrix

        self.tool_T_tip=np.array([[ 0. ,-1. , 0. , 0.],
                         [ 0. , 0. , 1. , 0.],
                         [-1. , 0. , 0. , 0.],
                         [ 0. , 0. , 0. , 1.]])

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
    
    def _load_policy_model(self, filepath):
        with open('rl/configs/agent/ddpg.yaml',"r") as f:
                agent_params=yaml.load(f.read(),Loader=yaml.FullLoader)
        agent_params=edict(agent_params)
        env_params = edict(
            obs=19,
            achieved_goal=3,
            goal=3,
            act=7,
            max_timesteps=10,
            max_action=1,
            act_rand_sampler=None,
        )
        

        self.agent=DDPG(env_params=env_params,agent_cfg=agent_params)
        checkpt_path=filepath
        checkpt = torch.load(checkpt_path, map_location='cpu')
        self.agent.load_state_dict(checkpt, strict=False)
       
        self.agent.g_norm.std=self.agent.g_norm_v.numpy()
        self.agent.g_norm.mean=self.agent.g_norm_mean.numpy()
        self.agent.o_norm.std=self.agent.o_norm_v.numpy()
        self.agent.o_norm.mean=self.agent.o_norm_mean.numpy()
       
        self.agent.cuda()
        self.agent.eval()

        opts.device='cuda:0'
        ckpt_dir=os.path.join(self.player_opts.task_base[self.task],"pretrained_models", self.player_opts.vmodel_file[self.task])
        self.v_model=vismodel(opts)
        ckpt=torch.load(ckpt_dir, map_location=opts.device)
        self.v_model.load_state_dict(ckpt['state_dict'],strict=False)
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
            #print(limg.ndim)
            padder = InputPadder(limg.shape, divis_by=32)
            image1, image2 = padder.pad(limg, rimg)
            disp = self.depth_model(image1, image2, iters=32, test_mode=True)
            disp = disp.cpu().numpy()
        
            disp = padder.unpad(disp).squeeze()
            depth_map = self.convert_disparity_to_depth(disp, self.calibration_data['baseline'], self.calibration_data['focal_length_left'])
        #return disp
        return depth_map
    
    def _load_fastsam(self, model_path="./FastSAM/weights/FastSAM-x.pt"):
        
        self.seg_model=FastSAM(model_path)
        
    
    def _seg_with_fastsam(self, input, object_point, background_point=[50,50]):
        point_prompt=str([object_point, background_point])
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
    
    def _get_action(self, seg, depth, robot_pos, robot_rot, jaw, goal):
        # the pos should be in ecm space
        '''
        input: seg (h,w); depth(h,w); robot_pos 3; robot_rot 3; jaw 1; goal 3
        '''

        seg=torch.from_numpy(seg).to("cuda:0").float()
        depth=torch.from_numpy(depth).to("cuda:0").float()

        robot_pos=torch.tensor(robot_pos).to(self.device)
        robot_rot=torch.tensor(robot_rot).to(self.device) 
        jaw=torch.tensor(jaw).to(self.device)

        goal=torch.tensor(goal).to(self.device)

        with torch.no_grad():
           
            v_output=self.v_model.get_obs(seg.unsqueeze(0), depth.unsqueeze(0))[0]
            
            rel_pos=v_output[:3]
            new_pos=robot_pos+rel_pos
            waypoint_pos_rot=v_output[3:]
            o_new=torch.cat([
                robot_pos, robot_rot, jaw,
                new_pos, rel_pos, waypoint_pos_rot
            ])
           
            o_norm=self.agent.o_norm.normalize(o_new,device=self.device)
            g_norm=self.agent.g_norm.normalize(goal, device=self.device)
            
            input_tensor=torch.cat((o_norm, g_norm), axis=0).to(torch.float32)
           
            action = self.agent.actor(input_tensor).cpu().data.numpy().flatten()
        return action

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
    
    def convert_pos(self,pos,matrix):
        '''
        input: ecm pose matrix 4x4
        output rcm pose matrix 4x4
        '''
        return np.matmul(matrix[:3,:3],pos)+matrix[:3,3]
       
    
    def convert_rot(self, euler_angles, matrix):
        # Convert Euler angles to rotation matrix
        # return: matrix
        roll, pitch, yaw = euler_angles
        R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        rotation_matrix = np.matmul(R_z, np.matmul(R_y, R_x))

        # Invert the extrinsic matrix
        extrinsic_matrix_inv = np.linalg.inv(matrix)

        # Extract the rotation part from the inverted extrinsic matrix
        rotation_matrix_inv = extrinsic_matrix_inv[:3, :3]

        # Perform the rotation
        position_rotated = np.matmul(rotation_matrix_inv, rotation_matrix)

        return position_rotated

    def rcm2tip(self, rcm_action):
        return np.matmul(rcm_action, self.tool_T_tip)
    
    def _set_action(self, action, robot_pos, rot):
        '''
        robot_pos in cam coodinate
        robot_rot in rcm; matrix
        '''
        action[:3] *= 0.01 * self.scaling
        ecm_pos=robot_pos+action[:3]
        
        psm_pose=np.zeros((4,4))
        
        psm_pose[3,3]=1
        psm_pose[:3,:3]=rot
        
        rcm_pos=self.convert_pos(ecm_pos,self.player_opts.basePSM_T_cam)
        psm_pose[:3,3]=rcm_pos
        
        return psm_pose


    
    def convert_point_to_camera_axis(self, x, y, depth, intrinsics_matrix, offset_x=0, offset_y=0):
        ''' 
        # Example usage
        x = 100
        y = 200
        depth = 5.0
        intrinsics_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

        xc, yc, zc = convert_point_to_camera_axis(x, y, depth, intrinsics_matrix)
        print(f"Camera axis coordinates: xc={xc}, yc={yc}, zc={zc}")
        '''
        # Extract camera intrinsics matrix components
        fx, fy, cx, cy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1], intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]

        # Normalize pixel coordinates
        xn = (x - cx) / fx
        yn = (y - cy) / fy

        # Convert to camera axis coordinates
        xc = xn * depth
        yc = yn * depth
        zc = depth
        # return np.array([xc, yc, zc])
        return np.array([xc-offset_x, yc-offset_y, zc])
    
    def goal_distance(self,goal_a, goal_b):
        assert goal_a.shape==goal_b.shape
        return np.linalg.norm(goal_a-goal_b,axis=-1)

    def is_success(self, curr_pos, desired_goal):
        d=self.goal_distance(curr_pos, desired_goal)
        d3=np.abs(curr_pos[2]-desired_goal[2])
        print('distance: ',d)
        print('distance z-axis: ',d3)
        if d<self.threshold or d3<0.003:
            return True
        return False
    
    def init_run(self):
        
        # edit for csr
        #self.p = dvrk.psm('PSM1')
        
        self.p= ArmProxy(node_, self.player_opts.psa_num)
        while(not self.p.is_connected):
            self.p.measured_cp()
        # To check if the arm is connected
        self.p.read_rtrk_arm_state()
        print("connection: ",self.p.is_connected)
        # end edit for csr

        self._finished=False
        #player=VisPlayer()

        self._load_depth_model()
        #player._load_dam()
        
        self._load_policy_model(filepath=os.path.join(self.player_opts.task_base[self.task],"pretrained_models", self.player_opts.policy_file[self.task]))
       
        self._load_fastsam()

        self.count=0

        self.cap_0=VideoCapture(self.player_opts.video_left) # left 708
        self.cap_2=VideoCapture(self.player_opts.video_right) # right 708

        # init 
        # open jaw
        #self.p.jaw.move_jp(np.array(-0.1)).wait()
        #print("open jaw")

        # 0. define the goal
        # TODO the goal in scaled image vs. goal in simualtor?
        for i in range(10):
            frame1=self.cap_0.read()
            frame2=self.cap_2.read()
        
        #point=SetPoints("test", frame1)

        # edit for csr camera
        self.fs = cv2.FileStorage(self.player_opts.fs_path, cv2.FILE_STORAGE_READ)
        # edit for csr camera end

        frame1, frame2 = my_rectify(frame1, frame2, self.fs)
        frame1_bgr=frame1.copy()


        frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)


        self.depth_img=self._get_depth(frame1, frame2)
        self.depth_img=cv2.resize(self.depth_img, self.img_size, interpolation=cv2.INTER_NEAREST)
        plt.imsave(os.path.join(self.player_opts.save_dir,'pred_depth_{}.png'.format(self.count)),self.depth_img)
        

        frame1=cv2.resize(frame1, self.img_size)
        frame2=cv2.resize(frame2, self.img_size)
        frame1_bgr=cv2.resize(frame1_bgr, self.img_size)
        
        plt.imsave( os.path.join(self.player_opts.save_dir,'frame1_{}.png'.format(self.count)),frame1)
        plt.imsave( os.path.join(self.player_opts.save_dir,'frame2_{}.png'.format(self.count)),frame2)
        
       
       
        point=SetPoints("Goal Selection", frame1_bgr)
        self.object_point=point[0]
        #if self.task=='gauze':
        #    self.place_point=point[1]
        #if self.task=='tissue_1':

        self.place_point=point[1]
        
        #bg_point=point[2]
        #bg_point[3]=

        seg=self._seg_with_fastsam(frame1,self.object_point, self.place_point)
        #print(seg)
        seg=np.array(seg==True).astype(int)
        plt.imsave(os.path.join(self.player_opts.save_dir,'seg_{}.png'.format(self.count)),seg)

        
        goal_x, goal_y=self.object_point[0], self.object_point[1]
        goal_depth=self.depth_img[goal_y][goal_x]
        #print(goal_depth)
        

        # self.goal=self.convert_point_to_camera_axis(goal_x*2.5, goal_y*2.5, goal_depth, self.intrinsics_matrix)
                                                    
        self.goal=self.convert_point_to_camera_axis(goal_x*2.5, goal_y*2.5, goal_depth-self.player_opts.goal_offset_z[self.task], self.intrinsics_matrix, self.player_opts.goal_offset_x[self.task], self.player_opts.goal_offset_y[self.task])
        #print("goal 1: ", self.goal)
        
        # trick to align the rotation with simualtor

        self.rcm_goal=self.convert_pos(self.goal,self.player_opts.basePSM_T_cam)#goal.copy()
      
        print("Selected Goal ecm: ",self.goal)
        print("Selected Goal rcm: ",self.rcm_goal)
        seg = np.array(seg==True).astype(np.uint8)
        # gradient_angle =  np.pi / 3
        # print('gradient_angle',gradient_angle)
        # self.R_z = np.array([
        #                     [np.cos(gradient_angle), -np.sin(gradient_angle), 0],
        #                     [np.sin(gradient_angle),  np.cos(gradient_angle), 0],
        #                     [0, 0, 1]
        #                 ])
        self.R_z =calculate_average_gradient(seg, self.object_point, 25)
        
        
        self.manipulator = Manipulator()
        return frame1
        
    def run_step(self):
        if self._finished:
            return True
        
        start_time = time.time()
        #time.sleep(.5)
        self.count+=1
        print("--------step {}----------".format(self.count))
        #time.sleep(2)

        frame1=self.cap_0.read()
        frame2=self.cap_2.read()


        frame1, frame2 = my_rectify(frame1, frame2, self.fs)

        frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        '''
        plt.imsave( os.path.join(self.player_opts.save_dir,'frame1_{}_ori.png'.format(self.count)),frame1)
        plt.imsave( os.path.join(self.player_opts.save_dir,'frame2_{}_ori.png'.format(self.count)),frame2)
        '''
        #print(frame1)
        '''
        if self.player_opts.use_blur:

            frame1=gaussian_blur(frame1)#add_gaussian_noise(frame1, 0.3)
            frame2=gaussian_blur(frame2)#add_gaussian_noise(frame2, 0.3)
        '''
        depth=self._get_depth(frame1, frame2)

        depth=cv2.resize(depth, self.img_size, interpolation=cv2.INTER_NEAREST)

        #print(frame1.shape)
        print('depth shape: ',depth.shape)
        #np.save('/home/kj/ar/GauzeRetrievel/test_record/depth.npy',depth)
        print(depth[self.object_point[1]][self.object_point[0]])
        #exit()
        
        plt.imsave(os.path.join(self.player_opts.save_dir,'pred_depth_{}.png'.format(self.count)),depth)
        #exit()
        

        frame1=cv2.resize(frame1, self.img_size)
        frame2=cv2.resize(frame2, self.img_size)

        

        plt.imsave( os.path.join(self.player_opts.save_dir,'frame1_{}.png'.format(self.count)),frame1)
        plt.imsave( os.path.join(self.player_opts.save_dir,'frame2_{}.png'.format(self.count)),frame2)
        
        
        
        seg=self._seg_with_fastsam(frame1,self.object_point, self.place_point)
        #print(seg)
        
        seg=np.array(seg==True).astype(int)
        
        plt.imsave(os.path.join(self.player_opts.save_dir,'seg_{}.png'.format(self.count)),seg)
        #seg=np.load('/home/kj/ar/peg_transfer/test_record/seg_from_depth.npy')
        print("finish seg")
        #exit()
        robot_pose=self.p.measured_cp()
        robot_pos=robot_pose.p
        print("pre action pos rcm: ",robot_pos)
        robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
        #robot_pos=player.rcm2tip(robot_pos)
        pre_robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
        # can be replaced with robot_pose.M.GetRPY()
        # start
        transform_2=robot_pose.M
        np_m=np.array([[transform_2[0,0], transform_2[0,1], transform_2[0,2]],
                            [transform_2[1,0], transform_2[1,1], transform_2[1,2]],
                            [transform_2[2,0], transform_2[2,1], transform_2[2,2]]])
        
        tip_psm_pose=np.zeros((4,4))
        
        tip_psm_pose[3,3]=1
        tip_psm_pose[:3,:3]=np_m
        tip_psm_pose[:3,3]=robot_pos
        #print('tip_psm_pose before: ',tip_psm_pose)
        tip_psm_pose=self.rcm2tip(tip_psm_pose)
        #print('tip_psm_pose aft: ',tip_psm_pose)
        
        np_m=tip_psm_pose[:3,:3]
        robot_pos=tip_psm_pose[:3,3]
        #print("pre action pos tip rcm: ",robot_pos)


        #robot_rot=np_m
        robot_rot=self.get_euler_from_matrix(np_m)        
        robot_rot=self.convert_rot(robot_rot, self.player_opts.cam_T_basePSM)
        robot_rot=self.get_euler_from_matrix(robot_rot)
        robot_pos=self.convert_pos(robot_pos,self.player_opts.cam_T_basePSM)
        #print("pre action pos tip ecm: ",robot_pos)
        # end

        jaw=np.array([0.0]).astype(np.float64)
        # edit for csr
        # if you need the jaw value, uncomment the next line
        # jaw=[np.array(self.p.measured_jp())[6]-0.5]
        
        # edit for csr end
        
        robot_rot = self.player_opts.init_rotate_ecm.copy()
        action=self._get_action(seg, depth ,robot_pos, robot_rot, jaw, self.goal)
        #print("finish get action")
        print("action: ",action)
        
        # edit for csr
        PSM2_rotate=self.player_opts.rotation
        
        print('time:',time.time()-start_time)
    
        
        action_split=action/self.action_len

       
        #return action_split, np_m, robot_pos
    
        for i in range(self.action_len):
        # 4. action -> state
            robot_pos, curr_robot_pos=self.perform_action(action_split,robot_pos, np_m,i)
            # print(robot_pos_new)
        
        return self.check_finished(curr_robot_pos, self.depth_img, seg)

    def check_finished(self, curr_robot_pos,depth ,seg):
        print("finish move")
        print('is sccess: ',self.is_success(curr_robot_pos, self.rcm_goal))
        if self.is_success(curr_robot_pos, self.rcm_goal) or self.count>self.player_opts.max_step[self.task]:
            
            self._finished=True

            step_num = 3
            tool_offset = self.player_opts.tool_offset[self.task]
            if self.task=='vessel':
                self.manipulator.vessel_clip(self.object_point, self.depth_img, seg, step_num, tool_offset)
            elif self.task=='gauze':
                self.manipulator.grasp_tissue(self.object_point,self.depth_img, seg, step_num, tool_offset)
                self.manipulator.place_tissue(self.place_point, self.depth_img, seg, step_num, tool_offset)
            elif self.task=='tissue_1':
                # tool_offset = [0.0, 0.0, 0.00954] # LND 
                # tool_offset = [0.0, 0.0, 0.02111] # FBF 
                self.manipulator.vessel_clip_long(self.object_point, depth, seg, step_num, tool_offset)
            elif self.task=='needle':
                self.manipulator.needle_grasp(self.object_point, self.depth_img, seg, step_num, tool_offset,rotation_pose=self.Rz_rotation)

        return self._finished
    
    def perform_action(self, action_split, robot_pos, np_m, i, action_len=15):
        frame1=self.cap_0.read()
        frame2=self.cap_2.read()
        frame1, frame2 = my_rectify(frame1, frame2, self.fs)
        frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame1=cv2.resize(frame1, self.img_size)

        
        state=self._set_action(action_split.copy(), robot_pos, np_m)
            
        #edit for csr
        PSM2_pose = PyCSR.Vector(state[0,-1], state[1,-1], state[2,-1])
        curr_robot_pos=np.array([state[0,-1], state[1,-1], state[2,-1]])
        
        print("target pos : ",curr_robot_pos)
        
        init_rotation = np.array([[ -0.445465,    0.888536,    0.109839],
                                [0.866506,    0.458745,   -0.196774],
                                [-0.225229,  0.00752064,   -0.974277]])
                                
        retified_rotation = np.dot(init_rotation, self.R_z)
        PSM2_rotation = PyCSR.Rotation(retified_rotation[0][0],retified_rotation[0][1],retified_rotation[0][2],
                                    retified_rotation[1][0],retified_rotation[1][1],retified_rotation[1][2],
                                    retified_rotation[2][0],retified_rotation[2][1],retified_rotation[2][2])
        self.Rz_rotation=retified_rotation
        
        
        move_goal = PyCSR.Frame(PSM2_rotation, PSM2_pose)
        
        #move_goal = PyCSR.Frame(self.player_opts.rotation, PSM2_pose)

        # move
        self.p.move_cp(move_goal, acc=1, duration=1, jaw=0.85)
        #edit for csr end
        #exit()
        #print('goal:',move_goal)
        #self.p.servo_cp(move_goal)
        if i==(action_len-1):
            return robot_pos, curr_robot_pos#, frame1
        time.sleep(0.6)
        robot_pos=self.convert_pos(curr_robot_pos,self.player_opts.cam_T_basePSM)
        return robot_pos, curr_robot_pos#, frame1
        
if __name__=="__main__":
    #lock = threading.Lock()
    from player_config import player_opts

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, type=str, help="choose from 'gauze', 'vessel','tissue','neddle'")
    args = parser.parse_args()
    if args.task=='tissue':
        player_opts.task='tissue_1'
    else:
        player_opts.task=args.task

    
    player=VisPlayer(player_opts)
    player.init_run()
    finished=False
    while not finished:
        #player.record_video
        finished=player.run_step()
    time.sleep(1.0)
    player.cap_0.release()
    player.cap_2.release()
    time.sleep(1.0)

    # if args.task=='tissue':
    #     player_opts.task='tissue_2'

    #     player=VisPlayer(player_opts)
    #     player.init_run()
    #     finished=False
    #     while not finished:
    #         #player.record_video
    #         finished=player.run_step()
    #     time.sleep(1.0)
    #     player.cap_0.release()
    #     player.cap_2.release()
    #     time.sleep(1.0)
    
