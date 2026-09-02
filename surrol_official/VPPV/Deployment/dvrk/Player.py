import torch
import torch.nn as nn
import numpy as np

import cv2
import dvrk
import crtk
import PyKDL

import yaml
import math
from scipy.spatial.transform import Rotation as R
from easydict import EasyDict as edict
import sys

sys.path.append('IGEV/core')
sys.path.append('IGEV')
from igev_stereo import IGEVStereo
from IGEV.core.utils.utils import InputPadder
from rl.agents.ddpg import DDPG

from FastSAM.fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from torchvision.transforms import Compose
import torch.nn.functional as F
import queue, threading

from vmodel import vismodel
from config import opts

from player_utils import my_rectify
from manipulation import Manipulator

ral = crtk.ral('dvrk_python_node')
psm1 = dvrk.psm(ral, 'PSM1')

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
# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.cap.set(3,1280)
    self.cap.set(4,720)

    self.cap.set(cv2.CAP_PROP_FPS,60)
    video_name='test_record_vppv/{}.mp4'.format(name.split('/')[-1])
    self.output_video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (800, 600))

    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  def _reader(self):
    while True:
            
        ret, frame = self.cap.read()
        frame = frame[:,160:1120]
        frame = frame[::-1]
        frame=cv2.resize(frame, (800, 600))

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
    frame = self.q.get()

    return self.q.get()
  
  def release(self):

    self.cap.release()
    self.output_video.release()


def transf_DH_modified(alpha, a, theta, d):
    trnsf = np.array([[math.cos(theta), -math.sin(theta), 0., a],
                    [math.sin(theta) * math.cos(alpha), math.cos(theta) * math.cos(alpha), -math.sin(alpha), -d * math.sin(alpha)],
                    [math.sin(theta) * math.sin(alpha), math.cos(theta) * math.sin(alpha), math.cos(alpha), d * math.cos(alpha)],
                    [0., 0., 0., 1.]])
    return trnsf

basePSM_T_cam = np.array([[-0.88328225, -0.4560953 ,  0.10857969,  0.05357333],
       [-0.46620461,  0.87896398, -0.10037717,  0.04117293],
       [-0.04965608, -0.13928173, -0.98900701, -0.03160624],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])
cam_T_basePSM = np.array([[-0.88328225, -0.46620461, -0.04965608,  0.06494594],
       [-0.4560953 ,  0.87896398, -0.13928173, -0.01615715],
       [ 0.10857969, -0.10037717, -0.98900701, -0.03294295],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

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
            'baseline': 0.004214,
            'focal_length_left': 693.12012738,
            'focal_length_right': 693.12012738
        }
        self.threshold=0.005
        #self.init_run()
        
    def _init_rcm(self):
        # TODO check matrix
        self.tool_T_tip=np.array([[ 0. ,-1. , 0. , 0.],
                         [ 0. , 0. , 1. , 0.],
                         [-1. , 0. , 0. , 0.],
                         [ 0. , 0. , 0. , 1.]])

        eul=np.array([np.deg2rad(-90), 0., 0.])
        eul= self.get_matrix_from_euler(eul)
        self.rcm_init_eul=np.array([-2.94573084 , 0.15808114 , 1.1354972])
        #object pos [-0.123593,   0.0267398,   -0.141579]
        # target pos [-0.0577594,   0.0043639,   -0.133283]
        self.rcm_init_pos=np.array([ -0.0617016, -0.00715477,  -0.0661465])

    def _load_depth_model(self, checkpoint_path='/home/kj/jw_demo/pegtransfer/pretrained_models/sceneflow.pth'):
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
    
    def _load_dam(self):
        encoder = 'vitl' # can also be 'vitb' or 'vitl'
        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()
        self.img_transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',    
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
            ])

    def _get_depth_with_dam(self, img):
        '''
        input: rgb image 1xHxW
        '''
        img=img/255.0
        h, w = img.shape[:2]
        
        img=self.img_transform({'image': img})['image']
        img=torch.from_numpy(img).unsqueeze(0)
        with torch.no_grad():
            depth = self.depth_anything(img)
        
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) # 0-1
        #print(depth.mean())
        
        depth = depth.cpu().numpy()

        return depth
    
    def _load_policy_model(self, filepath='./pretrained_models/state_dict.pt'):
        with open('/home/kj/jw_demo/pegtransfer/rl/configs/agent/ddpg.yaml',"r") as f:
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
        #self.agent.g_norm = checkpt['g_norm']
        #self.agent.o_norm = checkpt['o_norm']
        #self.agent.update_norm_test()
        #print('self.agent.g_norm.mean: ',self.agent.g_norm.mean)
        self.agent.g_norm.std=self.agent.g_norm_v.numpy()
        self.agent.g_norm.mean=self.agent.g_norm_mean.numpy()
        self.agent.o_norm.std=self.agent.o_norm_v.numpy()
        self.agent.o_norm.mean=self.agent.o_norm_mean.numpy()
        #print('self.agent.g_norm.mean: ',self.agent.g_norm.mean)
        #exit()

        '''
        
        self.agent.depth_norm.std=self.agent.d_norm_v.numpy()
        self.agent.depth_norm.mean=self.agent.d_norm_mean.numpy()
        s
        #print(self.agent.g_norm_v)
        '''
        self.agent.cuda()
        self.agent.eval()

        opts.device='cuda:0'
        self.v_model=vismodel(opts)
        ckpt=torch.load(opts.ckpt_dir, map_location=opts.device)
        self.v_model.load_state_dict(ckpt['state_dict'])
        self.v_model.to(opts.device)
        self.v_model.eval()

    def change_policy_model(self, filepath='./pretrained_models/state_dict.pt'):
        with open('/home/kj/jw_demo/pegtransfer/rl/configs/agent/ddpg.yaml',"r") as f:
                agent_params=yaml.load(f.read(),Loader=yaml.FullLoader)
        agent_params=edict(agent_params)
        env_params = edict(
            obs=19,
            achieved_goal=3,
            goal=3,
            act=3,
            max_timesteps=10,
            max_action=1,
            act_rand_sampler=None,
        )
        

        self.agent=DDPG(env_params=env_params,agent_cfg=agent_params)
        checkpt_path=filepath
        checkpt = torch.load(checkpt_path, map_location='cpu')
        self.agent.load_state_dict(checkpt, strict=True)
        #self.agent.g_norm = checkpt['g_norm']
        #self.agent.o_norm = checkpt['o_norm']
        #self.agent.update_norm_test()
        #print('self.agent.g_norm.mean: ',self.agent.g_norm.mean)
        self.agent.g_norm.std=self.agent.g_norm_v.numpy()
        self.agent.g_norm.mean=self.agent.g_norm_mean.numpy()
        self.agent.o_norm.std=self.agent.o_norm_v.numpy()
        self.agent.o_norm.mean=self.agent.o_norm_mean.numpy()
        #print('self.agent.g_norm.mean: ',self.agent.g_norm.mean)
        #exit()

        '''
        
        self.agent.depth_norm.std=self.agent.d_norm_v.numpy()
        self.agent.depth_norm.mean=self.agent.d_norm_mean.numpy()
        s
        #print(self.agent.g_norm_v)
        '''
        self.agent.cuda()
        self.agent.eval()

        opts.device='cuda:0'
        self.v_model=vismodel(opts)
        ckpt=torch.load(opts.ckpt_dir, map_location=opts.device)
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

    def _seg_with_red(self, grid_RGB):
        # input image RGB
        grid_HSV = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)
    
        # H、S、V range1：
        lower1 = np.array([0,59,25])
        upper1 = np.array([20,255,255])
        mask1 = cv2.inRange(grid_HSV, lower1, upper1)       # mask: binary
    
        # H、S、V range2：
        #lower2 = np.array([156,43,46])
        #upper2 = np.array([180,255,255])
        #mask2 = cv2.inRange(grid_HSV, lower2, upper2)
        
        mask3 = mask1 #+ mask2

        return mask3
    
    def _get_visual_state(self, seg, depth, robot_pos, robot_rot, jaw, goal):
        seg_d=np.concatenate([seg.reshape(1, self.img_size[0], self.img_size[1]), \
                              depth.reshape(1, self.img_size[0], self.img_size[1])],axis=0)
        
        inputs=torch.tensor(seg_d).unsqueeze(0).float().to(self.device)
        robot_pos=torch.tensor(robot_pos).to(self.device)
        robot_rot=torch.tensor(robot_rot).to(self.device)
        jaw=torch.tensor(jaw).to(self.device)
        goal=torch.tensor(goal).to(self.device)

        with torch.no_grad():
            #print(inputs.shape)
            v_output=self.agent.v_processor(inputs).squeeze()
           
            waypoint_pos_rot=v_output[3:]

        return waypoint_pos_rot[:3].cpu().data.numpy().copy(), waypoint_pos_rot[3:].cpu().data.numpy().copy()
    
    def _get_action(self, seg, depth, robot_pos, robot_rot, jaw, goal):
        # the pos should be in ecm space
        '''
        input: seg (h,w); depth(h,w); robot_pos 3; robot_rot 3; jaw 1; goal 3
        '''
        #depth=self.agent.depth_norm.normalize(depth.reshape(self.img_size*self.img_size),device=self.device).reshape(self.img_size,self.img_size)
        #plt.imsave('test_record/pred_depth_norm_{}.png'.format(count),depth)
        
        #image = self.img_transform({'image': rgb})['image']

        seg=torch.from_numpy(seg).to("cuda:0").float()
        depth=torch.from_numpy(depth).to("cuda:0").float()

        #seg_d=np.concatenate([seg.reshape(1, self.img_size[0], self.img_size[1]), \
        #                      depth.reshape(1, self.img_size[0], self.img_size[1])],axis=0)
        
        #inputs=torch.tensor(seg_d).unsqueeze(0).float().to(self.device)
        #image=torch.from_numpy(image).to(self.device).float()
        #seg=torch.from_numpy(seg).to(self.device).float()
        #with torch.no_grad():
        #    v_output=self.v_model.get_obs(seg.unsqueeze(0), image.unsqueeze(0))[0]#.cpu().data().numpy()

        robot_pos=torch.tensor(robot_pos).to(self.device)
        robot_rot=torch.tensor(robot_rot).to(self.device)
        #robot_rot=torch.tensor([0.9744, -0.009914,-0.000373]).to(self.device)
        #jaw=torch.tensor([0.6981]).to(self.device)
        jaw=torch.tensor(jaw).to(self.device)
        #print(jaw.shape)
        goal=torch.tensor(goal).to(self.device)

        with torch.no_grad():
            #print(inputs.shape)
            #v_output=self.agent.v_processor(inputs).squeeze()
            v_output=self.v_model.get_obs(seg.unsqueeze(0), depth.unsqueeze(0))[0]
            #print(v_output)
            #save_v=v_output.cpu().data.numpy()
            #np.save('test_record/v_output.npy',save_v)
            rel_pos=v_output[:3]
            #print(rel_pos.shape)
            #print(robot_pos.shape)
            new_pos=robot_pos+rel_pos
            #return new_pos.cpu().data.numpy()
            waypoint_pos_rot=v_output[3:]
            '''
            print(robot_rot.shape)
            print(robot_pos.shape)
            print(jaw.shape)
            print(new_pos.shape)
            print(rel_pos.shape)
            print(waypoint_pos_rot.shape)
            
            print(robot_pos.shape)
            print(robot_rot.shape)
            print(jaw.shape)
            print(new_pos.shape)
            print(rel_pos.shape)
            print(waypoint_pos_rot.shape)
            '''

            o_new=torch.cat([
                robot_pos, robot_rot, jaw,
                new_pos, rel_pos, waypoint_pos_rot
            ])
            print('o_new: ',o_new)
            o_norm=self.agent.o_norm.normalize(o_new,device=self.device)
            #print("goal ", goal)
            g_norm=self.agent.g_norm.normalize(goal, device=self.device)
            #print("g ",g)
            #g_norm=torch.tensor(g).float().to(self.device)
            input_tensor=torch.cat((o_norm, g_norm), axis=0).to(torch.float32)
            #save_input=input_tensor.cpu().data.numpy()
            #np.save('test_record/actor_input.npy',save_input)
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
    
    def wrap_angle(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi
    
    def convert_pos(self,pos,matrix):
        '''
        input: ecm pose matrix 4x4
        output rcm pose matrix 4x4
        '''
        return np.matmul(matrix[:3,:3],pos)+matrix[:3,3]
        #bPSM_T_j6=self.get_bPSM_T_j6(joint)
        #new_ma=matrix @ bPSM_T_j6
        #a=np.matmul(new_ma[:3,:3],pos)+new_ma[:3,3]
        #return a
    
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

    def get_bPSM_T_j6(self, joint_value):
        LRcc = 0.4318
        LTool = 0.4162
        LPitch2Yaw = 0.0091
        #                                 alpha  ,          a  ,        theta               ,        d
        base_T_j1 = transf_DH_modified( np.pi*0.5,          0. , joint_value[0] + np.pi*0.5 ,                  0. )
        j1_T_j2   = transf_DH_modified(-np.pi*0.5,          0. , joint_value[1] - np.pi*0.5 ,                  0. )
        j2_T_j3   = transf_DH_modified( np.pi*0.5,          0. ,                        0.0 , joint_value[2]-LRcc )
        j3_T_j4   = transf_DH_modified(       0. ,          0. ,             joint_value[3] ,               LTool )
        j4_T_j5   = transf_DH_modified(-np.pi*0.5,          0. , joint_value[4] - np.pi*0.5 ,                  0. )
        j5_T_j6   = transf_DH_modified(-np.pi*0.5 , LPitch2Yaw , joint_value[5] - np.pi*0.5 ,                  0. )
        
        j6_T_j6f  = np.array([[ 0.0, -1.0,  0.0,  0.0], # Offset from file `psm-pro-grasp.json`
                            [ 0.0,  0.0,  1.0,  0.0],
                            [-1.0,  0.0,  0.0,  0.0],
                            [ 0.0,  0.0,  0.0,  1.0]])
        
        bPSM_T_j2 = np.matmul(base_T_j1, j1_T_j2)
        bPSM_T_j3 = np.matmul(bPSM_T_j2, j2_T_j3)
        bPSM_T_j4 = np.matmul(bPSM_T_j3, j3_T_j4)
        bPSM_T_j5 = np.matmul(bPSM_T_j4, j4_T_j5)
        bPSM_T_j6 = np.matmul(bPSM_T_j5, j5_T_j6)
        bPSM_T_j6f = np.matmul(bPSM_T_j6, j6_T_j6f) # To make pose the same as the one in the dVRK
        return bPSM_T_j6f

    def rcm2tip(self, rcm_action):
        return np.matmul(rcm_action, self.tool_T_tip)
    
    def _set_action(self, action, robot_pos, rot):
        '''
        robot_pos in cam coodinate
        robot_rot in rcm; matrix
        '''
        action[:3] *= 0.01 * self.scaling
        #action[1]=action[1]*-1
        #print(action)
        
        ecm_pos=robot_pos+action[:3]
        print('aft robot pos tip ecm: ',ecm_pos)
        psm_pose=np.zeros((4,4))
        
        psm_pose[3,3]=1
        psm_pose[:3,:3]=rot
        #print('ecm pos: ',ecm_pos)
        rcm_pos=self.convert_pos(ecm_pos,basePSM_T_cam)
        print('aft robot pos tip rcm: ',rcm_pos)
        psm_pose[:3,3]=rcm_pos
        
        #rcm_action=self.rcm2tip(psm_pose)
        #return rcm_action
        
        return psm_pose


    '''
    def _set_action(self, action, rot, robot_pos):
        """
        delta_position (6), delta_theta (1) and open/close the gripper (1)
        in the ecm coordinate system
        input: robot_rot, robot_pos in ecm 
        """
        # TODO: need to ensure to use this scaling
        action[:3] *= 0.01 * self.scaling  # position, limit maximum change in position
        #ecm_pose=self.rcm2ecm(psm_pose)
        #ecm_pos=self.convert_pos(robot_pos, cam_T_basePSM)
        ecm_pos=robot_pos+action[:3]
        #ecm_pos[2]=ecm_pos[2]-2*action[2]

        #ecm_pose[:3,3]=ecm_pose[:3,3]+action[:3]
        #rot=self.get_euler_from_matrix(ecm_pose[:3,:3])
        #rot=self.convert_rot(robot_rot, cam_T_basePSM)
        #rot=self.get_euler_from_matrix(robot_rot)

        #action[3:6] *= np.deg2rad(20)
        #rot =(self.wrap_angle(rot[0]+action[3]),self.wrap_angle(rot[1]+action[4]),self.wrap_angle(rot[2]+action[5]))
        #rcm_action_matrix=self.convert_rot(rot,basePSM_T_cam) # ecm2rcm rotation
        
        rcm_pos=self.convert_pos(ecm_pos,basePSM_T_cam) # ecm2rcm position

        rot_matrix=self.get_matrix_from_euler(rot)
        #rcm_action_matrix=self.convert_rot(ecm_pose) #self.ecm2rcm(ecm_pose)
        
        #rcm_action_eul=self.get_euler_from_matrix(rcm_action_matrix)
        #rcm_action_eul=(self.rcm_init_eul[0], self.rcm_init_eul[1],rcm_action_eul[2])
        #rcm_action_matrix=self.get_matrix_from_euler(rcm_action_eul)

        psm_pose=np.zeros((4,4))
        psm_pose[3,3]=1
        psm_pose[:3,:3]=rot_matrix
        psm_pose[:3,3]=rcm_pos 

        # TODO: use get_bPSM_T_j6 function
        rcm_action=self.rcm2tip(psm_pose)
        rcm_action=psm_pose

        return rcm_action
    '''
    def convert_point_to_camera_axis(self, x, y, depth, intrinsics_matrix):
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

        return np.array([xc, yc, zc])
    
    def goal_distance(self,goal_a, goal_b):
        assert goal_a.shape==goal_b.shape
        return np.linalg.norm(goal_a-goal_b,axis=-1)

    def is_success(self, curr_pos, desired_goal):
        d=self.goal_distance(curr_pos, desired_goal)
        d3=np.abs(curr_pos[2]-desired_goal[2])
        print('distance: ',d)
        print('distance z-axis: ',d3)
        # if d3<0.004:
        #     return True
        return (d<self.threshold and d3<0.008).astype(np.float32)
    
    def init_run(self, cap_0=None, cap_2=None):

        # create the ROS Abstraction Layer with the name of the node
        # create a Python proxy for PSM1, name must match ROS namespace
        self.p = psm1

        self._finished=False

        self._load_depth_model()
        self._load_policy_model(filepath='/home/kj/jw_demo/pegtransfer/pretrained_models/s80_DDPG_demo0_traj_best.pt')
        # self._load_policy_model(filepath='/home/kj/ar/GauzeRetrievel/pretrained_models/s71_DDPG_demo0_traj_best.pt')
        # self._load_policy_model(filepath='/home/kj/jw_demo/pegtransfermove/pretrained_models/s80_DDPG_pegtransfermove_demo0_traj_best_0812.pt')
 
        self._load_fastsam()

        if cap_0 is None:
            self.cap_0=VideoCapture("/dev/video0") # left 6.20
        else:
            self.cap_0 = cap_0
        if cap_2 is None:
            self.cap_2=VideoCapture("/dev/video2") # right 6.20
        else:
            self.cap_2 = cap_2

        # init 
        # open jaw
        self.p.jaw.move_jp(np.array([0.5])).wait()
        print("open jaw")

        # 0. define the goal
        for i in range(10):
            frame1=self.cap_0.read()
            frame2=self.cap_2.read()
            # frame1 = frame1[:,160:1120]
            # frame1 = frame1[::-1]
            # frame2 = frame2[:,160:1120]
            # frame2 = frame2[::-1]
            # frame1=cv2.resize(frame1, (800, 600))
            # frame2=cv2.resize(frame2, (800, 600))
        
        self.fs = cv2.FileStorage("/home/kj/ar/EndoscopeCalibration/endoscope_calibration_csr_0degree.yaml", cv2.FILE_STORAGE_READ)

        frame1, frame2 = my_rectify(frame1, frame2, self.fs)
        frame1=cv2.resize(frame1, self.img_size)
        frame2=cv2.resize(frame2, self.img_size)
        point=SetPoints("Goal Selection", frame1)

        self.object_point=point[0]
        # self.place_point = point[1]
        self.rotation_pose_place = None
        frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        depth=self._get_depth(frame1, frame2)
        depth=cv2.resize(depth, self.img_size, interpolation=cv2.INTER_NEAREST)

        def convert_point_to_camera_axis(x, y, depth, intrinsics_matrix):

            fx, fy, cx, cy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1], intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]

            xn = (x - cx) / fx
            yn = (y - cy) / fy

            xc = xn * depth
            yc = yn * depth
            zc = depth

            return np.array([xc, yc, zc])
        
        goal = convert_point_to_camera_axis(self.object_point[0], self.object_point[1], depth[self.object_point[1]][self.object_point[0]], 
                                    np.array([[693.12012738 / 2.5, 0.0, 355.44816971 / 2.5], 
                                            [0.000000, 693.12012738 / 2.5, 327.7015152 / 2.5], 
                                            [0.000000, 0.000000, 1.000000]]))
        print('111111111111111111111',goal)
        self.rcm_goal = np.matmul(basePSM_T_cam[:3,:3],goal)+basePSM_T_cam[:3,3]
        self.goal = goal
        # goal= np.array([0.054872,   0.0252584,   -0.131618])

        # self.rcm_goal=goal.copy()
        # self.goal=self.convert_pos(goal, cam_T_basePSM)
        print("goal_x: ", self.object_point[0])
        print("goal_y: ", self.object_point[1])
        print("goal_depth: ", depth[self.object_point[1]][self.object_point[0]])
        print("Selected Goal ecm: ",self.goal)
        print("Selected Goal rcm: ",self.rcm_goal)
        
        self.count=0
        self.manipulator = Manipulator()

        self.depth_init, self.seg_init = None, None
        self.start_time = 0
        self.end_time = 0
        
    def run_step(self, m = None):

        if self._finished:
            return True
        
        self.count+=1
        print("--------step {}----------".format(self.count))

        frame1=self.cap_0.read()
        frame2=self.cap_2.read()
        # frame1 = frame1[:,160:1120]
        # frame1 = frame1[::-1]
        # frame2 = frame2[:,160:1120]
        # frame2 = frame2[::-1]
        # frame1=cv2.resize(frame1, (800, 600))
        # frame2=cv2.resize(frame2, (800, 600))

        frame1, frame2 = my_rectify(frame1, frame2, self.fs)

        frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

        depth=self._get_depth(frame1, frame2)

        depth=cv2.resize(depth, self.img_size, interpolation=cv2.INTER_NEAREST)

        import copy
        if self.depth_init is None:
            self.depth_init = copy.deepcopy(depth)
        
#########################################################################
        print(depth[self.object_point[1]][self.object_point[0]])
#########################################################################
        frame1=cv2.resize(frame1, self.img_size)
        frame2=cv2.resize(frame2, self.img_size)
        
        seg=self._seg_with_fastsam(frame1,self.object_point)

        seg=np.array(seg==True).astype(int)
        if self.seg_init is None:
            self.seg_init = copy.deepcopy(seg)

        print("finish seg")

        robot_pose=self.p.measured_cp()
        robot_pos=robot_pose.p
        print("pre action pos rcm: ",robot_pos)
        robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])

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
        robot_rot=self.convert_rot(robot_rot, cam_T_basePSM)
        robot_rot=self.get_euler_from_matrix(robot_rot)
        robot_pos=self.convert_pos(robot_pos,cam_T_basePSM)
        print("pre action pos tip ecm: ",robot_pos)
        # end

        jaw=self.p.jaw.measured_jp()
        action=self._get_action(seg, depth ,robot_pos, robot_rot, jaw, self.goal)
        print("finish get action")
        print("action: ",action)
        #obtained_object_position=player.convert_pos(action, basePSM_T_cam)
        #print('obtained_object_position: ',obtained_object_position)
        #PSM2_pose=PyKDL.Vector(obtained_object_position[0], obtained_object_position[1], obtained_object_position[2])
        
        # 4. action -> state
        # state=self._set_action(action, robot_pos, np_m)
        print("finish set action")
        # print("state: ",state)
        #z_delta=state[2,-1]-pre_robot_pos[2]
        #state[2,-1]=pre_robot_pos[2]-z_delta
        
        # 5. move 
        angle = -np.pi / 4 
        R_z = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]])
        rotation_pose = np.array([[0.645682,    -0.759185,   0.0820535],
                                [-0.763369,   -0.639067,   0.0941291],
                                [-0.0190237,   -0.123415,   -0.992173]])
        rotation_pose = np.dot(rotation_pose, R_z)
        PSM2_rotate = PyKDL.Rotation(rotation_pose[0,0], rotation_pose[0,1], rotation_pose[0,2],
                                    rotation_pose[1,0], rotation_pose[1,1], rotation_pose[1,2],
                                    rotation_pose[2,0], rotation_pose[2,1], rotation_pose[2,2])
        # PSM2_rotate=PyKDL.Rotation(0.645682,    -0.759185,   0.0820535,
        #                             -0.763369,   -0.639067,   0.0941291,
        #                             -0.0190237,   -0.123415,   -0.992173)

        action_len=15
        action_split=action/action_len
        import time
        if self.start_time == 0 :
            self.start_time = time.time()
        for i in range(action_len):
        # 4. action -> state
            '''
            robot_pose=self.p.measured_cp()
            robot_pos=robot_pose.p
            robot_pos=np.array([robot_pos[0],robot_pos[1],robot_pos[2]])
            robot_pos=self.convert_pos(robot_pos,cam_T_basePSM)
            '''
            # print(robot_pos_new)
            state=self._set_action(action_split.copy(), robot_pos, np_m)
            
            PSM2_pose = PyKDL.Vector(state[0,-1], state[1,-1], state[2,-1])
            curr_robot_pos=np.array([state[0,-1], state[1,-1], state[2,-1]])

            move_goal = PyKDL.Frame(PSM2_rotate, PSM2_pose)
            #move_goal=PyKDL.Frame(robot_pose.M,PSM2_pose)
            #if count>7:
            #    break


            self.p.move_cp(move_goal).wait(1)
            #print('goal:',move_goal)
            #self.p.servo_cp(move_goal)
            if i==(action_len-1):
                break
            import time
            time.sleep(0.2)
            robot_pos=self.convert_pos(curr_robot_pos,cam_T_basePSM)

        print("finish move")
        print('is sccess: ',self.is_success(curr_robot_pos, self.rcm_goal))

        count = 0
        if self.is_success(curr_robot_pos, self.rcm_goal) or self.count > count:

            self._finished=True
            step_num = 20
            # grasp_offset = [0.0, -0.0013, -0.0]
            grasp_offset = [0, 0, -0.013]
            self.manipulator.grasp_peg(self.object_point, self.depth_init, self.seg_init, step_num, grasp_offset,rotation_pose)
            self.end_time = time.time()
            print('total time is :', self.end_time - self.start_time)

        return self._finished

if __name__=="__main__":

    player=VisPlayer()
    player.init_run()

    finished = False
    place_state = False

    while not finished:
        finished = player.run_step()

    player.cap_0.release()
    player.cap_2.release()
    