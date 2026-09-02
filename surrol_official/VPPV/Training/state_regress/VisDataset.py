from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pickle
import sys,os
import numpy as np
import cv2
from torchvision.transforms import Compose
# from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

class ObsDataset(Dataset):
    def __init__(self, opts, view_matrix, status='train', add_noise=False):
        print("loading {} dataset".format(status))
        self.opts=opts # data_dir, obs_list, seg_dir, img_dir
        self._view_matrix=view_matrix
        self.data_list=np.load(os.path.join(self.opts.data_dir,"{}_list.npy".format(status)))
        # for debug
        #self.data_list=np.load('/research/d1/rshr/arlin/regress_data/debug_list.npy')
        #seg_file=os.path.join(os.path.join(self.opts.data_dir,"test_seg.pkl"))
        self.add_noise=add_noise
        
        with open(opts.obs_list,"rb") as f:
            self.obs_list=pickle.load(f)
        seg_file=os.path.join(os.path.join(self.opts.data_dir,"{}_seg.pkl".format(status)))
        with open(seg_file, "rb") as f:
            self.seg_file_list=pickle.load(f)
        self.seg_list=[]
        self.img_list=[]
        self.state_list=[]

        if self.opts.use_exist_depth:
            depth_file=os.path.join(os.path.join(self.opts.data_dir,"{}_depth.pkl".format(status)))
            # for debug
            #depth_file=os.path.join(os.path.join(self.opts.data_dir,"test_depth.pkl"))
            with open(depth_file,"rb") as f:
                self.depth_list=pickle.load(f)
        
        # self.transform = Compose([
        #     Resize(
        #         width=518,
        #         height=518,
        #         resize_target=False,
        #         keep_aspect_ratio=True,
        #         ensure_multiple_of=14,
        #         resize_method='lower_bound',
        #         image_interpolation_method=cv2.INTER_CUBIC,
        #     ),
        #     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     PrepareForNet(),
        # ])

        # load seg , image
        #for curr_id in self.img_list.keys():
        for curr_id in self.data_list:
            if self.__len__()%100==0:
                print(self.__len__())
            
            #print(curr_id)
            #seg_file=os.path.join(opts.seg_dir,"seg_{}.npy".format(curr_id))
            #seg=np.load(seg_file)
            #self.seg_list.append(seg)
            # print(self.seg_file_list)
            self.seg_list.append(self.seg_file_list[curr_id])

            if self.opts.use_exist_depth:
                self.img_list.append(self.depth_list[curr_id])

            else: 
                img_file=os.path.join(opts.img_dir,"img_{}.png".format(curr_id))
                #print(img_file)
                raw_image = cv2.imread(img_file)
                #print(raw_image.shape)
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
                image = self.transform({'image': image})['image']
                self.img_list.append(image)

            obs=self.obs_list[curr_id]['obs']
            state=self.world2cam(obs)
            #state=obs.copy()
            regress_state=state[10:]
            # normalization of the angles =======================
            norm_rot=self.normlize_angles(state[16:])
            regress_state[6:]=norm_rot
            # normalization of the angles =======================
            #regress_state[:3]=state[7:10]
            self.state_list.append(regress_state)


    def world2cam(self, observation):
        
        robot_state=observation[:7]
        object_pos=observation[7:10]
        waypoint_pos=observation[13:16]
        waypoint_rot=observation[16:]
        robot_pos=self._world2cam_pos(robot_state[:3])
        robot_rot=self._world2cam_rot(robot_state[3:6])
        object_pos=self._world2cam_pos(object_pos)
        waypoint_pos=self._world2cam_pos(waypoint_pos)
        waypoint_rot=self._world2cam_rot(waypoint_rot)
        object_rel_pos=object_pos-robot_pos
        
        new_observation=np.concatenate([
            robot_pos, robot_rot, np.array([robot_state[-1]]),
            object_pos, object_rel_pos, waypoint_pos, waypoint_rot
        ])
        
        return new_observation

    def normlize_angles(self, x):
        return np.arctan2(np.sin(x),np.cos(x))

    def _world2cam_pos(self, pos):
    
        point_world = np.array([pos[0], pos[1], pos[2], 1])
        
        cam_pos=self._view_matrix @ point_world
        return np.array([cam_pos[0],cam_pos[1],cam_pos[2]])
        #return cam_pos
        

    def _world2cam_rot(self,euler_world):
        
        rot_matrix_world = np.array([[np.cos(euler_world[1])*np.cos(euler_world[2]), np.sin(euler_world[0])*np.sin(euler_world[1])*np.cos(euler_world[2]) - np.cos(euler_world[0])*np.sin(euler_world[2]), np.cos(euler_world[0])*np.sin(euler_world[1])*np.cos(euler_world[2]) + np.sin(euler_world[0])*np.sin(euler_world[2])],
                                    [np.cos(euler_world[1])*np.sin(euler_world[2]), np.sin(euler_world[0])*np.sin(euler_world[1])*np.sin(euler_world[2]) + np.cos(euler_world[0])*np.cos(euler_world[2]), np.cos(euler_world[0])*np.sin(euler_world[1])*np.sin(euler_world[2]) - np.sin(euler_world[0])*np.cos(euler_world[2])],
                                    [-np.sin(euler_world[1]), np.sin(euler_world[0])*np.cos(euler_world[1]), np.cos(euler_world[0])*np.cos(euler_world[1])]])
        # Extract rotation matrix from camera extrinsic matrix
        #print(self._view_matrix)
        rot_matrix_camera = self._view_matrix[:3, :3]
        # Apply camera extrinsic rotation to rotation matrix in world axis
        rot_matrix_camera = rot_matrix_camera @ rot_matrix_world
        # Convert rotation matrix to Euler angles in camera axis
        euler_camera = np.array([np.arctan2(rot_matrix_camera[2, 1], rot_matrix_camera[2, 2]),
                                np.arctan2(-rot_matrix_camera[2, 0], np.sqrt(rot_matrix_camera[2, 1]**2 + rot_matrix_camera[2, 2]**2)),
                                np.arctan2(rot_matrix_camera[1, 0], rot_matrix_camera[0, 0])])
        return euler_camera

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        batch = {}
        if not self.add_noise:
            batch["img"] = torch.from_numpy(self.img_list[index])
            batch["seg"] = torch.from_numpy(self.seg_list[index])
            batch["state"] = torch.from_numpy(self.state_list[index])
            return batch
        else:
            # add gaussian noise to the image
            img = self.img_list[index]
            noise_intensity = np.random.random()*0.003  # set noise intensity
            noise = np.random.normal(0, noise_intensity, img.shape)  # generate Gaussian noise
            noisy_img = img + noise  # add noise to the image
            batch["img"] = torch.from_numpy(noisy_img.astype(np.float32))

            # add random noise to the seg image by flipping pixel values
            seg = self.seg_list[index]
            flipping_ratio = np.random.random()*0.3 # generate random flipping ratio
            noise_mask = np.random.choice([0, 1], size=seg.shape, p=[1 - flipping_ratio, flipping_ratio])  # generate random noise mask with flipping ratio
            noisy_seg = np.abs(seg - noise_mask)  # flip pixel values randomly
            batch["seg"] = torch.from_numpy(noisy_seg.astype(np.int32))
            # print("noisy_seg type: ", noisy_seg.dtype)
            # print("seg type: ", seg.dtype)
            
            batch["state"] = torch.from_numpy(self.state_list[index])
            return batch

