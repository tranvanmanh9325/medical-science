import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from regress_module import VisProcess, VisTR, VisRes
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
sys.path.append('./utils')



class vismodel(nn.Module):
    def __init__(
        self,
        opts
       
    ):
        super().__init__()
        self.opts=opts
        self.device=opts.device
       
        self.img_size=self.opts.img_size
        self.obj_num=1
        self.v_processor=VisRes()
        if not self.opts.use_exist_depth:
            self._load_dam()

    def _load_dam(self):
        encoder = 'vitb' # can also be 'vitb' or 'vitl'
        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()
        #self.depth_anything.to(self.device)

    def _get_depth_with_dam(self, img):
        '''
        input: rgb image 1xHxW
        '''
        
        #img=self.img_transform({'image': img})['image']
        #img=torch.from_numpy(img).unsqueeze(0)
        #img=transforms.Resize((518,518))(img)
        with torch.no_grad():
            depth = self.depth_anything(img)

        #print(depth.shape)
        depth = F.interpolate(depth[None], self.img_size, mode='bilinear', align_corners=False)[0]
        depth_min = torch.amin(depth, dim=(1, 2), keepdim=True)
        depth_max = torch.amax(depth, dim=(1, 2), keepdim=True)
        depth = (depth - depth_min) / (depth_max - depth_min)

        return depth

    
    def normlize_angles(self, x):
        return np.arctan2(np.sin(x),np.cos(x))
    
    def get_action(self, state, noise=False, v_action=False):
        #print("get_action")
        
        if not v_action:
            return super().get_action(state, noise)
        #self.count+=1

        self.v_processor.eval()
        
        rgb=self.to_torch(state['depth']).unsqueeze(0)
        
        depth=self._get_depth_with_dam(rgb)[0] #tensor
        depth_norm=self.depth_norm.normalize(depth.reshape(-1,256*256),device=self.device).reshape(1,256,256)
        seg=self.to_torch(transitions['seg'])
        seg_d=torch.concat((seg,depth_norm))
        
        inputs=seg_d.unsqueeze(0).float().to(self.device) # B 2 256 256
        #print(inputs.shape)
        
        with torch.no_grad():
            v_output=self.v_processor(inputs).squeeze() # 9

            #v_save=v_output.cpu().data.numpy()
            #np.save('test_record/v_output.npy', v_save)

            o, g = state['observation'], state['desired_goal']
            g=self.g_norm.normalize(g)
            #print("g: ",g)
            g_norm=torch.tensor(g).float().to(self.device)
            #print("g_norm: ",g_norm)
            
            if not self.regress_rbt_staet:
                robot_state=torch.tensor(o[:7]).to(self.device)
                #pos=v_output[:3]
                #rel_pos=pos-robot_state[:3]
                rel_pos=v_output[:3*self.obj_num]
                new_pos=robot_state[:3]+rel_pos[:3]
                
                if self.obj_num>1:
                    for i in range(1, self.obj_num):
                        pos=robot_state[:3]+rel_pos[i*3:3*self.obj_num]
                        new_pos=torch.concat((new_pos,pos))
                
                waypoint_pos_rot=v_output[3*self.obj_num:]
                
                
                #o=torch.from_numpy(np.concatenate([o[:,7:10].copy(),o[:,13:19]].copy(),axis=1)).to(self.device)
                
                #o_new=torch.concat((,o),dim=1) # B 19
                o_new=torch.concat((robot_state, new_pos))
                o_new=torch.concat((o_new, rel_pos))
                o_new=torch.concat((o_new, waypoint_pos_rot))
                o_norm=self.o_norm.normalize(o_new,device=self.device)
            else:
                o_norm=self.o_norm.normalize(v_output,device=self.device)
                o_norm=torch.tensor(o_norm).float().to(self.device)
                
            input_tensor=torch.concat((o_norm, g_norm), axis=0).to(torch.float32)
            #save_input=input_tensor.cpu().data.numpy()
            #np.save('test_record/actor_input.npy', save_input)
            #exit()
            
            action = self.actor(input_tensor).cpu().data.numpy().flatten()

        self.v_processor.train()
        return action
            

    def forward(self, seg, rgb, v_gt):
        if self.opts.use_exist_depth:
            d=rgb
        else:
            d=self._get_depth_with_dam(rgb)

        seg_d=torch.concat((seg.unsqueeze(1),d.unsqueeze(1)),dim=1)

        output=self.v_processor(seg_d)
        #print('output: ',type(output))
        #print('v_gt: ',v_gt.shape)
        
        v_loss=0.
        
        pos_loss=F.mse_loss(output[:,:3*self.obj_num],v_gt[:,:3*self.obj_num])
        v_loss+=pos_loss
        w_pos_loss=F.mse_loss(output[:,3*self.obj_num: 3*self.obj_num+3],v_gt[:,3*self.obj_num: 3*self.obj_num+3])
        v_loss+=w_pos_loss
        w_rot_loss=F.mse_loss(output[:,3*self.obj_num+3:],v_gt[:,3*self.obj_num+3: ])
        v_loss+=w_rot_loss
        metrics = AttrDict(
            v_pos=pos_loss.item(),
            w_pos_loss_loss=w_pos_loss.item(),
            w_rot_loss=w_rot_loss.item(),
            v_loss=v_loss.item()

        )
       
        return metrics, v_loss
    
    def get_obs(self, seg, rgb):
        if self.opts.use_exist_depth:
            d=rgb
        else:
            d=self._get_depth_with_dam(rgb)
        #d=self._get_depth_with_dam(rgb)
        seg_d=torch.concat((seg.unsqueeze(1),d.unsqueeze(1)),dim=1)#.to(self.device)

        output=self.v_processor(seg_d)
        #print(output.shape)
        return output