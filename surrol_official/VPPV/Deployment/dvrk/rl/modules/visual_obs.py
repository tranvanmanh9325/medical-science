import numpy as np
import torch
import torch.nn as nn
import timm
import torchvision
from torchvision.transforms import Resize

class VisProcess(nn.Module):
    
    def __init__(self,vo_dim):
        super().__init__()
        self.down=nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=3),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=3),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),   
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        
        self.up=nn.Sequential(
            nn.Linear(in_features=64*14*14, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=vo_dim)
        )
        #self.update_epoch = agent_cfg.update_epoch
        
    def forward(self,x):
        
        x=self.down(x) 
        #print(x.shape) # B 64 14 14
        x=torch.flatten(x, start_dim=1)
        x=self.up(x)
        return x
        
    '''
    def get_v_obs(self, v):
        
        depth=torch.from_numpy(depth).cuda()
        mask=torch.from_numpy(mask).cuda()
        robot_state=torch.from_numpy(robot_state).cuda()
        x=torch.concat((depth.unsqueeze(0),mask.unsqueeze(0)), dim=0)
        out_obs=self(x) # obj_pos, waypoint_pos, waypoint_rot
        rel_pos=out_obs[0:3]-robot_state[0:3]
        final_obs=torch.concat((out_obs.squeeze(),rel_pos.squeeze(),robot_state))
        
        return final_obs, out_obs[0:3].clone().detach().cpu().numpy()
    
    
    
    def update(self, replay_buffer, demo_buffer):
        #for name, params in self.v_processor.named_parameters():
        #    print('-->name: ', name, '--grad_requires: ', params.requires_grad, \
        #          '-->grad_value: ', params.grad)
        #print('-->agent.update2: ', obs.requires_grad)
        #metrics.update(self.update_vprocesser(obs, obs_demo))
        v_loss=0.
        for i in range(self.update_epoch):
            obs, action, reward, done, next_obs, _ = self.get_samples(replay_buffer)
            #print('-->agent.update1: ', obs.requires_grad)
            #print('-->obs_next agent.update1: ', next_obs.requires_grad)
            obs_demo, action_demo, reward_demo, done_demo, next_obs_demo, next_action_demo = self.get_samples(demo_buffer)
            v_loss+=F.mse_loss(obs,obs_demo)
            
        return v_loss
        #return metrics
    '''


class VisTR(nn.Module):
    def __init__(self):
        super(VisTR, self).__init__()
        # Load a pre-trained Vision Transformer
        self.pretrained = timm.create_model('vit_small_patch16_224', pretrained=True)
        # Modify the first layer to accept 2-channel input
        self.pretrained.patch_embed.proj = nn.Conv2d(2, self.pretrained.embed_dim, kernel_size=(16, 16), stride=(16, 16))
        # Add a regression head
        self.pretrained.head = nn.Linear(self.pretrained.embed_dim, 9)
        # Add a resize layer at the beginning
        self.resize = Resize((224, 224))

    def forward(self, x):
        x = self.resize(x)
        return self.pretrained(x)
    
class VisRes(nn.Module):
    def __init__(self):
        super(VisRes, self).__init__()
         # Load a pre-trained ResNet18
        self.pretrained = torchvision.models.resnet18(pretrained=True)
        # Modify the first layer to accept 2-channel input
        self.pretrained.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Remove the fully connected layer
        self.pretrained.fc = nn.Identity()

        # Add a new fully connected layer, but delay its creation until the forward pass
        # This allows us to dynamically adjust the input size based on the actual output size of the ResNet
        self.fc = nn.Linear(512,9)

    def forward(self, x):
        x = self.pretrained(x)
        # If the fully connected layer has not been created yet, create it now
        
        x = self.fc(x.view(x.size(0), -1))
        return x
