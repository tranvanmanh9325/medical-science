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

